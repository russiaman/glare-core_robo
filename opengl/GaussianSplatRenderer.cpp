/*=====================================================================
GaussianSplatRenderer.cpp
-------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatRenderer.h"


#include "GaussianSplatSaturationGrid.h" // SESSION074
#include "IncludeOpenGL.h"
#include "OpenGLEngine.h"
#include "RenderBuffer.h" // For the accumulation buffer's dimensions in getDiagnostics().
#include "OpenGLMeshRenderData.h"
#include "OpenGLShader.h"
#include "OpenGLTexture.h"
#include "VAO.h"
#include "VBO.h"
#include "VertexBufferAllocator.h"
#include "../maths/Matrix4f.h"
#include "../maths/mathstypes.h"
#include "../maths/vec2.h" // For the screen-space ellipse axes in splatFootprint().
#include "../utils/ArrayRef.h"
#include "../utils/AtomicInt.h" // SESSION080 DIAGNOSTIC: counts saturation phases running at once - see gs_sat_phases_running.
#include "../utils/BitUtils.h"
#include "../utils/ConPrint.h"
#include "../utils/Exception.h"
#include "../utils/RefCounted.h"
#include "../utils/StringUtils.h"
#include "../utils/Clock.h" // SESSION080 DIAGNOSTIC: Clock::getCurTimeRealSec() stamps GaussianSplatUnculledFrontier::built_time_real_s from the worker thread - see [gsr-sat-apply].
#include "../utils/Sort.h"
#include "../utils/Task.h"
#include "../utils/TaskManager.h"
#include "../utils/Timer.h" // For timing the synchronous traversal getFrustumStructureReport() runs.
#include "../utils/ThreadSafeRefCounted.h"
#include "../utils/Vector.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_map> // For grouping near-duplicate splats in getFrustumStructureReport().


// How far the camera has to move before a cloud's depth order is worth recomputing.  Camera rotation is deliberately
// not considered: the sort is by distance from the camera, which rotating on the spot doesn't change.
//
// The threshold scales with distance to the cloud.  Moving 10cm visibly reorders a cloud you are standing inside, but
// changes nothing about the order within one 500m away, and a single world-wide constant would re-sort every splat in
// the world for the latter.  The floor is that old constant, so a cloud you are close to behaves exactly as before.
static const float min_resort_move_threshold_ws = 0.1f;
static const float resort_threshold_dist_fraction = 0.01f; // SESSION063 K3 TEST: lowered 0.05->0.01 to re-kick U(P) ~5x more often on translation, to isolate whether translation holes are discrete-step staleness (helped) or the 450ms traversal latency under continuous motion (not helped). Revert if inconclusive.

// How many depth sorts may be in flight at once.  Each holds a scratch allocation proportional to its cloud, so this
// caps sort memory at roughly this many times the largest cloud, rather than letting it scale with the world.
static const int max_concurrent_sorts = 2;

// How many samples of a cloud's draw order the frustum-aware slicing keeps - see
// GaussianSplatRenderer::getVisibleSlicingEnabled().  A fixed budget rather than a fixed stride, so the per-frame cost
// of re-testing them against the frustum does not scale with the cloud: it is the same few thousand point-in-frustum
// tests whether the cloud holds ten thousand splats or ten million.  4096 samples place a boundary to within a
// four-thousandth of the draw order, which is far finer than a handful of slice boundaries can use.
static const int max_slice_samples = 4096;

// SESSION055/058: one shared Timer for ALL traversal-pipeline diagnostic prints in this file (kick log, rot-blocked
// log, session058 CPU-cost profiling) - see [[feedback_shared_diag_timer]]. Declared at file scope, above every diag
// print site (some of which - e.g. fillTraversalScratch() - are defined earlier in the file than kickOffTraversals()),
// so every timestamp shares the same zero point and lines from different sites stay chronologically comparable.
static Timer diag_timer;
// SESSION072: filter_debug_log/cpu_prof_log/kick_debug_log (below, near kickOffTraversals()) moved from build-time
// consts to live GaussianSplatRenderer members - see getFilterDebugLog() etc. in the header. [gsr-filter-kick] /
// [gsr-filter-drain] trace kickOffFilters()/drainFilterResults(); [gsr-prof] traces fillTraversalScratch() and the
// traversal-result VBO upload (session058 snapshot §3).


// SESSION081 ETAP 4: a minimal half-float codec for the packed occluder record below. Deliberately restricted: the
// three quantities it stores are all strictly positive and are clamped into the half's NORMAL range first, so there is
// no sign case, no denormal case and no inf/nan case, and each direction is one shift and one add. (IlmBase's half
// would do the job, but its decode is a 128KB table lookup - exactly the wrong shape for a loop whose entire purpose is
// to touch one cache line and nothing else.)
static const float gs_sat_half_min = 6.10352e-5f; // 2^-14, the smallest normal half.
static const float gs_sat_half_max = 65504.f;     // The largest finite half.

// Having no denormals means the representable window is [6.1e-5, 65504], which is not where two of the three stored
// quantities live, so those two are PRE-SCALED into it. This matters in one direction only: clamping a value UP
// inflates an occluder and can prune visible geometry, while clamping DOWN merely under-occludes. With these scales
// both clamps are harmless - the bottom one lands at sub-micron sizes (which occlude nothing either way) and the top
// one clamps down. Found by the session081 pack verification: without them, a splat with an extreme axis ratio has
// sqrt(a*b) BELOW 6.1e-5 and was being rounded up to it, a silent over-estimate of exactly the wrong sign.
static const float gs_sat_half_g_scale = 1024.f;     // Radius window becomes [6.0e-8, 63.9] m.
static const float gs_sat_half_alpha_scale = 1024.f; // Opacity window becomes [6.0e-8, 63.9], and opacity is <= 1.

static inline uint16 gsSatPackHalf(float v)
{
	const float clamped = myClamp(v, gs_sat_half_min, gs_sat_half_max);
	uint32 bits;
	std::memcpy(&bits, &clamped, 4);
	return (uint16)(((bits + 0x1000u) >> 13) - (112u << 10)); // Exponent rebias 127 -> 15, mantissa 23 -> 10 bits, rounded.
}

static inline float gsSatUnpackHalf(uint16 h)
{
	const uint32 bits = ((uint32)h + (112u << 10)) << 13;
	float v;
	std::memcpy(&v, &bits, 4);
	return v;
}


// SESSION081 ETAP 4: everything the saturation gather needs to know about one splat, in 12 bytes fetched as ONE cache
// line - replacing the three scattered reads (scales 12B, rotations 16B, alpha 4B, in three separate ~30M-entry arrays)
// that measured as essentially the WHOLE cost of the gather: 169.8ms -> 43.8ms with them removed (plan doc, P4.1). It
// also shrinks the cached geom rather than growing it - 12B/node against the 32B/node those three arrays cost.
//
// The stored model is the disc approximation validated in P4.2 (see the plan doc for its quality cost and the two ways
// back if it ever bites), written in the form the gather wants:
//
//   r = g * (1 + c^2*(inv_t^2 - 1))^(1/4),   c = |d . n|,   g = sqrt(a*b),   inv_t = a/b
//
// with a = sqrt(s_mid*s_max), b = s_min and n = the s_min axis. Face-on (c=1) gives a, edge-on (c=0) gives sqrt(a*b) -
// the same two sanity checks gsSatProjectedRadius() satisfies.
//
// Quantisation cost, measured against the unquantised disc model over 400k splats spanning 0.1mm-30m and axis ratios
// to the measured 63000:1: median 0.02%, p99.99 0.35%, worst 2.2%. The tail is the snorm16 normal, not the halves: a
// needle seen near edge-on has a genuinely steep dr/dc. It sits an order of magnitude below the disc model's OWN
// approximation error, which the owner validated visually - so this quantisation is not what would bite first.
struct GsSatPackedOccluder
{
	int16 nx, ny, nz;   // Thin-axis unit normal, snorm16 (n * 32767). Only |d.n| is ever used, so the sign is free.
	uint16 g_half;      // sqrt(a*b) * gs_sat_half_g_scale. One sigma, world space.
	uint16 inv_t_half;  // a/b, unscaled - always >= 1 (a = sqrt(s_mid*s_max) >= s_mid >= s_min = b), so it needs no room below 1.
	uint16 alpha_half;  // The STORED opacity * gs_sat_half_alpha_scale: alpha_gain/alpha_gamma are runtime knobs, so the gather applies them.
};

static_assert(sizeof(GsSatPackedOccluder) == 12, "GsSatPackedOccluder must stay 12 bytes - the whole point is one cache line per node.");


// SESSION081 ETAP 4: builds one record from a splat's world-space scales/rotation/alpha. Runs once per splat per
// topology change (see getOrBuildCachedGeom()), never per build.
static inline GsSatPackedOccluder gsSatPackOccluder(const Vec3f& sc, const Vec4f& rot, float alpha)
{
	int min_axis = 0;
	if(sc[1] < sc[min_axis]) min_axis = 1;
	if(sc[2] < sc[min_axis]) min_axis = 2;
	const float s_min = myMax(sc[min_axis], 1.0e-8f); // Same degenerate-axis floor gsSatProjectedRadius() uses.
	float s_p = 0.f, s_q = 0.f; // The two non-thin axes, in whatever order - only their product is wanted.
	bool first = true;
	for(int k=0; k<3; ++k)
		if(k != min_axis)
		{
			if(first) { s_p = myMax(sc[k], 1.0e-8f); first = false; }
			else        s_q = myMax(sc[k], 1.0e-8f);
		}
	const float a = std::sqrt(s_p * s_q); // = sqrt(s_mid*s_max): preserves the true face-on area, which goes as s_mid*s_max.

	// The thin axis in world space is the min_axis'th column of the quaternion's rotation matrix - same (x, y, z, w)
	// convention, and the same three columns, gsSatProjectedRadius() dots against.
	const float qx = rot[0], qy = rot[1], qz = rot[2], qw = rot[3];
	const float qxx = qx*qx, qyy = qy*qy, qzz = qz*qz;
	const float qxy = qx*qy, qxz = qx*qz, qyz = qy*qz, qwx = qw*qx, qwy = qw*qy, qwz = qw*qz;
	float nx, ny, nz;
	if(min_axis == 0)      { nx = 1.f - 2.f*(qyy + qzz); ny = 2.f*(qxy + qwz);       nz = 2.f*(qxz - qwy); }
	else if(min_axis == 1) { nx = 2.f*(qxy - qwz);       ny = 1.f - 2.f*(qxx + qzz); nz = 2.f*(qyz + qwx); }
	else                   { nx = 2.f*(qxz + qwy);       ny = 2.f*(qyz - qwx);       nz = 1.f - 2.f*(qxx + qyy); }

	const float len_sq = nx*nx + ny*ny + nz*nz; // The stored quaternion is unit only to float precision, and snorm16 assumes a unit vector.
	const float inv_len = len_sq > 1.0e-12f ? (1.f / std::sqrt(len_sq)) : 0.f;

	GsSatPackedOccluder p;
	p.nx = (int16)myClamp((int)(nx * inv_len * 32767.f + (nx >= 0.f ? 0.5f : -0.5f)), -32767, 32767);
	p.ny = (int16)myClamp((int)(ny * inv_len * 32767.f + (ny >= 0.f ? 0.5f : -0.5f)), -32767, 32767);
	p.nz = (int16)myClamp((int)(nz * inv_len * 32767.f + (nz >= 0.f ? 0.5f : -0.5f)), -32767, 32767);
	p.g_half = gsSatPackHalf(std::sqrt(a * s_min) * gs_sat_half_g_scale);
	p.inv_t_half = gsSatPackHalf(a / s_min);
	p.alpha_half = gsSatPackHalf(alpha * gs_sat_half_alpha_scale);
	return p;
}


// SESSION058: immutable, cacheable copy of one cloud's world-space node positions + feature_size, shared (via Reference<>)
// between every traversal kicked off since the cloud's last structural change, instead of each kick paying its own
// memcpy. Safe to share across the worker and later kicks because nothing ever mutates an instance of this class after
// it's built: bakeMember() (the only writer of the LIVE cloud.positions/feature_size arrays) always runs on a code path
// that bumps SplatCloud::topology_generation in the same call (appendMemberToCloud(), rebuildCloud(), and - since
// session057 - updateObjectTransform()'s re-bake), and fillTraversalScratch() only reuses a cached instance when its
// cloud's topology_generation still matches the one it was built against. A generation mismatch means a *new* instance
// is built and cached; the old one lives on, unmodified, for as long as some in-flight traversal (or a diagnostics
// dump) still holds a reference to it - see SplatCloud::cached_traversal_geom.
class GaussianSplatCachedGeom : public ThreadSafeRefCounted
{
public:
	js::Vector<Vec3f, 16> positions;
	js::Vector<float, 16> feature_size;
	js::Vector<float, 16> cull_radius; // SESSION059: world-space enclosing-sphere radius per node - see SplatCloud::cull_radius and GaussianSplatLodNode::bounding_radius_os.

	// SESSION081 ETAP 4: the saturation gather's per-splat input, packed - see GsSatPackedOccluder. Replaces the
	// separate scales (Vec3f), rotations (Vec4f) and alpha (float) arrays that used to live here; those three existed
	// only to feed that one loop, and feeding it from three ~30M-entry arrays cost three cache misses per node.
	//
	// SESSION077's point still stands and is what this record encodes: the grid used to treat every occluder as an
	// ISOTROPIC disc of radius max(scale.xyz). That is the right bound for CULLING (the splat cannot reach further in
	// any direction) and the wrong quantity for OCCLUSION, which needs the cross-section as seen from the anchor.
	// Measured on the owner's interior: mean max(scale)/min(scale) = 101, 67% of occluders past 10:1, worst 63000:1 -
	// so the isotropic disc overstated most footprints by orders of magnitude, by a factor that varies with viewing
	// angle, which is what made an opaque wall's mask break up by distance-from-normal instead of being uniform.
	js::Vector<GsSatPackedOccluder, 16> sat_occl;
};


// SESSION063: the orientation-independent unculled LoD frontier U(P), cached so a pure rotation can derive its draw list
// with a cheap per-orientation frustum filter instead of a fresh ~450ms traversal (session062 §9.3 - the split follows
// from S(P,R) = stable_frustum_filter(U(P), R), the traversal's split/keep decision and sort key depending only on
// camera position P, orientation R only on the frustum test). Built by GaussianSplatLodTraversalTask with
// frustum_cull_enabled = false, so `indices` is the full unculled frontier in front-to-back order; px/py/pz are the
// parallel SoA world positions and `radius` the per-node cull_radius, both gathered on the worker (the scattered read
// into the cloud's 30M-entry arrays costs ~50ms - measured session063 - and must stay off the main thread). The filter
// tests each node's centre against the current frustum with its own radius as the only margin (mr.S point 5: an already
// finally-selected node needs only its own 3-sigma footprint, not its whole subtree's bounding sphere), and a
// subsequence of a distance-sorted list stays sorted, so no re-sort is needed. Immutable once built; validity is keyed
// by the fields below (see drainTraversalResults()/GaussianSplatFilterTask).
class GaussianSplatUnculledFrontier : public ThreadSafeRefCounted
{
public:
	js::Vector<uint32, 16> indices;       // Frontier as cloud-array indices, front-to-back (nearest first).
	js::Vector<float, 16> px, py, pz;     // SoA world positions parallel to indices.
	js::Vector<float, 16> radius;         // Per-node cull_radius (world space), the filter's per-node plane margin.

	// SESSION063 K4: 1.0 for a coarse-floor node, 0.0 for a fine node - parallel to indices. Fine and coarse are merged
	// into ONE globally distance-sorted list (see the traversal), so the draw is front-to-back across both layers and a
	// near coarse node correctly occludes a far fine one. The filter reads this per node to dilate coarse nodes wider than
	// fine (coarse nodes are few/cheap, so a wide edge margin costs little) - which lets the dense fine set stay at a
	// tight, cheap dilation without leaving holes on motion. See filterUnculledFrontier() / kickOffFilters().
	js::Vector<float, 16> is_coarse;

	// SESSION085 ETAP 6: the prune's diagnostics (tested/dropped/aggr, the graded-read counters, apply timing and
	// concurrency) went with the prune. What the saturation stage does now is visible on [gsr-traversal]'s sat_bias=
	// instead - it is a traversal property, not a separate stage's.

	// Key this frontier was built for; drainTraversalResults() checks these before reusing it, so a settings change that
	// alters the selection can't be answered from a stale U(P).  Orientation is deliberately NOT here - that's the point.
	uint64 topology_generation;
	Vec4f anchor_pos_ws;
	float pixel_scale_limit;
	size_t max_splats_budget;
	float max_layer_density;
	int max_tree_depth;
	float focal_px;
	// SESSION085 ETAP 3: which barrier's LoD bias shaped this selection, identified by the barrier's own build timestamp,
	// or 0.0 when the bias was OFF and therefore had no influence.
	//
	// SESSION086: this is NOT part of the key set above any more. It was added to it in etap 3 and removed here - the
	// full argument sits at the reuse acceptance check in GaussianSplatLodTraversalTask's constructor, but the short
	// form is that the barrier cannot move the CUT (the cut predicate does not read it, and the bias stop is suppressed
	// on straddling nodes), only the detail level below it - and requiring it to match meant the far block was
	// inheritable only while the barrier held still, which at any usable R is never.
	//
	// Still stamped, and still read by the sort-staleness diagnostic in run(): that one measures how far the selection
	// moves between traversals, and two selections cut under different barriers would report the barrier, not the motion.
	double sat_bias_barrier_time;

	// SESSION080 DIAGNOSTIC: Clock::getCurTimeRealSec() at the moment this frontier was built on the worker thread (set
	// alongside anchor_pos_ws below) - answers "how stale is the prune drainTraversalResults() is about to apply", which
	// cloud->last_traversal_kick_time_s cannot: that field gets overwritten by kickOffTraversals() the moment message 1
	// clears traversal_in_flight, which can happen (a new kick for the same cloud) before message 2 - the saturation
	// follow-up carrying THIS frontier - has even arrived. See [gsr-sat-apply] in drainTraversalResults().
	double built_time_real_s;

	// SESSION081: the calibration knobs (thr/sub/R/close) and erosion stats (er_ceil/er_win/er_radmax/cl) moved to
	// GaussianSplatSaturationBarrier alongside sat_depth - they describe how the BARRIER was built, not this frontier.
	// See [gsr-sat-build].

	// SESSION080 DIAGNOSTIC: splits the traversal task's own cost (everything above the saturation stage) into the tree
	// walk/selection (the while(!stack.empty()) loop in run()) versus the front-to-back radix sort of its output - see
	// [gsr-traversal]. Answers whether a "keep the far frontier, only re-walk/re-sort the near shell" scheme (discussed
	// re: the region-radius topology) is worth it for the walk, the sort, or both - session054's own note that the sort
	// was ~76% of traversal time predates the switch from std::sort to radix and was never re-measured after.
	double expand_ms; // The DFS/selection loop: frustum cull, coarse capture, pixel_scale convergence checks, cap tests.
	double sort_ms;   // Sort::floatKeyAscendingSort() over the selection, plus unpacking into output/coarse_flags.
	// SESSION080 DIAGNOSTIC (plan2 §4.1): the share of sort_ms that is just allocating the sort's scratch buffer, which
	// is a local sized to the selection - so a fresh ~106MB allocation per traversal at full frontier size, first-touch
	// page faults included. Same question, and the same candidate fix (pool it on the traversal scratch), as
	// expand_splice_reserve_ms - measured together because one rebuild answers both.
	double sort_alloc_ms;
	// SESSION087: sort_ms is not one thing. Session086 recorded it as a serial radix that "was never parallelised", which
	// the code contradicted - the parallel branch has been taken since session080 - and reading the total as if it were
	// only the sort hid that a quarter of it was a serial unpack loop, plus a copy-back that bought nothing. Kept, unlike
	// that session's other diagnostics, because the sort block is an active work area and this is the split that makes
	// its cost readable: an interleaved A/B over these two measured -18 ms interior / -23 ms exterior per traversal.
	double sort_radix_ms;        // The sort call itself, nothing else.
	double sort_unpack_ms;       // Splitting the packed idx into output/coarse_flags. Parallel since session087.
	// SESSION080 DIAGNOSTIC: why the parallel expand did or did not pay off - see expandParallel(). The pair that matters
	// is sum vs max: task_sum_ms is the total work the tasks did, task_max_ms the longest single one, i.e. the critical
	// path. sum/max is the parallelism actually available in the seed split. If max ~= expand_ms one subtree dominates
	// and the seeds need cutting finer; if sum ~= expand_ms while max is much smaller, the tasks ran one after another
	// and the problem is the dispatch, not the split.
	size_t expand_seeds;        // Subtree roots the serial prologue handed to the pool.
	size_t expand_num_tasks;    // Tasks they were distributed into.
	double expand_prologue_ms;  // The serial top-of-tree walk before any of them started.
	double expand_task_max_ms;
	double expand_task_sum_ms;

	// SESSION086 DIAGNOSTIC: the two things par= cannot tell apart. par is task_sum/task_max, and a task's run_ms is
	// the window between a thread picking it up and the shared seed cursor running dry - so a task that no thread ever
	// got to while work remained contributes ~0 to the sum and drags par down exactly as an indivisible subtree would.
	// expand_workers counts the tasks that took at least one seed (how many threads we actually got); expand_seed_max_ms
	// is the costliest single seed (whether any one subtree is genuinely too big to split). The two call for opposite
	// fixes, so measuring them apart is the whole point of printing both.
	size_t expand_workers;
	double expand_seed_max_ms;
	// SESSION086: how finely the split ran, so the closed loop in updateExpandSeedTarget() can be read off the log -
	// expand_seed_max_ms above is the quantity it steers, this is where it steered to.
	size_t expand_seed_target;
	// SESSION080 DIAGNOSTIC (plan2 §4.1): the serial concatenation of the tasks' per-thread output vectors into
	// `decorated`, AFTER runTaskGroup() returns - not covered by task_max_ms/task_sum_ms, which are the tasks' own
	// wall-clock. Suspected (from expand_ms minus an estimate of the parallel section) to be roughly half of expand
	// at reuse=25, but that was an inference, not a measurement - see the field's use for why it needed its own timer
	// before anything was optimised on the strength of it.
	//
	// Split in two because the two halves want opposite fixes and the aggregate cannot tell them apart:
	//   reserve_ms - growing `decorated` to the final size. It is a local in run(), so this is a FRESH allocation every
	//     traversal, and at 13.2M nodes x 8 bytes it is a ~106MB one whose first-touch page faults are paid here. If
	//     this dominates, the fix is to pool the buffer on GaussianSplatLodTraversalScratch - what session079 did for
	//     sat_occluder_recs, EXCEPT that pooling was reverted in session081: it is only safe for a buffer fully
	//     consumed before result_queue->enqueue() publishes the frontier (like `decorated` here). sat_occluder_recs
	//     and sat_keep_mask are read by the LATER saturation phase, which runs after that enqueue - once scratch is
	//     back in the free pool a faster/second traversal can resize the same buffer out from under the first one's
	//     still-running saturation phase. See selected_indices/sort_scratch's own comment on the same struct for the
	//     line that still correctly distinguishes the two cases, and the sat_phase_scope comment at this bug's fix site.
	//   copy_ms - the element-by-element push_back loop itself. If THIS dominates, the fix is a memcpy per task block
	//     (offsets are known from the same prefix pass that computes `total`), optionally dispatched across the pool.
	double expand_splice_reserve_ms;
	double expand_splice_copy_ms;

	// SESSION080 DIAGNOSTIC: building this frontier's SoA (indices/px/py/pz/radius/is_coarse) from the sorted selection.
	// Sat between sort_ms and sat_gather_ms and was covered by NEITHER, so the stage timings did not sum to the traversal
	// task's real duration. It is a scattered read of positions[idx]/cull_radius[idx] over the whole selection - the same
	// shape as, and over the same indices as, the occluder gather that follows it.
	double soa_ms;

	// SESSION080 DIAGNOSTIC: camera-distance at the 10/25/50/75/90th percentile of this frontier, in metres. Read
	// straight out of the sorted selection, so it costs five indexings and five sqrts.
	//
	// This is the sizing question for any "reuse the far frontier, re-expand only the near shell" scheme: the owner's
	// hypothesis is about VISUAL importance (the nearest splats dominate what a re-sort must get right), but expand's
	// COST is distributed the other way - the tree unfolds deepest close to the camera. If most of the pool already
	// lies near, reusing the far tail saves little, and the scheme is not worth its complexity. These five numbers say
	// which regime the scene is in before anything is built.
	float dist_pctile[5];

	size_t traversal_output_n; // SESSION080 DIAGNOSTIC: decorated.size() at the point expand_ms/sort_ms were measured -
	// i.e. what the DFS actually walked/sorted, BEFORE the saturation prune. indices.size() is NOT this on the pruned
	// replacement (uf2): it shrinks to pool_after, which would misattribute expand_ms/sort_ms's cost to the wrong N.


	// SESSION080 DIAGNOSTIC (plan doc session080-plan.md STEP A): how much this frontier's front-to-back order has
	// moved since prev_frontier - the cloud's cached_ufrontier at kick time, i.e. whatever was
	// actually being drawn just before this traversal started. Matches nodes by identity (cloud_idx = member
	// offset + tree_local_idx), NOT by array position - see the computation site in run() for why. Zero/default
	// when there was no previous frontier (first kick for this cloud) or its topology_generation didn't match
	// (a structural change makes cloud_idx comparisons meaningless).
	//
	// Answers the plan doc's open question: under NORMAL WALKING (not a teleport - teleport tests read zero here
	// by construction, see the doc's §8) does the sort order drift enough between kicks to justify a depth-bucketed
	// segment scheme (plan §7.2), or does the existing ~1% resort_threshold_dist_fraction re-kick policy already
	// keep it close enough that none of that complexity is worth building.
	double sort_staleness_delta_ws;  // Camera displacement (m) since the previous frontier's anchor_pos_ws.
	size_t sort_staleness_prev_n;    // Size of the previous frontier compared against.
	size_t sort_staleness_common_n;  // Nodes present (same cloud_idx) in both frontiers - the denominator below.
	size_t sort_staleness_max_disp;  // Largest |new rank - old rank| among common nodes.
	double sort_staleness_mean_disp; // Mean |new rank - old rank| among common nodes.
	size_t sort_staleness_gt1k;      // Common nodes displaced more than 1000 positions.
	size_t sort_staleness_gt10k;     // Common nodes displaced more than 10000 positions.
	double sort_staleness_ms;        // Cost of computing the above, so this diagnostic's own overhead is visible.

	// SESSION080 §4.3: the far half of this frontier, held BY REFERENCE instead of copied into the arrays above.
	//
	// A frontier is therefore two segments: this object's own arrays [0, indices.size()), which the traversal that built
	// it walked fresh, followed by far_block's arrays [0, far_block->indices.size()), which it inherited untouched. Every
	// consumer that streams a frontier has to walk both - see filterUnculledFrontier(), and materialiseFlat() for the
	// consumers that still want one flat array.
	//
	// What makes the two halves fit together exactly is that the cut between them is FROZEN for the block's whole life.
	// far_block->anchor_pos_ws and far_block->reuse_split_dist_used are the anchor and distance the cut predicate
	// P(node) = dist(anchor, centre) - cull_radius >= split was evaluated against when the block was built, and every
	// later traversal in the generation re-evaluates that same P against those same frozen values. So all of them stop
	// at exactly the same nodes, and the block holds exactly the selection below that cut - no per-traversal scan of the
	// predecessor is needed, and none is done. (The previous cut of this, session080 STEP B, re-derived the cut from the
	// PREVIOUS traversal's own anchor, which moved every kick and so forced a full rescan and copy every time - 44.7ms.)
	//
	// Null means the frontier is self-contained: the whole selection is in its own arrays. That is the state of a
	// generation's first traversal, of every traversal while the reuse knob is 0, and of a far block itself.
	Reference<GaussianSplatUnculledFrontier> far_block;

	// SESSION088: on a far BLOCK, a copy of the saturation grid its selection was actually made under - the evidence the
	// LoD below the cut was chosen from. Empty on a normal frontier, and on a block built with the bias off.
	//
	// Why a copy of the grid rather than a Reference to the barrier: GaussianSplatSaturationBarrier is defined below this
	// class and only forward-declared above it, so a Reference member here would need its destructor instantiated against
	// an incomplete type. The grid is sat_grid_res^2 floats - 39KB at the 99x99 the owner's interior builds - copied once
	// per block build, which is nothing next to the block's own SoA.
	//
	// What it is FOR - see getSatBarrierAgreeTol(). Session080 sized the drift bound purely as an ordering error budget,
	// and session085's LoD bias then gave the block a second, far more sensitive dependence on the barrier that the same
	// one number was left to cover. This is the second dependence's own evidence, so it can be checked directly instead
	// of proxied by how far the camera has walked.
	js::Vector<float, 16> sat_depth_used;
	int sat_grid_res_used;
	Vec4f sat_anchor_used; // The barrier anchor that grid was measured from - two grids are only comparable tile-by-tile while their anchors are close, see gsSatBarrierDisagreement().
	double sat_built_time_used; // SESSION088 DIAGNOSTIC: the captured barrier's own built_time_real_s - printed alongside the LIVE barrier's, on [gsr-traversal], purely so a suspicious barrier_dis= reading can be checked against "were these actually two different barrier builds" before it is trusted.

	// SESSION080: diagnostics for the above - see [gsr-traversal]. reuse_n is far_block's size, repeated here so a log
	// line describes the whole frontier without dereferencing anything.
	size_t reuse_roots;          // Subtrees this walk stopped at and left to the far block.
	size_t reuse_n;              // Nodes the far block contributes, i.e. how much of the frontier was not walked.
	double reuse_ms;             // Cost of producing the far segment: 0 when it was inherited, the partition+SoA cost when it was built.
	float reuse_split_dist_used; // The frozen split distance this frontier's cut was made at - see far_block.
	float barrier_disagreement;  // SESSION088 DIAGNOSTIC: flip + depth, what the acceptance clause gates on - see gsSatBarrierDisagreement().
	float barrier_flip, barrier_depth, barrier_mean_rel_depth; // SESSION088: its two components, and the mean relative depth move behind the second.
	bool barrier_measured;       // SESSION088: false = nothing was compared. Printed as n/a, so a not-measured row can no longer be read as perfect agreement.
	double diag_live_barrier_time, diag_block_barrier_time; // SESSION088 DIAGNOSTIC, TEMPORARY - see the task member of the same name. Near-equal values with a nonzero disagreement would mean the comparator is broken; note these are KICK-time facts while this line is printed at DRAIN, so a one-cadence gap here is normal and is not evidence of anything.

	// SESSION080: the camera position the far block's cut was frozen at, carried on every frontier that references it so
	// the next traversal can measure drift in one subtraction. Equal to anchor_pos_ws on a self-contained frontier.
	//
	// This bounds something inheritance would otherwise let run away. The cut is frozen, so a node beyond it is never
	// re-examined however far the camera travels, and its front-to-back key stays the one it was given when the block was
	// built. One traversal of staleness is the tolerance this scheme is designed around; an unbounded number is not, and
	// the retreating case does not self-correct - retreating makes MORE of the tree far, so less of it gets refreshed,
	// not more. When drift passes reuse_max_drift_fraction of the split distance the block is discarded and the next
	// traversal walks in full - see GaussianSplatLodTraversalTask's reuse_enabled.
	Vec4f reuse_base_anchor_ws;

	GaussianSplatUnculledFrontier() // SESSION074: defaults are "stage never ran".
	:	built_time_real_s(0.0), // SESSION080
		expand_ms(0.0), sort_ms(0.0), // SESSION080
		expand_seeds(0), expand_num_tasks(0), expand_prologue_ms(0.0), expand_task_max_ms(0.0), expand_task_sum_ms(0.0), // SESSION080
		expand_workers(0), expand_seed_max_ms(0.0), expand_seed_target(0), // SESSION086
		expand_splice_reserve_ms(0.0), expand_splice_copy_ms(0.0), sort_alloc_ms(0.0), soa_ms(0.0), // SESSION080 (plan2 §4.1)
		sort_radix_ms(0.0), sort_unpack_ms(0.0), // SESSION087
		traversal_output_n(0), // SESSION080
		sat_bias_barrier_time(0.0), // SESSION085
		sort_staleness_delta_ws(0.0), sort_staleness_prev_n(0), sort_staleness_common_n(0), sort_staleness_max_disp(0), // SESSION080
		sort_staleness_mean_disp(0.0), sort_staleness_gt1k(0), sort_staleness_gt10k(0), sort_staleness_ms(0.0), // SESSION080
		sat_grid_res_used(0), sat_anchor_used(0.f), sat_built_time_used(0.0), // SESSION088 - see sat_depth_used.
		reuse_roots(0), reuse_n(0), reuse_ms(0.0), reuse_split_dist_used(0.f), barrier_disagreement(0.f), // SESSION080 §4.3; SESSION088
		barrier_flip(0.f), barrier_depth(0.f), barrier_mean_rel_depth(0.f), barrier_measured(false),
		diag_live_barrier_time(0.0), diag_block_barrier_time(0.0), reuse_base_anchor_ws(0.f)
	{
		for(int i=0; i<5; ++i) dist_pctile[i] = 0.f; // SESSION080
	}
};


// SESSION081: a ref-counted holder for the barrier build's working buffers, so the cloud that owns them and the task
// that is writing into them can both keep them alive - see SplatCloud::sat_build_scratch and GsSatBuildScratch. The
// refcount is the whole point of the wrapper: a cloud can be removed while its build is still running, and the running
// task must not be left writing into freed memory.
class GaussianSplatSaturationBuildScratch : public ThreadSafeRefCounted
{
public:
	GsSatBuildScratch scratch;
};


// SESSION081 - THE SATURATION BARRIER, decoupled from any one traversal's frontier. See the session081 snapshot's
// "Variant 2" for the motivation: sat_region_radius (R) already makes the grid a statement about a BALL of camera
// positions, not the single point it was built from, so once built it does not need rebuilding on every traversal -
// only once the camera has actually left that ball. Before this, the grid lived on GaussianSplatUnculledFrontier and
// was rebuilt by every traversal whose saturation stage ran, whether or not the previous grid was still valid; at
// R > 0 and a traversal now taking ~89ms against a build taking ~400-500ms, that meant most builds were wasted -
// superseded by a newer traversal before they finished, discarded (see [gsr-sat] DISCARDED in the session080
// snapshot), 76-85% of the time in a walking measurement.
//
// Built by GaussianSplatSaturationBuildTask, on its own cadence (kickOffSaturationBuilds()/
// drainSaturationBuildResults()), from a snapshot of the cloud's most recent UNPRUNED frontier - see
// SplatCloud::last_unpruned_ufrontier and SplatCloud::cached_sat_barrier. Immutable once built, like
// GaussianSplatUnculledFrontier. Read by GaussianSplatLodTraversalTask's saturation APPLY step (test/compact), which
// still runs once per traversal - only the BUILD moved out; see that task's ctor param.
class GaussianSplatSaturationBarrier : public ThreadSafeRefCounted
{
public:
	js::Vector<float, 16> sat_depth; // See GaussianSplatUnculledFrontier::sat_depth's old comment - same meaning, just anchored to THIS object's own anchor_pos_ws below, not necessarily to where the camera is now.
	int sat_grid_res;                // 0 means "no grid" - either the stage is disabled, or the source frontier had no occluders. Callers must check before indexing, same convention as before.
	Vec4f anchor_pos_ws;             // Where THIS barrier was built - the centre of the ball sat_region_radius_used is valid over.
	double built_time_real_s;        // Clock::getCurTimeRealSec() at build - see [gsr-sat-build]'s age, and the apply side's own anchor_age in [gsr-sat-apply].

	// The knobs this barrier was actually built with - see GaussianSplatUnculledFrontier's old _used fields for why
	// these are snapshotted rather than re-read from the live renderer settings (which may have moved on since, and -
	// SESSION081 - are also what kickOffSaturationBuilds() compares against the CURRENT settings to decide whether a
	// live-tuned knob has invalidated this barrier).
	float threshold_used, subdiv_used, region_radius_used;
	float coarse_pixel_scale_used; // SESSION088: "px" feeds sat_grid_res exactly like "sub" does (gsSatGridResForFocal()), so a change to it is exactly as stale-grid-shaped as a change to "sub" - and was never checked here. See setCoarsePixelScale().
	int closing_tiles_used;

	// DIAGNOSTIC, build-side - see [gsr-sat-build]. Same fields GaussianSplatUnculledFrontier used to carry for the
	// build it no longer does.
	size_t num_occluders;
	double gather_ms, grid_build_ms;
	// SESSION081 PLAN ETAP 0 DIAGNOSTIC: grid_build_ms broken into its sub-phases - see gsBuildSaturationGridParallel()'s
	// new out-params. Previously only the whole grid_build_ms was known, with no way to tell how much of it was the
	// occluder-record pass, the strip deposit, closing, or erosion. misc_ms is not stored - the caller computes
	// grid_build_ms - (rec_ms+dep_ms+close_ms+erode_ms) at print time as the "did we measure everything" check.
	double rec_ms, dep_ms, close_ms, erode_ms;
	size_t frontier_n; // SESSION081 PLAN ETAP 0 DIAGNOSTIC: size of the traversal frontier current at build time (near+far segments). SESSION085: no longer an INPUT to the gather (the tree walk supplies its own occluders) - kept because occl= is only readable against the render frontier it is being spent instead of.
	size_t num_recs;   // Records that survived the amp reject and reached phase 2 - what every strip re-reads (see num_strips).
	int num_strips;    // The strip count phase 2 actually ran with - num_strips*num_recs is the phase-2 re-read volume the plan's stage 5 targets.
	int concurrency_used; // task_manager->getConcurrency() at build time - num_strips is clamp(res/8, 1, this), so this says whether num_strips was capped by resolution or by thread count.
	size_t num_blocks;    // SESSION081 PLAN, ETAP 0 follow-up DIAGNOSTIC: how many gs_sat_rec_scratch_bytes-bounded blocks this build ran - see that constant's comment on the sync-barrier cost of blocking.
	size_t gate1_survivors; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC - see gsBuildSaturationGridParallel()'s out param of the same name.
	double rec_task_ms_min, rec_task_ms_max, rec_task_ms_mean; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC - see gsBuildSaturationGridParallel()'s out params of the same name.
	double sched_stats[9]; // SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC - layout in gsBuildSaturationGridParallel()'s out_sched_stats.
	size_t diag_writers, diag_tile_writes;
	size_t diag_tile_stats[3]; // [0] total per-tile iterations, [1] rejected off the Gaussian's tail, [2] skipped as already saturated - see gsSatDepositRec().
	size_t erode_stats[4];     // [0] er_ceil, [1] er_win, [2] er_radmax, [3] cl (closed) - see gsSatApplyRegionErosion()/gsSatApplyClosing().
	// The tree walk's own costs, reported apart from gather_ms. The sort is work the retired frontier sources did not
	// do - they inherited their front-to-back order from the traversal's own sort and paid nothing for it - so it stays
	// visible on its own rather than buried in a total.
	double walk_ms, walk_sort_ms;
	size_t walk_visited;
	size_t inf_hist[5];        // SESSION082 PLAN §1.1, TEMPORARY DIAGNOSTIC - see gsBuildSaturationGridParallel()'s out_inf_hist for the layout and the question.

	// SESSION077/078 debug overlay sources - see updateSatGridDebugTexture(). Empty unless the overlay was requested
	// for this build (same gate sat_overlay_requested used to drive on the traversal task).
	js::Vector<float, 16> sat_accum_t;
	js::Vector<float, 16> sat_amp_sum;

	GaussianSplatSaturationBarrier()
	:	sat_grid_res(0), anchor_pos_ws(0.f), built_time_real_s(0.0),
		threshold_used(0.f), subdiv_used(0.f), region_radius_used(0.f), coarse_pixel_scale_used(0.f), closing_tiles_used(0), // SESSION088: coarse_pixel_scale_used - see the field.
		num_occluders(0), gather_ms(0.0), grid_build_ms(0.0),
		frontier_n(0), rec_ms(0.0), dep_ms(0.0), close_ms(0.0), erode_ms(0.0), num_recs(0), num_strips(0), concurrency_used(0), num_blocks(0), gate1_survivors(0),
		rec_task_ms_min(0.0), rec_task_ms_max(0.0), rec_task_ms_mean(0.0),
		diag_writers(0), diag_tile_writes(0),
		walk_ms(0.0), walk_sort_ms(0.0), walk_visited(0)
	{
		for(int i=0; i<9; ++i) sched_stats[i] = 0.0; // SESSION081 SCHEDULING PROBE.
		for(int i=0; i<3; ++i) diag_tile_stats[i] = 0;
		for(int i=0; i<4; ++i) erode_stats[i] = 0;
		for(int i=0; i<5; ++i) inf_hist[i] = 0; // SESSION082 PLAN §1.1, TEMPORARY.
	}
};


// One registered splat object, and the range of its owning cloud's arrays that it occupies.
struct CloudMember
{
	GaussianSplatRenderer::Handle handle;
	GaussianSplatDataRef splat_data; // The original object-space data.  Kept so that a move, or a re-bake into a different cloud after a merge, starts from the source rather than accumulating error over repeated re-bakes.

	// Null unless GaussianSplatRenderer::applyCoplanarMerge() has replaced splat_data above with a merged copy, in which
	// case this holds what was loaded.  Two things need it: restoreUnmergedSplats(), and the merge itself, which always
	// starts from the loaded splats rather than from whatever the last press left behind - otherwise a second press at
	// the same tolerances would merge an already-merged cloud again, and the tolerances would mean a different thing
	// every time.
	GaussianSplatDataRef unmerged_splat_data;

	size_t offset, count; // This member's range within its cloud's arrays, and within its GPU texture and index VBO.

	js::AABBox aabb_ws; // Padded by the splat extent, not just bounding the splat centres - see bakeMember().

	// The pose, kept so that a merge can re-bake this member into a different cloud without the caller supplying it again.
	Vec4f translation_ws;
	Quat<float> rotation_ws;
	float uniform_scale_ws;

	// SESSION059: debug-only "leave this object out of the draw entirely" toggle - see GaussianSplatRenderer::setObjectHidden().
	// Deliberately in-memory only, never touches QSettings: every session must start with every object visible. Survives a
	// merge for free (mergeIntersectingClouds() copies the whole CloudMember by value), which is exactly the behaviour
	// wanted - hiding an object shouldn't depend on which cloud it currently happens to be merged into.
	bool hidden;
};


// One drawable cloud: a single GLObject holding one or more members' splats, with its own data texture, instance index
// VBO and depth sort.  Splat data is baked into world space, so ob->ob_to_world_matrix stays identity.
class SplatCloud : public RefCounted
{
public:
	SplatCloud()
	:	cloud_id(0), gpu_capacity_splats(0), total_splats(0), structure_generation(0), sort_in_flight(false),
		importance_layout_fingerprint(0), slice_sample_draw_count(0),
		have_last_sort_cam_pos(false), last_sort_cam_pos_ws(0.f), aabb_ws(js::AABBox::emptyAABBox()), added_to_engine(false),
		topology_generation(0), traversal_in_flight(false), have_last_traversal_cam_pos(false), last_traversal_cam_pos_ws(0.f), last_traversal_cam_forward_ws(0.f), last_traversal_kick_time_s(0.0),
		last_traversal_kicked_topology_generation(0), last_traversal_hit_budget_cap(false), last_traversal_hit_density_cap(false), last_traversal_hit_depth_cap(false), last_traversal_sat_bias_stops(0), last_traversal_sat_bias_tested(0), last_traversal_sat_bias_ceiling(1.f), last_traversal_dilation_elevated(false),
		cached_traversal_geom_generation(0), importance_num_views(0),
		sat_grid_debug_anchor_ws(0.f), // SESSION076 §9
		filter_in_flight(false), ufrontier_needs_filter(false), have_last_filter_cam_forward(false), last_filter_cam_forward_ws(0.f), // SESSION063
		last_applied_filter_frontier_time_s(0.0), // SESSION081
		filter_dilation_elevated(false), // SESSION066
		diag_filter_kick_time_s(0.0), sat_build_kick_time_s(0.0), // SESSION073 DIAGNOSTIC; SESSION088 - see the members.
		diag_filter_band_fine_rad(0.f), diag_applied_kick_forward_ws(0.f), diag_have_applied_filter(false), // SESSION073 DIAGNOSTIC
		sat_build_in_flight(false) // SESSION081
	{}

	uint64 cloud_id; // Stable, never reused.  Sort results carry it, so a result for a cloud that has since been merged away can be dropped.

	GLObjectRef ob;
	bool added_to_engine; // False between allocCloud() and the first member being baked in - see addCloudToEngineIfNeeded().
	Reference<VBO> instance_index_vbo;
	size_t gpu_capacity_splats; // Allocated capacity of the data texture and index VBO, in splats.  total_splats <= gpu_capacity_splats always.
	size_t total_splats;

	// World-space splat data for this cloud's members, concatenated in member order.
	js::Vector<Vec3f, 16> positions;
	js::Vector<Vec3f, 16> scales;
	js::Vector<Vec4f, 16> rotations; // (x, y, z, w)
	js::Vector<Vec4f, 16> colours;
	js::Vector<float, 16> feature_size; // SESSION055: 2*max(scale.xyz), maintained in step with scales at bake time so kickOffTraversals()'s fillTraversalScratch() doesn't rebuild it from scales on every kick - a session055 hot path once the frustum-cull re-kick started firing on rotation.
	js::Vector<float, 16> cull_radius; // SESSION059: world-space enclosing-sphere radius (uniform_scale_ws * the tree node's bounding_radius_os, or the same 3-sigma cutoff feature_size already gives for a no-tree member's own splats), maintained in step with feature_size at bake time. Used for the traversal's frustum-cull margin instead of feature_size - see GaussianSplatLodNode::bounding_radius_os's comment for why feature_size alone isn't a safe bound.

	// SESSION058: cached copy of positions+feature_size for the traversal path, valid as long as
	// cached_traversal_geom_generation == topology_generation - see GaussianSplatCachedGeom's comment for why that's a
	// safe invalidation signal. Null cached_traversal_geom means "never built yet"; fillTraversalScratch() handles both
	// that and a generation mismatch the same way (build and cache a fresh one).
	Reference<GaussianSplatCachedGeom> cached_traversal_geom;
	uint64 cached_traversal_geom_generation;

	std::vector<CloudMember> members;
	js::AABBox aabb_ws; // The union of the members' padded bounds.  Doubles as the merge test and, via ob, the cull and draw-order box.

	uint64 structure_generation; // Bumped by anything that renumbers the cloud, which invalidates an in-flight sort's indices.
	bool sort_in_flight; // True from when a sort is kicked off until its precise result is applied.  The coarse result doesn't clear it.
	bool have_last_sort_cam_pos;
	Vec4f last_sort_cam_pos_ws; // Camera position as of the last sort kicked off (not necessarily completed).

	// Bumped by anything that changes which nodes belong to this cloud or where (appendMemberToCloud() *and* rebuildCloud()) -
	// a strict superset of structure_generation's bump conditions (which rebuildCloud() alone triggers). A pure append doesn't
	// invalidate an in-flight sort's indices (the appended tail is already in identity order - see appendMemberToCloud()), but
	// it does mean a new member's tree root is missing from any traversal already in flight, so that result can't be trusted
	// once it lands; the newly appended member must appear in the *next* traversal even if the camera hasn't moved at all.
	uint64 topology_generation;
	bool traversal_in_flight; // True from when a traversal is kicked off until its result is applied (or dropped as stale).
	bool have_last_traversal_cam_pos;
	Vec4f last_traversal_cam_pos_ws; // Camera position as of the last traversal kicked off (not necessarily completed).
	Vec4f last_traversal_cam_forward_ws; // SESSION055: camera forward at that kick, so kickOffTraversals() can re-trigger on rotation now that the traversal is view-dependent - see lod_frustum_cull_enabled.
	double last_traversal_kick_time_s; // SESSION055: timestamp of that kick (diag_timer.elapsed()) so the next kick can compute the *empirical* rotation/translation rate since - see the dilation block in kickOffTraversals().
	uint64 last_traversal_kicked_topology_generation; // topology_generation as of the last traversal kicked off - a mismatch against the live value means the cloud's structure has changed since, so it's unconditionally overdue for a fresh one (see kickOffTraversals()), the same idea as !have_last_sort_cam_pos for the sort.
	bool last_traversal_hit_budget_cap; // Copied from the most recently applied traversal result's GaussianSplatLodTraversalScratch::hit_budget_cap (which itself doesn't persist - the scratch goes back to the pool) - surfaced in getDiagnostics() as a "detail is being truncated by the budget" warning.
	bool last_traversal_hit_density_cap; // As above, for GaussianSplatLodTraversalScratch::hit_density_cap.
	bool last_traversal_hit_depth_cap; // As above, for GaussianSplatLodTraversalScratch::hit_depth_cap.
	size_t last_traversal_sat_bias_stops, last_traversal_sat_bias_tested; // SESSION085 ETAP 3: as above, for the scratch's counters of the same name.
	float last_traversal_sat_bias_ceiling;  // SESSION085 ETAP 6: as above, for GaussianSplatLodTraversalScratch::sat_bias_ceiling_used.

	// SESSION063: the cached unculled frontier U(P) for the split filter architecture, when split_filter_enabled. Set by
	// drainTraversalResults() from a cull-off traversal; a rotation re-filters this instead of re-traversing. Null until
	// the first split-mode traversal for this cloud lands. See GaussianSplatUnculledFrontier.
	Reference<GaussianSplatUnculledFrontier> cached_ufrontier;

	// SESSION080 DIAGNOSTIC (plan doc STEP A), debug-only: the last UNPRUNED frontier this cloud received - i.e. what
	// message 1 carried, before the saturation follow-up replaced cached_ufrontier with its pruned copy.
	//
	// cached_ufrontier itself cannot serve as the "previous frontier" for the sort-staleness comparison, and this is the
	// whole reason this field exists: most of the time it holds the PRUNED frontier (message 2's uf2), whose ranks are
	// compacted by whatever saturation dropped - 55-80% of the pool at R=0. Ranking the new unpruned frontier against
	// that would measure the prune's compaction, not the camera's motion, and would swamp the signal being looked for.
	// Held only while filter_debug_log is on (drainTraversalResults() nulls it otherwise), because retaining it keeps a
	// whole extra frontier alive - see the assignment site.
	Reference<GaussianSplatUnculledFrontier> last_unpruned_ufrontier;

	// SESSION081: the current saturation barrier - see GaussianSplatSaturationBarrier and kickOffSaturationBuilds()/
	// drainSaturationBuildResults(). Null until the first build for this cloud lands. sat_build_in_flight is true from
	// the moment a build is kicked until its result is drained (mirrors traversal_in_flight/filter_in_flight above).
	Reference<GaussianSplatSaturationBarrier> cached_sat_barrier;
	bool sat_build_in_flight;

	// SESSION081: the barrier build's working buffers, kept alive between builds instead of being reallocated by each
	// one - see GsSatBuildScratch for the measurement that motivated this. Per CLOUD, not per renderer: two clouds can
	// have builds running at the same time (sat_build_in_flight only serialises builds within one cloud), and two
	// concurrent builds sharing one scratch would write over each other. Ref-counted and also held by the task, because
	// a build outlives its cloud if the cloud is removed mid-build - drainSaturationBuildResults() already handles that
	// case for the result, and this keeps the buffers the running task is still writing into alive to match.
	Reference<GaussianSplatSaturationBuildScratch> sat_build_scratch;

	// SESSION085 ETAP 6: the saturation APPLY pipeline's in-flight/throttle state went with the prune itself.

	// SESSION076 §9 / SESSION077: debug-only visualisation of cached_ufrontier's saturation grid, rebuilt by
	// drainTraversalResults() alongside cached_ufrontier whenever getSatDiagLog() is on. Null whenever the diag
	// checkbox is off, the frontier has no grid (sat_grid_res == 0), or none has landed yet - see
	// getSatGridDebugInfo(), which skips a cloud with a null binary tex. Two textures, one per overlay mode - see
	// GaussianSplatUnculledFrontier::sat_accum_t / sat_amp_sum for why one field can't serve both.
	Reference<OpenGLTexture> sat_grid_debug_tex;         // Binary view source: transmittance (pre closing/erosion).
	Reference<OpenGLTexture> sat_grid_debug_ramp_tex;    // Ramp view source: unbounded contribution sum.
	Reference<OpenGLTexture> sat_grid_debug_maskfix_tex; // SESSION081: "mask_fix" view source - see updateSatGridDebugTexture().
	Vec4f sat_grid_debug_anchor_ws;

	bool filter_in_flight;                 // True from a filter kick until its result is applied/dropped - see kickOffFilters().
	bool ufrontier_needs_filter;           // Set when a fresh U(P) lands, so kickOffFilters() produces the first S(P,R) for it even with no rotation.

	// SESSION081: the built_time_real_s of whatever U(P) frontier produced the S(P,R) currently on screen - the
	// freshness comparison drainFilterResults() makes against it is by TIMESTAMP, not identity, for exactly the same
	// reason as the saturation apply guard's own relaxation - see that comparison's comment in
	// drainSaturationApplyResults() for the fuller argument, and drainFilterResults()'s own comment for why it is
	// safe here too despite the filter choosing content (not just pruning it): a superseded frontier's own successor
	// is ALREADY installed as cached_ufrontier and has ALREADY triggered its own filter kick regardless of what this
	// guard decides, so accepting an intermediate, not-literally-latest result only ever replaces something OLDER on
	// screen with something newer - it never delays or replaces the eventual newest one. 0 = nothing applied yet.
	double last_applied_filter_frontier_time_s;
	bool have_last_filter_cam_forward;
	Vec4f last_filter_cam_forward_ws;      // Camera forward at the last filter kick, so a rotation past threshold re-filters - see kickOffFilters().
	bool filter_dilation_elevated;         // SESSION066: true if the last filter kick for this cloud used an above-floor dilation band, so kickOffFilters() knows to fire one tight "settle" re-filter once the slow-decay peak falls back to the floor - the filter analogue of last_traversal_dilation_elevated. Without it a wide band from an in-motion kick would stay applied after the camera stops (the observed stall).

	// SESSION073 DIAGNOSTIC (temporary - remove once the rotational-hole question is settled). Measures the dilation band's
	// ADEQUACY directly, in degrees, rather than inferring it from compute times: a hole appears exactly when the camera
	// turns further, while a draw list is on screen, than the band that list was dilated by. All three numbers the
	// comparison needs are already on the main thread at drain time - only one filter per cloud is ever in flight, and
	// think() drains before it kicks, so last_filter_cam_forward_ws still holds the forward of the kick that produced the
	// result being drained. See the [gsr-filter-drain] print in drainFilterResults().
	double diag_filter_kick_time_s;        // diag_timer.elapsed() at that kick, for the real kick->drain round-trip.
	// SESSION088: the same measurement for the BARRIER build - diag_timer.elapsed() at its kick, read back in
	// drainSaturationBuildResults() to feed sat_build_latency_measured. Kept here rather than on the task because the
	// round trip that matters is kick->APPLIED (what the LoD bias actually waits for), not the task's own run time,
	// which the build already reports as gather_ms/grid_build_ms/walk_ms and which excludes every queue it sat in.
	double sat_build_kick_time_s;
	float diag_filter_band_fine_rad;       // swept_fine of that kick - the angular half-width the fine layer was dilated by.
	Vec4f diag_applied_kick_forward_ws;    // Camera forward at the kick of the list CURRENTLY on screen, so the next drain can report the total angle that list went stale by before being replaced (the worst-case deficit, roughly double the kick->drain figure).
	bool diag_have_applied_filter;         // False until the first result has been applied, so the first drain doesn't report a bogus staleness against a zero vector.

	// SESSION059: true if the traversal just kicked for this cloud used a rotation_dilation_rate above the baseline
	// floor (i.e. cam_angular_speed_ema/peak was elevated at kick time) - see kickOffTraversals()'s "settle" re-kick.
	// A rotation past the 5deg re-kick threshold naturally corrects an over-wide selection on the next real kick, but a
	// single violent mouse flick can leave cam_angular_speed_peak elevated for ~1-3s (its decay is slow by design - see
	// think()) with NO further rotation happening at all: nothing re-triggers, so the over-dilated selection from the
	// flick's own kick stays applied indefinitely - session059 found this stuck at ~2x the GPU cost of a settled scene,
	// on a real-world (km-scale) capture where rotation_dilation_rate's distance-proportional term amplifies the effect.
	// This flag lets kickOffTraversals() notice, once w_effective has decayed back to baseline, that this cloud is
	// still carrying a stale wide margin and deserves one more kick to tighten it back up.
	bool last_traversal_dilation_elevated;

	// Per-splat importance, accumulated across getFrustumStructureReport() runs so that "does this splat matter from
	// anywhere?" can be asked of several viewpoints instead of one.  The pruning ceiling in that report is a single-camera
	// figure by construction - a splat hidden from here is the front layer from there - and until these existed there was
	// no way to tell the two apart.  Indexed by the same cloud-global splat index as positions/scales/colours.
	//
	// Allocated on the first report rather than with the cloud: 12 bytes a splat is 36 MB on a three-million-splat capture,
	// and a session that never opens the report should not pay it.
	js::Vector<float, 16> importance_best_contribution; // Most this splat ever added to an image, over the recorded views.
	js::Vector<float, 16> importance_sum_fill;          // Summed fill over the views that drew it, so a mean can be taken.
	js::Vector<uint32, 16> importance_views_drawn;      // How many recorded views drew it at all.
	uint64 importance_layout_fingerprint;               // What the arrays were filled against - see cloudLayoutFingerprint().  A mismatch means the indices no longer mean the same splats, so the record has to be thrown away.
	size_t importance_num_views;                        // How many reports have been folded in.  0 = nothing accumulated yet.

	// Frustum-aware draw slicing - see GaussianSplatRenderer::getVisibleSlicingEnabled().  An evenly spaced sample of
	// this cloud's current draw order, held as world positions rather than indices so that nothing here can index out of
	// an array that changed underneath it: the worst a stale sample can do is misplace a slice boundary, which cannot
	// change the picture.  Refreshed by noteDrawOrderForSlicing() every time the draw order is written.
	js::Vector<Vec3f, 16> slice_sample_positions;
	int slice_sample_draw_count; // num_instances_to_draw the samples above were taken from, so sample j can be mapped back to a draw index.

	// SESSION066: full CPU copy of the instance index VBO's current contents (the LoD draw list S(P,R) actually being
	// drawn), kept by noteDrawOrderForSlicing() alongside the slice sample. Only countSplatsInFrustum() reads it - to
	// report how many of the actually-drawn splats pass the frustum + size/distance slices this frame, i.e. what really
	// reaches the screen (unlike num_instances_to_draw, which is the pre-slice, dilation-band-inclusive selection size).
	js::Vector<uint32, 16> current_draw_indices;

	// Cumulative count of in-frustum samples, one entry per sample, rebuilt each frame by think() while the frustum-aware
	// slicing is on.  Empty when it is off, or when the samples cannot be used.
	js::Vector<int, 16> slice_visible_cdf;
};


// Identifies which splat each index refers to, so that the importance record above can tell "the cloud was renumbered
// underneath me" from "the cloud was rebuilt into exactly the same layout".
//
// topology_generation is not usable for this: it is bumped by any add or rebuild, and walking a world streams splat
// objects in and out, so a tour of four viewpoints discarded the record between the first and the second - silently, and
// without the layout having actually changed.  The member list is what the indices are actually derived from, so hashing
// it answers the question directly.
static uint64 cloudLayoutFingerprint(const SplatCloud& cloud)
{
	uint64 h = 1469598103934665603ull;
	const uint64 vals[1] = { (uint64)cloud.total_splats };
	for(int i=0; i<1; ++i) { h ^= vals[i]; h *= 1099511628211ull; }

	for(size_t m=0; m<cloud.members.size(); ++m)
	{
		// The source data's identity, and where its splats were placed.  Two members cannot share a range, so this
		// distinguishes any renumbering that moved anything.
		const uint64 member_vals[3] = { (uint64)(uintptr_t)cloud.members[m].splat_data.ptr(), (uint64)cloud.members[m].offset, (uint64)cloud.members[m].count };
		for(int i=0; i<3; ++i) { h ^= member_vals[i]; h *= 1099511628211ull; }
	}
	return h;
}


// Reusable working buffers for the background depth-sorts.  At large splat counts these run to hundreds of MB, so
// they're pooled and reused rather than reallocated per sort.  Reference counted rather than owned outright, so that an
// in-flight sort keeps them alive even if the renderer is torn down.
class GaussianSplatSortScratch : public ThreadSafeRefCounted
{
public:
	struct SortItem
	{
		uint32 key;
		uint32 splat_index;
	};

	js::Vector<Vec3f, 16> positions_snapshot; // A frozen copy of one cloud's positions, taken on the main thread when a sort is kicked off.  The worker only ever reads this, never the live arrays.

	js::Vector<SortItem, 16> items; // Sort input, and the precise stage's output.
	js::Vector<SortItem, 16> working_space; // Scratch space for the sort routines, and the coarse stage's output.

	// The two stages' results, as instance draw orders ready for VBO::updateData().  Kept separate so the precise stage
	// can't overwrite a coarse result the main thread hasn't consumed yet.
	js::Vector<uint32, 16> coarse_indices;
	js::Vector<uint32, 16> precise_indices;

	js::Vector<uint32, 16> temp_counts; // Bucket counts for Sort::radixSort32BitKey().
};


// One selected node, as the traversal's sort sees it: the squared camera distance it is keyed on, and the cloud index it
// carries. SESSION080: hoisted to file scope out of GaussianSplatLodTraversalTask so the buffers of these can be pooled
// on the scratch below, which is declared before that task - see the scratch's `decorated`/`sort_scratch`.
//
// SESSION063 K4: bit 31 of idx flags a coarse-floor node. Cloud indices are well under 2^31, and the radix sort keys
// only on dist_sq, so the flag rides along for free and is unpacked after the sort.
struct GsDistIdx { float dist_sq; uint32 idx; };


// Reusable working buffers for the background LoD traversals - pooled the same way GaussianSplatSortScratch is, and for
// the same reason (selected_indices can run to a non-trivial size for a large tree, so N clouds shouldn't mean N sets of
// these). SESSION058: geom below is no longer one of these per-scratch allocations - it's a Reference<> into a shared,
// per-cloud cache (see GaussianSplatCachedGeom), so it costs nothing extra to pool.
class GaussianSplatLodTraversalScratch : public ThreadSafeRefCounted
{
public:
	// One member's tree and where it landed in the cloud's arrays, as of when the snapshot was taken.
	struct MemberSnapshot
	{
		GaussianSplatDataRef splat_data; // Kept alive so the worker can read lod_tree from it directly - never mutated after decode, so safe to read cross-thread without copying its contents.
		size_t offset;
		size_t count;
		bool hidden; // SESSION059: mirrors CloudMember::hidden as of when this snapshot was taken - see GaussianSplatRenderer::setObjectHidden().
	};

	Reference<GaussianSplatCachedGeom> geom; // SESSION058: shared, cloud-topology-generation-scoped snapshot of positions/feature_size - see GaussianSplatCachedGeom. Replaces the old per-kick positions_snapshot/feature_size_snapshot copies.
	std::vector<MemberSnapshot> members_snapshot;

	js::Vector<uint32, 16> selected_indices; // Output: this frame's frontier, as cloud-array indices.  Unsorted (heap-pop order) until stage 5 wires the depth-sort up to the selection - see kickOffSorts()'s use of cloudHasLodTree().
	bool hit_budget_cap; // True if max_splats_budget stopped further expansion before pixel_scale converged - i.e. detail is being truncated by the budget, not just naturally coarse at this distance. Consumed by the diagnostics display from stage 6 onward.
	bool hit_density_cap; // True if getMaxLayerDensity() stopped at least one node's expansion this traversal - see GaussianSplatLodNode::layer_density.
	bool hit_depth_cap; // True if getMaxTreeDepth() stopped at least one node's expansion this traversal.
	float sat_bias_ceiling_used; // SESSION085 ETAP 6: the ceiling THIS traversal ran with, not the live setting - same rule the barrier follows for thr=/sub=/R=: a captured log line has to say what produced it. Found missing while reading a walking capture: the one knob that now decides the whole stage appeared in no log line at all.
	size_t num_sat_bias_stops, num_sat_bias_tested; // SESSION085 ETAP 3: counts, not a bool. The first cut reported only "did the bias stop anything", which was true while the mechanism was in fact near-inert - a bool cannot distinguish "fired a few thousand times" from "fired on half the walk". stops/tested is the diagnosis: a low ratio means the occlusion test is being asked and refusing.

	// SESSION079: scratch for the parallel saturation build's phase-1 records and the parallel read pass's keep mask
	// USED TO live here, pooled on the reused per-kick scratch for the same first-touch-cost reason as the fields
	// below. SESSION081 BUGFIX: moved back to locals in run() - both are read by the saturation phase, which runs
	// AFTER this scratch is handed back to the free pool (see the frontier's publish-then-saturate comment at the
	// enqueue site), so a second, faster traversal could resize either buffer while the first traversal's saturation
	// phase was still writing through a pointer into it. Reproduced as an access violation: saturation on, frontier
	// reuse > 0 (traversal fast enough to overlap), diag on (widens the race window further). Not pooled here any
	// more - see the local declarations at the saturation phase's own comment for the full account.

	// SESSION080 (plan2 §4.1): the traversal's selection and the sort's working space. Pooled here because BOTH are
	// fully consumed before result_queue->enqueue() publishes the frontier - unlike sat_occluder_recs/sat_keep_mask
	// above, this is the safe case: nothing outside run() can touch this scratch while these are still live, so there
	// is no equivalent of the bug the comment above describes. Sized to the frontier - ~106MB each at 13.2M nodes -
	// and as locals in run()
	// they were allocated, first-touched and freed on EVERY traversal. The allocation itself measured free
	// (splice_res_ms/sort_alloc_ms, 0.01-0.05ms: js::Vector::reserve does not touch the pages), so the cost that
	// pooling removes is the first-touch faulting spread through the copy that follows. run() clear()s them - which
	// keeps the capacity - rather than resizing, so a steady-state traversal reuses already-faulted pages.
	//
	// Scratches are pooled globally rather than per cloud, so a differently-sized cloud may inherit these; that is
	// harmless, since both are grown on demand and never read past the size the current traversal sets.
	js::Vector<GsDistIdx, 16> decorated;
	js::Vector<GsDistIdx, 16> sort_scratch;

	// SESSION080 §4.3: scratch for the near/far partition of a sorted selection, used only by the traversal that BUILDS
	// a far block (roughly one in ten - see reuse_max_drift_fraction). Pooled for the same reason as the two above.
	js::Vector<uint32, 16> partition_near, partition_far;
	js::Vector<float, 16> partition_near_coarse, partition_far_coarse;
};


namespace
{


const size_t texels_per_splat = 4; // See the texel layout in gaussian_splat_vert_shader.glsl.


// SESSION079: the adaptive pixel_scale_limit controller's fixed limits - see updateAdaptivePixelScale().
//
// The range is deliberately narrow at the bottom and generous at the top. Below ~1 the selected splat count climbs
// steeply for detail the display cannot resolve, so there is nothing down there worth the frames; the owner measured
// 2 against 4 as barely distinguishable over open ground and merely noticeable in a detailed interior, which says the
// useful working range sits above 1 and that giving up several of these costs less than it sounds.
const float gs_adaptive_pixel_scale_min = 1.0f;
const float gs_adaptive_pixel_scale_max = 10.0f;

// How often the controller takes a decision. Every frame contributes a timing sample, but acting on each one would
// steer by noise, and each change that lands can cost a re-traversal - so decisions are slow and the samples in
// between are averaged.
const double gs_adaptive_eval_interval_s = 1.0;

// The bottom of the band, as a fraction of the target rate. The owner's zones: 50fps is the aim and 40 is where this
// mechanism should start working, so 0.8. Between the two the controller deliberately does nothing - that band is the
// answer to vsync being unable to report headroom, and it has to be genuinely wide or the controller either freezes
// inside it or ratchets one way out of it.
const float gs_adaptive_floor_fraction = 0.8f;
const size_t splat_tex_width = 4096; // Gives ~16.7M splat capacity where GL_MAX_TEXTURE_SIZE >= 16384, which is common.
const int splat_index_attribute_loc = 1; // Forced in buildShadersIfNeeded().  Slot 1 is otherwise "normal_in", which splats have no use for.

const int coarse_key_bits = 16; // How many high bits of the sort key the coarse stage buckets on, i.e. 65536 evenly spaced depth slices.

const int max_concurrent_traversals = 2; // Mirrors max_concurrent_sorts above, for the same reason: caps traversal scratch memory at roughly this many clouds' worth rather than letting it scale with the world.


// SESSION080 DIAGNOSTIC: how many traversal tasks are inside their saturation phase (gather -> grid -> test) right now.
//
// This measures a quantity nothing in the pipeline currently bounds. max_concurrent_traversals above caps traversals,
// but num_traversals_in_flight is decremented when message 1 is drained - which happens BEFORE the saturation phase
// starts, since session076 deliberately publishes the frontier first so the picture does not wait for the prune. The
// saturation phase therefore runs entirely outside that accounting, and any number of them can be in flight at once:
// each traversal's own saturation continues while the next traversal, and the next, are kicked off and run.
//
// That is fine while a traversal's front half (expand+sort+soa) outlasts its saturation, because then the next kick
// cannot arrive before the previous prune is finished. Session080's STEP B inverts that relation by design - it makes
// the front half much cheaper without touching saturation - so this number is the direct test of whether saturations
// then start stacking up and competing for the pool with each other and with the per-frame filter. It is the hypothesis
// the pipeline restructuring would be built on, so it is measured rather than assumed.
//
// Reported per traversal as sat_conc= (value on entry, i.e. how many others were already running) and sat_conc_peak=
// (highest value seen while this one ran) - see [gsr-sat].
glare::AtomicInt gs_sat_phases_running(0);


// SESSION080 DIAGNOSTIC: keeps gs_sat_phases_running correct on every exit from the saturation phase, including the
// early-out when a traversal finds no occluders at all.
struct GsSatPhaseScope
{
	GsSatPhaseScope() : peak(0)
	{
		on_entry = (size_t)gs_sat_phases_running.increment(); // Returns the value BEFORE the increment: how many others were already inside.
		peak = on_entry + 1;
	}
	~GsSatPhaseScope() { gs_sat_phases_running.decrement(); }

	// Called at the phase's internal boundaries: the count can rise after we entered, and the peak is what says whether
	// this traversal actually had to share the pool, rather than merely how things looked at the instant it started.
	void sample() { peak = myMax(peak, (size_t)gs_sat_phases_running.getVal()); }

	size_t on_entry;
	size_t peak;
};


// Whether any member of this cloud has a built LoD tree.  Drives two things: whether kickOffTraversals() bothers picking
// this cloud at all (nothing to choose between without a tree), and, temporarily, whether kickOffSorts() skips it (see
// that function's use of this) - a cloud with no tree anywhere in it behaves exactly as before LoD existed.
bool cloudHasLodTree(const SplatCloud& cloud)
{
	for(size_t m=0; m<cloud.members.size(); ++m)
		if(!cloud.members[m].splat_data->lod_tree.empty())
			return true;
	return false;
}

// The world-space radius a splat covers, from its centre.  gaussian_splat_vert_shader.glsl cuts the splat off at 3
// sigma, so this matches what actually gets drawn.  Rotation is irrelevant: this bounds the splat in every direction.
const float splat_cutoff_sigmas = 3.f;


// How many texture rows, each splat_tex_width texels wide, are needed to hold num_splats splats.
size_t texHeightForSplatCount(size_t num_splats)
{
	return myMax<size_t>(1, Maths::roundedUpDivide(num_splats * texels_per_splat, splat_tex_width));
}


// Packs a range of splats into RGBA32F texels: 4 texels per splat.
void packSplatTexels(const js::Vector<Vec3f, 16>& positions, const js::Vector<Vec3f, 16>& scales, const js::Vector<Vec4f, 16>& rotations,
	const js::Vector<Vec4f, 16>& colours, size_t begin_splat, size_t end_splat, float* texel_data)
{
	for(size_t i=begin_splat; i<end_splat; ++i)
	{
		const Vec3f& pos   = positions[i];
		const Vec3f& scale = scales[i];
		const Vec4f& rot   = rotations[i];
		const Vec4f& col   = colours[i];

		float* const t = texel_data + (i - begin_splat) * texels_per_splat * 4;
		t[ 0] = pos.x;      t[ 1] = pos.y;      t[ 2] = pos.z;      t[ 3] = scale.x;
		t[ 4] = scale.y;    t[ 5] = scale.z;    t[ 6] = rot[0];     t[ 7] = rot[1];
		t[ 8] = rot[2];     t[ 9] = rot[3];     t[10] = col[0];     t[11] = col[1];
		t[12] = col[2];     t[13] = col[3];     // t[14] and t[15] are left as zero.
	}
}


// The instanced quad geometry: a local-space unit square.  The vertex shader scales and orients this per splat from the
// projected 2D covariance, so this only needs to bound [-1, 1] in both axes.
//
// Built per cloud rather than shared between them: the cloud's bounds live in mesh_data->aabb_os, which is what the
// engine derives GLObject::aabb_ws from, so a shared mesh would give every cloud one shared bounding box.
Reference<OpenGLMeshRenderData> makeInstancedQuadMeshData(VertexBufferAllocator& allocator)
{
	Reference<OpenGLMeshRenderData> mesh_data = new OpenGLMeshRenderData();
	mesh_data->setIndexType(GL_UNSIGNED_SHORT);
	mesh_data->has_uvs = false;
	mesh_data->has_shading_normals = false;
	mesh_data->num_materials_referenced = 1;
	// The quad's own bounds.  Only a placeholder: the owning cloud overwrites aabb_os with its world-space bounds in
	// rebuildCloudAABB(), before the object is ever handed to the engine.  It must still be a valid finite box rather
	// than emptyAABBox(), whose infinities produce NaNs when transformed (inf * 0 in transformedAABBFast()).
	mesh_data->aabb_os = js::AABBox(Vec4f(-1, -1, 0, 1), Vec4f(1, 1, 0, 1));

	mesh_data->batches.resize(1);
	mesh_data->batches[0].material_index = 0;
	mesh_data->batches[0].prim_start_offset_B = 0;
	mesh_data->batches[0].num_indices = 6;

	VertexAttrib pos_attrib;
	pos_attrib.enabled = true;
	pos_attrib.num_comps = 3;
	pos_attrib.type = GL_FLOAT;
	pos_attrib.normalised = false;
	pos_attrib.stride = (uint32)(sizeof(float) * 3);
	pos_attrib.offset = 0;
	mesh_data->vertex_spec.attributes.push_back(pos_attrib);

	// Placeholder for the per-instance splat index attribute: disabled, with no VBO, until ensureGpuCapacity() and
	// rebuildVAO() build the index VBO and rebuild the VAO with it enabled.  This mirrors how GLMeshBuilding.cpp adds
	// disabled instance-matrix attributes that GLObject::enableInstancing() enables later.
	VertexAttrib splat_index_attrib;
	splat_index_attrib.enabled = false;
	splat_index_attrib.num_comps = 1;
	splat_index_attrib.type = GL_UNSIGNED_INT;
	splat_index_attrib.normalised = false;
	splat_index_attrib.integer_attribute = true;
	splat_index_attrib.instancing = true;
	splat_index_attrib.stride = (uint32)sizeof(uint32);
	splat_index_attrib.offset = 0;
	assert(mesh_data->vertex_spec.attributes.size() == (size_t)splat_index_attribute_loc);
	mesh_data->vertex_spec.attributes.push_back(splat_index_attrib);

	const float quad_verts[4 * 3] = {
		-1, -1, 0,
		 1, -1, 0,
		 1,  1, 0,
		-1,  1, 0
	};
	const uint16 quad_indices[6] = { 0, 1, 2, 0, 2, 3 };

	allocator.allocateBufferSpaceAndVAO(*mesh_data, mesh_data->vertex_spec, quad_verts, sizeof(quad_verts), quad_indices, sizeof(quad_indices));

	return mesh_data;
}


// Result of a background depth-sort, handed back to the main thread.  think() checks cloud_id, to drop results for a
// cloud that has since been merged away or removed, and then generation, to drop results whose cloud has been
// renumbered since.
class GaussianSplatSortResultMsg : public ThreadMessage
{
public:
	enum Stage
	{
		Stage_Coarse, // Fast approximate order.  Splats sharing a depth slice are in arbitrary relative order, which is far finer than the splats themselves.
		Stage_Precise // Exact front-to-back order.
	};

	uint64 cloud_id;
	uint64 generation;
	Stage stage;
	Reference<GaussianSplatSortScratch> scratch; // Holds the result buffer, and keeps it alive even if the renderer was torn down while the sort ran.

	// Front-to-back (nearest first) instance order, ready to write into the cloud's instance_index_vbo.
	const js::Vector<uint32, 16>& sortedIndices() const { return (stage == Stage_Coarse) ? scratch->coarse_indices : scratch->precise_indices; }
};


// Sorts one cloud front-to-back by camera distance, entirely on a worker thread.  No GL calls here.
class GaussianSplatSortTask : public glare::Task
{
public:
	GaussianSplatSortTask(uint64 cloud_id_, uint64 generation_, const Reference<GaussianSplatSortScratch>& scratch_, const Matrix4f& world_to_cam_,
		ThreadSafeQueue<Reference<ThreadMessage> >* result_queue_)
	:	cloud_id(cloud_id_), generation(generation_), scratch(scratch_), world_to_cam(world_to_cam_), result_queue(result_queue_)
	{}

	virtual void run(size_t /*thread_index*/) override
	{
		typedef GaussianSplatSortScratch::SortItem SortItem;
		struct SortItemGetKey { inline uint32 operator () (const SortItem& item) const { return item.key; } };

		const js::Vector<Vec3f, 16>& positions = scratch->positions_snapshot; // The frozen snapshot, never the live arrays.
		const size_t num_splats = positions.size();

		js::Vector<SortItem, 16>& items = scratch->items;
		js::Vector<SortItem, 16>& working_space = scratch->working_space;
		items.resizeNoCopy(num_splats);
		working_space.resizeNoCopy(num_splats);

		// Pass 1: distance from the camera to each splat, and the range those distances span.  Using distance rather
		// than depth along the camera's forward axis is what makes the order invariant to camera rotation.  The distance
		// is stashed in the key field as raw bits, and pass 2 turns it into the integer key in place.
		float min_dist = std::numeric_limits<float>::max();
		float max_dist = 0;
		for(size_t i=0; i<num_splats; ++i)
		{
			const Vec3f& p = positions[i];
			const float dist = maskWToZero(world_to_cam * Vec4f(p.x, p.y, p.z, 1.f)).length(); // world_to_cam is a rigid transform, so this is the true world-space distance.
			items[i].key = bitCast<uint32>(dist);
			items[i].splat_index = (uint32)i;
			min_dist = myMin(min_dist, dist);
			max_dist = myMax(max_dist, dist);
		}

		// Pass 2: quantise those distances linearly over the whole uint32 key range, so that ascending key order is
		// nearest-first, as the front-to-back "under" blend needs - see OpenGLEngine::drawSplatClouds().  A linear key,
		// rather than the float's own bit pattern (which sorts identically but spaces values by exponent), is what makes
		// the coarse stage meaningful.
		const float dist_range = myMax(max_dist - min_dist, 1.0e-9f); // Guards the degenerate equidistant case, where every key ends up 0 anyway.
		const double key_scale = (double)std::numeric_limits<uint32>::max() / (double)dist_range;
		for(size_t i=0; i<num_splats; ++i)
			items[i].key = (uint32)((double)(bitCast<float>(items[i].key) - min_dist) * key_scale);

		// Stage 1: a single counting-sort pass over the top coarse_key_bits of the key.  Much cheaper than the precise
		// sort below, and already fine-grained enough to look right on its own, so it's posted immediately rather than
		// leaving the view in the pre-move order until the precise sort finishes.
		{
			struct CoarseBucketChooser { inline size_t operator () (const SortItem& item) const { return item.key >> (32 - coarse_key_bits); } };
			Sort::serialCountingSortWithNumBuckets(items.data(), working_space.data(), num_splats, (size_t)1 << coarse_key_bits, CoarseBucketChooser());

			scratch->coarse_indices.resizeNoCopy(num_splats);
			for(size_t i=0; i<num_splats; ++i)
				scratch->coarse_indices[i] = working_space[i].splat_index;

			enqueueResult(GaussianSplatSortResultMsg::Stage_Coarse);
		}

		// Stage 2: the precise sort.  Stage 1 only wrote to working_space, which radixSort32BitKey() treats as scratch
		// anyway, so items is still in its original order here.
		scratch->temp_counts.resizeNoCopy(6144); // The size Sort::radixSort32BitKey() requires.
		Sort::radixSort32BitKey(items.data(), working_space.data(), num_splats, SortItemGetKey(), scratch->temp_counts.data(), scratch->temp_counts.size());

		scratch->precise_indices.resizeNoCopy(num_splats);
		for(size_t i=0; i<num_splats; ++i)
			scratch->precise_indices[i] = items[i].splat_index;

		enqueueResult(GaussianSplatSortResultMsg::Stage_Precise);
	}

private:
	void enqueueResult(GaussianSplatSortResultMsg::Stage stage)
	{
		Reference<GaussianSplatSortResultMsg> msg = new GaussianSplatSortResultMsg();
		msg->cloud_id = cloud_id;
		msg->generation = generation;
		msg->stage = stage;
		msg->scratch = scratch;
		result_queue->enqueue(msg);
	}

	uint64 cloud_id;
	uint64 generation;
	Reference<GaussianSplatSortScratch> scratch; // Keeps the snapshot and working buffers alive for the duration of the task.
	Matrix4f world_to_cam;
	ThreadSafeQueue<Reference<ThreadMessage> >* result_queue;
};


// Result of a background LoD traversal, handed back to the main thread.  drainTraversalResults() checks cloud_id, to drop
// results for a cloud that has since been merged away or removed, and then topology_generation, to drop results computed
// against a member set the cloud no longer has (see SplatCloud::topology_generation's comment for why this needs to be a
// stricter counter than the sort's structure_generation).
class GaussianSplatLodTraversalResultMsg : public ThreadMessage
{
public:
	uint64 cloud_id;
	uint64 topology_generation;
	Reference<GaussianSplatLodTraversalScratch> scratch; // Holds the result buffer, and keeps it alive even if the renderer was torn down while the traversal ran.

	// SESSION063: non-null when the task was asked to build the unculled frontier U(P) (split_filter_enabled). Carries the
	// SoA copy the worker gathered, so drainTraversalResults() can adopt it as the cloud's cache with no main-thread copy.
	//
	// SESSION081: this is now the ONLY message a traversal sends - saturation build and apply both moved out to their
	// own fully independent, differently-throttled pipelines (see GaussianSplatSaturationBuildTask and
	// GaussianSplatSaturationApplyTask), so there is no "follow-up" for a traversal to promise or send any more. A
	// traversal is, once again, purely "walk the tree, publish what it found."
	Reference<GaussianSplatUnculledFrontier> unculled_frontier;

	GaussianSplatLodTraversalResultMsg() : cloud_id(0), topology_generation(0) {}
};


// SESSION081: result of a background saturation barrier build - see GaussianSplatSaturationBarrier and
// GaussianSplatSaturationBuildTask. No staleness guard needed, unlike the traversal follow-up above: a barrier build
// is not "derived from" any one frontier in a way that a later one invalidates - drainSaturationBuildResults() only
// has to check the cloud still exists and its topology hasn't changed since the build was kicked (the same check
// every other result-drain in this file makes).
class GaussianSplatSaturationBuildResultMsg : public ThreadMessage
{
public:
	uint64 cloud_id;
	uint64 topology_generation;
	Reference<GaussianSplatSaturationBarrier> barrier;
};




// Why the traversal stopped unfolding the tree at a particular frontier node.  Only recorded when a caller asks for it
// (see GaussianSplatRenderer::getFrustumStructureReport()); the normal per-frame traversal doesn't collect any of this.
//
// The distinction that matters is between the reasons a knob can move and the one it can't: Converged/DensityCap/DepthCap/
// BudgetCap all mean "a coarser stand-in exists above this node and the traversal chose not to use it", so the LoD
// parameters still have room; Leaf means the tree has nothing coarser to offer at this position without going up a level
// that the traversal has already rejected as too big, i.e. the hierarchy is fully unfolded here and only changing the
// source cloud (merging or pruning splats) can reduce the fill cost.
enum FrontierStopReason
{
	FrontierStop_Leaf = 0,   // An original, unmerged splat - the finest detail this tree has.
	FrontierStop_Converged,  // pixel_scale fell to or below pixel_scale_limit, so expanding further would buy nothing visible.
	FrontierStop_DensityCap, // max_layer_density stopped expansion here - see GaussianSplatLodNode::layer_density.
	FrontierStop_DepthCap,   // max_tree_depth stopped expansion here.
	FrontierStop_BudgetCap,  // max_splats_budget stopped expansion, and this node was drained from the heap as-is.
	FrontierStop_NoTree,     // The member has no LoD tree at all, so every one of its splats is always selected.
	FrontierStop_SatBias,        // SESSION085 ETAP 3: the saturation barrier's LoD bias stopped expansion here - the node is behind saturated geometry, so its own merged stand-in is drawn instead of its subtree. Only produced when the bias ceiling is above 1.
	FrontierStop_OutOfFrustum, // SESSION055: the node's centre is outside the frustum (dilated by 1.5*feature_size to keep large nodes whose centre is just past a plane), so it and its subtree were skipped. Only produced when frustum-cull is on (see GaussianSplatRenderer::setFrustumCullEnabled). getFrustumStructureReport() disables cull, so this bucket stays 0 there - it exists so the runtime path can bucket cheaply and so the count matches what the fast path actually did.
	FrontierStop_OutOfDistRange, // SESSION072: the node's whole bounding sphere is outside the distance-slice shell (or, inverted, entirely inside it) - see GaussianSplatRenderer::getDistClampEnabled(). Only produced when the dist-clamp checkbox is on; getFrustumStructureReport() always leaves it off, same reasoning as FrontierStop_OutOfFrustum above.
	FrontierStop_NumReasons
};


// One selected node, as recorded for the structure report.  Carries where the node came from rather than just its cloud
// index, so the report can read the node's own tree fields (child_count, layer_density) without a reverse lookup.
//
// Recorded into a vector of its own rather than parallel to GaussianSplatLodTraversalScratch::selected_indices, because
// run() re-sorts that one front-to-back at the end and a parallel array would have to be permuted with it.  These records
// are self-describing, so their order doesn't matter.
struct FrontierNodeRecord
{
	uint32 cloud_idx;      // Index into the cloud's world-space arrays: member offset + tree_local_idx.
	uint32 member_idx;     // Index into GaussianSplatLodTraversalScratch::members_snapshot.
	uint32 tree_local_idx; // Index into that member's lod_tree.  Meaningless when stop_reason is FrontierStop_NoTree.
	uint16 depth;          // Levels below the tree root.  0 for a root, and for a member with no tree.
	uint16 stop_reason;    // A FrontierStopReason.
};


// Best-first LoD frontier selection for one cloud, entirely on a worker thread.  No GL calls here.
//
// Runs one shared max-heap across every member of the cloud that has a built LoD tree, rather than one heap per member.
// The reason a shared heap matters at all is to avoid reintroducing cross-object depth-blending bugs between members
// whose bounds intersect on screen without intersecting in 3D - but mergeIntersectingClouds() already guarantees that
// members with intersecting bounds share a cloud, so scoping the shared heap to "this cloud" (rather than "the whole
// world", which an earlier, pre-partitioning version of this renderer needed) protects exactly the members where it's
// load-bearing, and no more.  See GaussianSplatRenderer.h's Partitioning section.
//
// No frustum culling here, deliberately - the selection covers the whole cloud, including what is behind the camera.
// Nodes are prioritised by pixel_scale, which is size over distance; the direction the camera is facing is not an input
// to this task at all.  Two reasons, and they compound:
//
//  - Rotation stays free.  The selection, and the sort at the end of it, are both by distance from the camera rather
//    than by anything view-dependent, so turning on the spot changes neither and costs nothing whatsoever.  A
//    frustum-culled frontier would have to be recomputed every time the camera turned, which in a first-person client
//    is the most common camera motion by a wide margin - trading "three times the work, rarely" for "a third of the
//    work, constantly".
//  - A stale answer stays correct.  This runs asynchronously and lands a frame or two later, so whatever it produces is
//    always slightly behind the live camera.  A stale *complete* frontier is stale only in its choice of LoD level,
//    which is invisible; a stale *frustum-culled* one is missing the nodes that entered view while it was in flight,
//    which is a hole along the screen edge that grows with how fast the camera is turning.  Recovering from that needs
//    either a dilated frustum (giving back most of the saving) or a synchronous traversal (giving back the reason this
//    is a background task in the first place).
//
// The prize would be small in any case.  An off-screen node costs one index in the VBO, one instance, and four vertex
// shader invocations before clipping discards it - and no fill at all, which is where two thirds of this pass's time
// goes.  Vertex work does not show up in the measured profile even with the covariance recomputed per quad vertex.
//
// Frustum culling does happen, one level up: drawSplatClouds() tests each cloud's AABB once per frame, which is the
// right granularity for a world of many separate capture objects.  It rejects nothing in the degenerate case of a
// single cloud with the camera standing inside it - which is exactly what the scene this was tuned against is, so the
// off-screen share looks worse here than it will in a populated world.
//
// One consequence is not benign, and is called out at max_splats_budget's own accessor: the budget is spent on nodes
// behind the camera as readily as on ones in front of it.
//
// The intended fix is not a frustum test in this function.  It is the planned move to cluster-based LoD (a Nanite-style
// cluster DAG replacing this per-splat voxel tree - see the roadmap's "axis 3"), where the unit of selection is a
// cluster rather than a single splat.  A cluster can be rejected whole, for the cost of one AABB test, at a granularity
// coarse enough that a frame-late answer does not show as a hole - so the problem this comment describes disappears as
// a side effect of that change rather than needing its own mechanism.  That rewrite renumbers everything this task
// walks, which is the other half of why a per-node frustum test added now would be work thrown away.


// SESSION080: one slice of the frontier SoA build - see the call site in GaussianSplatLodTraversalTask::run(). Carries no
// chunk struct of its own because, unlike the occluder gather below, it reports nothing back: every slot it writes is
// determined by its own index, so there is no count to return and no compaction afterwards.
class GsFrontierSoATask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		for(size_t i=i_begin; i<i_end; ++i)
		{
			const uint32 idx = output[i];
			const Vec3f& p = positions[idx];
			out_indices[i] = idx;
			out_px[i] = p.x; out_py[i] = p.y; out_pz[i] = p.z;
			out_radius[i] = cull_radii[idx];
			out_is_coarse[i] = coarse_flags[i];
		}
	}

	size_t i_begin, i_end;
	const uint32* output; const float* coarse_flags;
	const Vec3f* positions; const float* cull_radii;
	uint32* out_indices;
	float* out_px; float* out_py; float* out_pz; float* out_radius; float* out_is_coarse;
};


// SESSION087: one slice of the post-sort unpack - see the call site in GaussianSplatLodTraversalTask::run(). Splits the
// sorted DistIdx stream into the two arrays the rest of the pipeline consumes. Measured serial at 12-16 ms (a quarter of
// sort_ms) purely because it was the one pass in the sort block nobody had split; it is the same pure map as
// GsFrontierSoATask - slot i is written from element i alone - so it parallelises without any reasoning about order.
class GsSortUnpackTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		for(size_t i=i_begin; i<i_end; ++i)
		{
			const uint32 packed = sorted[i].idx;
			out_indices[i] = packed & 0x7FFFFFFFu;             // Real cloud index (bit 31 stripped).
			out_coarse_flags[i] = (packed >> 31) ? 1.f : 0.f;  // 1 = coarse-floor node, for the filter's per-node dilation.
		}
	}

	size_t i_begin, i_end;
	const GsDistIdx* sorted;
	uint32* out_indices; float* out_coarse_flags;
};


// SESSION081 STAGE 4B: how far ahead the occluder gather prefetches the packed occluder record. Swept, not assumed -
// ns per occluder on the owner's reference scene, 5-7 barrier builds each, teleporting between viewpoints:
//
//   none 7.05 (6.62-7.91) | 8: 5.66 (5.49-11.62) | 16: 5.67 (5.31-11.76) | 32: 5.34 (5.14-5.46) | 64: 5.72 (5.38-6.31)
//
// 32 wins on the median, but what actually decides it is the SPREAD. The saturation build shares a pool with the
// traversal (session081 decoupling), and at 8 and 16 a build that overlaps a full ~13M-node traversal blows out to
// 7.5-11.6 ns - while every one of 32's samples was taken under exactly that contention and stayed inside 5.14-5.46.
// Issuing the prefetch further ahead leaves slack for the core to be pulled away and the line still to arrive in time.
// 64 is worse again, as expected: ~64 prefetches in flight against ~10-12 line-fill buffers per core, so lines are
// evicted before use. Both neighbours being worse is what makes this a measured optimum rather than a stopping point.
static const size_t gs_sat_gather_prefetch_dist = 32;


// SESSION085: GsSatGatherChunk / GsSatGatherTask - the frontier-derived occluder gather - are removed with the
// Fine and CoarseFloor sources they existed to serve. The tree walk builds its occluder set itself (see
// occluderTreeWalk() and GsSatWalkGatherTask), so nothing reads the frontier's SoA for occluders any more. This
// also retires the duplicated radius/alpha expression the two gathers had to keep in step: there is one left.








// SESSION082: one entry in the occluder tree walk's DFS stack - see GaussianSplatSaturationBuildTask::occluderTreeWalk().
// Deliberately smaller than the traversal's HeapItem: the walk carries no pixel_scale, no depth, and none of the
// once-per-branch inherited flags, because its stop rule reads only the node itself.
struct GsWalkItem { uint32 member_idx; uint32 tree_local_idx; };


// SESSION082: front-to-back sort key for that walk. Same shape as the traversal's DistIdxKey/DistIdxLess pair over the
// same shared GsDistIdx record; a second copy rather than a hoist because those two are members of
// GaussianSplatLodTraversalTask, and this walk exists precisely so the barrier does not depend on that class.
struct GsWalkDistKey  { inline float operator () (const GsDistIdx& x) const { return x.dist_sq; } };
// SESSION082 SELF-CHECK, TEMPORARY: orders by node index so the parallel and serial walks can be compared as sets - see the check in occluderTreeWalk().
struct GsWalkIdxLess  { inline bool operator () (const GsDistIdx& a, const GsDistIdx& b) const { return a.idx < b.idx; } };
struct GsWalkDistLess { inline bool operator () (const GsDistIdx& a, const GsDistIdx& b) const { return a.dist_sq < b.dist_sq; } };


// SESSION082: the tree walk's gather - fills the grid's five input arrays from the walk's own, already-sorted index
// list. SESSION085: the only occluder gather there is, now that the frontier-derived sources are retired. It reads
// positions[idx] as well as occl[idx] - two scattered reads per node, where the frontier path had one because it could
// copy positions straight out of the frontier's SoA. Measured session085 at ~10.6 ns/node against that path's
// ~5.4 ns/node, over 14-55x fewer nodes: the per-node cost roughly doubled and the total fell by an order of magnitude.
//
// The duplicated radius/alpha expression that used to sit here and in GsSatGatherTask, with a standing obligation to
// keep the two in step, is resolved: this is the surviving copy.
class GsSatWalkGatherTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/) override
	{
		for(size_t i=i_begin; i<i_end; ++i)
		{
			// Same reasoning, and the same measured distance, as the retired frontier gather's prefetch: the stride is whatever the
			// index list says, so the hardware prefetcher cannot follow it but we can. Both scattered reads are issued,
			// since this loop stalls on either of them.
			const size_t pf_i = i + gs_sat_gather_prefetch_dist;
			if(pf_i < i_end)
			{
				_mm_prefetch((const char*)&occl[src[pf_i].idx], _MM_HINT_T0);
				_mm_prefetch((const char*)&positions[src[pf_i].idx], _MM_HINT_T0);
			}

			const uint32 idx = src[i].idx;
			const Vec3f& pos = positions[idx];
			out_px[i] = pos.x; out_py[i] = pos.y; out_pz[i] = pos.z;

			const GsSatPackedOccluder p = occl[idx];

			const float ddx = pos.x - cam_pos_ws.x[0];
			const float ddy = pos.y - cam_pos_ws.x[1];
			const float ddz = pos.z - cam_pos_ws.x[2];
			const float d_len = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
			const float inv_d = d_len > 1.0e-6f ? (1.f / d_len) : 0.f; // Degenerate (node at the anchor) - the grid build skips such nodes anyway.

			const float nx = (float)p.nx * (1.f / 32767.f), ny = (float)p.ny * (1.f / 32767.f), nz = (float)p.nz * (1.f / 32767.f);
			const float cos_n = myMin(std::fabs(ddx*nx + ddy*ny + ddz*nz) * inv_d, 1.f);
			const float inv_t = gsSatUnpackHalf(p.inv_t_half);
			const float w = 1.f + cos_n*cos_n * (inv_t*inv_t - 1.f);

			out_radius[i] = (gs_sat_occluder_sigmas / gs_sat_half_g_scale) * gsSatUnpackHalf(p.g_half) * std::sqrt(std::sqrt(w));
			out_alpha[i] = adjustSplatAlpha(gsSatUnpackHalf(p.alpha_half) * (1.f / gs_sat_half_alpha_scale), alpha_gain, alpha_gamma); // The DRAWN opacity, not the stored one - see GaussianSplatRenderer::getAlphaGain().
		}
	}

	const GsDistIdx* src;        // The walk's selection, already sorted front-to-back.
	const Vec3f* positions;      // GaussianSplatCachedGeom::positions - world-baked node centres.
	const GsSatPackedOccluder* occl;
	float* out_px; float* out_py; float* out_pz; float* out_radius; float* out_alpha;
	Vec4f cam_pos_ws;
	float alpha_gain, alpha_gamma;
	size_t i_begin, i_end;
};


// SESSION081: builds a GaussianSplatSaturationBarrier from a snapshot of a cloud's most recent UNPRUNED frontier -
// see that class. Runs on kickOffSaturationBuilds()'s own cadence, independent of any one traversal: this is exactly
// the gather+build(grid+closing+erosion) code that used to run inline inside GaussianSplatLodTraversalTask::run(),
// unchanged in substance, only re-targeted to write a GaussianSplatSaturationBarrier instead of a
// GaussianSplatUnculledFrontier and to read its inputs from ctor fields instead of the traversal's own locals.
//
// anchor_pos_ws is the camera position AT KICK TIME (kickOffSaturationBuilds() reads it fresh, the same way
// kickOffTraversals() does) - it becomes the new barrier's own anchor, which is deliberately NOT the same point
// source_frontier was built at: the frontier can be arbitrarily older (whatever the cloud's last_unpruned_ufrontier
// currently holds), while the barrier's ball guarantee has to be centred on where the camera actually is now.
class GaussianSplatSaturationBuildTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/) override
	{
		// SESSION085: the frontier is no longer READ for occluders (the tree walk supplies its own - see below); its size
		// is still recorded, because frontier_n is what [gsr-sat-build] reports the walk's occluder count against.
		const GaussianSplatUnculledFrontier& uf = *source_frontier;
		const size_t n = uf.indices.size() + (uf.far_block.nonNull() ? uf.far_block->indices.size() : 0);

		Reference<GaussianSplatSaturationBarrier> barrier = new GaussianSplatSaturationBarrier();
		barrier->frontier_n = n; // SESSION081 PLAN ETAP 0 DIAGNOSTIC - see the field's comment.
		barrier->anchor_pos_ws = anchor_pos_ws;
		barrier->built_time_real_s = Clock::getCurTimeRealSec();
		barrier->threshold_used = saturation_threshold;
		barrier->subdiv_used = grid_subdiv;
		barrier->coarse_pixel_scale_used = coarse_pixel_scale; // SESSION088 - see the field's comment.
		barrier->region_radius_used = region_radius;
		barrier->closing_tiles_used = closing_tiles;

		// ---- Gather: collect this build's occluders. SESSION085: always the tree walk - see occluderTreeWalk(), and
		// the note above GaussianSplatRenderer for the measurements that retired the two frontier-derived sources. ----
		js::Vector<float, 16> occl_px, occl_py, occl_pz, occl_radius, occl_alpha;
		Timer sat_gather_timer;
		occluderTreeWalk(*barrier, occl_px, occl_py, occl_pz, occl_radius, occl_alpha);
		// SESSION085: less the self-check, which occluderTreeWalk() times for exactly this - see walk_verify_ms.
		barrier->gather_ms = sat_gather_timer.elapsed() * 1.0e3 - walk_verify_ms;
		barrier->num_occluders = occl_px.size();

		// ---- Build: the grid itself (+closing +erosion) - see GaussianSplatSaturationGrid.h. Empty occluder set is a
		// valid, deliberate outcome (an isolated interior, or the very first build for a cloud) - the barrier is still
		// sent, with sat_grid_res left at 0 ("no grid"), so the cloud gets a real, non-null barrier and
		// kickOffSaturationBuilds() does not re-kick every single frame trying to build one from the same empty input. ----
		if(!occl_px.empty())
		{
			barrier->sat_grid_res = gsSatGridResForFocal(focal_px, coarse_pixel_scale, grid_subdiv);
			Timer sat_grid_timer;

			if(task_manager != NULL && !overlay_requested)
			{
				if(diag_log) barrier->concurrency_used = (int)task_manager->getConcurrency(); // SESSION081 PLAN ETAP 0
				gsBuildSaturationGridParallel(occl_px.data(), occl_py.data(), occl_pz.data(), occl_radius.data(), occl_alpha.data(), occl_px.size(),
					anchor_pos_ws, barrier->sat_grid_res, saturation_threshold, region_radius, closing_tiles, barrier->sat_depth, *task_manager,
					diag_log ? &barrier->diag_writers : NULL, diag_log ? &barrier->diag_tile_writes : NULL,
					diag_log ? barrier->diag_tile_stats : NULL,
					build_scratch.nonNull() ? &build_scratch->scratch : NULL, // SESSION081: caller-owned working buffers, so the build stops reallocating ~370MB per call - see GsSatBuildScratch.
					barrier->erode_stats,
					diag_log ? &barrier->close_ms : NULL, diag_log ? &barrier->erode_ms : NULL, // SESSION081 PLAN ETAP 0
					diag_log ? &barrier->rec_ms : NULL, diag_log ? &barrier->dep_ms : NULL,
					diag_log ? &barrier->num_recs : NULL, diag_log ? &barrier->num_strips : NULL,
					diag_log ? &barrier->num_blocks : NULL,
					diag_log ? &barrier->gate1_survivors : NULL, // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY.
					diag_log ? &barrier->rec_task_ms_min : NULL, diag_log ? &barrier->rec_task_ms_max : NULL, diag_log ? &barrier->rec_task_ms_mean : NULL, // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY.
					diag_log ? barrier->inf_hist : NULL, // SESSION082 PLAN §1.1, TEMPORARY.
					diag_log ? barrier->sched_stats : NULL); // SESSION081 SCHEDULING PROBE, TEMPORARY.
			}
			else
				gsBuildSaturationGrid(occl_px.data(), occl_py.data(), occl_pz.data(), occl_radius.data(), occl_alpha.data(), occl_px.size(),
					anchor_pos_ws, barrier->sat_grid_res, saturation_threshold, region_radius, closing_tiles, barrier->sat_depth,
					diag_log ? &barrier->diag_writers : NULL, diag_log ? &barrier->diag_tile_writes : NULL,
					overlay_requested ? &barrier->sat_accum_t : NULL, overlay_requested ? &barrier->sat_amp_sum : NULL,
					diag_log ? barrier->diag_tile_stats : NULL,
					barrier->erode_stats,
					diag_log ? &barrier->close_ms : NULL, diag_log ? &barrier->erode_ms : NULL); // SESSION081 PLAN ETAP 0 - no rec_ms/dep_ms here, this path has no record/strip split.

			barrier->grid_build_ms = sat_grid_timer.elapsed() * 1.0e3;
		}

		Reference<GaussianSplatSaturationBuildResultMsg> msg = new GaussianSplatSaturationBuildResultMsg();
		msg->cloud_id = cloud_id;
		msg->topology_generation = topology_generation;
		msg->barrier = barrier;
		result_queue->enqueue(msg);
	}

	// SESSION082: the walk itself, factored out so the serial path, the parallel prologue and every pool task run
	// literally the same code rather than copies that can drift apart - the same reason expandStack() is shared.
	//
	// The stop rule is the whole of it: descend only while the node still subtends more than one grid tile. Nothing is
	// inherited down a branch and nothing is shared between branches, which is what makes the parallel split exact
	// rather than approximate.
	//
	// breadth_first + pause_at_stack_size mirror expandStack()'s and exist for the same reason: a DFS dives, so its
	// stack can never accumulate the breadth a seed split needs. Order changes only the sequence nodes are appended
	// in, never WHICH nodes are selected - and the sort that follows settles the sequence anyway.
	void walkStack(std::vector<GsWalkItem>& stack, js::Vector<GsDistIdx, 16>& out, size_t& visited,
		bool breadth_first, size_t pause_at_stack_size)
	{
		const js::Vector<Vec3f, 16>& positions = geom->positions;
		const js::Vector<float, 16>& feature_sizes = geom->feature_size;

		size_t head = 0; // Read cursor; breadth-first only.
		for(;;)
		{
			const size_t pending = breadth_first ? (stack.size() - head) : stack.size();
			if(pending == 0)
				break;
			if(pause_at_stack_size != 0 && pending >= pause_at_stack_size)
				break; // Enough independent subtrees for the caller to spread; the pending entries are the seeds.

			const GsWalkItem top = breadth_first ? stack[head++] : stack.back(); // Copied, not referenced: the pushes below can reallocate.
			if(!breadth_first)
				stack.pop_back();
			++visited;

			const WalkMember& m = walk_members[top.member_idx];
			const GaussianSplatLodNode& node = m.splat_data->lod_tree[top.tree_local_idx];
			const uint32 cloud_idx = (uint32)(m.offset + top.tree_local_idx);

			const Vec3f& p = positions[cloud_idx];
			const float dist_sq = anchor_pos_ws.getDist2(Vec4f(p.x, p.y, p.z, 1.f));
			const float dist = myMax(std::sqrt(dist_sq), 1.0e-6f); // Clamped away from zero so a node sitting exactly at the anchor cannot produce an infinite angular size.

			if(node.child_count != 0 && (feature_sizes[cloud_idx] / dist) > walk_tile_ang)
			{
				for(uint32 c = node.child_start; c < (uint32)node.child_start + node.child_count; ++c)
				{
					GsWalkItem child; child.member_idx = top.member_idx; child.tree_local_idx = c;
					stack.push_back(child);
				}
				continue;
			}

			GsDistIdx d; d.dist_sq = dist_sq; d.idx = cloud_idx;
			out.push_back(d);
		}

		if(head > 0) // Breadth-first only: drop what was consumed, so `stack` holds exactly the unfinished seeds.
			stack.erase(stack.begin(), stack.begin() + head);
	}


	// SESSION082: one seed subtree's worth of the walk, on a pool thread, into its own output vector. No shared mutable
	// state at all - see occluderTreeWalk()'s determinism note.
	class GsOccWalkTask : public glare::Task
	{
	public:
		GsOccWalkTask() : parent(NULL), visited(0) {}

		virtual void run(size_t /*thread_index*/) override
		{
			parent->walkStack(stack, selected, visited, /*breadth_first=*/false, /*pause_at_stack_size=*/0);
		}

		GaussianSplatSaturationBuildTask* parent; // Outlives every task: runTaskGroup() blocks until they have all finished.
		std::vector<GsWalkItem> stack;            // This task's seeds, and its own working stack thereafter.
		js::Vector<GsDistIdx, 16> selected;
		size_t visited;
	};


	// SESSION082: the tree-walk occluder source, the only one since session085. Chooses the barrier's
	// occluders by walking the LoD trees here, in the build, instead of filtering a set the render traversal chose.
	//
	// Three things follow from doing it here rather than inside the traversal, and together they are the point:
	//   - it is anchored at THIS build's anchor_pos_ws, so the front-to-back order it produces is exact for the barrier
	//     being built, rather than inherited from wherever the source frontier happened to be traversed from;
	//   - the descent rule is the barrier's own. A node wider than one grid tile can still change which tile saturates
	//     and at what depth; a node narrower than a tile cannot - it and its parent land in the same tile - so below
	//     that size the merged stand-in carries the same information as the subtree for a fraction of the nodes. The
	//     occluder count therefore follows the GRID's resolution, and moves with it automatically if the grid is
	//     re-sized, instead of following the render's quality target;
	//   - nothing in it needs a frontier to exist. That is the property the Nanite-streaming direction needs: a walk of
	//     a resident hierarchy is this same shape, where filtering a render frontier is not.
	//
	// The one cost the frontier sources do not pay is the sort - they inherit their order from the traversal's own -
	// so it is measured and reported on its own (walk_sort_ms) rather than folded into a total.
	//
	// The walk itself is serial. Its branches are independent, exactly as the traversal's are (see expandParallel()),
	// so it parallelises the same way if the measurement shows that is worth doing; leaving it serial here keeps the
	// first number reported an honest, unoptimised one.
	void occluderTreeWalk(GaussianSplatSaturationBarrier& barrier,
		js::Vector<float, 16>& occl_px, js::Vector<float, 16>& occl_py, js::Vector<float, 16>& occl_pz,
		js::Vector<float, 16>& occl_radius, js::Vector<float, 16>& occl_alpha)
	{
		const js::Vector<Vec3f, 16>& positions = geom->positions;

		// The resolution this barrier is about to be built at - the same call run() makes below, so the walk's stop
		// rule and the grid's tile size cannot drift apart.
		const int res = gsSatGridResForFocal(focal_px, coarse_pixel_scale, grid_subdiv);
		walk_tile_ang = myMax(gsSatGridTileAngle(res), 1.0e-6f); // Stored, not local: every walk task reads it - see walkStack().
		walk_verify_ms = 0.0; // SESSION085: set only if the self-check below runs, and the caller subtracts it unconditionally.

		Timer walk_timer;
		js::Vector<GsDistIdx, 16> selected;
		size_t visited = 0;

		std::vector<GsWalkItem> stack;
		for(size_t mi=0; mi<walk_members.size(); ++mi)
		{
			const WalkMember& m = walk_members[mi];
			// A member with no tree contributes no occluders at all here, where the frontier sources would have carried
			// its splats individually - see the session082 plan's open items. Hidden members contribute nothing either
			// way, matching the traversal.
			if(m.hidden || m.splat_data.isNull() || m.splat_data->lod_tree.empty())
				continue;
			GsWalkItem it; it.member_idx = (uint32)mi; it.tree_local_idx = 0; // Root is always node 0 - see buildGaussianSplatLodTree().
			stack.push_back(it);
		}

		// SESSION082: spread the walk across the pool the same way expandParallel() spreads the traversal's - walk the
		// top serially just far enough to have a supply of independent subtrees, then run one task per bucket of them.
		// The branches are independent by construction: every array read is a frozen snapshot, the stop rule reads only
		// the node itself, and nothing is inherited down a branch. Unlike the traversal this walk has no coarse-capture
		// flag, no far-cut mark and no budget cap, so there is no shared state at all and hence no equivalent of that
		// function's over-budget serial fallback.
		//
		// Deterministic despite the threading: which seeds land in which bucket is fixed by the loop below, and the
		// fragments are concatenated in TASK order, so thread scheduling cannot reorder anything. The only difference
		// from a purely serial walk is the relative order of nodes at exactly equal dist_sq, which the stable sort
		// below would otherwise have left in DFS order - the same cosmetic difference expandParallel() documents.
		const size_t concurrency = (task_manager != NULL) ? myMax<size_t>(1, (size_t)task_manager->getConcurrency()) : 1;
		if(task_manager != NULL)
		{
			// Seeds cut far finer than the thread count, for the reason expandParallel() gives: subtree cost is wildly
			// uneven (the member the camera stands inside dwarfs the others), so an even split of the SUBTREES is not
			// an even split of the work. Many small tasks let the pool even that out itself.
			walkStack(stack, selected, visited, /*breadth_first=*/true, /*pause_at_stack_size=*/concurrency * 16);
			if(!stack.empty())
			{
				const size_t num_tasks = myMin(stack.size(), concurrency * 4);
				glare::TaskGroupRef group = new glare::TaskGroup();
				group->tasks.resize(num_tasks);
				for(size_t t=0; t<num_tasks; ++t)
				{
					Reference<GsOccWalkTask> task = new GsOccWalkTask();
					task->parent = this;
					for(size_t si=t; si<stack.size(); si += num_tasks) // Round-robin, so a task gets a spread of the tree rather than a contiguous run of siblings, whose costs are correlated.
						task->stack.push_back(stack[si]);
					group->tasks[t] = task;
				}
				stack.clear();
				task_manager->runTaskGroup(group);

				for(size_t t=0; t<num_tasks; ++t) // Task order, not completion order - see the determinism note above.
				{
					const GsOccWalkTask* const task = static_cast<const GsOccWalkTask*>(group->tasks[t].ptr());
					visited += task->visited;
					const size_t base = selected.size();
					selected.resize(base + task->selected.size());
					if(!task->selected.empty())
						std::memcpy(selected.data() + base, task->selected.data(), task->selected.size() * sizeof(GsDistIdx));
				}
			}
		}
		else
			walkStack(stack, selected, visited, /*breadth_first=*/false, /*pause_at_stack_size=*/0);
		barrier.walk_ms = walk_timer.elapsed() * 1.0e3; // Stopped HERE, before the self-check below - that check runs a whole second walk, and timing it as part of the first would make walk_ms meaningless under "diag", which is the mode most likely to be captured.
		barrier.walk_visited = visited;

		// SESSION082 SELF-CHECK, TEMPORARY: the parallel walk must select exactly the set the serial one does. Runs the
		// serial walk a second time and compares, under the existing "diag" checkbox rather than a switch of its own -
		// that checkbox already means "spend real time measuring what is normally left alone", which is precisely this.
		//
		// Compared as a SET (sorted by node index), not element-for-element in walk order: the two differ in the order
		// they append equidistant nodes, which is the one documented and deliberate difference between them. What must
		// not differ is which nodes were chosen and at what distance - a discrepancy there would mean a branch was
		// walked twice or missed, which is exactly the class of defect a counter cannot see (session081's parallel
		// compaction race is the precedent).
		if(diag_log && task_manager != NULL)
		{
			// SESSION085: timed so the caller can take it back out of gather_ms. walk_ms already excluded it (see above), but
			// gather_ms is wall time around this whole function, so with "diag" on the self-check's second full walk landed
			// inside the very number used to judge the walk's cost - measured 3-5x inflation (250ms reported against 50ms
			// real). A diagnostic must not change the measurement it is printed beside.
			Timer verify_timer;
			js::Vector<GsDistIdx, 16> serial_sel;
			size_t serial_visited = 0;
			std::vector<GsWalkItem> serial_stack;
			for(size_t mi=0; mi<walk_members.size(); ++mi)
			{
				const WalkMember& m = walk_members[mi];
				if(m.hidden || m.splat_data.isNull() || m.splat_data->lod_tree.empty())
					continue;
				GsWalkItem it; it.member_idx = (uint32)mi; it.tree_local_idx = 0;
				serial_stack.push_back(it);
			}
			walkStack(serial_stack, serial_sel, serial_visited, /*breadth_first=*/false, /*pause_at_stack_size=*/0);

			bool match = (serial_sel.size() == selected.size()) && (serial_visited == visited);
			if(match)
			{
				std::vector<GsDistIdx> a(selected.begin(), selected.end()), b(serial_sel.begin(), serial_sel.end());
				std::sort(a.begin(), a.end(), GsWalkIdxLess());
				std::sort(b.begin(), b.end(), GsWalkIdxLess());
				for(size_t i=0; i<a.size(); ++i)
					if(a[i].idx != b[i].idx || a[i].dist_sq != b[i].dist_sq)
					{ match = false; break; }
			}
			walk_verify_ms = verify_timer.elapsed() * 1.0e3; // SESSION085 - see the Timer above.
			conPrint("[gsr-occ-verify] cloud=" + toString(cloud_id) + " parallel=" + uInt64ToStringCommaSeparated(selected.size()) +
				" serial=" + uInt64ToStringCommaSeparated(serial_sel.size()) +
				" visited " + uInt64ToStringCommaSeparated(visited) + "/" + uInt64ToStringCommaSeparated(serial_visited) +
				(match ? " PASS" : " ***FAIL***"));
		}

		// Nearest first - the order the grid's accumulation depends on (see gsBuildSaturationGrid()). Same two sorts,
		// same threshold between them, as the traversal uses on its own output.
		Timer sort_timer;
		const size_t count = selected.size();
		if(count > 0)
		{
			js::Vector<GsDistIdx, 16> sort_scratch;
			sort_scratch.resizeNoCopy(count);
			const size_t parallel_sort_min_elements = 16384;
			if(task_manager != NULL && count >= parallel_sort_min_elements)
				Sort::radixSortWithParallelPartition<GsDistIdx, GsWalkDistKey>(*task_manager, selected.data(), (uint32)count, GsWalkDistKey(), sort_scratch.data(), /*put_result_in_working_space=*/false);
			else
				Sort::floatKeyAscendingSort(selected.data(), count, GsWalkDistLess(), GsWalkDistKey(), sort_scratch.data(), /*put_result_in_working_space=*/false);
		}
		barrier.walk_sort_ms = sort_timer.elapsed() * 1.0e3;

		occl_px.resizeNoCopy(count); occl_py.resizeNoCopy(count); occl_pz.resizeNoCopy(count);
		occl_radius.resizeNoCopy(count); occl_alpha.resizeNoCopy(count);
		if(count == 0)
			return;

		// Chunked exactly like the frontier gather, and for the same reason: the cost is DRAM miss latency on the
		// scattered reads, so what threads buy is outstanding misses rather than arithmetic.
		const size_t num_chunks = myMax<size_t>(1, myMin(concurrency * 4, count / 16384));
		glare::TaskGroupRef group = new glare::TaskGroup();
		for(size_t c=0; c<num_chunks; ++c)
		{
			Reference<GsSatWalkGatherTask> t = new GsSatWalkGatherTask();
			t->src = selected.data(); t->positions = positions.data(); t->occl = geom->sat_occl.data();
			t->out_px = occl_px.data(); t->out_py = occl_py.data(); t->out_pz = occl_pz.data();
			t->out_radius = occl_radius.data(); t->out_alpha = occl_alpha.data();
			t->cam_pos_ws = anchor_pos_ws;
			t->alpha_gain = alpha_gain; t->alpha_gamma = alpha_gamma;
			t->i_begin = (count * c) / num_chunks;
			t->i_end   = (count * (c + 1)) / num_chunks;
			if(task_manager != NULL) group->tasks.push_back(t); else t->run(0);
		}
		if(task_manager != NULL)
			task_manager->runTaskGroup(group);
	}


	uint64 cloud_id, topology_generation;
	Reference<GaussianSplatUnculledFrontier> source_frontier; // Its own arrays + far_block - see the segment enumeration above.
	Reference<GaussianSplatCachedGeom> geom;
	Reference<GaussianSplatSaturationBuildScratch> build_scratch; // SESSION081 - see GsSatBuildScratch. Held by reference so it survives its cloud being removed mid-build.
	Vec4f anchor_pos_ws;
	float saturation_threshold, grid_subdiv, region_radius;
	int closing_tiles;
	float focal_px, coarse_pixel_scale; // Needed for gsSatGridResForFocal() - see GaussianSplatRenderer::kickOffSaturationBuilds().
	float alpha_gain, alpha_gamma;
	bool diag_log, overlay_requested;
	// SESSION082: the cloud's members as of kick time, so the tree walk can run here without a frontier and without the
	// traversal's scratch. Filled by kickOffSaturationBuilds() rather than taken from GaussianSplatCachedGeom, because
	// `hidden` can change without a topology-generation bump (see GaussianSplatRenderer::setObjectHidden()) and would go
	// stale in a generation-scoped cache.
	struct WalkMember { GaussianSplatDataRef splat_data; size_t offset; bool hidden; };
	std::vector<WalkMember> walk_members;
	float walk_tile_ang; // SESSION082: angular size of one grid tile, derived in occluderTreeWalk() from this build's own gsSatGridResForFocal() call so the walk's stop rule and the grid's resolution cannot drift apart.
	double walk_ms, walk_sort_ms; // SESSION082: the walk's own two costs, reported apart from gather_ms - see the barrier's fields of the same name.
	double walk_verify_ms;        // SESSION085: cost of the "diag" self-check below, subtracted back out of gather_ms by the caller so a diagnostic cannot inflate the number it is printed beside. 0 when the check did not run.
	size_t walk_visited;          // SESSION082: nodes the walk touched, against the number it selected - the walk's own overhead ratio.
	glare::TaskManager* task_manager; // May be NULL - see gsBuildSaturationGridParallel()'s caller-side branch above.
	ThreadSafeQueue<Reference<ThreadMessage> >* result_queue;
};




// SESSION088: how much of the saturation barrier has MOVED between two grids, as a fraction of tiles in [0,1].
//
// This is the evidence a frozen far block's detail level rests on. The block's selection was made by asking the bias a
// question about the barrier; if the answer would still be the same everywhere, the selection is still the one this
// traversal would make, however far the camera has walked since. If it has changed, no amount of standing still makes
// the block right. That is the quantity, and until now it was proxied by camera displacement - see
// GaussianSplatRenderer::getSatBarrierAgreeTol() and the drift bound in the ctor below.
//
// A tile disagrees when its SATURATION STATE flipped (a direction that was blocked is now open, or the reverse - the
// case that actually changes a bias verdict), or when both are blocked but the depth moved by more than rel_tol of the
// nearer of the two. Relative, not absolute, because the bias reads a RATIO of distance to barrier depth: a 2m shift at
// 4m is a different statement about occlusion than the same 2m at 200m.
//
// Grids from different anchors are only comparable tile-for-tile while the anchors are close, since a tile is a
// DIRECTION and the two are measured from different origins. That is normally guaranteed - a barrier is rebuilt as soon
// as the camera leaves its R-ball, so consecutive ones sit within R of each other - but a long-lived block can hold a
// grid many rebuilds old. The failure is in the safe direction: distant anchors compare different rays, report heavy
// disagreement, and the block is discarded. Never the reverse.
//
// Cost: one linear pass over sat_grid_res^2 floats - 9,801 on the owner's interior. Run once per traversal kick on the
// main thread, against a walk that costs 60-180ms.
// SESSION088, SECOND CUT: reported as TWO components rather than one number, because the first cut's single figure was
// dominated by the half the bias barely reads, and that was invisible while they were summed.
//
//   flip  - the tile's saturation STATE changed: a direction that was blocked is now open, or the reverse. This is the
//           change that can actually flip a bias verdict, since gsSatBuriedRatioSq() returns "not behind the barrier"
//           for an unsaturated tile whatever the numbers around it say.
//   depth - both are still blocked, but the depth moved by more than rel_tol. The bias reads depth only through a
//           ratio, and its curve is ~89% of the way to the ceiling by ratio 3, so a 10% depth wobble moves the
//           multiplier by a fraction of a percent. Nearly all of this is churn the bias would never have noticed.
//
// The measurement that forced the split: consecutive builds on the owner's interior reported sat_tiles 79.6% -> 70.9%,
// i.e. ~9% of tiles changed state, while the summed figure read 68%. So ~59 of those 68 points were the depth clause.
// Judging a reuse threshold on the sum would have been judging it on the noise.
//
// mean_rel_depth is the average |da| / min(a,b) over tiles blocked in BOTH grids - the magnitude behind the depth
// count, so rel_tol can be chosen against the distribution instead of guessed at.
//
// Grids from different anchors are only comparable tile-for-tile while the anchors are close, since a tile is a
// DIRECTION measured from an origin. A long-lived block can hold a grid many rebuilds old; that failure is in the safe
// direction - distant anchors compare different rays, report heavy disagreement, and the block is discarded.
//
// Cost: one linear pass over sat_grid_res^2 floats - 9,801 on the owner's interior - once per traversal kick, against a
// walk that costs 60-180ms.
static void gsSatBarrierDisagreement(const js::Vector<float, 16>& cur, const js::Vector<float, 16>& ref, float rel_tol,
	float& out_flip_frac, float& out_depth_frac, float& out_mean_rel_depth)
{
	const size_t n = cur.size();
	if(n == 0 || n != ref.size())
	{
		// No grid, or grids of different resolutions (the subdiv knob moved): nothing comparable. Report total
		// disagreement on the flip component so the caller's threshold rejects, whatever it is set to.
		out_flip_frac = 1.f; out_depth_frac = 0.f; out_mean_rel_depth = 0.f;
		return;
	}

	size_t num_flip = 0, num_depth = 0, num_both_sat = 0;
	double rel_sum = 0.0;
	for(size_t i=0; i<n; ++i)
	{
		const float a = cur[i], b = ref[i];
		const bool a_sat = std::isfinite(a), b_sat = std::isfinite(b);
		if(a_sat != b_sat)
			++num_flip;
		else if(a_sat)
		{
			++num_both_sat;
			const float denom = myMin(a, b);
			const float rel = (denom > 0.f) ? (std::fabs(a - b) / denom) : 0.f;
			rel_sum += rel;
			if(rel > rel_tol)
				++num_depth;
		}
	}
	out_flip_frac  = (float)((double)num_flip  / (double)n);
	out_depth_frac = (float)((double)num_depth / (double)n);
	out_mean_rel_depth = (num_both_sat > 0) ? (float)(rel_sum / (double)num_both_sat) : 0.f;
}


class GaussianSplatLodTraversalTask : public glare::Task
{
public:
	// SESSION054: build (dist_sq, idx) pairs directly during traversal, so the sort phase reuses the distance already
	// computed for pixel_scale rather than re-scanning positions[] a second time. Squared distance is a monotone key,
	// preserves ascending sort order, and skips 3-8M sqrt calls that the previous getDist()-per-item scan cost.
	// SESSION080: at class scope rather than inside run(), so the parallel expand's tasks can name it too.
	typedef GsDistIdx DistIdx; // SESSION080: hoisted to file scope so the scratch can pool buffers of it - see GsDistIdx.
	struct DistIdxLess { inline bool operator () (const DistIdx& a, const DistIdx& b) const { return a.dist_sq < b.dist_sq; } }; // Nearest first (matches GaussianSplatSortResultMsg's convention for the front-to-back "under" blend). Used only by the small-N std::sort fallback inside floatKeyAscendingSort.
	struct DistIdxKey  { inline float operator () (const DistIdx& x) const { return x.dist_sq; } }; // Sort::floatKeyAscendingSort keys on this float; squared distance is non-negative so FloatFlip's positive-branch monotone mapping applies.

	// result_queue_ may be null, and frontier_record_ non-null, for a synchronous run made purely to inspect the frontier -
	// see GaussianSplatRenderer::getFrustumStructureReport().  Both are null/absent for the normal per-frame traversal,
	// where the enqueued result is the whole point and nothing wants the per-node breakdown.
	GaussianSplatLodTraversalTask(uint64 cloud_id_, uint64 topology_generation_, const Reference<GaussianSplatLodTraversalScratch>& scratch_,
		const Vec4f& cam_pos_ws_, float pixel_scale_limit_, size_t max_splats_budget_, float max_layer_density_, int max_tree_depth_, float focal_px_,
		const Planef* frustum_clip_planes_, int num_frustum_clip_planes_, bool frustum_cull_enabled_, // SESSION055: planes are copied into num_frustum_clip_planes below rather than pointed at, because OpenGLScene's own array is mutated by the draw path each frame and a worker running across a frame boundary would otherwise read torn values.
		const float* translation_dilation_, float rotation_dilation_rate_, // SESSION055: per-plane translation dilation (metres) + rotation dilation rate (rad, multiplied by dist-to-node inside cull) - see kickOffTraversals()'s anisotropic dilation block.
		ThreadSafeQueue<Reference<ThreadMessage> >* result_queue_,
		js::Vector<FrontierNodeRecord, 16>* frontier_record_ = NULL,
		bool build_unculled_frontier_ = false, // SESSION063: also emit the SoA U(P) into the result msg, for the split filter path.
		bool coarse_floor_enabled_ = false, float coarse_pixel_scale_ = 20.f, // SESSION063 K4: also capture a coarse floor into U(P) - see GaussianSplatUnculledFrontier::is_coarse.
		bool dist_clamp_enabled_ = false, float dist_clamp_min_ = 0.f, float dist_clamp_max_ = 0.f, bool dist_clamp_invert_ = false, // SESSION072: distance-slice early-cull, mirrors the frustum-cull block below - see GaussianSplatRenderer::getDistClampEnabled(). Defaults off, so getFrustumStructureReport()'s call site (which omits these) always sees the whole tree.
		bool sat_diag_log_ = false, // SESSION076 DIAGNOSTIC: gated by its own "sat diag" checkbox.
		bool coarse_layer_drawn_ = true, // SESSION076: whether anything will actually DRAW the captured coarse layer (the "coarse" checkbox). False means it was captured solely to feed the saturation grid, which lets this task both skip capturing nodes the grid cannot use and evict the rest once the grid is built - see the capture block and the compaction loop in run(). Defaults true, i.e. the pre-session076 behaviour, so callers that don't care are unaffected.
		glare::TaskManager* task_manager_ = NULL, // SESSION079: the pool this task is itself running on.
		const Reference<GaussianSplatUnculledFrontier>& prev_frontier_ = Reference<GaussianSplatUnculledFrontier>(), // SESSION080 DIAGNOSTIC (plan doc STEP A) - see the field's comment.
		bool sort_staleness_diag_enabled_ = false,
		float reuse_split_dist_ = 0.f, // SESSION080 STEP B: 0 = walk the whole tree, the pre-session080 behaviour - see GaussianSplatRenderer::getFrontierReuseSplitDist() and reuse_enabled below.
		const Reference<GaussianSplatSaturationBarrier>& sat_barrier_ = Reference<GaussianSplatSaturationBarrier>(), // SESSION085 ETAP 3: the cloud's barrier at kick time, for the LoD bias. Null (the default) leaves the bias off, so every existing call site is unaffected.
		float sat_bias_ceiling_ = 1.f, // SESSION085 ETAP 3: 1 = off - see GaussianSplatRenderer::getSatBiasCeiling().
		float reuse_drift_fraction_ = 0.25f, // SESSION086: was a static const - see GaussianSplatRenderer::getFrontierReuseDriftFraction(). 0.25 is session080's original value.
		size_t expand_seed_target_ = 0, // SESSION086: 0 = nothing measured yet, expandParallel() falls back to concurrency*16.
		float sat_barrier_agree_tol_ = 1.f, // SESSION088: 1 = accept any barrier change, i.e. the pre-session088 behaviour - see GaussianSplatRenderer::getSatBarrierAgreeTol().
		float sat_bias_exponent_ = 1.f) // SESSION088: 1 = session085's fixed 1/ratio^2 curve - see GaussianSplatRenderer::getSatBiasExponent().
	:	cloud_id(cloud_id_), topology_generation(topology_generation_), scratch(scratch_), cam_pos_ws(cam_pos_ws_),
		pixel_scale_limit(pixel_scale_limit_), max_splats_budget(max_splats_budget_), max_layer_density(max_layer_density_), max_tree_depth(max_tree_depth_), focal_px(focal_px_),
		num_frustum_clip_planes(num_frustum_clip_planes_), frustum_cull_enabled(frustum_cull_enabled_),
		rotation_dilation_rate(rotation_dilation_rate_),
		result_queue(result_queue_),
		frontier_record(frontier_record_),
		build_unculled_frontier(build_unculled_frontier_),
		coarse_floor_enabled(coarse_floor_enabled_), coarse_pixel_scale(coarse_pixel_scale_),
		dist_clamp_enabled(dist_clamp_enabled_), dist_clamp_min(dist_clamp_min_), dist_clamp_max(dist_clamp_max_), dist_clamp_invert(dist_clamp_invert_),
		sat_diag_log(sat_diag_log_), coarse_layer_drawn(coarse_layer_drawn_),
		task_manager(task_manager_), // SESSION079
		expand_seeds(0), expand_num_tasks(0), expand_prologue_ms(0.0), expand_task_max_ms(0.0), expand_task_sum_ms(0.0), // SESSION080 DIAGNOSTIC
		expand_workers(0), expand_seed_max_ms(0.0), // SESSION086
		expand_seed_target(expand_seed_target_), // SESSION086 - see GaussianSplatRenderer::updateExpandSeedTarget().
		expand_splice_reserve_ms(0.0), expand_splice_copy_ms(0.0), // SESSION080 DIAGNOSTIC (plan2 §4.1)
		prev_frontier(prev_frontier_), sort_staleness_diag_enabled(sort_staleness_diag_enabled_), // SESSION080 DIAGNOSTIC
		sat_barrier(sat_barrier_), sat_bias_ceiling(sat_bias_ceiling_), sat_bias_exponent(sat_bias_exponent_), // SESSION085 ETAP 3, SESSION088
		// Precomputed here, not per node: the DFS asks this millions of times. A barrier with sat_grid_res 0 is the "no
		// grid" convention (nothing saturated, or no barrier yet), and the bias would be inert for every node anyway.
		sat_bias_active(sat_bias_ceiling_ > 1.f && sat_barrier_.nonNull() && sat_barrier_->sat_grid_res != 0),
		reuse_drift_fraction(reuse_drift_fraction_), // SESSION086
		sat_barrier_agree_tol(sat_barrier_agree_tol_), diag_barrier_disagreement(0.f), // SESSION088 - see getSatBarrierAgreeTol(); the rest are measured in the block below.
		diag_barrier_flip(0.f), diag_barrier_depth(0.f), diag_barrier_mean_rel_depth(0.f), diag_barrier_measured(false),
		diag_live_barrier_time(0.0), diag_block_barrier_time(0.0),
		reuse_enabled(false), far_cut_is_frozen(false), reuse_prev_anchor_ws(0.f), reuse_split_dist(0.f) // SESSION080 §4.3 - derived below.
	{
		if(num_frustum_clip_planes < 0)
			num_frustum_clip_planes = 0;
		if(num_frustum_clip_planes > (int)staticArrayNumElems(frustum_clip_planes))
			num_frustum_clip_planes = (int)staticArrayNumElems(frustum_clip_planes); // Bounds guard against a scene ever growing past 6 planes; the cull just misses planes past the 6th, cannot false-cull.
		for(int i=0; i<num_frustum_clip_planes; ++i)
			frustum_clip_planes[i] = frustum_clip_planes_[i];
		for(int i=0; i<(int)staticArrayNumElems(translation_dilation); ++i)
			translation_dilation[i] = translation_dilation_ ? translation_dilation_[i] : 0.f;

		// SESSION080 §4.3: decide once, here, which of the three modes this traversal runs in - see reuse_enabled and
		// far_cut_is_frozen. Doing it in the constructor rather than in the walk keeps the per-node test down to two
		// bools and keeps every precondition in one readable place.
		//
		//   reuse off                          - walk everything, one self-contained frontier. The knob at 0, or a
		//                                        setting the cut cannot coexist with.
		//   building a block (frozen = false)  - walk everything, but mark the part beyond the cut so run() can split it
		//                                        off into a fresh far block. The generation's first traversal.
		//   frozen cut (frozen = true)         - walk only the near part and reference the existing block.
		if(reuse_split_dist_ > 0.f && build_unculled_frontier &&
			// The cut's geometry is defined on the unculled tree. With any of these on, the walk prunes for reasons the
			// frozen predicate knows nothing about, so a later traversal's near part and the block would no longer be
			// complementary.
			!frustum_cull_enabled && !dist_clamp_enabled && !coarse_floor_enabled)
		{
			reuse_enabled = true;
			reuse_split_dist = reuse_split_dist_;

			const GaussianSplatUnculledFrontier* const block = prev_frontier.nonNull() ? prev_frontier->far_block.ptr() : NULL;

			// SESSION088: measure how far the barrier has moved since this block's selection was made - see
			// gsSatBarrierDisagreement() and the acceptance clause that reads it below. Measured whenever there is a block
			// and a live bias, whatever the tolerance is set to, so the number is in the log to be read BEFORE it is
			// trusted to gate anything. Costs one pass over ~9,801 floats against a 60-180ms walk.
			if(block != NULL && sat_bias_active && !block->sat_depth_used.empty())
			{
				gsSatBarrierDisagreement(sat_barrier->sat_depth, block->sat_depth_used, 0.10f,
					diag_barrier_flip, diag_barrier_depth, diag_barrier_mean_rel_depth);
				// SESSION088, THIRD CUT: the gate reads the FLIP component alone. Measured on the owner's two scenes, the
				// summed figure ran at 68% (interior) / 36% (exterior) per barrier cadence while the part that can
				// actually change a bias verdict ran at 20% / 8.6% - so the sum was 1.75-2.4x dominated by depth churn the
				// bias reads only through a ratio whose curve is already ~89% saturated by ratio 3. Gating on the sum would
				// have meant no threshold below 68% could keep a block on the interior at all, which is exactly the shape
				// of session086's failed distance calibration, arrived at for the same reason: judging the mechanism on a
				// quantity it does not actually depend on. bd_depth stays measured and printed, just not gated on.
				diag_barrier_disagreement = diag_barrier_flip;
				diag_barrier_measured = true; // SESSION088: without this a not-measured row is indistinguishable from a perfect-agreement row, since every field above defaults to 0. The first cut's "instrument verified" check was exactly that collision and proved nothing.
				diag_live_barrier_time = sat_barrier->built_time_real_s; // SESSION088 DIAGNOSTIC, TEMPORARY - see the members.
				diag_block_barrier_time = block->sat_built_time_used;
			}

			if(block != NULL &&
				// The block must describe the same selection, or the nodes in it are not the ones our own stop rules
				// would have picked and the two halves would disagree about what a cut even means. Same key set
				// drainTraversalResults() checks, and for the same reason.
				prev_frontier->topology_generation == topology_generation &&
				prev_frontier->pixel_scale_limit == pixel_scale_limit &&
				prev_frontier->max_splats_budget == max_splats_budget &&
				prev_frontier->max_layer_density == max_layer_density &&
				prev_frontier->max_tree_depth == max_tree_depth &&
				prev_frontier->focal_px == focal_px &&
				// SESSION086: the barrier is deliberately NOT part of this key, though session085 etap 3 briefly made it
				// one. Reasoning, in the order it has to hold:
				//
				//  - COVERAGE is bias-independent, which is the only property the two halves must agree on. The cut
				//    predicate P below reads the frozen anchor and the node's own sphere and nothing else, and the bias
				//    stop sits under `!force_descend` exactly as the caps do, so a straddling node is never stopped by
				//    it. The build-mode and frozen-mode walks therefore resolve the cut at the same nodes whatever the
				//    barrier says - which is the property session083 (b) established and the bias does not disturb.
				//  - What the barrier does change is the DETAIL LEVEL chosen below the cut. Requiring it to match made
				//    the far block inheritable only while the barrier held still, and the barrier is rebuilt every time
				//    the camera leaves its R-ball (session085: ~180ms at R=0.5, ~49ms at R=0) - far more often than a
				//    traversal completes. So the key did not make reuse conservative, it made it never happen.
				//  - And the staleness it was guarding against is one this scheme already accepts by construction: the
				//    far segment's whole selection is frozen for up to reuse_max_drift_fraction of the split distance.
				//    A bias from a barrier one or two rebuilds old is the same kind of error, in the same segment,
				//    bounded by the same trigger - and since the prune's removal (etap 6) its failure mode is geometry
				//    that is momentarily too coarse, never geometry that is missing.
				//
				// The field itself stays: the sort-staleness diagnostic in run() still keys on it, where comparing two
				// frontiers cut under different barriers really would measure the barrier instead of the motion.
				prev_frontier->reuse_split_dist_used == reuse_split_dist_ && // The knob moved: the frozen cut is at the wrong distance now.
				// The accumulated-drift bound, i.e. the full-rebuild safety trigger. The cut is frozen, so a node beyond
				// it is never re-examined however far the camera travels; without this it would keep the key it was given
				// when the block was built, for an unbounded number of traversals. See
				// GaussianSplatUnculledFrontier::reuse_base_anchor_ws for why the retreating case cannot self-correct.
				//
				// Expressed as a fraction of the split distance rather than as an absolute, because that is the quantity
				// it is answerable to: an ordering error of size E among nodes at range D matters in proportion to E/D,
				// and the split distance is the range at which we have declared order to stop mattering much. So capping
				// drift at a fixed fraction of it caps the relative error, at any scene scale and any knob setting, with
				// no second number to tune. A traversal that trips this walks the whole tree and builds a fresh block for
				// the ones after it - visible in the log as reuse_n dropping to 0 for one traversal.
				cam_pos_ws_.getDist(block->anchor_pos_ws) <= reuse_split_dist_ * reuse_drift_fraction_ &&
				// SESSION088: the LoD half of that same bound, now answering to its own evidence - see
				// GaussianSplatRenderer::getSatBarrierAgreeTol() and gsSatBarrierDisagreement().
				//
				// The clause above is an ORDERING budget and session080 derived it as one. Session085's LoD bias then gave
				// the block a second dependence on the barrier, far more sensitive than order and not a function of camera
				// displacement at all, and it was left to the same single number - which is why session086 could find no
				// value that worked: at 0.10 and 0.05 the far field visibly lagged, and the 0.02 that did not was no
				// cheaper than never reusing. A distance cannot express "the occlusion in that direction changed", so no
				// setting of it was going to.
				//
				// This asks the barrier instead. tol 1 accepts any change, which is the pre-session088 behaviour exactly;
				// below 1 the block is discarded when the grid has moved further than that, whether it took the camera
				// five metres or one. Ordering and detail are now bounded by the quantity each is actually about, which is
				// what makes raising reuse_drift_fraction (the order budget, which is genuinely tolerant) safe to try.
				(!sat_bias_active || block->sat_depth_used.empty() || diag_barrier_disagreement <= sat_barrier_agree_tol))
			{
				far_cut_is_frozen = true;
				reuse_prev_anchor_ws = block->anchor_pos_ws; // The FROZEN anchor, not the predecessor's - that is the whole point.
				inherited_far_block = prev_frontier->far_block;
			}
			else
				reuse_prev_anchor_ws = cam_pos_ws_; // Building a block: the cut is frozen at where the camera is right now.
		}
	}

	// SESSION080 §4.3: fill one frontier's SoA from a compacted index list. Both segments of a split frontier are built
	// with this - the near one into the published frontier, the far one into the block it references - so there is a
	// single copy of the gather rather than two that can drift apart.
	//
	// A pure map: element i reads positions[src[i]]/cull_radii[src[i]] and writes only slot i, so the chunks need nothing
	// from each other and the result is bit-identical to the serial loop. Chunked like the occluder gather, and for the
	// same reason: the cost is DRAM miss latency on the scattered reads into the 30M-entry geometry arrays, not
	// arithmetic, so what threads buy is outstanding misses.
	void buildFrontierSoA(GaussianSplatUnculledFrontier& out, const uint32* src, const float* src_coarse, size_t count,
		const js::Vector<Vec3f, 16>& positions, const js::Vector<float, 16>& cull_radii)
	{
		out.indices.resizeNoCopy(count);
		out.px.resizeNoCopy(count); out.py.resizeNoCopy(count); out.pz.resizeNoCopy(count);
		out.radius.resizeNoCopy(count); out.is_coarse.resizeNoCopy(count);
		if(count == 0)
			return;

		const size_t concurrency = (task_manager != NULL) ? myMax<size_t>(1, (size_t)task_manager->getConcurrency()) : 1;
		const size_t num_chunks = myMax<size_t>(1, myMin(concurrency * 4, count / 16384));
		if(task_manager != NULL && num_chunks > 1)
		{
			glare::TaskGroupRef group = new glare::TaskGroup();
			group->tasks.resize(num_chunks);
			for(size_t c=0; c<num_chunks; ++c)
			{
				Reference<GsFrontierSoATask> t = new GsFrontierSoATask();
				t->i_begin = (count * c)       / num_chunks;
				t->i_end   = (count * (c + 1)) / num_chunks;
				t->output = src; t->coarse_flags = src_coarse;
				t->positions = positions.data(); t->cull_radii = cull_radii.data();
				t->out_indices = out.indices.data();
				t->out_px = out.px.data(); t->out_py = out.py.data(); t->out_pz = out.pz.data();
				t->out_radius = out.radius.data(); t->out_is_coarse = out.is_coarse.data();
				group->tasks[c] = t;
			}
			task_manager->runTaskGroup(group);
		}
		else
			for(size_t i=0; i<count; ++i)
			{
				const uint32 idx = src[i];
				const Vec3f& p = positions[idx];
				out.indices[i] = idx;
				out.px[i] = p.x; out.py[i] = p.y; out.pz[i] = p.z;
				out.radius[i] = cull_radii[idx];
				out.is_coarse[i] = src_coarse[i];
			}
	}


	virtual void run(size_t /*thread_index*/) override
	{
		// SESSION076: pin the geometry snapshot for the whole of run(). The saturation phase now runs AFTER this task has
		// published its frontier (see the two enqueues at the end), by which point drainTraversalResults() may already have
		// returned `scratch` to the free pool and a fresh traversal may have refilled it with a DIFFERENT geom (see
		// fillTraversalScratch()) - the references below would then dangle mid-pass. Holding our own Reference keeps the
		// snapshot alive independently of the scratch that handed it to us.
		const Reference<GaussianSplatCachedGeom> geom_ref = scratch->geom;
		const js::Vector<Vec3f, 16>& positions = geom_ref->positions; // SESSION058: shared cached snapshot, never the live cloud arrays - see GaussianSplatCachedGeom.
		const js::Vector<float, 16>& feature_sizes = geom_ref->feature_size; // SESSION054: replaces per-push Vec3f scales[] lookup + 3-way max in makeHeapItem.
		const js::Vector<float, 16>& cull_radii = geom_ref->cull_radius; // SESSION059: enclosing-sphere bound for the frustum-cull margin below - NOT the same quantity as feature_size, see GaussianSplatLodNode::bounding_radius_os's comment.

		js::Vector<uint32, 16>& output = scratch->selected_indices;

		// SESSION054: replaced std::priority_queue with a plain LIFO stack (DFS). Profiling in session053/054 showed the
		// tree is expanded to convergence, not truncated by budget, in every case observed - so the heap's best-first
		// ordering was buying us nothing on the expand side (all nodes get visited regardless of order) while costing an
		// O(log N) per push/pop. Peak stack size for DFS is O(depth * branching), typically ~100 entries, versus the heap
		// which held the entire active frontier (millions of entries) - much better cache locality too. The trade: when
		// max_splats_budget clips the traversal, which specific nodes get sacrificed is no longer "smallest pixel_scale
		// first" but arbitrary DFS order. Acceptable per owner: budget-clip was not observed firing in the profiled scenes,
		// and even when it does, the visual policy change is small next to the perf win.
		std::vector<HeapItem> stack;
		stack.reserve(4096); // Peak is O(depth * branching); 4096 covers deep trees comfortably without reallocating.

		// SESSION063 K4: fine frontier nodes (is_coarse bit 0) and coarse-floor nodes (bit 31 of idx set) go into the SAME
		// list and are sorted together by distance, so the draw is globally front-to-back across both layers - a near coarse
		// node correctly occludes a far fine one. The bit is packed into the top of idx (cloud indices are well under 2^31)
		// so the radix sort, which keys only on dist_sq, carries it for free; it's unpacked when the output is built.
		//
		// SESSION080 (plan2 §4.1): pooled on the scratch rather than a local - see the field's comment. clear() keeps the
		// capacity, so a steady-state traversal writes into pages that are already faulted in.
		js::Vector<DistIdx, 16>& decorated = scratch->decorated;
		decorated.clear();
		js::Vector<float, 16> coarse_flags; // SESSION063 K4: parallel to output after the sort - 1 per coarse-floor node, 0 per fine node.

		// SESSION076 DIAGNOSTIC: coarse-capture breakdown, counted inline in the DFS below (the only place that still
		// knows WHY each node was captured - the flag array downstream records only that it was). All three stay zero
		// unless the "sat diag" checkbox is on. See GaussianSplatUnculledFrontier::sat_diag_coarse_a.
		size_t diag_coarse_a = 0, diag_coarse_b = 0;
		size_t diag_reuse_roots = 0; // SESSION080 STEP B DIAGNOSTIC: subtrees inherited whole from the previous frontier - see GaussianSplatUnculledFrontier::reuse_roots.

		Timer expand_timer; // SESSION080 DIAGNOSTIC - see GaussianSplatUnculledFrontier::expand_ms.

		for(size_t mi=0; mi<scratch->members_snapshot.size(); ++mi)
		{
			const GaussianSplatLodTraversalScratch::MemberSnapshot& m = scratch->members_snapshot[mi];
			if(m.hidden) // SESSION059: debug "hide this object" toggle - see GaussianSplatRenderer::setObjectHidden(). Contributes nothing to the frontier at all, root or leaves.
				continue;
			if(m.splat_data->lod_tree.empty())
			{
				// No tree built for this member (yet, or ever) - nothing to choose between, so every one of its splats is
				// always selected, exactly as it would be with no LoD at all.
				for(size_t i=0; i<m.count; ++i)
				{
					const size_t cloud_idx = m.offset + i;
					const Vec3f& p = positions[cloud_idx];
					DistIdx d; d.dist_sq = cam_pos_ws.getDist2(Vec4f(p.x, p.y, p.z, 1.f)); d.idx = (uint32)cloud_idx;
					decorated.push_back(d);
					recordFrontierNode((uint32)cloud_idx, (uint32)mi, (uint32)i, /*depth=*/0, FrontierStop_NoTree);
				}
			}
			else
			{
				stack.push_back(makeHeapItem((uint32)mi, /*tree_local_idx=*/0, m.offset, /*depth=*/0, positions, feature_sizes)); // Root is always node 0 - see buildGaussianSplatLodTree().
			}
		}

		bool hit_budget_cap = false;
		bool hit_density_cap = false;
		bool hit_depth_cap = false;
		size_t num_sat_bias_stops = 0, num_sat_bias_tested = 0; // SESSION085 ETAP 3

		// SESSION080: spread the DFS across the pool. Measured at 52% of the whole async pipeline once session079 had
		// parallelised gather/grid/test and session080 the sort - by some distance the largest remaining serial stage.
		//
		// The branches are independent by construction: positions/feature_sizes/cull_radii are read-only snapshots,
		// `stack` is per-walk, coarse_captured propagates only downwards inside one branch, and the density/depth caps are
		// per-node. The ONE thing that couples them is the budget cap, whose test reads the running decorated.size() +
		// stack.size(); that is handled below rather than approximated - see expandParallel()'s comment.
		//
		// Not taken when: no pool (the synchronous getFrustumStructureReport() path), or frontier_record is wanted (that
		// path's per-node records are pushed to one shared vector and it is a debug report, not a hot path).
		const bool expand_in_parallel = (task_manager != NULL) && (frontier_record == NULL);
		if(expand_in_parallel && !stack.empty())
			expandParallel(stack, decorated, positions, feature_sizes, cull_radii, diag_coarse_a, diag_coarse_b, diag_reuse_roots,
				hit_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);
		else
			expandStack(stack, decorated, positions, feature_sizes, cull_radii, /*enforce_budget=*/true, /*breadth_first=*/false, /*pause_at_stack_size=*/0,
				diag_coarse_a, diag_coarse_b, diag_reuse_roots, hit_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);

		const double expand_ms = expand_timer.elapsed() * 1.0e3; // SESSION080 DIAGNOSTIC - stops here, before the sort.

		// Sort the selection front-to-back by camera distance. Each node's dist_sq was computed once in makeHeapItem (or
		// inline in the NoTree branch) and carried through, so this pass is pure sort - no positions[] lookup or sqrt.
		// SESSION054: uses glare-core's serial 11-bit-chunk radix sort (Sort::floatKeyAscendingSort), which was ~76% of
		// traversal time as std::sort on 3-8M elements. floatKeyAscendingSort falls back to std::sort itself for N<320.
		// Two-stage (coarse+precise) GaussianSplatSortTask isn't reused: that machinery exists to keep an unbounded
		// whole-cloud sort off the main thread's critical path via a fast approximate first pass; a traversal's output is
		// already budget-bounded, so one exact radix pass here is both simpler and fast enough - see kickOffSorts()'s
		// cloudHasLodTree() guard, which leaves an LoD-active cloud to this sort instead of the old one.
		// SESSION063 K4: one global front-to-back sort over fine + coarse nodes together (see the decorated declaration).
		// SESSION080 DIAGNOSTIC: that 76% figure predates the radix switch and was never re-measured after - see
		// expand_ms/sort_ms and [gsr-traversal]. Re-measured session080: 29-36% of traversal time, second only to the
		// expand/DFS loop above - still the largest easy target, since gather/grid/test were already parallelised in
		// session079.
		//
		// SESSION080: parallel radix when a task_manager is available and the pool clears the dispatch overhead -
		// Sort::radixSortWithParallelPartition() is the SAME 11-bit-chunk radix as the serial path below, just with
		// each of its 3 passes split across 32 partition tasks instead of running on this one thread; the output order
		// is identical. Existing precedent for this exact call shape: NonBinningBVHBuilder.cpp's use of it to sort BVH
		// build centres. Threshold mirrors the sat-test/gather passes' own min_chunk_nodes (session079) - below it the
		// 3-pass task dispatch overhead isn't worth it, so small/no-tree-yet clouds fall through to the serial sort.
		Timer sort_timer;
		// SESSION080 DIAGNOSTIC (plan2 §4.1): timed apart from the sort itself - see sort_alloc_ms. Pooled on the scratch
		// alongside `decorated`, for the same reason; resizeNoCopy because the sort overwrites it wholesale.
		Timer sort_alloc_timer;
		js::Vector<DistIdx, 16>& sort_scratch = scratch->sort_scratch;
		sort_scratch.resizeNoCopy(decorated.size());
		const double sort_alloc_ms = sort_alloc_timer.elapsed() * 1.0e3;
		const size_t parallel_sort_min_elements = 16384;
		// SESSION087: hoisted out of the if below only so the two costs inside sort_ms - the sort proper and the unpack
		// that follows it - can be timed apart. Reading the total as one number is what hid the unpack for six sessions.
		const bool sort_parallel = (task_manager != NULL) && (decorated.size() >= parallel_sort_min_elements);
		Timer sort_radix_timer; // SESSION087 DIAGNOSTIC - see GaussianSplatUnculledFrontier::sort_radix_ms.
		// SESSION087: put_result_in_working_space=true. Both radix variants run an odd number of 11-bit passes (3, no
		// branching - checked in Sort.h, not assumed) and so end with the sorted data in the working space; false made them
		// pay a final full-array copy back into `decorated` (memcpy in the parallel variant, an element loop in the serial
		// one) purely so the result would sit in the buffer the two readers below happened to name. Those readers now name
		// sort_scratch instead, which is the same data without the copy - 34 MB read + 34 MB written per traversal at 4M
		// nodes, and it bought nothing.
		// floatKeyAscendingSort honours the flag on its small-N std::sort path too, so the fallback stays correct.
		if(sort_parallel)
			Sort::radixSortWithParallelPartition<DistIdx, DistIdxKey>(*task_manager, decorated.data(), (uint32)decorated.size(), DistIdxKey(), sort_scratch.data(), /*put_result_in_working_space=*/true);
		else
			Sort::floatKeyAscendingSort(decorated.data(), decorated.size(), DistIdxLess(), DistIdxKey(), sort_scratch.data(), /*put_result_in_working_space=*/true);
		const double sort_radix_ms = sort_radix_timer.elapsed() * 1.0e3; // SESSION087 DIAGNOSTIC
		// The sorted stream. NOTE for anyone extending this block: `decorated` is NOT it any more - it holds an
		// intermediate radix pass's output and is dead from here on. Both readers below (the unpack and dist_pctile) take
		// `sorted`, and anything added after them must too.
		const DistIdx* const sorted = sort_scratch.data();
		const size_t sorted_n = decorated.size();

		output.resizeNoCopy(sorted_n);
		coarse_flags.resizeNoCopy(sorted_n);

		// SESSION087: was a serial loop, measured at 12-16 ms - a quarter of sort_ms and the last unsplit pass in this
		// block. Chunked exactly like buildFrontierSoA() (same pure-map argument: slot i depends on element i alone, so the
		// result is bit-identical to the serial version), with the same chunk sizing.
		Timer sort_unpack_timer; // SESSION087 DIAGNOSTIC
		const size_t unpack_concurrency = (task_manager != NULL) ? myMax<size_t>(1, (size_t)task_manager->getConcurrency()) : 1;
		const size_t unpack_chunks = myMax<size_t>(1, myMin(unpack_concurrency * 4, sorted_n / 16384));
		if(task_manager != NULL && unpack_chunks > 1)
		{
			glare::TaskGroupRef unpack_group = new glare::TaskGroup();
			unpack_group->tasks.resize(unpack_chunks);
			for(size_t c=0; c<unpack_chunks; ++c)
			{
				Reference<GsSortUnpackTask> t = new GsSortUnpackTask();
				t->i_begin = (sorted_n * c)       / unpack_chunks;
				t->i_end   = (sorted_n * (c + 1)) / unpack_chunks;
				t->sorted = sorted;
				t->out_indices = output.data(); t->out_coarse_flags = coarse_flags.data();
				unpack_group->tasks[c] = t;
			}
			task_manager->runTaskGroup(unpack_group);
		}
		else
			for(size_t i=0; i<sorted_n; ++i)
			{
				const uint32 packed = sorted[i].idx;
				output[i] = packed & 0x7FFFFFFFu;                  // Real cloud index (bit 31 stripped).
				coarse_flags[i] = (packed >> 31) ? 1.f : 0.f;      // 1 = coarse-floor node, for the filter's per-node dilation.
			}
		const double sort_unpack_ms = sort_unpack_timer.elapsed() * 1.0e3; // SESSION087 DIAGNOSTIC
		const double sort_ms = sort_timer.elapsed() * 1.0e3; // SESSION080 DIAGNOSTIC - includes the unpack loop above, not just the sort call.

		// SESSION080 DIAGNOSTIC: where this frontier's nodes actually sit, in metres - see GaussianSplatUnculledFrontier::
		// dist_pctile. The sorted stream is ascending by dist_sq, so this is five lookups; deliberately not gated on a
		// debug flag for that reason. Sizes the "reuse the far tail" question before any of it is built.
		// SESSION087: reads `sorted`, not `decorated` - see the sort call's put_result_in_working_space note.
		float dist_pctile[5] = { 0.f, 0.f, 0.f, 0.f, 0.f };
		if(sorted_n > 0)
		{
			const int pct[5] = { 10, 25, 50, 75, 90 };
			for(int k=0; k<5; ++k)
				dist_pctile[k] = std::sqrt(sorted[myMin(sorted_n - 1, (sorted_n * pct[k]) / 100)].dist_sq);
		}

		scratch->hit_budget_cap = hit_budget_cap;
		scratch->hit_density_cap = hit_density_cap;
		scratch->hit_depth_cap = hit_depth_cap;
		scratch->num_sat_bias_stops = num_sat_bias_stops; scratch->num_sat_bias_tested = num_sat_bias_tested; // SESSION085 ETAP 3
		scratch->sat_bias_ceiling_used = sat_bias_active ? sat_bias_ceiling : 1.f; // SESSION085 ETAP 6 - 1 means the bias was off or had no barrier, which is what the reader needs to know.

		// A null queue means nobody is waiting for this frontier to be drawn - the caller ran the task itself and reads the
		// scratch directly.  Enqueueing anyway would have drainTraversalResults() apply a selection, and decrement an
		// in-flight count, for a traversal it never kicked off.
		if(result_queue != NULL)
		{
			Reference<GaussianSplatLodTraversalResultMsg> msg = new GaussianSplatLodTraversalResultMsg();
			msg->cloud_id = cloud_id;
			msg->topology_generation = topology_generation;
			msg->scratch = scratch;

			// SESSION063: split filter path - gather the selected frontier's positions + cull_radius into an SoA U(P) here
			// on the worker (never the main thread - the scattered read into the 30M-entry position array is ~50ms). The
			// key fields let drainTraversalResults() reject a stale cache; orientation is deliberately absent.
			//
			// SESSION076: declared out here, not inside the branch, because the saturation phase below runs AFTER this
			// frontier has been enqueued and needs to read it - see the two enqueues at the end of this block.
			Reference<GaussianSplatUnculledFrontier> uf;
			if(build_unculled_frontier)
			{
				Timer soa_timer; // SESSION080 DIAGNOSTIC - see GaussianSplatUnculledFrontier::soa_ms.
				uf = new GaussianSplatUnculledFrontier();
				const size_t n = output.size();

				// SESSION080 §4.3: split the sorted selection into the near part this traversal keeps and the far part that
				// becomes (or already is) a far block - see GaussianSplatUnculledFrontier::far_block.
				//
				// The mark rode through the sort in bit 30 of each index (set in expandStack()), so this is a stable partition
				// of an already-sorted array: both halves come out sorted, which is what lets the two segments be streamed one
				// after the other without a merge. far_n is 0 in every mode except the one that builds a block.
				double reuse_ms_accum = 0.0;
				Timer partition_timer;
				size_t far_n = 0;
				// Only the traversal that BUILDS a block ever sets the mark: with reuse off nothing sets it, and with the
				// cut frozen the walk stops at the cut instead of descending past it. Skipping the count in those two
				// cases is not an optimisation of the common path so much as the removal of a pure waste - measured at
				// 15.9ms scanning 13.2M indices for a bit that is never set.
				if(reuse_enabled && !far_cut_is_frozen)
				{
					for(size_t i=0; i<n; ++i)
						if(output[i] & 0x40000000u)
							++far_n;
				}
				const size_t near_n = n - far_n;

				if(far_n > 0) // Building a block: compact the two halves into [near][far], stripping the mark.
				{
					scratch->partition_near.resizeNoCopy(near_n); scratch->partition_near_coarse.resizeNoCopy(near_n);
					scratch->partition_far.resizeNoCopy(far_n);   scratch->partition_far_coarse.resizeNoCopy(far_n);
					size_t wn = 0, wf = 0;
					for(size_t i=0; i<n; ++i)
					{
						if(output[i] & 0x40000000u) { scratch->partition_far[wf] = output[i] & ~0x40000000u; scratch->partition_far_coarse[wf] = coarse_flags[i]; ++wf; }
						else                        { scratch->partition_near[wn] = output[i];                scratch->partition_near_coarse[wn] = coarse_flags[i]; ++wn; }
					}
					assert(wn == near_n && wf == far_n);
				}
				reuse_ms_accum += partition_timer.elapsed() * 1.0e3;

				// The near segment - this frontier's own arrays. When no block is being built this is the whole selection and
				// `output` is used directly, so the common path allocates and copies nothing extra.
				const uint32* const near_src       = (far_n > 0) ? scratch->partition_near.data()        : output.data();
				const float*  const near_src_coarse = (far_n > 0) ? scratch->partition_near_coarse.data() : coarse_flags.data();
				buildFrontierSoA(*uf, near_src, near_src_coarse, near_n, positions, cull_radii);

				if(far_n > 0)
				{
					// The far segment, materialised ONCE here and then referenced by every traversal of this generation.
					Timer far_soa_timer;
					Reference<GaussianSplatUnculledFrontier> block = new GaussianSplatUnculledFrontier();
					buildFrontierSoA(*block, scratch->partition_far.data(), scratch->partition_far_coarse.data(), far_n, positions, cull_radii);
					// The frozen pair the cut was made at, which every later traversal re-evaluates its predicate against, and
					// which the drift trigger measures from. reuse_prev_anchor_ws is cam_pos_ws in this mode - see the ctor.
					block->anchor_pos_ws = reuse_prev_anchor_ws;
					block->reuse_split_dist_used = reuse_split_dist;
					block->topology_generation = topology_generation;
					// SESSION088: the barrier this block's LoD was actually chosen under, kept so later traversals can ask
					// whether that choice still stands - see GaussianSplatUnculledFrontier::sat_depth_used. Captured only
					// when the bias was live: with it off the selection does not depend on the barrier at all, and an empty
					// grid is the signal the acceptance clause reads as "this block owes the barrier nothing".
					if(sat_bias_active)
					{
						block->sat_depth_used = sat_barrier->sat_depth;
						block->sat_grid_res_used = sat_barrier->sat_grid_res;
						block->sat_anchor_used = sat_barrier->anchor_pos_ws;
						block->sat_built_time_used = sat_barrier->built_time_real_s;
					}
					uf->far_block = block;
					reuse_ms_accum += far_soa_timer.elapsed() * 1.0e3;
				}
				else
					uf->far_block = inherited_far_block; // Null when reuse is off; the existing block when its cut is frozen.

				uf->topology_generation = topology_generation;
				uf->anchor_pos_ws = cam_pos_ws;
				uf->built_time_real_s = Clock::getCurTimeRealSec(); // SESSION080 DIAGNOSTIC - see the field's comment.
				uf->pixel_scale_limit = pixel_scale_limit;
				uf->sat_bias_barrier_time = satBiasBarrierTime(); // SESSION085 ETAP 3 - see the field.
				uf->max_splats_budget = max_splats_budget;
				uf->max_layer_density = max_layer_density;
				uf->max_tree_depth = max_tree_depth;
				uf->focal_px = focal_px;
				uf->expand_ms = expand_ms; // SESSION080 DIAGNOSTIC
				uf->sort_ms = sort_ms;
				uf->sort_radix_ms = sort_radix_ms; uf->sort_unpack_ms = sort_unpack_ms; // SESSION087
				uf->traversal_output_n = n;
				uf->expand_seeds = expand_seeds;
				uf->expand_num_tasks = expand_num_tasks;
				uf->expand_prologue_ms = expand_prologue_ms;
				uf->expand_task_max_ms = expand_task_max_ms;
				uf->expand_workers = expand_workers; uf->expand_seed_max_ms = expand_seed_max_ms; // SESSION086 DIAGNOSTIC
				uf->expand_seed_target = expand_seed_target;
				uf->expand_task_sum_ms = expand_task_sum_ms;
				uf->expand_splice_reserve_ms = expand_splice_reserve_ms; // SESSION080 DIAGNOSTIC (plan2 §4.1)
				uf->expand_splice_copy_ms = expand_splice_copy_ms;
				uf->sort_alloc_ms = sort_alloc_ms;
				uf->soa_ms = soa_timer.elapsed() * 1.0e3; // SESSION080 DIAGNOSTIC - the whole segment build, partition included.
				uf->reuse_roots = diag_reuse_roots; // SESSION080 §4.3 DIAGNOSTIC: subtrees the walk stopped at, or marked as the block's.
				uf->reuse_n = uf->far_block.nonNull() ? uf->far_block->indices.size() : 0;
				// SESSION085: this frontier's own size, both segments - see the field. Nothing has pruned it yet, so the
				uf->reuse_ms = reuse_ms_accum; // Partition + the far SoA build, i.e. what BUILDING a block costs. 0 when one was inherited - which is the whole point of §4.3.
				uf->reuse_split_dist_used = reuse_enabled ? reuse_split_dist : 0.f;
				uf->barrier_disagreement = diag_barrier_disagreement; // SESSION088 DIAGNOSTIC - what the acceptance clause measured for THIS kick, see the field.
				uf->barrier_flip = diag_barrier_flip; uf->barrier_depth = diag_barrier_depth;
				uf->barrier_mean_rel_depth = diag_barrier_mean_rel_depth; uf->barrier_measured = diag_barrier_measured;
				uf->diag_live_barrier_time = diag_live_barrier_time; uf->diag_block_barrier_time = diag_block_barrier_time; // SESSION088 DIAGNOSTIC, TEMPORARY
				// SESSION080 §4.3: the frozen anchor the far block's cut was made at, carried so the next traversal can measure
				// drift against it in one subtraction. Equal to cam_pos_ws on a self-contained frontier.
				uf->reuse_base_anchor_ws = uf->far_block.nonNull() ? uf->far_block->anchor_pos_ws : cam_pos_ws;
				for(int k=0; k<5; ++k) uf->dist_pctile[k] = dist_pctile[k]; // SESSION080 DIAGNOSTIC

				msg->unculled_frontier = uf;
			}

			// SESSION076: publish the frontier NOW, before any saturation work. This is the whole point of the split: the
			// tree walk is what the picture is waiting on, and the saturation grid + its prune were adding ~300ms to that
			// wait for a result that only ever REMOVES nodes. Sending the unpruned frontier first means the camera's new
			// viewpoint is on screen at DFS-only latency, and the prune arrives afterwards as a refinement.
			// SESSION085 ETAP 6: nothing follows a traversal any more. The saturation prune that used to arrive later as a
			// refinement - and the "is it safe to draw the unpruned frontier meanwhile?" question that came with it - are
			// gone: the barrier is applied as an LoD bias DURING this walk, so what is published here is already the
			// saturated selection. drainTraversalResults() installs it unconditionally.
			result_queue->enqueue(msg);

			// SESSION080 DIAGNOSTIC (plan doc STEP A): how far this frontier's front-to-back order has moved since the
			// previous traversal's - see GaussianSplatUnculledFrontier::sort_staleness_* for what question it answers.
			//
			// Placed HERE, immediately after the enqueue above and before the saturation stage, on purpose:
			//  - after the enqueue, because this is pure diagnostic work and anything ahead of that line delays the
			//    picture by its own duration (§2.8 of the plan doc - the trap the soa/gather merge idea fell into);
			//  - before the saturation stage rather than inside it, because the numbers describe the TRAVERSAL, and
			//    nesting them in the uf2 block would silently tie them to the saturation stage being switched on.
			//
			// Both sides are UNPRUNED frontiers: prev_frontier is fed from SplatCloud::
			// last_unpruned_ufrontier, not cached_ufrontier - ranking against a saturation-pruned list would measure
			// that prune's compaction (55-80% of the pool at R=0) rather than the camera's motion.
			//
			// Matching is by identity, not by array position: cloud_idx (= member offset + tree_local_idx) names a
			// specific LoD-tree node - see HeapItem's comment - so it is stable across traversals whenever the same node
			// was selected both times. Nodes selected only once are not "displaced", they are LoD churn, and are counted
			// out of common_n rather than into the displacement. The lookup is a flat rank array indexed by cloud_idx
			// (bound: positions.size(), the geometry snapshot this task already pins) rather than a hash map, which at
			// ~13M entries would cost more to build than the walk it serves.
			double staleness_delta_ws = 0.0, staleness_mean_disp = 0.0, staleness_ms = 0.0;
			size_t staleness_prev_n = 0, staleness_common_n = 0, staleness_max_disp = 0, staleness_gt1k = 0, staleness_gt10k = 0;
			if(uf.nonNull() && sort_staleness_diag_enabled && prev_frontier.nonNull() &&
				// The previous frontier has to have been built for the same SELECTION, or the differences are the
				// settings change rather than the motion. This is the key set GaussianSplatUnculledFrontier documents
				// and that drainTraversalResults() checks for the same reason; checked here rather than by clearing
				// last_unpruned_ufrontier at each setter, so a setter added later cannot silently skip it.
				prev_frontier->topology_generation == topology_generation &&
				prev_frontier->pixel_scale_limit == pixel_scale_limit &&
				prev_frontier->max_splats_budget == max_splats_budget &&
				prev_frontier->max_layer_density == max_layer_density &&
				prev_frontier->max_tree_depth == max_tree_depth &&
				prev_frontier->focal_px == focal_px &&
				prev_frontier->sat_bias_barrier_time == satBiasBarrierTime()) // SESSION085 ETAP 3 - same key set as the reuse check above.
			{
				Timer staleness_timer;
				const GaussianSplatUnculledFrontier& prev_uf = *prev_frontier; // Not `prev` - that name shadows glare::Task::prev, the intrusive task-list link.
				staleness_prev_n = prev_uf.indices.size();

				// Coarse-floor nodes are excluded from BOTH sides. A coarse node re-uses the cloud_idx of the fine node
				// it stands for whenever its branch is terminal (see the capture block in expandStack()), so including
				// them would make cloud_idx ambiguous as a key; and excluding them also makes this measurement
				// independent of the "coarse" checkbox, which changes the captured set but never the fine selection.
				js::Vector<uint32, 16> old_rank;
				old_rank.resizeNoCopy(positions.size());
				std::memset(old_rank.data(), 0xFF, old_rank.size() * sizeof(uint32)); // 0xFFFFFFFF = "not in the previous frontier".
				uint32 prev_fine_rank = 0;
				for(size_t i=0; i<staleness_prev_n; ++i)
				{
					if(prev_uf.is_coarse[i] != 0.f)
						continue;
					const uint32 idx = prev_uf.indices[i];
					if(idx < old_rank.size()) // Defensive only - the topology_generation match above already means the two frontiers index the same cloud.
						old_rank[idx] = prev_fine_rank;
					++prev_fine_rank; // Rank within the FINE subsequence, so both sides are numbered on the same scale.
				}

				const size_t new_n = uf->indices.size();
				double disp_sum = 0.0;
				uint32 new_fine_rank = 0;
				for(size_t i=0; i<new_n; ++i)
				{
					if(uf->is_coarse[i] != 0.f)
						continue;
					const uint32 idx = uf->indices[i];
					const uint32 rank = new_fine_rank++;
					if(idx >= old_rank.size())
						continue;
					const uint32 old_r = old_rank[idx];
					if(old_r == 0xFFFFFFFFu)
						continue; // Selected this time but not last - LoD churn, not displacement.
					const size_t disp = (rank > old_r) ? (rank - old_r) : (old_r - rank);
					++staleness_common_n;
					disp_sum += (double)disp;
					if(disp > staleness_max_disp) staleness_max_disp = disp;
					if(disp > 1000) ++staleness_gt1k;
					if(disp > 10000) ++staleness_gt10k;
				}

				staleness_delta_ws = cam_pos_ws.getDist(prev_uf.anchor_pos_ws);
				staleness_prev_n = prev_fine_rank; // Report the fine count, the same population the comparison ran over.
				staleness_mean_disp = staleness_common_n > 0 ? (disp_sum / (double)staleness_common_n) : 0.0;
				staleness_ms = staleness_timer.elapsed() * 1.0e3;
			}

		}
	}

private:
	inline void recordFrontierNode(uint32 cloud_idx, uint32 member_idx, uint32 tree_local_idx, uint32 depth, FrontierStopReason reason)
	{
		if(frontier_record == NULL)
			return;
		FrontierNodeRecord rec;
		rec.cloud_idx = cloud_idx;
		rec.member_idx = member_idx;
		rec.tree_local_idx = tree_local_idx;
		rec.depth = (uint16)myMin(depth, (uint32)65535);
		rec.stop_reason = (uint16)reason;
		frontier_record->push_back(rec);
	}

	// One entry under consideration in run()'s stack: either a tree root not yet examined, or a node whose parent was just
	// expanded.  member_idx/tree_local_idx together identify the node; the corresponding cloud-array index (needed to read
	// its baked world-space position/scale, and to write it to the output) is member.offset + tree_local_idx.
	// SESSION054: carries dist_sq so the sort phase can reuse the distance computed here rather than re-scanning positions[].
	struct HeapItem
	{
		float pixel_scale;
		float dist_sq; // Squared camera-distance already computed here; sort phase uses it directly, avoiding a second lookup of positions[].
		uint32 member_idx;
		uint32 tree_local_idx;
		uint32 depth; // 0 for a tree root, parent's depth + 1 for each expansion - see max_tree_depth's use in run().
		bool coarse_captured; // SESSION063 K4: true once some ancestor on this branch was recorded into the coarse floor, so it's captured once per branch (the first node fine enough at the coarse pixel_scale). Set false at the root by makeHeapItem, propagated to children in run().

		// SESSION080 §4.3: true once this node or an ancestor satisfied the frozen far-cut predicate, i.e. this node
		// belongs to the far block rather than to the walked near part. Propagated exactly like coarse_captured above.
		// Only ever set while a block is being BUILT: in frozen mode the walk stops at the cut instead of descending
		// past it, so nothing below it is ever reached. See the classification block in expandStack().
		bool below_far_cut;
	};

	// SESSION086: orders seeds by how much walking they still have ahead of them - see the targeted split in
	// expandParallel() for why pixel_scale is the cost signal.
	struct SeedPixelScaleLess { bool operator() (const HeapItem& a, const HeapItem& b) const { return a.pixel_scale < b.pixel_scale; } };

	HeapItem makeHeapItem(uint32 member_idx, uint32 tree_local_idx, size_t member_offset, uint32 depth, const js::Vector<Vec3f, 16>& positions, const js::Vector<float, 16>& feature_sizes) const
	{
		const size_t cloud_idx = member_offset + tree_local_idx;
		const Vec3f& p = positions[cloud_idx];
		const float dist_sq = cam_pos_ws.getDist2(Vec4f(p.x, p.y, p.z, 1.f));
		const float dist = myMax(std::sqrt(dist_sq), 1.0e-6f); // Clamped away from zero so a node exactly at the camera doesn't produce an infinite pixel_scale.
		const float feature_size = feature_sizes[cloud_idx]; // SESSION054: precomputed at kick time in fillTraversalScratch() - replaces the old Vec3f scales[] load + 3-way max on the hot per-push path.
		HeapItem item;
		item.pixel_scale = (feature_size / dist) * focal_px;
		item.dist_sq = dist_sq;
		item.member_idx = member_idx;
		item.tree_local_idx = tree_local_idx;
		item.depth = depth;
		item.coarse_captured = false; // SESSION063 K4: set by the caller for children; false at the root.
		item.below_far_cut = false; // SESSION080 §4.3: a root is never below the cut - see the field.
		return item;
	}

	// SESSION085 ETAP 3 - THE SATURATION LoD BIAS. How much this node's pixel_scale_limit is multiplied by, given how
	// deeply the barrier says it is buried. 1 = untouched.
	//
	// The idea in one line: instead of DROPPING a node behind saturated geometry (which is a hole whenever the barrier is
	// wrong), stop the walk there and draw the node's own merged stand-in - what is given up is the subtree it would have
	// unfolded into, not the geometry. See snapshots/2026-09-03-session085-progressiveSaturation_PLAN.md.
	//
	// The measure is gsSatBuriedRatioSq() - the node's centre tile, node as a point - and NOT gsSatOccluded(), which is
	// the prune's test. That was etap 3's first cut and it left the mechanism near-inert; the reasoning, and the measured
	// numbers, are recorded on gsSatBuriedRatioSq() itself. The short version: a prune must be certain, so it charges the
	// node's whole extent and demands every tile it touches agree - and this is asked about ANCESTORS, which are exactly
	// the nodes that cannot pay that. A bias removes nothing, so it does not owe that certainty.
	//
	// The multiplier reads as tree levels, since feature_size roughly halves per level: 2 is about one level coarser,
	// 4 two levels, 8 three. What maps the barrier ratio onto it is the curve below, not the ratio itself - see there.
	//
	// Cost: called ONLY on nodes that have already failed the plain converged test, i.e. ones that would otherwise
	// descend - never on the common early-out path - and it is one direction encode plus one array read.
	inline float satBiasFor(uint32 cloud_idx, const js::Vector<Vec3f, 16>& positions) const
	{
		const GaussianSplatSaturationBarrier& b = *sat_barrier;
		const Vec3f& p = positions[cloud_idx];
		// From the BARRIER's anchor, not the camera: sat_depth is measured from where the barrier was built, and the two
		// are not the same point (that is what region_radius is about). Using cam_pos_ws here would be a silent error.
		const Vec4f offset(p.x - b.anchor_pos_ws[0], p.y - b.anchor_pos_ws[1], p.z - b.anchor_pos_ws[2], 0.f);
		const float dist_sq = offset[0]*offset[0] + offset[1]*offset[1] + offset[2]*offset[2];

		const float ratio_sq = gsSatBuriedRatioSq(offset, dist_sq, b.sat_depth, b.sat_grid_res);
		if(ratio_sq <= 1.f)
			return 1.f; // Not behind the barrier, or exactly on it - no bias either way.

		// f = 1 + (ceiling - 1) * (1 - 1/ratio^2). Continuous, f(1) = 1 exactly, monotonic, and it approaches the ceiling
		// quickly - about 89% of the way there by ratio 3.
		//
		// The first cut used the ratio ITSELF as the multiplier, and that was measured too weak to matter. The arithmetic,
		// recorded because it is the argument for this shape: 689,864 stops out of 3,419,595 tests took the frontier from
		// 11.5M to 7.45M, i.e. 5.8 nodes saved per stop - about 1.3 tree levels. That is exactly what the ratio can buy,
		// since climbing the tree is logarithmic in the multiplier (feature_size halves per level) and the ratio is only
		// 2-10 over most of a frontier. Meanwhile the PRUNE, reading the same barrier, took the same frontier to 2M: it
		// draws the strongest possible conclusion from that evidence, while the bias was drawing one of the weakest.
		//
		// The reasoning behind the shape: the ratio is CONFIDENCE that the node is invisible, not a measure of how much
		// detail it needs. Past a few multiples of the barrier the confidence is not meaningfully increasing any more -
		// either the barrier is right, in which case nothing there is visible and the coarsening is free, or it is wrong,
		// in which case a bigger ratio would not have saved us. So the sensible response is to reach the ceiling and stop,
		// leaving the grading for the region near the barrier - which is where it is actually needed, both because the
		// verdict is genuinely uncertain there and because a discontinuity there would be a moving seam (see the plan's
		// etap 4 note on session064-style boiling).
		//
		// This makes the CEILING the operative knob rather than a backstop, which is the right shape for the project's
		// "no manual per-scene tuning" rule: one number, calibrated once, with the per-node behaviour fixed in code.
		//
		// Works on ratio_sq directly - 1/ratio^2 is 1/ratio_sq - so the sqrt the first cut needed is gone from a path the
		// walk takes millions of times per traversal.
		//
		// SESSION088: the falloff's exponent is a knob now - 1/ratio_sq becomes 1/ratio_sq^exponent, i.e. 1/ratio^(2*e).
		// At 1 this is exactly the line it replaced. See GaussianSplatRenderer::getSatBiasExponent() for why flattening
		// it is what the far-field artifact responds to, and for what it costs. The == 1 branch is not an optimisation
		// for the current default (0.2, which takes the pow) - it is there so that setting the knob back to 1 restores
		// session085's arithmetic bit-for-bit rather than to within pow()'s rounding, which is what an A/B against the
		// original curve needs.
		const float denom = (sat_bias_exponent == 1.f) ? ratio_sq : std::pow(ratio_sq, sat_bias_exponent);
		return 1.f + (sat_bias_ceiling - 1.f) * (1.f - 1.f / denom);
	}


	// SESSION080: the DFS itself, lifted out of run() unchanged so that the serial path and each parallel task run
	// literally the same code rather than two copies that can drift apart.
	//
	// enforce_budget=false omits ONLY the budget-cap branch - see expandParallel() for why that is exact rather than an
	// approximation. pause_at_stack_size > 0 stops the walk once at least that many unconsumed entries are pending and
	// leaves them in `stack` for the caller to distribute; 0 means run to exhaustion.
	//
	// breadth_first swaps the LIFO pop for a FIFO one (a read cursor over the same vector; the consumed prefix is erased
	// on the way out). It exists solely to make pause_at_stack_size reachable: DFS is defined by diving, so its stack
	// stays at O(depth * branching) - ~100 entries, per session054's note above - and can never accumulate the breadth a
	// seed split needs. Only the prologue uses it; the walks that do the actual work stay depth-first, which is what
	// keeps their stacks small and cache-friendly. Order does not change the result - every decision in the loop below
	// depends on the node itself and on coarse_captured inherited down its own branch, never on what was visited before -
	// with the single exception of the budget cap, which is why breadth_first is only ever paired with enforce_budget=false.
	void expandStack(std::vector<HeapItem>& stack, js::Vector<DistIdx, 16>& decorated,
		const js::Vector<Vec3f, 16>& positions, const js::Vector<float, 16>& feature_sizes, const js::Vector<float, 16>& cull_radii,
		bool enforce_budget, bool breadth_first, size_t pause_at_stack_size,
		size_t& diag_coarse_a, size_t& diag_coarse_b, size_t& diag_reuse_roots,
		bool& hit_budget_cap, bool& hit_density_cap, bool& hit_depth_cap, size_t& num_sat_bias_stops, size_t& num_sat_bias_tested)
	{
		size_t head = 0; // Read cursor; breadth-first only.
		for(;;)
		{
			const size_t pending = breadth_first ? (stack.size() - head) : stack.size();
			if(pending == 0)
				break;
			if(pause_at_stack_size != 0 && pending >= pause_at_stack_size)
				break; // Enough independent subtrees for the caller to spread; the pending entries are the seeds.

			// Copied, not referenced: the pushes below can reallocate `stack`.
			const HeapItem top = breadth_first ? stack[head++] : stack.back();
			if(!breadth_first)
				stack.pop_back();

			const GaussianSplatLodTraversalScratch::MemberSnapshot& m = scratch->members_snapshot[top.member_idx];
			const GaussianSplatLodNode& node = m.splat_data->lod_tree[top.tree_local_idx];
			const uint32 cloud_idx_u32 = (uint32)(m.offset + top.tree_local_idx);

			// SESSION055/059: frustum-cull check. Each plane's margin has three parts:
			//   base_margin = cull_radius          - SESSION059: enclosing-sphere radius of this node's whole subtree
			//                                       (see GaussianSplatLodNode::bounding_radius_os), NOT feature_size.
			//                                       feature_size is a statistical fit of this node's own merged
			//                                       appearance and can be smaller than the true spread of its
			//                                       descendants - using it here was session055's original choice and
			//                                       worked at the small scenes tested then, but under-culls (drops
			//                                       visible subtrees) at real-world/km scale - see session059 snapshot.
			//   translation_dilation[i]           - anisotropic pad for camera movement toward this plane over the async
			//                                       traversal latency window; ~zero for planes the camera moves away from.
			//   rotation_dilation_rate * dist     - rotational pad: r*theta tangential shift at distance r from camera.
			// The whole point of the cull is to skip the subtree entirely when the parent is outside, so on cull we neither
			// push children nor add the node to decorated. Overrides the "no frustum test here" property called out at the
			// top of the class; the paired re-kick-on-rotation trigger in kickOffTraversals() puts back the property that
			// turning on the spot updates the selection.
			// SESSION072: shared by the frustum-cull block below and the distance-slice block right after it - both need
			// distance-to-camera, and dist_sq is already sitting in top from makeHeapItem() (session054), so computing the
			// one sqrt here instead of inside each block avoids paying it twice when both culls are active.
			const bool need_dist_to_node = frustum_cull_enabled || dist_clamp_enabled;
			const float dist_to_node = need_dist_to_node ? std::sqrt(top.dist_sq) : 0.f;

			if(frustum_cull_enabled)
			{
				const Vec3f& p = positions[cloud_idx_u32];
				const float base_margin = cull_radii[cloud_idx_u32];
				const float rot_pad = rotation_dilation_rate * dist_to_node;
				const Vec4f pos4(p.x, p.y, p.z, 1.f);
				bool outside = false;
				for(int i=0; i<num_frustum_clip_planes; ++i)
				{
					const float margin_i = base_margin + translation_dilation[i] + rot_pad;
					if(dot(frustum_clip_planes[i].getNormal(), pos4) >= frustum_clip_planes[i].getD() + margin_i)
					{ outside = true; break; }
				}
				if(outside)
				{
					recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_OutOfFrustum);
					continue;
				}
			}

			// SESSION072: distance-slice early-cull - moves the "isolate a distance shell" debug tool (GaussianSplatSettingsWidget's
			// distClamp* controls, previously a per-instance vertex-shader discard only - see countSplatsInFrustum()) up to a real
			// subtree prune here, gated by its own checkbox rather than by "min/max still at the keep-everything default" (that
			// implicit check stays where it always was, in the shader and the two CPU diagnostics - this is a separate, additive
			// early-out). A whole subtree is only skippable when EVERY point in it is provably on the excluded side: normal mode
			// prunes when the node's whole bounding sphere lies entirely outside [min, max]; inverted mode (hiding the shell)
			// prunes when the whole sphere lies entirely inside it - the mirror image of the same margin test frustum-cull uses.
			if(dist_clamp_enabled)
			{
				const float radius = cull_radii[cloud_idx_u32];
				const bool outside = dist_clamp_invert
					? (dist_to_node - radius >= dist_clamp_min && dist_to_node + radius <= dist_clamp_max)
					: (dist_to_node + radius < dist_clamp_min || dist_to_node - radius > dist_clamp_max);
				if(outside)
				{
					recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_OutOfDistRange);
					continue;
				}
			}

			// SESSION080 §4.3: the far-block cut. One O(1) predicate, evaluated against a FROZEN anchor and split
			// distance - see GaussianSplatUnculledFrontier::far_block for why freezing them is the whole trick.
			//
			//   P(node) = dist(frozen_anchor, centre) - cull_radius >= frozen_split
			//
			// cull_radius is the subtree's enclosing sphere (NOT feature_size - see GaussianSplatLodNode::
			// bounding_radius_os), so P is inherited downwards: if P holds for a node it holds for every descendant,
			// whose sphere is contained in this one's. That is what makes the cut a clean antichain.
			//
			// Two modes, and they must agree node-for-node or the picture gets a hole or a double-draw:
			//
			//   far_cut_is_frozen (a later traversal in the generation): P true -> STOP and emit nothing. The far block
			//     already holds this subtree's selection, made when the block was built. Descending would cover the same
			//     paths twice.
			//   building the block (the generation's first traversal): P true -> mark the subtree FAR and carry on under
			//     the normal rules. Whatever this walk emits down there is what the block will hold, and the partition
			//     after the sort splits it off by that mark.
			//
			// Because P depends only on the frozen pair and the node's own geometry, the first node satisfying it on any
			// root->leaf path is the SAME node in both modes. So the block holds exactly one selected node per path below
			// the cut, and later walks stop at exactly the boundary that produced it. Coverage is exact, not approximate.
			//
			// STRADDLE (sphere spans the split: P false, but dist + radius >= split) is the case that needs care. Some
			// descendants satisfy P and some do not, so emitting here would cover paths the block also covers - the walk
			// is forced past its normal stop rules until every branch resolves one way or the other.
			//
			// SESSION083: two corrections here, both about a straddling node whose own radius is large.
			//
			// (a) A straddling LEAF now resolves to FAR, not near. It cannot descend, and it fails P, so the old rule
			//     emitted it into the NEAR segment keyed on its centre distance. But P fails for it only because its own
			//     radius is subtracted: a leaf with a 700m 3-sigma radius fails P at any centre distance below split+700m,
			//     while its centre may sit hundreds of metres away. Near and far are concatenated, not merged, so such a
			//     leaf lands at the END of near - i.e. still ahead of the whole far segment, which starts at `split`. Under
			//     the front-to-back T *= (1-a) compositor that is not a small ordering error that blends away: a huge,
			//     hundreds-of-metres-distant splat eats the transmittance of everything the far segment was about to draw
			//     just past the split. That is the "cracks" artifact - visible even at zero drift, i.e. exactly at the
			//     anchor, which is why it never looked like the drift-shell error the split's own error budget accounts
			//     for. Resolving it to far puts it back among nodes of its own range, sorted with them, at no cost.
			//
			//     Only a leaf needs this. Any straddling INTERNAL node is descended past, so it never reaches an emit
			//     point in the first place, and every node that does emit below it has resolved to one side or the other.
			//
			// (b) force_descend now applies while BUILDING the block too, not only in the frozen mode. It has to, for (a)
			//     to be safe: the block is what a later frozen walk defers to, so any node a frozen walk skips must have
			//     been reached and emitted when the block was built. A straddling leaf is reachable only through
			//     straddling ancestors (a node with dist + radius < split contains only descendants with the same
			//     property, so a fully-near ancestor can never hold a straddling leaf). If a straddling ancestor is
			//     allowed to stop on its own rules at build time - pixel_scale there is measured from the anchor, and a
			//     later walk measures it from somewhere else - the build never reaches the leaf, the block never receives
			//     it, and the frozen walk's skip turns into a hole. Descending in both modes makes the cut resolve at
			//     the same nodes in both, which is the property the paragraph above claims and previously only assumed.
			//     The extra descent is confined to the straddle shell and is work the frozen walks already did.
			bool force_descend = false;
			bool child_below_far_cut = top.below_far_cut; // Once below the cut, the whole branch is - see the field.
			if(reuse_enabled && !top.below_far_cut)
			{
				const Vec3f& p = positions[cloud_idx_u32];
				const float dist_frozen = reuse_prev_anchor_ws.getDist(Vec4f(p.x, p.y, p.z, 1.f));
				const float radius = cull_radii[cloud_idx_u32];

				// P holds, or the node straddles and is a leaf that cannot be resolved by descending - see (a) above.
				// Both resolve to FAR, and both do it identically in the two modes, so coverage stays exactly once.
				if((dist_frozen - radius >= reuse_split_dist) ||
					(node.child_count == 0 && dist_frozen + radius >= reuse_split_dist))
				{
					++diag_reuse_roots;
					if(far_cut_is_frozen)
						continue; // The block covers this subtree - see above.
					child_below_far_cut = true; // Building the block: descend, but everything from here down is its.
				}
				else
					force_descend = dist_frozen + radius >= reuse_split_dist; // STRADDLE, both modes - see (b) above.
			}

			// SESSION080 §4.3: which segment a node emitted at THIS point belongs to, packed into bit 30 of idx so it
			// survives the sort the way the coarse flag survives it in bit 31 (cloud indices are far below 2^30). Only
			// ever set while building a block; in frozen mode the walk emits nothing below the cut, so it stays 0 and
			// the partition after the sort is a no-op. Unpacked, and the bit cleared, in run()'s partition pass.
			const uint32 far_bit = child_below_far_cut ? 0x40000000u : 0u;

			// SESSION063 K4: complete coarse-floor cut. A node becomes its branch's single coarse representative if no
			// ancestor already took the role AND it is either coarse enough (pixel_scale <= coarse_pixel_scale) or terminal
			// (the branch stops here - converged/leaf/capped). So every branch contributes exactly one coarse node and the
			// coarse layer covers the scene as fully as the fine set, only coarser. A branch that terminates finer than
			// coarse_pixel_scale takes its own (coarsest-available) terminal node, duplicating the fine node there - fine:
			// drawn after the fine set, the duplicate is gated wherever the fine node already covered. The terminal test
			// mirrors the stop checks below (read-only, no side effects; the real branches still set the hit_*_cap flags).
			bool captured_here = false;
			if(coarse_floor_enabled && !top.coarse_captured)
			{
				const bool terminal =
					(top.pixel_scale <= pixel_scale_limit) ||
					(node.child_count == 0) ||
					(max_layer_density > 0.f && node.layer_density > max_layer_density) ||
					(max_tree_depth > 0 && top.depth >= (uint32)max_tree_depth) ||
					(enforce_budget && (decorated.size() + stack.size() + node.child_count > max_splats_budget));
				const bool by_threshold = top.pixel_scale <= coarse_pixel_scale;

				// SESSION076: the capture-time skip that used to sit here is gone with the binary write rule it depended
				// on - see gsSatGridMinWritingPixelScaleFactor's removal note in GaussianSplatSaturationGrid.h. Under the
				// Gaussian model every node contributes in proportion to its integrated occlusion, so there is no
				// pixel_scale below which a node is provably useless to the grid.
				if(by_threshold || terminal)
				{
					DistIdx cd; cd.dist_sq = top.dist_sq; cd.idx = cloud_idx_u32 | 0x80000000u | far_bit; // Bit 31 marks a coarse-floor node; bit 30 the far segment - see far_bit.
					decorated.push_back(cd);
					captured_here = true;

					// SESSION076 DIAGNOSTIC: split the capture into its two populations and predict, from pixel_scale
					// alone, whether the grid will accept this node - see GaussianSplatUnculledFrontier::sat_diag_coarse_a
					// for what the split means and why the prediction is printed next to the measured count.
					if(sat_diag_log)
					{
						if(by_threshold) ++diag_coarse_a; else ++diag_coarse_b;
					}
				}
			}
			const bool child_coarse_captured = top.coarse_captured || captured_here;

			// Converged - already fine enough, no need to expand further.
			// SESSION080 STEP B: force_descend suppresses this and the two caps below - see the STRADDLE case above. The
			// leaf branch is deliberately NOT suppressed, and no longer needs to be: force_descend is now only ever set
			// on a node with children (SESSION083 (a) resolves a straddling leaf to far before we get here), so a leaf
			// reaching this point has already been classified and emitting it covers its path exactly once.
			if(top.pixel_scale <= pixel_scale_limit && !force_descend)
			{
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_Converged);
				continue;
			}

			if(node.child_count == 0)
			{
				// A leaf can't be expanded regardless of pixel_scale - it's already the finest detail this tree has.
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_Leaf);
				continue;
			}

			// SESSION085 ETAP 3 - THE SATURATION LoD BIAS. Deliberately placed AFTER the two tests above rather than folded
			// into the converged one, and the ordering is the whole reason this costs nothing when it is off:
			//
			//  - A node that already converged has stopped; the bias only ever RAISES the limit, so it could not have
			//    changed that answer. Testing it first would pay gsSatOccluded() on every node in the walk for nothing.
			//  - A leaf is emitted whichever way this goes, so asking is wasted there too.
			//
			// What is left is exactly the population the bias can act on: nodes that would otherwise DESCEND. And when
			// sat_bias_active is false the whole thing is one bool test, so the off path is bit-for-bit what it was.
			if(sat_bias_active && !force_descend)
			{
				++num_sat_bias_tested;
				if(top.pixel_scale <= pixel_scale_limit * satBiasFor(cloud_idx_u32, positions))
				{
					++num_sat_bias_stops;
					DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
					decorated.push_back(d);
					recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_SatBias);
					continue;
				}
			}

			// Density-capped: expanding this node would recurse into a region estimated to be this dense with overdraw
			// (see GaussianSplatLodNode::layer_density) - stop here and use this node's own merged approximation
			// instead, regardless of how coarse its pixel_scale still looks. max_layer_density <= 0 disables this check
			// (the "0 = unlimited" convention GaussianSplatSettingsWidget's other debug knobs use).
			if(max_layer_density > 0.f && node.layer_density > max_layer_density && !force_descend)
			{
				hit_density_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_DensityCap);
				continue;
			}

			// Depth-capped: a hard, global ceiling on how many levels traversal may unfold, independent of pixel_scale/
			// density/budget - "the tree will never unfold finer than this, anywhere". max_tree_depth <= 0 disables it.
			if(max_tree_depth > 0 && top.depth >= (uint32)max_tree_depth && !force_descend)
			{
				hit_depth_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_DepthCap);
				continue;
			}

			// Budget cap: expanding would push us over the ceiling. Keep this node's own merged representation instead.
			// Current node is already popped, so the projected total after pushing children is decorated.size() + stack.size() + child_count.
			// SESSION080 STEP B: force_descend suppresses this too. Stopping here would cover paths the far pass also
			// covers, which is a double draw - a worse outcome than overshooting a soft target by the size of one
			// straddle band. hit_budget_cap is still raised on the unsuppressed path, so the over-budget rebuild in
			// expandParallel() still sees the same signal it always did.
			if(enforce_budget && (decorated.size() + stack.size() + node.child_count > max_splats_budget) && !force_descend)
			{
				hit_budget_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32 | far_bit; // SESSION080 §4.3: bit 30 - see far_bit.
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_BudgetCap);
				continue;
			}

			for(uint32 c = node.child_start; c < (uint32)node.child_start + node.child_count; ++c)
			{
				HeapItem child = makeHeapItem(top.member_idx, c, m.offset, top.depth + 1, positions, feature_sizes);
				child.coarse_captured = child_coarse_captured; // SESSION063 K4: propagate the once-per-branch capture flag.
				child.below_far_cut = child_below_far_cut;     // SESSION080 §4.3: likewise, once-per-branch - see the field.
				stack.push_back(child);
			}
		}

		if(head > 0) // Breadth-first only: drop what was consumed, so `stack` holds exactly the unfinished seeds.
			stack.erase(stack.begin(), stack.begin() + head);
	}

	// SESSION080: seed subtrees' worth of DFS, on a pool thread, into its own output vector. The only shared mutable
	// state is the seed cursor described below; every array it reads is a frozen snapshot, and the diag counters and cap
	// flags are per-task and reduced by the caller.
	//
	// SESSION086: the seeds are PULLED one at a time off a shared cursor rather than dealt out to the tasks up front.
	//
	// Session080 dealt them round-robin - seed s to task s % num_tasks - reasoning that round-robin decorrelates sibling
	// costs. It does, but the assignment is still fixed before any work starts, so a task that draws an expensive subtree
	// runs long while the tasks that drew cheap ones finish and idle. Measured session086 over six captures: par
	// (task_sum/task_max) 3.0-5.3 against ~17 threads, with task_max within 8% of the whole of expand_ms - one bucket
	// held roughly a third of the work and the group waited on it. Pulling bounds the imbalance by the cost of the
	// largest single SEED rather than the largest BUCKET; it needs no cost estimate for a subtree (the tree does not
	// carry one - subtree sizes are not stored, and the breadth-first linearisation makes them non-local); and it adapts
	// on its own when the pool is already busy with a barrier build, which a static deal cannot.
	//
	// run_ms now means "how long this thread was working", not "how long one bucket took", so par reads as true
	// utilisation. Fragment order across tasks was already documented as not mattering - see expandParallel().
	class GsExpandTask : public glare::Task
	{
	public:
		GsExpandTask() : diag_coarse_a(0), diag_coarse_b(0), diag_reuse_roots(0), hit_density_cap(false), hit_depth_cap(false), num_sat_bias_stops(0), num_sat_bias_tested(0), run_ms(0.0), seeds_taken(0), seed_max_ms(0.0) {}

		virtual void run(size_t /*thread_index*/) override
		{
			Timer task_timer; // SESSION080 DIAGNOSTIC - see GaussianSplatUnculledFrontier::expand_task_max_ms.
			bool unused_budget_cap = false; // Not enforced here - see expandParallel().
			while(1)
			{
				const int64 seed_i = next_seed->increment(); // Returns the value BEFORE the increment - see AtomicInt::increment().
				if(seed_i >= (int64)seeds->size())
					break;

				// expandStack() runs the stack down to empty in DFS mode and APPENDS to `decorated`, so taking one seed at a
				// time accumulates into the same output block a whole bucket used to fill.
				++seeds_taken; // SESSION086 DIAGNOSTIC
				Timer seed_timer; // SESSION086 DIAGNOSTIC - see GaussianSplatUnculledFrontier::expand_seed_max_ms.
				stack.push_back((*seeds)[(size_t)seed_i]);
				parent->expandStack(stack, decorated, *positions, *feature_sizes, *cull_radii,
					/*enforce_budget=*/false, /*breadth_first=*/false, /*pause_at_stack_size=*/0,
					diag_coarse_a, diag_coarse_b, diag_reuse_roots, unused_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);
				seed_max_ms = myMax(seed_max_ms, seed_timer.elapsed() * 1.0e3);
			}
			run_ms = task_timer.elapsed() * 1.0e3;
		}

		GaussianSplatLodTraversalTask* parent; // Outlives every task: runTaskGroup() below blocks until they have all finished.
		const js::Vector<Vec3f, 16>* positions;
		const js::Vector<float, 16>* feature_sizes;
		const js::Vector<float, 16>* cull_radii;
		const std::vector<HeapItem>* seeds; // SESSION086: shared and read-only, owned by expandParallel() for the group's lifetime.
		glare::AtomicInt* next_seed;        // SESSION086: the shared cursor into `seeds`.
		std::vector<HeapItem> stack; // This task's working stack - empty between seeds.
		js::Vector<DistIdx, 16> decorated;
		size_t diag_coarse_a, diag_coarse_b, diag_reuse_roots;
		bool hit_density_cap, hit_depth_cap;
		size_t num_sat_bias_stops, num_sat_bias_tested; // SESSION085 ETAP 3 - see the parent's.
		double run_ms; // SESSION080 DIAGNOSTIC
		size_t seeds_taken; double seed_max_ms; // SESSION086 DIAGNOSTIC - see the frontier's expand_workers/expand_seed_max_ms.
	};

	// SESSION080: the parallel expand. Walks the top of the tree serially just far enough to have a good supply of
	// independent subtrees, hands one task per subtree to the pool, then concatenates the fragments.
	//
	// Fragment order does not matter: the caller sorts the whole of `decorated` by dist_sq immediately afterwards. The
	// only observable difference from the serial walk is the relative order of nodes at EXACTLY equal dist_sq, which the
	// (stable) radix sort would otherwise have left in DFS order - cosmetic, since they are at the same depth in the
	// front-to-back blend.
	//
	// The budget cap is the one piece of shared state, and it is handled exactly rather than approximately. The tasks run
	// with it switched off; afterwards, if the total is within budget, the serial walk provably could not have fired it
	// either - at every step of that walk decorated.size() + stack.size() + child_count is a lower bound on its own final
	// output size (every stacked entry yields at least one node), so if the final total is <= budget the test never
	// tripped. If the total DOES exceed budget the parallel result is discarded and the whole walk is redone serially,
	// which keeps the DFS-order truncation policy bit-for-bit. That path has never been observed to run (budget_cap has
	// measured false in every capture - see [gsr-traversal]), so its cost is theoretical.
	void expandParallel(std::vector<HeapItem>& stack, js::Vector<DistIdx, 16>& decorated,
		const js::Vector<Vec3f, 16>& positions, const js::Vector<float, 16>& feature_sizes, const js::Vector<float, 16>& cull_radii,
		size_t& diag_coarse_a, size_t& diag_coarse_b, size_t& diag_reuse_roots,
		bool& hit_budget_cap, bool& hit_density_cap, bool& hit_depth_cap, size_t& num_sat_bias_stops, size_t& num_sat_bias_tested)
	{
		// SESSION080/086: the seed set the pool pulls from. Two numbers, meaning different things: `spread_seeds` is how
		// far the plain breadth-first prologue runs before the targeted splitting below takes over, `target_seeds` is how
		// many independent pieces the walk is finally handed out in.
		//
		// Both are granularity, not tuning. What decides the BALANCE is the split rule below, which is driven by a measured
		// property of each seed rather than by a number chosen per scene - see there.
		const size_t concurrency = myMax<size_t>(1, (size_t)task_manager->getConcurrency());
		const size_t spread_seeds = concurrency * 4;
		// SESSION086: not a fixed count any more - the renderer carries it and adjusts it from the previous traversal's
		// own measurements. See GaussianSplatRenderer::updateExpandSeedTarget(). 0 means "nothing measured yet".
		const size_t target_seeds = myMax(spread_seeds, expand_seed_target > 0 ? expand_seed_target : concurrency * 16);

		// run()'s seed loop has already written any no-tree member's splats into `decorated`, and this function never
		// revisits those - so the over-budget fallback below must rewind to here, not to empty.
		const size_t initial_decorated = decorated.size();

		// Walk the top serially until the stack holds a first spread of subtree roots. Same code, same decisions - this is
		// simply the first part of the identical DFS, and anything it finishes on the way lands in `decorated` directly.
		Timer prologue_timer; // SESSION080 DIAGNOSTIC
		expandStack(stack, decorated, positions, feature_sizes, cull_radii, /*enforce_budget=*/false, /*breadth_first=*/true, /*pause_at_stack_size=*/spread_seeds,
			diag_coarse_a, diag_coarse_b, diag_reuse_roots, hit_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);

		// SESSION086: TARGETED SPLITTING. Keep expanding whichever seed has the largest pixel_scale until there are enough
		// of them, instead of expanding everything uniformly and stopping on a global count.
		//
		// Why the plain prologue cannot do this. Its stop condition is "enough entries are pending", a property of the whole
		// QUEUE, so it halts the instant the count is reached and whatever monster is sitting unexpanded in that queue stays
		// whole. Measured session086, which is what expand_workers/expand_seed_max_ms were added to settle: with the
		// distribution already dynamic and workers=15 of 17 threads confirmed available, seed_max_ms was 131.0 against a
		// task_max_ms of 131 - one seed WAS the whole window, ~15% of all the work in a single piece. Deepening the plain
		// prologue 16x (272 -> 4354 seeds) moved it only from 142 to 131: uniform deepening splits the entire tree to pay
		// for splitting one branch of it, and mostly misses the branch.
		//
		// pixel_scale as the cost signal. It is not a guess at subtree size - it is the quantity the walk STOPS on
		// (pixel_scale <= pixel_scale_limit), so a seed's pixel_scale says how many levels it still has to descend, and the
		// node count grows roughly as its square. The tree stores no subtree sizes and its breadth-first linearisation makes
		// them non-local, so this is the only cost signal available for free, and it happens to be the right one. It can be
		// fooled by an unusually dense subtree at a modest pixel_scale (layer_density is not consulted); that failure shows
		// up as an unsplit straggler, which is exactly what seed_max_ms reports, so it would not be silent.
		//
		// Cost. One iteration expands ONE node - pause_at_stack_size=2 stops as soon as that node's children are pending -
		// so this is a few hundred to a few thousand node expansions, not a walk (measured 0.02-0.21ms).
		//
		// HOW MANY splits is not decided here. Ordering by pixel_scale was confirmed right by measurement - the costliest
		// seed reported px 851 with the limit at single digits, i.e. it WAS the heap's first pick - but a stop rule of
		// "until there are N seeds" is not a statement about balance at all, and on a scene where hundreds of nodes need
		// splitting a fixed N runs out long before the worst one is small. So the count comes from the closed loop in
		// updateExpandSeedTarget(), which raises it while the measured costliest seed exceeds a thread's fair share of the
		// measured total and lowers it when it does not. Nothing here is chosen per scene.
		if(stack.size() < target_seeds)
		{
			std::vector<HeapItem> split_tmp; // Holds one node's children at a time; kept outside the loop so the capacity is reused.
			split_tmp.reserve(16);
			std::make_heap(stack.begin(), stack.end(), SeedPixelScaleLess());
			while(!stack.empty() && stack.size() < target_seeds)
			{
				std::pop_heap(stack.begin(), stack.end(), SeedPixelScaleLess()); // Largest pixel_scale moves to the back.
				split_tmp.clear();
				split_tmp.push_back(stack.back());
				stack.pop_back();

				// One level. If the node stops here instead (converged, leaf, capped) it is emitted into `decorated` by the very
				// same code that would have emitted it during the walk, and split_tmp comes back empty - nothing to re-add.
				expandStack(split_tmp, decorated, positions, feature_sizes, cull_radii, /*enforce_budget=*/false, /*breadth_first=*/true, /*pause_at_stack_size=*/2,
					diag_coarse_a, diag_coarse_b, diag_reuse_roots, hit_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);

				for(size_t i=0; i<split_tmp.size(); ++i)
				{
					stack.push_back(split_tmp[i]);
					std::push_heap(stack.begin(), stack.end(), SeedPixelScaleLess());
				}
			}
		}
		expand_prologue_ms = prologue_timer.elapsed() * 1.0e3;
		expand_seeds = stack.size();

		if(stack.empty()) // The whole tree fitted in the serial prologue (a small or heavily culled scene).
			return;

		// SESSION086: one task per THREAD, each pulling seeds off a shared cursor, rather than session080's fixed
		// round-robin deal of seeds into concurrency*4 buckets - see GsExpandTask for the measurement that motivated it.
		// More tasks than threads no longer buys anything once the balancing is dynamic; it would only add task overhead.
		// `stack` stays alive as the seed array for the whole group, so it is cleared after the run rather than before it.
		const size_t num_tasks = myMin(stack.size(), concurrency);
		expand_num_tasks = num_tasks;
		glare::AtomicInt next_seed(0);
		glare::TaskGroupRef group = new glare::TaskGroup();
		group->tasks.resize(num_tasks);
		for(size_t t=0; t<num_tasks; ++t)
		{
			Reference<GsExpandTask> task = new GsExpandTask();
			task->parent = this;
			task->positions = &positions; task->feature_sizes = &feature_sizes; task->cull_radii = &cull_radii;
			task->seeds = &stack;
			task->next_seed = &next_seed;
			group->tasks[t] = task;
		}

		task_manager->runTaskGroup(group);
		stack.clear(); // Only now: the tasks read it as their seed array - see above.

		// SESSION080: what the serial part (run()'s no-tree seeds plus this function's prologue) already put in - NOT
		// initial_decorated, which predates the prologue. This is where the tasks' blocks start, and it is the same
		// value `total` is seeded with below, so the two agree by construction.
		const size_t serial_prefix_n = decorated.size();

		size_t total = serial_prefix_n;
		for(size_t t=0; t<num_tasks; ++t)
		{
			const GsExpandTask* const task = static_cast<const GsExpandTask*>(group->tasks[t].ptr());
			total += task->decorated.size();
			expand_task_sum_ms += task->run_ms; // SESSION080 DIAGNOSTIC - see the fields' comment.
			expand_task_max_ms = myMax(expand_task_max_ms, task->run_ms);
			if(task->seeds_taken > 0) ++expand_workers; // SESSION086 DIAGNOSTIC - see the fields' comment.
			expand_seed_max_ms = myMax(expand_seed_max_ms, task->seed_max_ms); // SESSION086 - the quantity updateExpandSeedTarget() steers on.
		}

		// Over budget: the parallel walk's truncation would not match the serial one's, so throw it away and redo the
		// whole thing serially. See this function's comment for why reaching here at all is a theoretical case.
		if(total > max_splats_budget)
		{
			decorated.resize(initial_decorated); // NOT 0 - see initial_decorated's comment.
			diag_coarse_a = diag_coarse_b = diag_reuse_roots = 0;
			hit_density_cap = hit_depth_cap = false;
			num_sat_bias_stops = num_sat_bias_tested = 0; // SESSION085 ETAP 3
			stack.clear();
			for(size_t mi=0; mi<scratch->members_snapshot.size(); ++mi)
			{
				const GaussianSplatLodTraversalScratch::MemberSnapshot& m = scratch->members_snapshot[mi];
				if(m.hidden || m.splat_data->lod_tree.empty())
					continue; // The no-tree members' nodes are already in `decorated` from run()'s seed loop, which this does not redo.
				stack.push_back(makeHeapItem((uint32)mi, /*tree_local_idx=*/0, m.offset, /*depth=*/0, positions, feature_sizes));
			}
			expandStack(stack, decorated, positions, feature_sizes, cull_radii, /*enforce_budget=*/true, /*breadth_first=*/false, /*pause_at_stack_size=*/0,
				diag_coarse_a, diag_coarse_b, diag_reuse_roots, hit_budget_cap, hit_density_cap, hit_depth_cap, num_sat_bias_stops, num_sat_bias_tested);
			return;
		}

		// SESSION080 DIAGNOSTIC (plan2 §4.1): serial - see expand_splice_reserve_ms/expand_splice_copy_ms. Timed on its
		// own, separately from expand_ms as a whole, because a prior estimate of this cost (task_max_ms/task_sum_ms
		// subtracted from expand_ms) was an inference, not a measurement, and plan2 §4.1 explicitly calls for a real
		// timer before optimising it. The reserve is timed apart from the copy because the two want opposite fixes.
		//
		// SESSION080 (plan2 §4.1): resize, not reserve + push_back. `total` is already exact - the loop above summed
		// every task's output - so the destination can be sized once and the blocks memcpy'd into their own slices.
		// DistIdx is trivially copyable (two scalars), the slices are disjoint and written in task order, so the result
		// is bit-identical to the element-by-element append it replaces.
		//
		// js::Vector::resize(n) placement-news each new element as `T` without parentheses, which is deliberately NOT
		// value-initialisation for a POD (see the note in Vector.h), so this does not memset 106MB. The loop around it
		// should compile away entirely for a trivial T - and expand_splice_reserve_ms, which brackets exactly this line,
		// is what says whether it did: it measured 0.02-0.05ms as a bare reserve, so any jump here is that loop.
		Timer splice_reserve_timer;
		decorated.resize(total);
		expand_splice_reserve_ms = splice_reserve_timer.elapsed() * 1.0e3;

		Timer splice_copy_timer;
		size_t write_pos = serial_prefix_n; // Where the tasks' output starts - see that variable.
		for(size_t t=0; t<num_tasks; ++t)
		{
			const GsExpandTask* const task = static_cast<const GsExpandTask*>(group->tasks[t].ptr());
			const size_t task_n = task->decorated.size();
			if(task_n > 0)
			{
				std::memcpy(decorated.data() + write_pos, task->decorated.data(), task_n * sizeof(DistIdx));
				write_pos += task_n;
			}
			diag_coarse_a += task->diag_coarse_a;
			diag_coarse_b += task->diag_coarse_b;
			diag_reuse_roots += task->diag_reuse_roots; // SESSION080 STEP B
			hit_density_cap = hit_density_cap || task->hit_density_cap;
			hit_depth_cap   = hit_depth_cap   || task->hit_depth_cap;
			num_sat_bias_stops  += task->num_sat_bias_stops;  // SESSION085 ETAP 3 - summed, not or-ed: see the scratch field.
			num_sat_bias_tested += task->num_sat_bias_tested;
		}
		assert(write_pos == total); // The sizing pass and this one walk the same tasks in the same order.
		expand_splice_copy_ms = splice_copy_timer.elapsed() * 1.0e3;
	}

	uint64 cloud_id;
	uint64 topology_generation;
	Reference<GaussianSplatLodTraversalScratch> scratch;
	Vec4f cam_pos_ws;
	float pixel_scale_limit;
	size_t max_splats_budget;
	float max_layer_density;
	int max_tree_depth;
	float focal_px;
	Planef frustum_clip_planes[6]; // SESSION055: local copy; matches OpenGLEngine.h's cap. Empty when frustum_cull_enabled=false (getFrustumStructureReport() takes that path).
	int num_frustum_clip_planes;
	bool frustum_cull_enabled;
	float translation_dilation[6]; // SESSION055: per-plane world-space margin (metres) - see kickOffTraversals()'s anisotropic dilation block.
	float rotation_dilation_rate;  // SESSION055: rad; multiplied by dist-to-node inside cull so a rotation of theta at range r dilates the plane by r*theta.
	ThreadSafeQueue<Reference<ThreadMessage> >* result_queue;
	js::Vector<FrontierNodeRecord, 16>* frontier_record; // Null (the normal case) means don't record anything - see recordFrontierNode().
	bool build_unculled_frontier; // SESSION063: gather the SoA U(P) at the end of run() and hand it back on the result msg.
	bool coarse_floor_enabled;    // SESSION063 K4: also capture the coarse floor.
	float coarse_pixel_scale;     // SESSION063 K4: pixel_scale threshold for the coarse floor cut (>> pixel_scale_limit).
	bool dist_clamp_enabled;      // SESSION072: distance-slice early-cull toggle - see GaussianSplatRenderer::getDistClampEnabled().
	float dist_clamp_min, dist_clamp_max;
	bool dist_clamp_invert;
	bool sat_diag_log;               // SESSION076 DIAGNOSTIC: see the ctor param.
	bool coarse_layer_drawn;         // SESSION076: see the ctor param.

	// SESSION085 ETAP 3 - the saturation LoD bias. The barrier the cloud had at kick time (may be NULL: a cloud's first
	// traversal, or the stage off), used READ-ONLY and immutable once built, so sharing it with a worker is safe.
	// Note this is the PREVIOUS build's barrier - one traversal behind, exactly as the apply stage already consumes it.
	Reference<GaussianSplatSaturationBarrier> sat_barrier;
	float sat_bias_ceiling;  // Cap on the multiplier - see GaussianSplatRenderer::getSatBiasCeiling(). 1 = off.
	float sat_bias_exponent; // How fast the multiplier climbs to that ceiling - see GaussianSplatRenderer::getSatBiasExponent(). 1 = session085's original curve.
	bool sat_bias_active;    // Precomputed once at kick: ceiling > 1 AND a usable barrier exists. Keeps the DFS's per-node test to one bool.

	// SESSION085 ETAP 3: the value this traversal answers to for reuse purposes - see
	// GaussianSplatUnculledFrontier::sat_bias_barrier_time. A function rather than two computations, because the
	// ctor (which decides whether a far block may be inherited) and run() (which stamps the produced frontier) have
	// to agree exactly; if they ever disagreed, a block would be inherited under one key and labelled with another.
	inline double satBiasBarrierTime() const { return sat_bias_active ? sat_barrier->built_time_real_s : 0.0; }
	glare::TaskManager* task_manager; // SESSION079: may be NULL - see the ctor param.

	// SESSION080 DIAGNOSTIC: filled by expandParallel(), copied onto the frontier at the end of run() - see
	// GaussianSplatUnculledFrontier's matching fields for what the numbers are for. All stay zero on the serial path.
	size_t expand_seeds, expand_num_tasks;
	double expand_prologue_ms, expand_task_max_ms, expand_task_sum_ms;
	size_t expand_workers; double expand_seed_max_ms; // SESSION086 DIAGNOSTIC - see the frontier's matching fields.
	size_t expand_seed_target;  // SESSION086 - see GaussianSplatRenderer::updateExpandSeedTarget().
	double expand_splice_reserve_ms, expand_splice_copy_ms; // SESSION080 DIAGNOSTIC (plan2 §4.1)

	// SESSION080: the previous traversal's UNPRUNED frontier for this cloud, captured by kickOffTraversals() before this
	// task was created - SplatCloud::last_unpruned_ufrontier, NOT cached_ufrontier (see that field's comment for why the
	// pruned one will not do). NULL on a cloud's first kick. Serves two consumers:
	//   - STEP A, the sort-staleness diagnostic, gated by sort_staleness_diag_enabled (see
	//     GaussianSplatUnculledFrontier::sort_staleness_*);
	//   - STEP B, frontier reuse, gated by reuse_enabled below.
	Reference<GaussianSplatUnculledFrontier> prev_frontier;
	bool sort_staleness_diag_enabled;

	// SESSION080 §4.3 - see GaussianSplatUnculledFrontier::far_block for the idea and the geometry. reuse_enabled folds
	// the knob together with every precondition the scheme needs, so the hot loop tests one bool.
	//
	// Preconditions, each of which would otherwise break the cut invariant or the block's meaning:
	//   - frustum cull is off (it is, in split-filter mode): a culled traversal's frontier is not a complete cut, so a
	//     block split out of one could hold a hole;
	//   - the distance clamp is off, for the same reason - it prunes whole subtrees out of the cut;
	//   - the coarse floor is off. Not a correctness problem but a scope one: the coarse layer is captured once per
	//     branch on the way down, and a branch that stops at the cut is never walked below it, so the two capture rules
	//     would have to be reconciled. The owner runs coarse off; revisit if that changes.
	bool reuse_enabled;

	// SESSION080 §4.3: false = this traversal BUILDS a block (walks everything, marking the far part); true = a block
	// already exists and this traversal stops at its frozen cut and references it. See the classification in
	// expandStack(), which is the only place the distinction is acted on.
	bool far_cut_is_frozen;

	// The FROZEN anchor the cut predicate is evaluated against - the block's own anchor when inheriting one, and this
	// traversal's camera position when building one. Deliberately NOT cam_pos_ws in the frozen case: holding it still is
	// what makes every traversal in a generation stop at the same nodes. See expandStack().
	Vec4f reuse_prev_anchor_ws;
	float reuse_split_dist;

	// The block being inherited, held so run() can hand it to the frontier it publishes. Null while building one.
	Reference<GaussianSplatUnculledFrontier> inherited_far_block;

	// How far the camera may drift from a block's frozen anchor before a traversal gives up on it and walks the whole
	// tree - see the clause that uses it in the constructor, and GaussianSplatUnculledFrontier::reuse_base_anchor_ws for
	// what it protects against. A fraction of the split distance, not an absolute: 1/4 keeps the accumulated ordering
	// error inside the far region well under the range at which that region sits, which is the ratio that decides
	// whether it can be seen.
	float reuse_drift_fraction; // SESSION086: was `static const float reuse_max_drift_fraction = 0.25f` - see GaussianSplatRenderer::getFrontierReuseDriftFraction() for why it had to become adjustable.
	// SESSION088: the LoD half of the drift bound - see GaussianSplatRenderer::getSatBarrierAgreeTol(). 1 accepts any
	// barrier change, i.e. exactly the pre-session088 behaviour.
	float sat_barrier_agree_tol;
	float diag_barrier_disagreement; // What the acceptance clause gates on: the FLIP component only, see where it is assigned. Meaningless unless diag_barrier_measured - see below.
	float diag_barrier_flip, diag_barrier_depth, diag_barrier_mean_rel_depth; // SESSION088: the two components and the magnitude behind the second - see gsSatBarrierDisagreement().
	bool diag_barrier_measured; // SESSION088: false = no block, no bias, or no captured grid, so nothing was compared. Distinct from "compared and agreed", which the first cut conflated with it - every field here defaults to 0.
	// SESSION088 DIAGNOSTIC, TEMPORARY: the two build_time_real_s the disagreement above was computed FROM - the live
	// barrier's own, and the one the block's grid was captured under. Exists purely to let a suspicious barrier_dis=
	// reading be checked against ground truth: equal timestamps with nonzero disagreement means the comparison itself is
	// broken (same barrier, should read 0), not that the barrier moved. Remove once the instrument is trusted.
	double diag_live_barrier_time, diag_block_barrier_time;
};


} // end anonymous namespace


GaussianSplatRenderer::GaussianSplatRenderer(OpenGLEngine& opengl_engine_)
:	opengl_engine(&opengl_engine_), next_handle(1), next_cloud_id(1), num_sorts_in_flight(0),
	num_traversals_in_flight(0), num_filters_in_flight(0), lod_pixel_scale_limit(1.0f), lod_max_splats_budget(10000000), lod_resort_move_threshold_ws(0.1f),
	lod_max_layer_density(0.0f), lod_max_tree_depth(0), lod_frustum_cull_enabled(true), split_filter_enabled(false),
	adaptive_pixel_scale_enabled(false), adaptive_pixel_scale(2.0f), adaptive_target_fps(50.f), // SESSION079 - see updateAdaptivePixelScale(). Off by default: it overrides a value the user may have set by hand.
	adaptive_frame_ms_ema(0.0), adaptive_frame_ms_sum(0.0), adaptive_frame_samples(0),
	adaptive_last_frame_time_s(-1.0), adaptive_last_eval_time_s(0.0), adaptive_last_force_time_s(-1.0),
	// SESSION079: latency 0.17 -> 0.2, max rot rate 40 -> 200 deg/s, min 45 -> 50, all confirmed on the owner's
	// bridge-and-forest scene, which is the first one heavy enough to stress them.
	//
	// The max rate mattered most and was the most wrong. At 40 deg/s the band is pinned to 8 deg while the camera turns
	// at 350; over the forest the filter's kick-to-drain round trip is 85ms against the interior's 34 (7.2M survivors
	// against 2.9M - open sky occludes nothing, so the saturation prefilter drops 16.8% there against 76.8% indoors), so
	// the applied list went 31 deg stale against a 16 deg band and tore a visible hole along the frustum edge. The old
	// default was confirmed on the interior, where the same knob never binds - a value fitted to the one scene that
	// could not disprove it.
	//
	// The latency change follows the same measurement: a list is on screen from its own kick until the NEXT drain, so
	// the envelope to cover is roughly twice the round trip, ~170ms over the forest. See [gsr-filter-drain]'s deficit=,
	// which is exactly stale minus band and goes positive precisely when a hole is visible.
	filter_dilation_latency(0.2f), filter_min_rot_rate_deg_per_s(50.f), filter_max_rot_rate_deg_per_s(200.f), filter_min_trans_rate_m_per_s(2.0f), // SESSION063 K3, SESSION072, SESSION079 - see above.
	split_coarse_floor_enabled(true), split_coarse_pixel_scale(30.f), filter_coarse_dilation_latency(0.9f), coarse_layer_debug(false), // SESSION063 K4
	filter_debug_log(false), kick_debug_log(false), cpu_prof_log(false), // SESSION072: default off - see getFilterDebugLog()'s comment.
	filter_frustum_planes_enabled(true), // SESSION074 - see getFilterFrustumPlanesEnabled().
	sat_prefilter_threshold(0.98f), // SESSION079 - see getSatPrefilterThreshold().
	sat_diag_log(false), sat_probe_log(false), sat_probe_last_print_s(0.0), // SESSION088 DIAGNOSTIC: the probe is off by default like every other trace - see getSatProbeLog().
	sat_bias_exponent(0.2f), // SESSION088: the owner's calibrated value - see getSatBiasExponent() for what it trades. 1 would be session085's original fixed curve.
	sat_debug_overlay_mode(GaussianSplatSatDebugOverlayMode_Off), sat_grid_subdiv(0.3f), sat_region_radius(0.5f), sat_region_closing_tiles(0), sat_bias_ceiling(1.f), // SESSION076/078/081: diagnostics off by default - see getSatDiagLog()/getSatDebugOverlayMode(). SESSION085 ETAP 6: sat_bias_ceiling 1 = the LoD bias off, i.e. the pre-bias behaviour; the calibrated value is still owner-selected rather than a default. SESSION086: sat_region_radius frozen at 0.5 - it is no longer only a conservatism knob, it also sets how long a barrier survives camera motion, which is what keeps the barrier rebuild off the traversal's threads. See getSatRegionRadius().
	// SESSION080 STEP B: 0 = walk everything, the pre-session080 behaviour, bit-for-bit. SESSION086 turned it on at 25,
	// measured it against the LoD bias, and turned it back off - the mechanism is sound but incompatible with the bias
	// as cut. The measurement and the verdict are in getFrontierReuseSplitDist(); the drift fraction that measured it
	// stays at session080's 0.25 and no longer has a UI knob.
	frontier_reuse_split_dist(0.f), frontier_reuse_drift_fraction(0.25f), sat_barrier_agree_tol(1.f), expand_seed_target(0), // SESSION086: 0 = nothing measured yet - see updateExpandSeedTarget(). SESSION088: tol 1 = accept any barrier change, as before.
	filter_latency_measured_enabled(false), sat_predict_gain(0.f), // SESSION088: both default to today's behaviour - see getFilterLatencyMeasuredEnabled()/getSatPredictGain(). The two GsMeasuredLatency members self-initialise.
	sat_predict_prev_kick_pos_ws(0.f), sat_predict_prev_kick_time_s(0.0), have_sat_predict_prev_kick(false), diag_sat_predict_lead_m(0.f), // SESSION088 - see kickOffSaturationBuilds().
	splat_point_size_px(1.f),
	splat_merge_spread_widen(3.0f), // SESSION071: analytic minimum is sqrt(3) (see widenedMergedScale()); owner default set higher for extra margin.
	// SESSION071: GaussianSplatMergeColourParams defaults to Energy - owner-confirmed better at every pixel scale limit tested; Legacy is kept only as the A/B comparison.
	have_prev_think_cam_state(false), prev_think_cam_pos_ws(0.f), prev_think_cam_forward_ws(0.f), cam_velocity_ema_ws(0.f), cam_angular_speed_ema(0.f), cam_angular_speed_peak(0.f), cam_inst_angular_speed(0.f), cam_angular_axis_ema_ws(0.f), // SESSION072
	splat_size_clamp_min(0.0f), splat_size_clamp_max(0.0f), splat_size_clamp_invert(false),
	splat_dist_clamp_min(0.0f), splat_dist_clamp_max(1000.0f), splat_dist_clamp_invert(false), splat_dist_clamp_enabled(false), splat_alpha_cutoff(1.0f / 255.0f),
	splat_alpha_gain(1.0f), splat_alpha_gamma(1.0f), // Identity: the cloud as captured - see getAlphaGain().
	last_report_reached_rasteriser(0), last_report_in_frustum(0),
	splat_num_draw_slices(1), splat_draw_slice_limit(0), splat_layer_cap(0), splat_layer_cap_opaque(true), splat_coverage_cap(0.f), splat_ablation_stage(0), splat_quad_radius_scale(1.f), cap_fill_mask_tex_uniform_loc(-2), splat_area_scale_gamma(1.f), splat_area_scale_ref_px(20.f), splat_coverage_shrink_strength(0.f), splat_coverage_shrink_mode(0), splat_coverage_reduce_mode(0), splat_show_coverage_map_level(-1), coverage_mask_tex_uniform_loc(-2), splat_ewa_fix_enabled(true), splat_near_fade_width(0.3f), splat_near_epsilon(0.1f), splat_dof_depth_mode(0), splat_dof_depth_prepass_alpha_min(0.3f), splat_hide_test_conservative(true), splat_layer_estimate_requested(false), splat_slice_growth(1.0f), splat_visible_slicing(false), splat_saturation_gate_enabled(false), splat_saturation_threshold(1.0f - 1.0f / 255.0f),
	splat_saturation_mask_downscale(4), splat_mask_tex_uniform_loc(-2), splat_hide_count_tex_uniform_loc(-2),
	splat_accum_buffer_8bit(false),
	splat_accum_buffer_scale(1.f), splat_accum_upsample_bilinear(true), // SESSION067 - 1 = full resolution, i.e. exactly the pre-knob behaviour; the upsample setting is not consulted at that scale.
	splat_area_slice_mode(0), splat_area_slice_px(256.f), // SESSION067 DIAGNOSTIC - off; 256 px is the threshold the measured histogram puts 85% of the fill above - see getAreaSliceMode().
	splat_deconv_enabled(false), splat_deconv_gain(1.f), splat_rcas_enabled(false), splat_rcas_sharpness(0.5f), // SESSION068 - both off; gain 1.0 = the analytically derived strength; sharpness 0.5 = mid of RCAS's safe range.
	splat_taa_enabled(false), splat_taa_frame_count(0), splat_taa_frame_index_monotonic(0), splat_taa_write_index(0), splat_taa_current_jitter_px(0.f, 0.f), // SESSION069 - off; every last-* below is (re)set the first time updateTAAState() sees an active frame, so their initial values don't matter as long as they don't misidentify frame 0 as unchanged.
	splat_low_pass_variance_current(0.3f), // SESSION069 fix - matches the pre-fix hardcoded value until think() runs once.
	splat_taa_last_view_matrix(Matrix4f::identity()), splat_taa_last_viewport_dims(0, 0), splat_taa_last_accum_dims(0, 0),
	splat_taa_last_settings_hash(0), splat_taa_last_num_clouds(0), splat_taa_last_content_apply_snapshot(0), splat_taa_content_apply_counter(0),
	splat_show_overdraw_mode(0), splat_hide_overdraw_enabled(false), splat_hide_alpha_enabled(false),
	splat_hide_overdraw_mask_valid(false), splat_hide_overdraw_mask_view(Matrix4f::identity()),
	splat_hide_overdraw_mask_viewport_w(0), splat_hide_overdraw_mask_viewport_h(0), splat_hide_overdraw_mask_block(0),
	splat_hide_overdraw_mask_threshold(0.f), splat_hide_overdraw_mask_mode(0), splat_hide_overdraw_mask_splats_drawn(0), splat_hide_overdraw_mask_rebuilds(0),
	splat_overdraw_range_min(2.0f), splat_overdraw_range_max(100.0f)
{}


GaussianSplatRenderer::~GaussianSplatRenderer()
{}


void GaussianSplatRenderer::buildShadersIfNeeded()
{
	if(shader_prog)
		return;

	const std::string shader_dir           = opengl_engine->getShadersDir();
	const std::string version_directive    = opengl_engine->getVersionDirective();
	const std::string preprocessor_defines = opengl_engine->getPreprocessorDefines();

	{
		OpenGLProgramExtraArgs extra_args;
		extra_args.input_vert_attribute_bindings.push_back(OpenGLProgramExtraArgs::AttributeBinding({"splat_index_in", splat_index_attribute_loc}));

		shader_prog = new OpenGLProgram(
			"gaussian splat prog",
			new OpenGLShader(shader_dir + "/gaussian_splat_vert_shader.glsl", version_directive, preprocessor_defines, GL_VERTEX_SHADER),
			new OpenGLShader(shader_dir + "/gaussian_splat_frag_shader.glsl", version_directive, preprocessor_defines, GL_FRAGMENT_SHADER),
			opengl_engine->getAndIncrNextProgramIndex(),
			/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support,
			extra_args
		);
	}
	opengl_engine->addProgram(shader_prog);

	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "viewport_dims_px");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "focal_len_px");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_tex_width");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "splat_size_clamp_min_max");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_size_clamp_invert");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_alpha_cutoff");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_show_overdraw");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int, "splat_saturation_mask_block"); // 0 disables the test - see setSplatMaskBlockSize().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int, "splat_saturation_mask_max_level");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "splat_dist_clamp_min_max");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_dist_clamp_invert");
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_mask_centre_test"); // See setSplatMaskBlockSize().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "splat_alpha_gain_gamma"); // See getAlphaGain(). NOTE: user_uniform_vals is sized to match this list in allocCloud().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,  "splat_frag_mask_block"); // DIAGNOSTIC ONLY - the per-pixel layer cap's test, see getLayerCap().  Set by the draw path, like the mask uniforms above.
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_ablation_stage"); // DIAGNOSTIC ONLY - see getAblationStage().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_quad_radius_scale"); // DIAGNOSTIC ONLY - see getQuadRadiusScale().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_area_scale_gamma"); // DIAGNOSTIC ONLY - see getAreaScaleGamma().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_area_scale_ref_px"); // DIAGNOSTIC ONLY - see getAreaScaleRefPx().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_coverage_shrink_strength"); // DIAGNOSTIC ONLY - see getCoverageShrinkStrength().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_ewa_fix_enabled"); // See getEWAProjectionFixEnabled().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_near_fade_width"); // See getNearFadeWidth().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_coverage_shrink_mode"); // DIAGNOSTIC ONLY - see getCoverageShrinkMode().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_near_epsilon"); // See getNearEpsilon().  NOTE: user_uniform_vals is sized to match this list in allocCloud().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_hide_count_active");    // SESSION066 DIAGNOSTIC - the "Clip" per-splat cull, see setHideCountCull(). 0 = off (no sample).
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_hide_count_threshold"); // SESSION066 DIAGNOSTIC - red-zone threshold (getOverdrawRangeMax()).
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_area_slice_mode"); // SESSION067 DIAGNOSTIC - the projected-area slice, see getAreaSliceMode().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_area_slice_px");   // SESSION067 DIAGNOSTIC - its threshold, converted to buffer pixels in think().  NOTE: user_uniform_vals is sized to match this list in allocCloud().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2,  "splat_jitter_px");       // SESSION069 - subpixel jitter in accum-buffer pixels.  NOTE: user_uniform_vals is sized to match this list in allocCloud().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_low_pass_variance"); // SESSION069 fix - replaces the hardcoded +0.3 anti-alias low-pass; shrunk while TAA is active, see think(). NOTE: user_uniform_vals is sized to match this list in allocCloud().
	shader_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_point_size_px");   // SESSION071 DIAGNOSTIC - size of the point quad the ablation stages 2-5 emit, see getPointSizePx(). NOTE: user_uniform_vals is sized to match this list in allocCloud().


	// Splats blend into an accumulation buffer of their own rather than straight onto the main colour buffer, so that
	// the blend happens in the display-referred space 3DGS fits them in; this program does the full-viewport pass that
	// resolves that buffer and composites it.  See OpenGLEngine::drawSplatClouds().
	const std::string key_defs = preprocessorDefsForKey(ProgramKey(ProgramKey::ProgramName_splat_resolve, ProgramKeyArgs())); // Needed to define MATERIALISE_EFFECT to 0 etc. for frag_utils_glsl.
	resolve_prog = new OpenGLProgram(
		"gaussian splat resolve prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_frag_shader.glsl", version_directive, key_defs + preprocessor_defines + opengl_engine->frag_utils_glsl, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(resolve_prog);

	// Registered via appendUserUniformInfo() rather than resolved with a direct getUniformLocation() call here, since
	// resolve_prog's build may still be in flight (parallel_shader_compile_support) at this point - appendUserUniformInfo
	// is what resolves a pending uniform's location once the build actually completes; see its own comment. Read back
	// by setResolveOverdrawUniforms(), in this same order, once resolveSplatAccumBuffer() has bound the program.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_show_overdraw");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_overdraw_range_min");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_overdraw_range_max");
	// DIAGNOSTIC ONLY - the coverage map view, see getShowCoverageMapLevel(). The sampler goes through the same list as
	// the rest: a sampler uniform is set with glUniform1i to the texture unit it should read, so it needs its location
	// resolved the same way and at the same time as the ints beside it.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_show_coverage_map_level");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_coverage_map_block");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_coverage_map_texture");

	// SplatDoFDepthMode_Weighted - see setResolveDoFDepthUniforms(). Indices 6-8, read back there and by
	// getResolveDoFDepthTexUniformLoc() in this same order.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_write_weighted_depth");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_dof_near_clip_dist");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_dof_depth_texture");

	// SESSION067 - the downscaled accumulation buffer's upsample, see setResolveUpsampleUniforms(). Indices 9-11, read
	// back there in this same order.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_accum_upsample");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2,  "splat_accum_dims_px");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2,  "splat_resolve_dims_px");

	// SESSION068 - post-processing enhancers on the upscaled buffer, see setResolveEnhanceUniforms(). Indices 12-15,
	// read back there in this same order.  Independent on/off flags for A/B testing either filter alone or both together.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_deconv_enabled");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_deconv_gain");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_rcas_enabled");
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_rcas_sharpness");

	// SESSION069 - see setResolveTAAActiveUniform().  1 = the resolve is writing into the TAA current texture (blending
	// off, discard replaced by vec4(0) so history knows this pixel was empty this frame); 0 = the pre-TAA path,
	// writing straight into the frame's colour buffer with the front-to-back "under" blend.  Index 16.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_taa_active");

	// SESSION069 fix - see setResolveTAAJitterUniform().  Same value as the vertex shader's splat_jitter_px this frame,
	// used to un-shift the accum-buffer read so the reconstruction is registered per-frame instead of always landing on
	// the same full-res pixels.  (0,0) whenever TAA is inactive.  Index 17.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2,  "splat_jitter_px");

	// SESSION069 fix - see setResolveLowPassVarianceUniform().  Same value as the vertex shader's splat_low_pass_variance
	// this frame - applyMatchedDeconv() has to invert the blur that was ACTUALLY applied, not a hardcoded 0.3, or it
	// over-sharpens whenever TAA has shrunk the low-pass.  Index 18.
	resolve_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_low_pass_variance");


	// Marks the pixels the composite has already finished with, between draw slices - see
	// OpenGLEngine::markSaturatedSplatPixels().  Shares the resolve pass's full-viewport quad vertex shader, since it is
	// the same geometry doing the same job.  No frag_utils_glsl here: this shader reads one channel and discards, and
	// has no use for the display transform the resolve shader needs.
	saturation_mask_prog = new OpenGLProgram(
		"gaussian splat saturation mask prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_saturation_mask_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(saturation_mask_prog);

	// Same reason as resolve_prog's uniforms above: the build may still be in flight here, and appendUserUniformInfo() is
	// what resolves the location once it completes.  Read back by setSaturationMaskUniforms().
	saturation_mask_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_saturation_threshold");
	saturation_mask_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_mask_block_size");
	saturation_mask_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_mask_from_layer_count");


	// DIAGNOSTIC ONLY - fills the pixels the layer cap cut short out to full coverage, see OpenGLEngine::fillCappedSplatPixels().
	// Same full-viewport quad vertex shader as the passes above.
	cap_fill_prog = new OpenGLProgram(
		"gaussian splat cap fill prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_cap_fill_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(cap_fill_prog);
	cap_fill_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int, "splat_mask_block_size"); // Read back by setCapFillUniforms(), for the same reason the uniforms above are registered rather than looked up here.


	// Halves the mask by minimum, once per level, so that the vertex shader can ask about a whole quad's worth of screen
	// with one fetch - see gaussian_splat_mask_reduce_frag_shader.glsl.  Same full-viewport quad vertex shader again.
	mask_reduce_prog = new OpenGLProgram(
		"gaussian splat mask reduce prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_mask_reduce_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(mask_reduce_prog);


	// SESSION067 - reduces the scene's depth buffer onto the (smaller) accumulation buffer's depth attachment, taking the
	// farthest sample of each footprint - see getDepthDownsampleProgram() for why this is a pass and not a blit.  Same
	// full-viewport quad vertex shader again.  Built unconditionally alongside the rest, but only run at scales below 1.
	depth_downsample_prog = new OpenGLProgram(
		"gaussian splat depth downsample prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_depth_downsample_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(depth_downsample_prog);

	// Same reason as the programs above: the build may still be in flight here, so the locations are resolved by
	// appendUserUniformInfo() once it completes.  Read back by setDepthDownsampleUniforms(), in this order.
	depth_downsample_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "splat_depth_src_dims_px");
	depth_downsample_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2, "splat_depth_dst_dims_px");


	// SESSION069 - TAA accumulate: running-mean blend of this frame's resolved splat layer into the history texture.
	// Full-viewport quad, shares the same vertex shader as the resolve.
	taa_accumulate_prog = new OpenGLProgram(
		"gaussian splat taa accumulate prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_taa_accum_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(taa_accumulate_prog);

	// Order of appendUserUniformInfo is the contract with the getters above - do not reorder.
	taa_accumulate_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "current_texture");
	taa_accumulate_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "history_texture");
	taa_accumulate_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "taa_weight");


	// SESSION069 - TAA composite: draws the freshly accumulated history onto the frame with (GL_ONE, GL_ONE_MINUS_SRC_ALPHA),
	// exactly as the pre-TAA resolve did directly.
	taa_composite_prog = new OpenGLProgram(
		"gaussian splat taa composite prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_taa_composite_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(taa_composite_prog);

	taa_composite_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int, "history_texture");


	// SESSION070 - depth-only draw of the composite's quad, for SplatDoFDepthMode_Weighted under TAA - see
	// getTAADepthWritebackProgram()'s own comment.
	taa_depth_writeback_prog = new OpenGLProgram(
		"gaussian splat taa depth writeback prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_taa_depth_writeback_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(taa_depth_writeback_prog);

	taa_depth_writeback_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Int,   "splat_dof_depth_texture");
	taa_depth_writeback_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Vec2,  "splat_resolve_dims_px");
	taa_depth_writeback_prog->appendUserUniformInfo(UserUniformInfo::UniformType_Float, "splat_dof_near_clip_dist");


	// Halves the coverage pyramid by mean instead of minimum, once per level - see getCoverageShrinkStrength() and
	// gaussian_splat_mask_reduce_mean_frag_shader.glsl. Same full-viewport quad vertex shader again.
	mean_reduce_prog = new OpenGLProgram(
		"gaussian splat mean mask reduce prog",
		new OpenGLShader(shader_dir + "/gaussian_splat_resolve_vert_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_VERTEX_SHADER),
		new OpenGLShader(shader_dir + "/gaussian_splat_mask_reduce_mean_frag_shader.glsl", version_directive, key_defs + preprocessor_defines, GL_FRAGMENT_SHADER),
		opengl_engine->getAndIncrNextProgramIndex(),
		/*wait_for_build_to_complete=*/!opengl_engine->parallel_shader_compile_support
	);
	opengl_engine->addProgram(mean_reduce_prog);
}


int GaussianSplatRenderer::getCapFillMaskTexUniformLoc()
{
	if(cap_fill_mask_tex_uniform_loc == -2)
	{
		assert(cap_fill_prog.nonNull() && cap_fill_prog->isBuilt());
		cap_fill_mask_tex_uniform_loc = cap_fill_prog->getUniformLocation("splat_saturation_mask_texture");
	}
	return cap_fill_mask_tex_uniform_loc;
}


void GaussianSplatRenderer::setCapFillUniforms(int block_size) const
{
	glUniform1i(cap_fill_prog->user_uniform_info[0].loc, block_size);
}


int GaussianSplatRenderer::getSplatMaskTexUniformLoc()
{
	if(splat_mask_tex_uniform_loc == -2)
	{
		assert(shader_prog.nonNull() && shader_prog->isBuilt()); // getUniformLocation() on a program still being linked would return -1 and be cached as such.
		splat_mask_tex_uniform_loc = shader_prog->getUniformLocation("splat_saturation_mask_texture");
	}
	return splat_mask_tex_uniform_loc;
}


int GaussianSplatRenderer::getCoverageMaskTexUniformLoc()
{
	if(coverage_mask_tex_uniform_loc == -2)
	{
		assert(shader_prog.nonNull() && shader_prog->isBuilt());
		coverage_mask_tex_uniform_loc = shader_prog->getUniformLocation("splat_coverage_mask_texture");
	}
	return coverage_mask_tex_uniform_loc;
}


int GaussianSplatRenderer::getHideCountTexUniformLoc() // SESSION066 - mirrors getSplatMaskTexUniformLoc().
{
	if(splat_hide_count_tex_uniform_loc == -2)
	{
		assert(shader_prog.nonNull() && shader_prog->isBuilt());
		splat_hide_count_tex_uniform_loc = shader_prog->getUniformLocation("splat_hide_count_texture");
	}
	return splat_hide_count_tex_uniform_loc;
}


void GaussianSplatRenderer::setHideCountCull(bool active, float threshold) // SESSION066 - the Clip per-splat cull, indices 23/24.
{
	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->ob->materials[0].user_uniform_vals[23].intval   = active ? 1 : 0;
		clouds[i]->ob->materials[0].user_uniform_vals[24].floatval = threshold;
	}
}


void GaussianSplatRenderer::setSplatMaskBlockSize(int block_size, int max_level, bool centre_test)
{
	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->ob->materials[0].user_uniform_vals[7].intval = block_size;
		clouds[i]->ob->materials[0].user_uniform_vals[8].intval = max_level;
		clouds[i]->ob->materials[0].user_uniform_vals[11].intval = centre_test ? 1 : 0;
	}
}


bool GaussianSplatRenderer::hideOverdrawMaskNeedsRebuild(const Matrix4f& view_matrix, int viewport_w, int viewport_h, int mask_block, uint64 num_splats_drawn) const
{
	if(!splat_hide_overdraw_mask_valid)
		return true;
	if(viewport_w != splat_hide_overdraw_mask_viewport_w || viewport_h != splat_hide_overdraw_mask_viewport_h ||
		mask_block != splat_hide_overdraw_mask_block || splat_overdraw_range_max != splat_hide_overdraw_mask_threshold ||
		getHideMode() != splat_hide_overdraw_mask_mode || num_splats_drawn != splat_hide_overdraw_mask_splats_drawn)
		return true;

	// Compared with a tolerance rather than exactly: the camera transform is rebuilt from the player's state every frame,
	// and a standing player can still jitter it in the last bits.  Exact comparison would rebuild every frame and give
	// back the extra pass this exists to avoid, silently.
	const float tolerance = 1.0e-5f;
	for(int i=0; i<4; ++i)
	{
		const Vec4f a = view_matrix.getColumn(i), b = splat_hide_overdraw_mask_view.getColumn(i);
		for(int c=0; c<4; ++c)
			if(std::fabs(a[c] - b[c]) > tolerance)
				return true;
	}
	return false;
}


void GaussianSplatRenderer::noteHideOverdrawMaskBuilt(const Matrix4f& view_matrix, int viewport_w, int viewport_h, int mask_block, uint64 num_splats_drawn)
{
	splat_hide_overdraw_mask_valid = true;
	splat_hide_overdraw_mask_view = view_matrix;
	splat_hide_overdraw_mask_viewport_w = viewport_w;
	splat_hide_overdraw_mask_viewport_h = viewport_h;
	splat_hide_overdraw_mask_block = mask_block;
	splat_hide_overdraw_mask_threshold = splat_overdraw_range_max;
	splat_hide_overdraw_mask_mode = getHideMode();
	splat_hide_overdraw_mask_splats_drawn = num_splats_drawn;
	splat_hide_overdraw_mask_rebuilds++;
}


void GaussianSplatRenderer::setSplatFragMaskBlockSize(int block_size)
{
	for(size_t i=0; i<clouds.size(); ++i)
		clouds[i]->ob->materials[0].user_uniform_vals[13].intval = block_size;
}


void GaussianSplatRenderer::setSaturationMaskUniforms(int block_size, bool from_layer_count, bool layer_count_threshold, bool coverage_cap_threshold) const
{
	// One threshold uniform serves both modes, since only one of them is ever running: coverage for the gate, a layer
	// count for the hide-overdraw diagnostic.  Reusing getOverdrawRangeMax() as that layer count is deliberate - it is
	// the value the overdraw ramp paints solid red at, so what the diagnostic removes is exactly what was red on screen.
	glUniform1f(saturation_mask_prog->user_uniform_info[0].loc, from_layer_count ? (layer_count_threshold ? (float)splat_layer_cap : splat_overdraw_range_max) :
		(coverage_cap_threshold ? splat_coverage_cap : splat_saturation_threshold));
	glUniform1i(saturation_mask_prog->user_uniform_info[1].loc, block_size);
	glUniform1i(saturation_mask_prog->user_uniform_info[2].loc, from_layer_count ? 1 : 0);
}


void GaussianSplatRenderer::setResolveOverdrawUniforms(int coverage_map_block) const
{
	glUniform1i(resolve_prog->user_uniform_info[0].loc, splat_show_overdraw_mode);
	glUniform1f(resolve_prog->user_uniform_info[1].loc, splat_overdraw_range_min);
	glUniform1f(resolve_prog->user_uniform_info[2].loc, splat_overdraw_range_max);

	// DIAGNOSTIC ONLY - see getShowCoverageMapLevel(). The sampler is told which unit to read here rather than at the
	// bind, so that the texture the caller binds and the unit the shader looks at cannot drift apart.
	glUniform1i(resolve_prog->user_uniform_info[3].loc, splat_show_coverage_map_level);
	glUniform1i(resolve_prog->user_uniform_info[4].loc, coverage_map_block);
	glUniform1i(resolve_prog->user_uniform_info[5].loc, RESOLVE_COVERAGE_MAP_TEXTURE_UNIT_INDEX);
}


int GaussianSplatRenderer::getResolveCoverageMapTexUniformLoc() const
{
	// The six entries are appended together in buildShadersIfNeeded(); a shorter list means the program was never built.
	return (resolve_prog.nonNull() && (resolve_prog->user_uniform_info.size() >= 6)) ? resolve_prog->user_uniform_info[5].loc : -1;
}


void GaussianSplatRenderer::setResolveDoFDepthUniforms(bool write_weighted_depth, float near_clip_dist) const
{
	glUniform1i(resolve_prog->user_uniform_info[6].loc, write_weighted_depth ? 1 : 0);
	glUniform1f(resolve_prog->user_uniform_info[7].loc, near_clip_dist);
	glUniform1i(resolve_prog->user_uniform_info[8].loc, RESOLVE_DOF_DEPTH_TEXTURE_UNIT_INDEX);
}


int GaussianSplatRenderer::getResolveDoFDepthTexUniformLoc() const
{
	return (resolve_prog.nonNull() && (resolve_prog->user_uniform_info.size() >= 9)) ? resolve_prog->user_uniform_info[8].loc : -1;
}


// SESSION067 - see the declaration.  The upsample mode is decided here rather than in the shader from the two sizes,
// because "the buffer is full resolution" has to mean bit-for-bit the old path: at equal sizes the shader takes its
// texelFetch branch, so a frame at scale 1 is not merely visually but numerically what it was before this existed.
void GaussianSplatRenderer::setResolveUpsampleUniforms(const Vec2i& accum_dims, const Vec2i& viewport_dims) const
{
	const bool downscaled = (accum_dims.x != viewport_dims.x) || (accum_dims.y != viewport_dims.y);

	glUniform1i(resolve_prog->user_uniform_info[9].loc, downscaled ? (splat_accum_upsample_bilinear ? 2 : 1) : 0); // 0 = one-to-one, 1 = nearest, 2 = bilinear - see the shader.
	glUniform2f(resolve_prog->user_uniform_info[10].loc, (float)accum_dims.x, (float)accum_dims.y);
	glUniform2f(resolve_prog->user_uniform_info[11].loc, (float)viewport_dims.x, (float)viewport_dims.y);
}


// SESSION068 - see setResolveEnhanceUniforms() in the header.  Both enable flags are forced to 0 at scale 1 so the
// default kadr stays bit-for-bit identical to the pre-enhancer path, whatever is checked in the UI.  gain/sharpness
// still ride along - harmless with the flags off, and matches how the upsample setter above passes its dimensions
// unconditionally.
void GaussianSplatRenderer::setResolveEnhanceUniforms(const Vec2i& accum_dims, const Vec2i& viewport_dims) const
{
	const bool downscaled = (accum_dims.x != viewport_dims.x) || (accum_dims.y != viewport_dims.y);

	glUniform1i(resolve_prog->user_uniform_info[12].loc, (downscaled && splat_deconv_enabled) ? 1 : 0);
	glUniform1f(resolve_prog->user_uniform_info[13].loc, splat_deconv_gain);
	glUniform1i(resolve_prog->user_uniform_info[14].loc, (downscaled && splat_rcas_enabled) ? 1 : 0);
	glUniform1f(resolve_prog->user_uniform_info[15].loc, splat_rcas_sharpness);
}


// SESSION067 - see getDepthDownsampleProgram().  Both sizes are passed for the same reason the resolve's are: the
// destination is a rounded scaling of the source, so the ratio has to be formed from the sizes actually allocated.
void GaussianSplatRenderer::setDepthDownsampleUniforms(const Vec2i& src_dims, const Vec2i& dst_dims) const
{
	glUniform2f(depth_downsample_prog->user_uniform_info[0].loc, (float)src_dims.x, (float)src_dims.y);
	glUniform2f(depth_downsample_prog->user_uniform_info[1].loc, (float)dst_dims.x, (float)dst_dims.y);
}


//========================================================================================================
// SESSION069 - TAA (temporal accumulation with subpixel jitter).  See getTAAEnabled().
//========================================================================================================

// Halton (2, 3) - a low-discrepancy sequence used to pick subpixel offsets in TAA.  Preferred over a regular grid because
// it works for fractional buffer scales (any k, not only powers of two) and degrades gracefully when fewer than
// numJitterSamples() frames have accumulated - the picture converges rather than snapping into place at frame N.
static float halton(int index, int base)
{
	float f = 1.f, r = 0.f;
	int i = index + 1; // Halton is defined from 1; index 0 would give zero shift, i.e. a wasted "no jitter" frame in every reset cycle.
	while(i > 0) { f /= (float)base; r += (float)(i % base) * f; i /= base; }
	return r;
}


// FNV-1a 64-bit over a raw byte range.  Used only to detect whether ANY splat setting affecting the picture has changed
// since the last frame - the actual values are meaningless.  A hash rather than an explicit compare list so it can't
// silently fall behind next time a knob is added; the trade-off is the theoretical possibility of a collision leaving a
// changed setting undetected, which for our two-word-per-setting inputs is negligible in practice.
static uint64 fnv1a64(const void* data, size_t len)
{
	const unsigned char* p = static_cast<const unsigned char*>(data);
	uint64 h = 0xcbf29ce484222325ULL;
	for(size_t i = 0; i < len; ++i) { h ^= (uint64)p[i]; h *= 0x100000001b3ULL; }
	return h;
}


int GaussianSplatRenderer::numJitterSamples() const
{
	// Enough jitter positions to cover the frame-pixel grid at accum scale s: each frame pixel row/column falls into
	// ceil(1/s) accum texels, and the 2D combination is that squared.  More than that adds latency to full quality with
	// no gain; fewer leaves subpixel gaps that will never fill in.  Clamped to a hard cap of 16 in case s dips very low
	// (0.1) and the sequence length would otherwise be 100 - beyond ~16 the eye doesn't tell the difference between
	// "still resolving" and "resolved" anyway, and the diminishing 1/N weight makes newer samples imperceptible.
	const float s = getEffectiveAccumBufferScale();
	const int n = myMax(1, (int)std::ceil(1.f / s));
	return myClamp(n * n, 1, 16);
}


bool GaussianSplatRenderer::isTAAActiveForResolve() const
{
	// Off if the setting is off.  Off if the accumulation buffer is 1:1 with the frame - there is no sub-frame-pixel
	// information to reconstruct at that scale, and the pre-TAA path is bit-for-bit correct there.
	//
	// (Session069) An earlier draft of this method also required !SplatDoFDepthMode_Weighted, on the theory that the
	// resolve's gl_FragDepth write would need the frame's depth attachment - but that turned out to gate TAA off in
	// practice, because DoF weighted mode is persisted and the owner had it on.  Since the resolve in TAA mode writes
	// to a colour-only framebuffer, gl_FragDepth simply has nowhere to land there and is dropped by the driver (per the
	// GL spec: depth writes to a framebuffer without a depth attachment are no-ops, and depth test with no depth buffer
	// always passes) - harmless, since it isn't the only place the weighted depth gets written any more.  (Session070)
	// OpenGLEngine::resolveSplatAccumBuffer() now routes it through a dedicated depth-only draw of the composite's
	// quad instead - see GaussianSplatRenderer::getTAADepthWritebackProgram()'s comment - so DoF still receives
	// per-splat depth correctly while TAA is on.
	if(!splat_taa_enabled) return false;
	if(!opengl_engine) return false;
	const Vec2i viewport_dims = opengl_engine->getViewportDims();
	const Vec2i accum_dims = accumBufferDimsForViewport(viewport_dims);
	if(accum_dims == viewport_dims) return false;
	return true;
}


// Called from think() once per frame, right before the per-cloud loop writes user_uniform_vals[27].  Decides whether
// the accumulation window has to be reset, then picks the jitter offset the vertex shader will use this frame.  The
// counter and jitter are both zeroed when TAA is inactive so any resume-with-scale-1 frame lands the vertex shader's
// shift at (0, 0).
void GaussianSplatRenderer::updateTAAState(const Vec2i& accum_dims)
{
	const bool active = isTAAActiveForResolve();

	if(!active)
	{
		splat_taa_current_jitter_px.set(0.f, 0.f);
		splat_taa_frame_count = 0;
		// last_* memory left alone: the next active frame will reset by comparing to whatever was current at that
		// time, which is exactly the semantics we want across an on/off toggle.
		return;
	}

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Matrix4f cur_view = scene ? scene->last_view_matrix : Matrix4f::identity();
	const Vec2i viewport_dims = opengl_engine->getViewportDims();

	// Settings hash - EVERY splat setting that affects the picture goes in, plus the buffer scale itself (a scale change
	// is already caught by accum_dims below, but included here to make the hash self-contained for future callers).
	// New knobs added to the class must be appended here or the accumulation will smear across their changes.
	struct SettingsBlob {
		float alpha_cutoff, quad_radius_scale, area_scale_gamma, area_scale_ref_px, coverage_shrink_strength, near_fade_width, near_epsilon;
		float size_clamp_min, size_clamp_max, dist_clamp_min, dist_clamp_max, alpha_gain, alpha_gamma;
		float slice_growth, saturation_threshold, area_slice_px;
		float accum_buffer_scale, deconv_gain, rcas_sharpness;
		int   ablation_stage, coverage_shrink_mode, coverage_reduce_mode, show_overdraw_mode, dof_depth_mode;
		int   size_clamp_invert, dist_clamp_invert, area_slice_mode, ewa_fix_enabled, saturation_gate_enabled;
		int   accum_buffer_8bit, accum_upsample_bilinear, deconv_enabled, rcas_enabled;
		int   layer_cap, layer_cap_opaque, saturation_mask_downscale;
	} blob = {};
	blob.alpha_cutoff = splat_alpha_cutoff; blob.quad_radius_scale = splat_quad_radius_scale;
	blob.area_scale_gamma = splat_area_scale_gamma; blob.area_scale_ref_px = splat_area_scale_ref_px;
	blob.coverage_shrink_strength = splat_coverage_shrink_strength; blob.near_fade_width = splat_near_fade_width;
	blob.near_epsilon = splat_near_epsilon;
	blob.size_clamp_min = splat_size_clamp_min; blob.size_clamp_max = splat_size_clamp_max;
	blob.dist_clamp_min = splat_dist_clamp_min; blob.dist_clamp_max = splat_dist_clamp_max;
	blob.alpha_gain = splat_alpha_gain; blob.alpha_gamma = splat_alpha_gamma;
	blob.slice_growth = splat_slice_growth; blob.saturation_threshold = splat_saturation_threshold;
	blob.area_slice_px = splat_area_slice_px;
	blob.accum_buffer_scale = splat_accum_buffer_scale;
	blob.deconv_gain = splat_deconv_gain; blob.rcas_sharpness = splat_rcas_sharpness;
	blob.ablation_stage = splat_ablation_stage;
	blob.coverage_shrink_mode = splat_coverage_shrink_mode; blob.coverage_reduce_mode = splat_coverage_reduce_mode;
	blob.show_overdraw_mode = splat_show_overdraw_mode; blob.dof_depth_mode = splat_dof_depth_mode;
	blob.size_clamp_invert = splat_size_clamp_invert ? 1 : 0; blob.dist_clamp_invert = splat_dist_clamp_invert ? 1 : 0;
	blob.area_slice_mode = splat_area_slice_mode; blob.ewa_fix_enabled = splat_ewa_fix_enabled ? 1 : 0;
	blob.saturation_gate_enabled = splat_saturation_gate_enabled ? 1 : 0;
	blob.accum_buffer_8bit = splat_accum_buffer_8bit ? 1 : 0; blob.accum_upsample_bilinear = splat_accum_upsample_bilinear ? 1 : 0;
	blob.deconv_enabled = splat_deconv_enabled ? 1 : 0; blob.rcas_enabled = splat_rcas_enabled ? 1 : 0;
	blob.layer_cap = splat_layer_cap; blob.layer_cap_opaque = splat_layer_cap_opaque ? 1 : 0;
	blob.saturation_mask_downscale = splat_saturation_mask_downscale;
	const uint64 settings_hash = fnv1a64(&blob, sizeof(blob));

	// Any single one of these being different from last frame invalidates the accumulation.  Bitwise view compare is
	// deliberate: half a pixel of camera dither is exactly what would show up as a smear if it were tolerated, and a
	// spurious reset costs nothing (4 frames back to full quality).
	const bool view_changed     = !(cur_view == splat_taa_last_view_matrix);
	const bool viewport_changed = (viewport_dims != splat_taa_last_viewport_dims);
	const bool accum_changed    = (accum_dims != splat_taa_last_accum_dims);
	const bool settings_changed = (settings_hash != splat_taa_last_settings_hash);
	const bool clouds_changed   = (clouds.size() != splat_taa_last_num_clouds);
	const bool content_applied  = (splat_taa_content_apply_counter != splat_taa_last_content_apply_snapshot);

	const bool reset = view_changed || viewport_changed || accum_changed || settings_changed || clouds_changed || content_applied;

	if(reset)
	{
		splat_taa_frame_count = 0;
		// splat_taa_write_index is NOT reset - keeping it ping-ponging avoids a case where the composite pass reads
		// history[write] on the very first frame after a reset (weight = 1, so the read is discarded anyway, but we
		// still need a valid texture bound to satisfy WebGL's sampler validation).
	}
	else
	{
		splat_taa_frame_count = myMin(splat_taa_frame_count + 1, numJitterSamples() - 1);
		// After N frames the weight 1/(N+1) is small enough that new frames barely register; capping the counter keeps
		// it stable and avoids stagnation when the sequence has repeated all its offsets.
	}

	// Pick this frame's jitter offset from the Halton sequence.  Indexed by the accumulated frame count (not the
	// monotonic index) so that a reset restarts the sequence from 0 and every accumulation window sees the same,
	// well-distributed set of offsets.  In [-0.5, +0.5) accum-buffer pixels.
	const int k = splat_taa_frame_count;
	splat_taa_current_jitter_px.set(halton(k, 2) - 0.5f, halton(k, 3) - 0.5f);

	// Flip the write index for next frame's accumulate pass - the freshly written history becomes next frame's read.
	splat_taa_write_index = 1 - splat_taa_write_index;

	// Snapshot for next frame's compares.
	splat_taa_last_view_matrix = cur_view;
	splat_taa_last_viewport_dims = viewport_dims;
	splat_taa_last_accum_dims = accum_dims;
	splat_taa_last_settings_hash = settings_hash;
	splat_taa_last_num_clouds = clouds.size();
	splat_taa_last_content_apply_snapshot = splat_taa_content_apply_counter;

	splat_taa_frame_index_monotonic++;
}


void GaussianSplatRenderer::setResolveTAAActiveUniform(int taa_active) const
{
	glUniform1i(resolve_prog->user_uniform_info[16].loc, taa_active);
}


void GaussianSplatRenderer::setResolveTAAJitterUniform(const Vec2f& jitter_accum_px) const
{
	glUniform2f(resolve_prog->user_uniform_info[17].loc, jitter_accum_px.x, jitter_accum_px.y);
}


int GaussianSplatRenderer::getTAADepthWritebackDepthTexUniformLoc() const
{
	return taa_depth_writeback_prog.nonNull() ? taa_depth_writeback_prog->user_uniform_info[0].loc : -1;
}


void GaussianSplatRenderer::setTAADepthWritebackUniforms(const Vec2i& resolve_dims, float near_clip_dist) const
{
	glUniform2f(taa_depth_writeback_prog->user_uniform_info[1].loc, (float)resolve_dims.x, (float)resolve_dims.y);
	glUniform1f(taa_depth_writeback_prog->user_uniform_info[2].loc, near_clip_dist);
}


void GaussianSplatRenderer::setResolveLowPassVarianceUniform(float variance) const
{
	glUniform1f(resolve_prog->user_uniform_info[18].loc, variance);
}


void GaussianSplatRenderer::setTAAAccumulateWeight(float weight) const
{
	// Only the weight is set per-frame; sampler bindings are handled by resolveSplatAccumBuffer() via the sampler-loc
	// getters exposed above.  Index 2 in the same order as the appendUserUniformInfo() list below (0 = current, 1 = history).
	if(taa_accumulate_prog.nonNull())
		glUniform1f(taa_accumulate_prog->user_uniform_info[2].loc, weight);
}


int GaussianSplatRenderer::getTAAAccumulateCurrentTexUniformLoc() const
{
	return taa_accumulate_prog.nonNull() ? taa_accumulate_prog->user_uniform_info[0].loc : -1;
}


int GaussianSplatRenderer::getTAAAccumulateHistoryTexUniformLoc() const
{
	return taa_accumulate_prog.nonNull() ? taa_accumulate_prog->user_uniform_info[1].loc : -1;
}


int GaussianSplatRenderer::getTAACompositeHistoryTexUniformLoc() const
{
	return taa_composite_prog.nonNull() ? taa_composite_prog->user_uniform_info[0].loc : -1;
}


size_t GaussianSplatRenderer::maxSplatsPerCloud() const
{
	// The texture width is fixed at splat_tex_width, which any conformant implementation supports.  The height is capped
	// at the real driver limit rather than the guaranteed minimum, since real hardware commonly supports far more.
	const size_t max_tex_h = (size_t)myMax(1, opengl_engine->max_texture_size);
	return (splat_tex_width * max_tex_h) / texels_per_splat;
}


size_t GaussianSplatRenderer::numSplatsInWorld() const
{
	size_t num = 0;
	for(size_t i=0; i<clouds.size(); ++i)
		num += clouds[i]->total_splats;
	return num;
}


size_t GaussianSplatRenderer::numObjectsInWorld() const
{
	return handle_to_cloud.size();
}


void GaussianSplatRenderer::forceTraversalRefresh()
{
	for(size_t i=0; i<clouds.size(); ++i)
		clouds[i]->have_last_traversal_cam_pos = false; // Makes kickOffTraversals() treat every cloud as unconditionally overdue, the same way a cloud that has never been traversed is - see its ratio computation.
}


// Point-in-frustum test, same accept/reject convention as OpenGLEngine.cpp's AABBIntersectsFrustum() (outside if
// dot(normal, point) >= plane.getD() for any plane), just for one point rather than an AABB's 8 corners.
static inline bool pointInFrustum(const Planef* frustum_clip_planes, int num_frustum_clip_planes, const Vec4f& pos_ws)
{
	for(int i=0; i<num_frustum_clip_planes; ++i)
		if(dot(frustum_clip_planes[i].getNormal(), pos_ws) >= frustum_clip_planes[i].getD())
			return false;
	return true;
}


// SESSION063: the per-orientation frustum filter of the split architecture (session062 §9.3). Streams the SoA unculled
// frontier U(P) - one globally distance-sorted list of fine + coarse nodes (K4) - and keeps each node whose centre is
// inside every plane once that plane is pushed out by three margins, mirroring the base_margin+translation+rotation
// dilation the old cull-traversal used (see the cull block in GaussianSplatLodTraversalTask::run()), computed for the
// filter's own (much shorter) latency in kickOffFilters():
//   - radius                     : the node's own cull_radius (its 3-sigma footprint), always applied.
//   - trans_dilation[i]          : per-plane world-space margin (metres) for camera translation during the latency window.
//   - rotational margin          : SESSION072 - was isotropic (rate*dist, same on all 6 planes); now max() of two
//                                   per-plane terms - see below.
// SESSION072: anisotropic rotational dilation (session067 §8/§15 plan A). The isotropic rate*dist margin assumed the
// worst case (a node exactly perpendicular to the rotation axis) on EVERY plane at once, which is what let a violent
// flick multiply the whole draw list regardless of which edge it actually threatened. Replaced by max() of:
//   - rate_*_baseline * dist : isotropic FLOOR only now - covers the static->moving transition, where nothing has
//                              rotated yet in any direction, so there is no axis to be anisotropic about.
//   - dot(d, cn[pl]) * mag   : the measured term. The camera's rotation over the latency window is an axis-aligned
//                              swept-angle vector: direction rotation_axis (unit), magnitude swept_fine/swept_coarse
//                              (the angle swept during the fine/coarse latency window). The tangential displacement a
//                              node at offset d = node_pos - cam_pos picks up from that rotation is swept x d; the
//                              component that threatens plane i is dot(n_i, swept x d). Precomputing cn_i = n_i x axis
//                              ONCE per plane (below, outside the node loop) turns this per-node-per-plane into a plain
//                              dot(d, cn_i) scaled by the swept magnitude - the scalar triple product identity
//                              n.(a x b) == b.(n x a). By Cauchy-Schwarz |dot(d,cn_i)*mag| <= mag*|d| = the old
//                              isotropic pad, with equality only when d is exactly perpendicular to the rotation axis -
//                              so this can only shrink the margin relative to the old code, never grow it.
//
// SESSION072 measured cost (owner's A/B, [gsr-filter-drain] compute= vs iso=): the first cut of this took ~2.5x the
// isotropic pass (~150ms vs ~60ms on a 15.5M-node U(P)) because it did TWO dot products per plane per node - one for
// the fine swept vector, one for the coarse-minus-fine difference. Only one is needed: the fine and coarse swept
// vectors share the rotation axis and differ only in magnitude (both are axis * w_eff * their own latency), and the
// cross product is linear in its second argument, so n x swept_coarse is just a scalar multiple of n x swept_fine.
// Passing the unit axis plus the two magnitudes separately (rather than two pre-scaled vectors) makes that explicit:
// one cross product per plane, one dot product per plane per node, and the per-node fine/coarse magnitude blend hoists
// entirely OUT of the plane loop. Also handles a zero fine latency cleanly, which a ratio between the two would not.
//
// SESSION063 K4: fine vs coarse is still chosen PER NODE from uf.is_coarse - coarse-floor nodes get the wider _coarse
// terms so their big cheap splats catch motion-revealed edges, while the dense fine set stays tight. Because fine and
// coarse are one globally sorted list, survivors come out globally front-to-back (a near coarse node correctly occludes a
// far fine one - the fix for the green bleed of the old draw-coarse-last approach). coarse_only_debug keeps only coarse
// nodes, for the isolation view. Writes survivor indices in input order (no re-sort); shrinks out_indices to the count.
// SESSION080 §4.3: a frontier is now two segments - its own arrays followed by its far block's - so the streaming pass
// below runs once per segment, appending into the same output. Split out of filterUnculledFrontier(), which became the
// two-call wrapper; the body is unchanged from the single-segment version it replaces.
static size_t filterFrontierSegment(const GaussianSplatUnculledFrontier& uf, const Planef* planes, int num_planes,
	const Vec4f& cam_pos_ws, float rate_fine_baseline, float rate_coarse_baseline,
	const Vec4f& rotation_axis, float swept_fine, float swept_coarse, // SESSION072: unit rotation axis + the angle swept during each layer's latency window - see above.
	const float* trans_dilation, bool coarse_only_debug,
	bool draw_coarse_layer, // SESSION075: whether coarse-floor nodes may survive at all - see kickOffFilters()'s call site. Independent of whether U(P) captured them (see kickOffTraversals()'s coarse_capture_needed): captured-but-not-drawn is now a valid state, for saturation-only measurement.
	uint32* const out, // SESSION080 §4.3: caller-owned, pre-sized for both segments - see filterUnculledFrontier().
	size_t* out_num_coarse) // SESSION072 DIAGNOSTIC: accumulates how many of the survivors were coarse-floor nodes - see [gsr-filter-drain].
{
	// SESSION074: no saturation work happens here any more. The verdict is orientation-independent, so it is applied
	// once when the frontier is built and this function simply streams whatever survived - see
	// GaussianSplatUnculledFrontier::sat_num_occluders's block comment for why that move mattered.
	size_t num_coarse_out = 0;
	const size_t n = uf.indices.size();
	const int npl = myMin(num_planes, 6);

	__m128 pl_nx[6], pl_ny[6], pl_nz[6], pl_d[6], pl_d_raw[6];
	__m128 cn_x[6], cn_y[6], cn_z[6]; // SESSION072: SoA broadcasts of cn[pl] below, for the SIMD node loop.
	Vec4f cn[6];                      // Same values, kept as plain Vec4f for the scalar tail loop.
	for(int pl=0; pl<npl; ++pl)
	{
		const Vec4f& nrm = planes[pl].getNormal();
		pl_nx[pl] = _mm_set1_ps(nrm.x[0]); pl_ny[pl] = _mm_set1_ps(nrm.x[1]); pl_nz[pl] = _mm_set1_ps(nrm.x[2]);
		pl_d[pl]     = _mm_set1_ps(planes[pl].getD() + (trans_dilation ? trans_dilation[pl] : 0.f)); // Dilated plane (translation dilation folds into d).
		pl_d_raw[pl] = _mm_set1_ps(planes[pl].getD()); // SESSION063 K4: the tight (undilated) plane, for confining the coarse floor to the dilation band - see below.

		cn[pl] = crossProduct(nrm, rotation_axis); // Unit axis, so this carries direction only - the swept magnitude is applied per node below.
		cn_x[pl] = _mm_set1_ps(cn[pl].x[0]); cn_y[pl] = _mm_set1_ps(cn[pl].x[1]); cn_z[pl] = _mm_set1_ps(cn[pl].x[2]);
	}
	const __m128 swept_fine_v = _mm_set1_ps(swept_fine);
	const __m128 swept_diff_v = _mm_set1_ps(swept_coarse - swept_fine); // mag = swept_fine + is_coarse * (swept_coarse - swept_fine).
	const __m128 cam_x = _mm_set1_ps(cam_pos_ws.x[0]);
	const __m128 cam_y = _mm_set1_ps(cam_pos_ws.x[1]);
	const __m128 cam_z = _mm_set1_ps(cam_pos_ws.x[2]);
	const __m128 rate_fine_baseline_v = _mm_set1_ps(rate_fine_baseline);
	const __m128 rate_baseline_diff_v = _mm_set1_ps(rate_coarse_baseline - rate_fine_baseline); // baseline_rate = rate_fine_baseline + is_coarse * (rate_coarse_baseline - rate_fine_baseline).

	const float* const px = uf.px.data();
	const float* const py = uf.py.data();
	const float* const pz = uf.pz.data();
	const float* const rad = uf.radius.data();
	const float* const isc = uf.is_coarse.data();
	const uint32* const idx = uf.indices.data();
	size_t num_out = 0;

	const size_t n4 = n & ~size_t(3);
	for(size_t i=0; i<n4; i+=4)
	{
		const __m128 X = _mm_loadu_ps(px + i);
		const __m128 Y = _mm_loadu_ps(py + i);
		const __m128 Z = _mm_loadu_ps(pz + i);
		const __m128 R = _mm_loadu_ps(rad + i);
		const __m128 IC = _mm_loadu_ps(isc + i); // 1.0 for coarse nodes, 0.0 for fine.
		const __m128 baseline_rate = _mm_add_ps(rate_fine_baseline_v, _mm_mul_ps(IC, rate_baseline_diff_v));
		const __m128 dx = _mm_sub_ps(X, cam_x), dy = _mm_sub_ps(Y, cam_y), dz = _mm_sub_ps(Z, cam_z);
		const __m128 dist = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), _mm_mul_ps(dz, dz)));
		const __m128 baseline_pad = _mm_mul_ps(baseline_rate, dist); // SESSION072: isotropic floor only now - see the function comment.
		// SESSION072: the swept magnitude this node's layer uses. Hoisted out of the plane loop - it depends only on
		// is_coarse, not on which plane, which is exactly what makes the single-cross-product form above worth having.
		const __m128 swept_mag = _mm_add_ps(swept_fine_v, _mm_mul_ps(IC, swept_diff_v));
		__m128 outside = _mm_setzero_ps();      // Outside the DILATED frustum (the keep test).
		__m128 outside_tight = _mm_setzero_ps(); // Outside the TIGHT (undilated, radius-only) frustum - for the coarse band test.
		for(int pl=0; pl<npl; ++pl)
		{
			const __m128 dotv = _mm_add_ps(_mm_add_ps(_mm_mul_ps(X, pl_nx[pl]), _mm_mul_ps(Y, pl_ny[pl])), _mm_mul_ps(Z, pl_nz[pl]));
			// SESSION072: anisotropic rotational pad for this plane - see the function comment. One dot product against
			// the axis-only cn[pl], scaled by the per-node swept magnitude computed above.
			const __m128 measured_dir = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, cn_x[pl]), _mm_mul_ps(dy, cn_y[pl])), _mm_mul_ps(dz, cn_z[pl]));
			const __m128 measured = _mm_mul_ps(measured_dir, swept_mag);
			const __m128 sub = _mm_add_ps(R, _mm_max_ps(baseline_pad, measured)); // max(), not sum - see the function comment.
			outside       = _mm_or_ps(outside,       _mm_cmpge_ps(_mm_sub_ps(dotv, sub), pl_d[pl]));     // dot - (radius + pad) >= d + trans.
			outside_tight = _mm_or_ps(outside_tight, _mm_cmpge_ps(_mm_sub_ps(dotv, R),   pl_d_raw[pl])); // dot - radius >= d.
		}
		if(coarse_only_debug)
			outside = _mm_or_ps(outside, _mm_cmpeq_ps(IC, _mm_setzero_ps())); // Debug: keep only coarse nodes (show full coarse coverage, band restriction off).
		else if(!draw_coarse_layer)
			// SESSION075: coarse layer captured (for saturation) but not meant to draw - reject every coarse node
			// outright, regardless of the band test below. See this function's draw_coarse_layer parameter comment.
			outside = _mm_or_ps(outside, _mm_cmpgt_ps(IC, _mm_setzero_ps()));
		else
			// SESSION063 K4: confine the coarse floor to the dilation band. A coarse node INSIDE the tight frustum is
			// rejected - there the fine set already covers, and letting the coarse layer draw over the whole visible frame
			// slightly changed the image everywhere fine wasn't fully saturated. Coarse now survives only in the margin
			// beyond the tight frustum (off-screen until motion reveals it), which is where it's actually needed.
			outside = _mm_or_ps(outside, _mm_and_ps(_mm_cmpgt_ps(IC, _mm_setzero_ps()), _mm_cmpeq_ps(outside_tight, _mm_setzero_ps())));
		const int m = _mm_movemask_ps(outside) & 0xF; // bit j set = point i+j is outside/rejected.
		if((m & 1) == 0) { out[num_out++] = idx[i+0]; if(isc[i+0] != 0.f) ++num_coarse_out; } // SESSION072 DIAGNOSTIC counting - see out_num_coarse.
		if((m & 2) == 0) { out[num_out++] = idx[i+1]; if(isc[i+1] != 0.f) ++num_coarse_out; }
		if((m & 4) == 0) { out[num_out++] = idx[i+2]; if(isc[i+2] != 0.f) ++num_coarse_out; }
		if((m & 8) == 0) { out[num_out++] = idx[i+3]; if(isc[i+3] != 0.f) ++num_coarse_out; }
	}
	for(size_t i=n4; i<n; ++i) // Tail (n not a multiple of 4).
	{
		const bool is_coarse = isc[i] != 0.f;
		if(coarse_only_debug && !is_coarse) continue;
		const float baseline_rate = rate_fine_baseline + isc[i] * (rate_coarse_baseline - rate_fine_baseline);
		const float ddx = px[i]-cam_pos_ws.x[0], ddy = py[i]-cam_pos_ws.x[1], ddz = pz[i]-cam_pos_ws.x[2];
		const float baseline_pad = baseline_rate * std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
		const float swept_mag = swept_fine + isc[i] * (swept_coarse - swept_fine); // SESSION072: as in the SIMD loop - depends only on is_coarse, so it is hoisted out of the plane loop.
		bool inside = true, inside_tight = true;
		for(int pl=0; pl<npl; ++pl)
		{
			const Vec4f& nrm = planes[pl].getNormal();
			const float dpn = nrm.x[0]*px[i] + nrm.x[1]*py[i] + nrm.x[2]*pz[i];
			// SESSION072: same anisotropic pad as the SIMD loop above, scalar form.
			const float measured = (cn[pl].x[0]*ddx + cn[pl].x[1]*ddy + cn[pl].x[2]*ddz) * swept_mag;
			const float sub = rad[i] + myMax(baseline_pad, measured);
			const float d_dil = planes[pl].getD() + (trans_dilation ? trans_dilation[pl] : 0.f);
			if(dpn - sub    >= d_dil)              inside = false;
			if(dpn - rad[i] >= planes[pl].getD())  inside_tight = false;
			if(!inside) break;
		}
		if(!inside) continue;
		if(!coarse_only_debug && is_coarse && !draw_coarse_layer) continue; // SESSION075: captured-but-not-drawn coarse - see draw_coarse_layer's comment.
		if(!coarse_only_debug && is_coarse && inside_tight) continue; // Band restriction: coarse only survives beyond the tight frustum.
		out[num_out++] = idx[i];
		if(is_coarse) ++num_coarse_out; // SESSION072 DIAGNOSTIC counting - see out_num_coarse.
	}
	if(out_num_coarse)
		*out_num_coarse += num_coarse_out; // Accumulates: the caller runs this once per segment.
	return num_out;
}


// SESSION080 §4.3: filter a whole frontier - its own arrays, then its far block's, appending both into one draw list.
//
// The two segments are streamed back to back rather than merged, and that needs no re-sort: each is sorted, and the cut
// that separates them put the far one beyond a split distance from the near one's nodes, so the concatenation is
// front-to-back to within the same one-traversal tolerance everything on screen already carries (the selection being
// drawn was always built for where the camera was a traversal ago). Same property the single-array frontier relied on
// when its far tail was a copy rather than a reference; only the storage changed.
static size_t filterUnculledFrontier(const GaussianSplatUnculledFrontier& uf, const Planef* planes, int num_planes,
	const Vec4f& cam_pos_ws, float rate_fine_baseline, float rate_coarse_baseline,
	const Vec4f& rotation_axis, float swept_fine, float swept_coarse,
	const float* trans_dilation, bool coarse_only_debug, bool draw_coarse_layer,
	js::Vector<uint32, 16>& out_indices,
	size_t* out_num_coarse = NULL)
{
	const GaussianSplatUnculledFrontier* const far_seg = uf.far_block.ptr(); // Not `far`: that is still a macro in the Windows headers.
	const size_t total_in = uf.indices.size() + (far_seg ? far_seg->indices.size() : 0);
	out_indices.resizeNoCopy(total_in); // Worst case every node survives.
	if(out_num_coarse)
		*out_num_coarse = 0;

	size_t num_out = filterFrontierSegment(uf, planes, num_planes, cam_pos_ws, rate_fine_baseline, rate_coarse_baseline,
		rotation_axis, swept_fine, swept_coarse, trans_dilation, coarse_only_debug, draw_coarse_layer,
		out_indices.data(), out_num_coarse);

	if(far_seg)
		num_out += filterFrontierSegment(*far_seg, planes, num_planes, cam_pos_ws, rate_fine_baseline, rate_coarse_baseline,
			rotation_axis, swept_fine, swept_coarse, trans_dilation, coarse_only_debug, draw_coarse_layer,
			out_indices.data() + num_out, out_num_coarse);

	out_indices.resize(num_out); // Shrink to survivor count (keeps prefix, no realloc) so .size() is authoritative.
	return num_out;
}


// SESSION063: result of a background filter pass (see GaussianSplatFilterTask). Carries a Reference to the exact U(P) it
// filtered so drainFilterResults() can drop a result whose U(P) has since been replaced by a fresh traversal.
class GaussianSplatFilterResultMsg : public ThreadMessage
{
public:
	uint64 cloud_id;
	Reference<GaussianSplatUnculledFrontier> frontier; // The U(P) this result was filtered from - staleness check on drain.
	js::Vector<uint32, 16> survivors; // The draw list S(P,R), globally front-to-back (fine + coarse interleaved by depth).
	size_t num_coarse_survivors; // SESSION072 DIAGNOSTIC - see [gsr-filter-drain].
	double filter_compute_ms; // SESSION072 DIAGNOSTIC: pure filterUnculledFrontier() time on the worker thread - excludes task scheduling/queueing, unlike the kick-to-drain gap in the logs. See [gsr-filter-drain].
};


// SESSION063: the cheap per-orientation half of the split architecture. Filters a cached, orientation-independent U(P)
// against the current frustum on a worker thread (~13ms on ~7.5M, measured session063), so a pure rotation produces a
// fresh draw list without the ~450ms traversal. The frontier is immutable and refcounted, so reading it here while the
// main thread holds its own reference is safe.
class GaussianSplatFilterTask : public glare::Task
{
public:
	GaussianSplatFilterTask(uint64 cloud_id_, const Reference<GaussianSplatUnculledFrontier>& frontier_,
		const Planef* planes_, int num_planes_, const Vec4f& cam_pos_ws_, float rate_fine_baseline_, float rate_coarse_baseline_,
		const Vec4f& rotation_axis_, float swept_fine_, float swept_coarse_, // SESSION072: anisotropic rotational dilation - see kickOffFilters().
		const float* trans_dilation_, bool coarse_only_debug_, bool draw_coarse_layer_, // SESSION075
		ThreadSafeQueue<Reference<ThreadMessage> >* result_queue_)
	:	cloud_id(cloud_id_), frontier(frontier_), num_planes(num_planes_), cam_pos_ws(cam_pos_ws_),
		rate_fine_baseline(rate_fine_baseline_), rate_coarse_baseline(rate_coarse_baseline_),
		rotation_axis(rotation_axis_), swept_fine(swept_fine_), swept_coarse(swept_coarse_),
		coarse_only_debug(coarse_only_debug_), draw_coarse_layer(draw_coarse_layer_), result_queue(result_queue_)
	{
		if(num_planes < 0) num_planes = 0;
		if(num_planes > (int)staticArrayNumElems(planes)) num_planes = (int)staticArrayNumElems(planes);
		for(int i=0; i<num_planes; ++i)
			planes[i] = planes_[i];
		for(int i=0; i<(int)staticArrayNumElems(trans_dilation); ++i)
			trans_dilation[i] = trans_dilation_ ? trans_dilation_[i] : 0.f;
	}

	virtual void run(size_t /*thread_index*/) override
	{
		Reference<GaussianSplatFilterResultMsg> msg = new GaussianSplatFilterResultMsg();
		msg->cloud_id = cloud_id;
		msg->frontier = frontier;
		Timer filter_compute_timer; // SESSION072 DIAGNOSTIC: isolates filterUnculledFrontier()'s own cost from task scheduling - see msg->filter_compute_ms.
		filterUnculledFrontier(*frontier, planes, num_planes, cam_pos_ws, rate_fine_baseline, rate_coarse_baseline,
			rotation_axis, swept_fine, swept_coarse, trans_dilation, coarse_only_debug, draw_coarse_layer,
			msg->survivors, &msg->num_coarse_survivors);
		msg->filter_compute_ms = filter_compute_timer.elapsed() * 1.0e3;
		result_queue->enqueue(msg);
	}

private:
	uint64 cloud_id;
	Reference<GaussianSplatUnculledFrontier> frontier;
	Planef planes[6];
	int num_planes;
	Vec4f cam_pos_ws;                                 // SESSION063 K3: for the per-node rotation dilation (rate * dist-to-camera).
	float rate_fine_baseline, rate_coarse_baseline;   // SESSION063 K4/SESSION072: isotropic floor only now - see rotation_axis/swept_* for the measured (anisotropic) component.
	Vec4f rotation_axis;                              // SESSION072: unit rotation axis (world space) - anisotropic rotational dilation, see kickOffFilters().
	float swept_fine, swept_coarse;                   // SESSION072: angle swept about that axis during the fine/coarse dilation latency window. Kept separate from the axis (rather than as two pre-scaled vectors) so the filter needs one cross product per plane instead of two - see filterUnculledFrontier().
	bool coarse_only_debug;                           // SESSION063 K4: keep only coarse nodes (isolation view).
	bool draw_coarse_layer;                           // SESSION075: whether captured coarse nodes may survive at all - see filterUnculledFrontier()'s parameter comment.
	float trans_dilation[6];                          // per-plane translation margin (metres).
	ThreadSafeQueue<Reference<ThreadMessage> >* result_queue;
};


GaussianSplatRenderer::FrustumCounts GaussianSplatRenderer::countSplatsInFrustum() const
{
	const OpenGLScene* scene = opengl_engine->getCurrentScene();
	const Planef* frustum_clip_planes = scene->frustum_clip_planes;
	const int num_frustum_clip_planes = scene->num_frustum_clip_planes;

	const bool clamp_active = (splat_size_clamp_min > 0.f) || (splat_size_clamp_max > 0.f);

	FrustumCounts result;
	result.in_frustum = 0;
	result.total = 0;
	result.frontier = 0;
	result.drawn = 0;
	result.visible = 0;
	for(size_t c=0; c<clouds.size(); ++c)
	{
		const SplatCloud& cloud = *clouds[c];

		// One splat's on-screen test: in frustum AND passing both slices, mirroring the vertex shader (size and distance
		// slices, gaussian_splat_vert_shader.glsl / user_uniform_vals[9,10]) including the invert flags. dist_to_cam is the
		// length of the view-space position, exactly as the shader takes it. A slice at its keep-everything default
		// (size 0/0; distance min 0 / max 1000) excludes nothing.
		const auto passesFrustumAndSlices = [&](uint32 idx) -> bool
		{
			const Vec3f& pos = cloud.positions[idx];
			if(!pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(pos.x, pos.y, pos.z, 1.f)))
				return false;

			if(clamp_active)
			{
				const Vec3f& scale = cloud.scales[idx];
				const float feature_size = 2.f * myMax(scale.x, myMax(scale.y, scale.z));
				const bool below_min = (splat_size_clamp_min > 0.f) && (feature_size < splat_size_clamp_min);
				const bool above_max = (splat_size_clamp_max > 0.f) && (feature_size > splat_size_clamp_max);
				const bool outside_range = below_min || above_max;
				if(splat_size_clamp_invert ? !outside_range : outside_range)
					return false;
			}

			if(splat_dist_clamp_enabled)
			{
				const Vec4f pos_vs = scene->last_view_matrix * Vec4f(pos.x, pos.y, pos.z, 1.f);
				const float dist_to_cam = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length();
				const bool inside_dist_range = (dist_to_cam >= splat_dist_clamp_min) && (dist_to_cam <= splat_dist_clamp_max);
				if(splat_dist_clamp_invert ? inside_dist_range : !inside_dist_range)
					return false;
			}

			return true;
		};

		result.total += cloud.total_splats; // Leaves + merged internal nodes - matches what the traversal iterates over.

		// SESSION085 ETAP 6: ONE frontier stage again. There used to be two - before and after the saturation prune - but
		// the barrier is applied during the walk now, so the frontier a traversal produces is already the saturated one and
		// a separate "after saturation" figure would just repeat this number. Both segments, since reuse splits a frontier
		// into near + inherited far. 0 for a cloud with nothing cached yet.
		if(cloud.cached_ufrontier.nonNull())
		{
			const GaussianSplatUnculledFrontier& uf = *cloud.cached_ufrontier;
			result.frontier += uf.indices.size() + (uf.far_block.nonNull() ? uf.far_block->indices.size() : 0);
		}

		result.drawn += (size_t)myMax(0, cloud.ob->num_instances_to_draw); // SESSION066: the live LoD draw list S(P,R) size - the pre-slice, dilation-band-inclusive selection the pixel_scale limit / camera position pick this frame.

		// in_frustum: over the whole baked tree (the geometric ceiling, LoD-independent).
		for(size_t i=0; i<cloud.total_splats; ++i)
			if(passesFrustumAndSlices((uint32)i))
				result.in_frustum++;

		// visible: over the LoD draw list actually being drawn (current_draw_indices), so this is what really reaches the
		// screen this frame - it drops as the LoD/pixel_scale, frustum, or slices tighten, unlike drawn/in_frustum.
		for(size_t i=0; i<cloud.current_draw_indices.size(); ++i)
			if(passesFrustumAndSlices(cloud.current_draw_indices[i]))
				result.visible++;
	}
	return result;
}


// Focal length in pixels for the current view, averaged over the two axes - a splat's on-screen size only differs
// meaningfully per axis with a non-square viewport/sensor, which is close enough for the traversal's coarse pixel_scale
// budget.  Shared by kickOffTraversals() and getFrustumStructureReport() rather than written out twice, because the report
// classifies nodes as converged by comparing against the same pixel_scale the traversal computed: derived differently,
// the two would disagree exactly at the boundary the report is trying to describe.
static float focalPxForScene(const OpenGLScene& scene, const Vec2i& viewport_dims)
{
	const float focal_x = (float)viewport_dims.x * scene.lens_sensor_dist / scene.use_sensor_width;
	const float focal_y = (float)viewport_dims.y * scene.lens_sensor_dist / scene.use_sensor_height;
	return (focal_x + focal_y) * 0.5f;
}


// One line per bucket: label, count, share of the total, and a bar scaled to the largest bucket.  Empty buckets in the
// middle are printed rather than skipped - a gap in a distribution is itself information.
static std::string histogramLines(const std::string& indent, const std::vector<std::string>& labels, const std::vector<size_t>& counts)
{
	assert(labels.size() == counts.size());

	size_t total = 0, largest = 0;
	for(size_t i=0; i<counts.size(); ++i)
	{
		total += counts[i];
		largest = myMax(largest, counts[i]);
	}
	if(total == 0)
		return indent + "(empty)\n";

	size_t label_w = 0;
	for(size_t i=0; i<labels.size(); ++i)
		label_w = myMax(label_w, labels[i].size());

	const size_t max_bar_chars = 40;

	std::string s;
	for(size_t i=0; i<counts.size(); ++i)
	{
		const size_t bar_len = (counts[i] * max_bar_chars) / largest;
		s += indent + rightSpacePad(labels[i], (unsigned int)label_w) + "  " +
			leftPad(uInt64ToStringCommaSeparated(counts[i]), ' ', 13) + "  " +
			leftPad(doubleToStringNDecimalPlaces(100.0 * (double)counts[i] / (double)total, 1), ' ', 5) + "%  " +
			std::string(bar_len, '#') + "\n";
	}
	return s;
}


// A column-major 3x3, so that the projection below can be a line-by-line transcription of
// gaussian_splat_vert_shader.glsl rather than a translation into the engine's row-major Matrix3.  The maths here decides
// how much fill cost the report attributes to each splat, and a silently transposed covariance would produce numbers that
// look entirely plausible and are wrong - the exact failure mode the measurement rules exist to prevent.
struct Mat3Cols
{
	Vec4f col[3]; // w components are 0 throughout and never read.
};

static inline Mat3Cols mat3Mul(const Mat3Cols& a, const Mat3Cols& b)
{
	Mat3Cols r;
	for(int j=0; j<3; ++j)
		r.col[j] = a.col[0] * b.col[j][0] + a.col[1] * b.col[j][1] + a.col[2] * b.col[j][2];
	return r;
}

static inline Mat3Cols mat3Transpose(const Mat3Cols& a)
{
	Mat3Cols r;
	for(int j=0; j<3; ++j)
		r.col[j] = Vec4f(a.col[0][j], a.col[1][j], a.col[2][j], 0.f);
	return r;
}

static inline Vec4f mat3MulVec(const Mat3Cols& a, const Vec4f& v)
{
	return a.col[0] * v[0] + a.col[1] * v[1] + a.col[2] * v[2];
}


// What one splat costs the rasteriser and the blender at the current camera, worked out the same way the vertex shader
// does.  Everything is in pixels of the current viewport.
struct SplatFootprint
{
	bool drawn;             // False where the vertex shader would push the quad out of the clip volume outright: behind the near plane, degenerate covariance, or opacity at or below the alpha cutoff.
	float radius1_px;       // Screen-space semi-axes, after the same clamp the shader applies.
	float radius2_px;
	float max_radius_px;    // The bound those radii were clamped against, which is per-splat - see the shader's honest-size bound.  Reported so a caller can say whether a radius is a projection or just the bound.
	float quad_area_px;     // 4*r1*r2, clipped to the viewport - fragments the rasteriser produces.
	float ellipse_area_px;  // pi*r1*r2, clipped the same way - the part inside the alpha cutoff, i.e. fragments that survive the fragment shader's discard and reach the blender.  The gap between the two is what an octagonal or otherwise tighter quad could remove.

	// Screen-space ellipse, in the form the software rasteriser below needs: centre, unit axes, and the standard deviation
	// along each of them (the radii are sigma_cutoff of these).
	float centre_x_px, centre_y_px;
	Vec2f axis1, axis2;
	float sigma1_px, sigma2_px;
	float opacity;

	// Integral of this splat's alpha over its footprint - what it actually contributes to the composite, and the figure
	// comparable to the overdraw view's mode 2.
	//
	// Not area * opacity: opacity is the value at the centre only, and alpha falls off as a Gaussian from there. Over an
	// ellipse cut off at k sigma the mean alpha is opacity * 2*(1 - exp(-k^2/2)) / k^2, which at the fixed 3-sigma cap is
	// 0.22 of the peak and at the k a typical low-opacity splat gets from the alpha cutoff (about 2.15) is 0.39. Using the
	// peak would overstate the total by between two and five times - enough to make a comparison against the measured
	// overdraw view agree only after an eyeballed correction, which is not a measurement.
	float alpha_integral_px;
};


// view_matrix is the engine's OpenGL-convention view matrix (scene->last_view_matrix); the splat's data is already baked
// into world space, so the shader's model_matrix is the identity here and drops out.
static SplatFootprint splatFootprint(const Vec3f& pos_ws, const Vec3f& scale, const Vec4f& rotation, float opacity,
	const Matrix4f& view_matrix, const Vec2f& focal_len_px, const Vec2i& viewport_dims, float alpha_cutoff, bool ewa_fix_enabled, float near_fade_width)
{
	SplatFootprint fp;
	fp.drawn = false;
	fp.radius1_px = fp.radius2_px = fp.max_radius_px = fp.quad_area_px = fp.ellipse_area_px = fp.alpha_integral_px = 0.f;

	const Vec4f pos_vs = view_matrix * Vec4f(pos_ws.x, pos_ws.y, pos_ws.z, 1.f);

	const float depth = -pos_vs[2];
	const float near_epsilon = 0.1f;
	if(depth < near_epsilon)
		return fp;

	// The vertex shader's near fade and frustum cull, in that order - see gaussian_splat_vert_shader.glsl for what they
	// are for and where the plane equations come from.  Mirrored here because this function's whole value is that it
	// answers what the shader does.
	const float bound_radius_vs = 3.f * myMax(scale.x, myMax(scale.y, scale.z)); // model_matrix is the identity here - see the comment above this function.
	if(ewa_fix_enabled)
	{
		const float dist_to_centre = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length();
		const float near_fade = myClamp((1.f - bound_radius_vs / myMax(dist_to_centre, 1.0e-6f)) / myMax(near_fade_width, 1.0e-6f), 0.f, 1.f);
		if(near_fade <= 0.f)
			return fp;
		opacity *= near_fade;
	}

	{
		const Vec2f half_viewport_px((float)viewport_dims.x * 0.5f, (float)viewport_dims.y * 0.5f);
		const float dist_outside_x = (std::fabs(focal_len_px.x * pos_vs[0]) + half_viewport_px.x * pos_vs[2]) / std::sqrt(focal_len_px.x*focal_len_px.x + half_viewport_px.x*half_viewport_px.x);
		const float dist_outside_y = (std::fabs(focal_len_px.y * pos_vs[1]) + half_viewport_px.y * pos_vs[2]) / std::sqrt(focal_len_px.y*focal_len_px.y + half_viewport_px.y*half_viewport_px.y);
		if(ewa_fix_enabled && (myMax(dist_outside_x, dist_outside_y) > bound_radius_vs))
			return fp;
	}

	const float qx = rotation[0], qy = rotation[1], qz = rotation[2], qw = rotation[3];
	const Vec4f r_col0(1.f - 2.f*(qy*qy + qz*qz),        2.f*(qx*qy + qz*qw),        2.f*(qx*qz - qy*qw), 0.f);
	const Vec4f r_col1(       2.f*(qx*qy - qz*qw), 1.f - 2.f*(qx*qx + qz*qz),        2.f*(qy*qz + qx*qw), 0.f);
	const Vec4f r_col2(       2.f*(qx*qz + qy*qw),        2.f*(qy*qz - qx*qw), 1.f - 2.f*(qx*qx + qy*qy), 0.f);

	Mat3Cols R;   R.col[0] = r_col0; R.col[1] = r_col1; R.col[2] = r_col2;
	Mat3Cols RS; RS.col[0] = r_col0 * (scale.x*scale.x); RS.col[1] = r_col1 * (scale.y*scale.y); RS.col[2] = r_col2 * (scale.z*scale.z);
	const Mat3Cols cov_os = mat3Mul(RS, mat3Transpose(R));

	Mat3Cols W;
	for(int j=0; j<3; ++j)
	{
		const Vec4f c = view_matrix.getColumn(j);
		W.col[j] = Vec4f(c[0], c[1], c[2], 0.f);
	}
	const Mat3Cols cov_vs = mat3Mul(mat3Mul(W, cov_os), mat3Transpose(W));

	// SESSION083: clamp the linearisation point the Jacobian is evaluated at - mirrors gaussian_splat_vert_shader.glsl's
	// tan-clamp, see that comment for the reasoning. Does not move or resize anything; only bounds the affine
	// approximation's error for splats whose centre sits at a wide angle off the view axis.
	const float tan_half_fov_x = (float)viewport_dims.x / (2.f * focal_len_px.x);
	const float tan_half_fov_y = (float)viewport_dims.y / (2.f * focal_len_px.y);
	const float lim_x = 1.3f * tan_half_fov_x, lim_y = 1.3f * tan_half_fov_y;
	const float vx = myClamp(pos_vs[0] / depth, -lim_x, lim_x) * depth;
	const float vy = myClamp(pos_vs[1] / depth, -lim_y, lim_y) * depth;
	const Vec4f j_row0(focal_len_px.x / depth, 0.f, focal_len_px.x * vx / (depth*depth), 0.f);
	const Vec4f j_row1(0.f, focal_len_px.y / depth, focal_len_px.y * vy / (depth*depth), 0.f);

	const Vec4f cov_vs_j0 = mat3MulVec(cov_vs, j_row0);
	const Vec4f cov_vs_j1 = mat3MulVec(cov_vs, j_row1);

	const float cov2d_a = (j_row0[0]*cov_vs_j0[0] + j_row0[1]*cov_vs_j0[1] + j_row0[2]*cov_vs_j0[2]) + 0.3f;
	const float cov2d_b =  j_row0[0]*cov_vs_j1[0] + j_row0[1]*cov_vs_j1[1] + j_row0[2]*cov_vs_j1[2];
	const float cov2d_c = (j_row1[0]*cov_vs_j1[0] + j_row1[1]*cov_vs_j1[1] + j_row1[2]*cov_vs_j1[2]) + 0.3f;

	const float det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
	if(det <= 0.f)
		return fp;

	const float mid = 0.5f * (cov2d_a + cov2d_c);
	const float half_span = std::sqrt(myMax(mid*mid - det, 0.f));
	const float lambda1 = mid + half_span;
	const float lambda2 = myMax(mid - half_span, 0.f);

	Vec2f axis1;
	if(cov2d_b != 0.f)
	{
		axis1 = Vec2f(cov2d_b, lambda1 - cov2d_a);
		axis1 = normalise(axis1);
	}
	else
		axis1 = (cov2d_a >= cov2d_c) ? Vec2f(1.f, 0.f) : Vec2f(0.f, 1.f);
	const Vec2f axis2(-axis1.y, axis1.x);

	const float sigma_cutoff = (opacity > alpha_cutoff) ? myMin(std::sqrt(2.f * std::log(opacity / alpha_cutoff)), 3.f) : 0.f;
	if(sigma_cutoff <= 0.f)
		return fp; // Invisible even at its centre - the shader makes a degenerate zero-area quad of it.

	float max_radius_px = 2.f * (float)myMax(viewport_dims.x, viewport_dims.y);

	// The vertex shader's honest-size bound on the radius - see gaussian_splat_vert_shader.glsl for the derivation.
	const float bound_radius_ws = sigma_cutoff * myMax(scale.x, myMax(scale.y, scale.z));
	const float dist_to_cam = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length();
	const float sin_half_angle = bound_radius_ws / myMax(dist_to_cam, 1.0e-6f);
	if(ewa_fix_enabled && (sin_half_angle < 0.999f))
		max_radius_px = myMin(max_radius_px, myMax(focal_len_px.x, focal_len_px.y) * sin_half_angle / std::sqrt(1.f - sin_half_angle * sin_half_angle));

	const float radius1 = myMin(sigma_cutoff * std::sqrt(lambda1), max_radius_px);
	const float radius2 = myMin(sigma_cutoff * std::sqrt(lambda2), max_radius_px);

	// Clip to the viewport through the quad's axis-aligned bounding box, and scale the area by how much of that box is on
	// screen.  Approximate for a rotated quad, but the alternative - a real polygon clip - would be a lot of code for a
	// correction that only matters at the screen edge, whereas leaving it out entirely would let one near-camera splat
	// with a quad several screens wide dominate the totals with area that never becomes a fragment.
	// Screen position straight from view space - screen = focal * (x, y) / depth, centred - rather than through the
	// projection matrix, which would only be divided back out again.
	const float centre_x_px = ((focal_len_px.x * vx / depth) + (float)viewport_dims.x * 0.5f);
	const float centre_y_px = ((focal_len_px.y * vy / depth) + (float)viewport_dims.y * 0.5f);
	const float ext_x = std::fabs(radius1 * axis1.x) + std::fabs(radius2 * axis2.x);
	const float ext_y = std::fabs(radius1 * axis1.y) + std::fabs(radius2 * axis2.y);

	const float box_w = 2.f * ext_x, box_h = 2.f * ext_y;
	const float vis_w = myMax(0.f, myMin(centre_x_px + ext_x, (float)viewport_dims.x) - myMax(centre_x_px - ext_x, 0.f));
	const float vis_h = myMax(0.f, myMin(centre_y_px + ext_y, (float)viewport_dims.y) - myMax(centre_y_px - ext_y, 0.f));
	const float visible_fraction = (box_w > 0.f && box_h > 0.f) ? ((vis_w / box_w) * (vis_h / box_h)) : 0.f;

	fp.drawn = true;
	fp.radius1_px = radius1;
	fp.radius2_px = radius2;
	fp.max_radius_px = max_radius_px;
	fp.centre_x_px = centre_x_px;
	fp.centre_y_px = centre_y_px;
	fp.axis1 = axis1;
	fp.axis2 = axis2;
	fp.sigma1_px = radius1 / sigma_cutoff; // The shader builds its conic from the possibly-clamped radii the same way - see its inv_cutoff_sq.
	fp.sigma2_px = radius2 / sigma_cutoff;
	fp.opacity = opacity;
	fp.quad_area_px    = 4.f * radius1 * radius2 * visible_fraction;
	fp.ellipse_area_px = Maths::pi<float>() * radius1 * radius2 * visible_fraction;

	// See alpha_integral_px's comment for the 2*(1 - exp(-k^2/2)) / k^2 factor.
	const float k_sq = sigma_cutoff * sigma_cutoff;
	const float mean_alpha_fraction = 2.f * (1.f - std::exp(-0.5f * k_sq)) / k_sq;
	fp.alpha_integral_px = fp.ellipse_area_px * opacity * mean_alpha_fraction;
	return fp;
}


// Bucket boundaries for the part of the report that asks not "how many splats" but "where does the fill actually come
// from".  All geometric, because every one of these quantities spans orders of magnitude in a real capture.
static const size_t num_area_buckets = 10;
static const char* const area_bucket_labels[num_area_buckets] = { "< 16 px", "16 - 64", "64 - 256", "256 - 1K", "1K - 4K", "4K - 16K", "16K - 64K", "64K - 256K", "256K - 1M", "1M +" };
static size_t areaBucket(float area_px)
{
	if(area_px < 16.f)      return 0;
	if(area_px < 64.f)      return 1;
	if(area_px < 256.f)     return 2;
	if(area_px < 1024.f)    return 3;
	if(area_px < 4096.f)    return 4;
	if(area_px < 16384.f)   return 5;
	if(area_px < 65536.f)   return 6;
	if(area_px < 262144.f)  return 7;
	if(area_px < 1048576.f) return 8;
	return 9;
}

static const size_t num_opacity_buckets = 8;
static const char* const opacity_bucket_labels[num_opacity_buckets] = { "0 - 0.02", "0.02 - 0.05", "0.05 - 0.1", "0.1 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0" };
static size_t opacityBucket(float opacity)
{
	if(opacity < 0.02f) return 0;
	if(opacity < 0.05f) return 1;
	if(opacity < 0.1f)  return 2;
	if(opacity < 0.2f)  return 3;
	if(opacity < 0.4f)  return 4;
	if(opacity < 0.6f)  return 5;
	if(opacity < 0.8f)  return 6;
	return 7;
}

// Smallest scale axis over the largest.  Near 0 is a flat disc or a needle - which is what a 3DGS optimiser fits to a flat
// surface, and what a coplanar merge would have to work with; near 1 is a blob.
static const size_t num_flatness_buckets = 7;
static const char* const flatness_bucket_labels[num_flatness_buckets] = { "< 0.02 (very flat)", "0.02 - 0.05", "0.05 - 0.1", "0.1 - 0.2", "0.2 - 0.4", "0.4 - 0.7", "0.7 - 1.0 (blob)" };
static size_t flatnessBucket(float ratio)
{
	if(ratio < 0.02f) return 0;
	if(ratio < 0.05f) return 1;
	if(ratio < 0.1f)  return 2;
	if(ratio < 0.2f)  return 3;
	if(ratio < 0.4f)  return 4;
	if(ratio < 0.7f)  return 5;
	return 6;
}

// Distance from the camera, in metres.  Here to answer "is the fill coming from the wall in front of me or from
// everything behind it", which is the question a single-room interior capture actually poses.
static const size_t num_distance_buckets = 9;
static const char* const distance_bucket_labels[num_distance_buckets] = { "< 1 m", "1 - 2", "2 - 4", "4 - 8", "8 - 16", "16 - 32", "32 - 64", "64 - 128", "128 m +" };
static size_t distanceBucket(float dist)
{
	if(dist < 1.f)   return 0;
	if(dist < 2.f)   return 1;
	if(dist < 4.f)   return 2;
	if(dist < 8.f)   return 3;
	if(dist < 16.f)  return 4;
	if(dist < 32.f)  return 5;
	if(dist < 64.f)  return 6;
	if(dist < 128.f) return 7;
	return 8;
}


// As histogramLines(), but for the buckets where the count is the less interesting of the two numbers: a bucket holding
// 1% of the splats and 60% of the fill is the whole point of this half of the report, and a count-only histogram hides
// exactly that.  The bar tracks the area share, not the count.
static std::string weightedHistogramLines(const std::string& indent, const std::vector<std::string>& labels, const std::vector<size_t>& counts, const std::vector<double>& areas)
{
	assert(labels.size() == counts.size() && labels.size() == areas.size());

	size_t total_count = 0;
	double total_area = 0, largest_area = 0;
	for(size_t i=0; i<counts.size(); ++i)
	{
		total_count += counts[i];
		total_area += areas[i];
		largest_area = myMax(largest_area, areas[i]);
	}
	if(total_count == 0)
		return indent + "(empty)\n";

	size_t label_w = 0;
	for(size_t i=0; i<labels.size(); ++i)
		label_w = myMax(label_w, labels[i].size());

	const size_t max_bar_chars = 40;

	std::string s;
	for(size_t i=0; i<counts.size(); ++i)
	{
		const size_t bar_len = (largest_area > 0) ? (size_t)((areas[i] * (double)max_bar_chars) / largest_area) : 0;
		s += indent + rightSpacePad(labels[i], (unsigned int)label_w) + "  " +
			leftPad(uInt64ToStringCommaSeparated(counts[i]), ' ', 13) + " splats " +
			leftPad(doubleToStringNDecimalPlaces(100.0 * (double)counts[i] / (double)total_count, 1), ' ', 5) + "%   fill " +
			leftPad(doubleToStringNDecimalPlaces((total_area > 0) ? (100.0 * areas[i] / total_area) : 0.0, 1), ' ', 5) + "%  " +
			std::string(bar_len, '#') + "\n";
	}
	return s;
}


static const char* const stop_reason_labels[FrontierStop_NumReasons] =
{
	"leaf (finest detail there is)",
	"converged (pixel_scale <= limit)",
	"density cap",
	"depth cap",
	"budget cap",
	"member has no LoD tree",
	"saturation LoD bias", // SESSION085 ETAP 3 - only produced when the bias ceiling is above 1.
	"out of frustum", // SESSION055 - only produced by the fast path with cull enabled; getFrustumStructureReport() disables cull so this stays 0 there.
	"out of distance slice" // SESSION072 - only produced by the fast path with the dist-clamp checkbox on; getFrustumStructureReport() always leaves it off.
};


// SESSION081: the geom-snapshot cache-hit/miss logic, extracted from fillTraversalScratch() below so
// kickOffSaturationBuilds() can obtain the same snapshot without needing a whole GaussianSplatLodTraversalScratch to
// hold it - a saturation build reads positions/scales/rotations/alpha exactly like a traversal does, just without the
// rest of the scratch (selected_indices, members_snapshot, the caps). Behaviour unchanged from before the extraction.
// SESSION081 ETAP 4: one slice of the packed-occluder build - see getOrBuildCachedGeom() and GsSatPackedOccluder.
class GsSatPackTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		for(size_t i=i_begin; i<i_end; ++i)
			out[i] = gsSatPackOccluder(scales[i], rotations[i], colours[i][3]);
	}

	const Vec3f* scales; const Vec4f* rotations; const Vec4f* colours;
	GsSatPackedOccluder* out;
	size_t i_begin, i_end;
};


Reference<GaussianSplatCachedGeom> GaussianSplatRenderer::getOrBuildCachedGeom(SplatCloud& cloud) const
{
	// SESSION058: reuse the cached snapshot if the cloud hasn't structurally changed since it was built - see
	// GaussianSplatCachedGeom's comment for why topology_generation is a safe invalidation signal (every writer of
	// cloud.positions/cloud.feature_size bumps it in the same call). A hit is just a Reference<> copy: no allocation,
	// no memcpy, regardless of cloud size.
	if(!cloud.cached_traversal_geom.isNull() && cloud.cached_traversal_geom_generation == cloud.topology_generation)
		return cloud.cached_traversal_geom;

	// Freeze a fresh snapshot of the cloud's current world-space node data, so a worker never touches the live,
	// growable/mutable state - same rule the sort task follows, for the same reason. The old cached_traversal_geom
	// (if any) is left alone, not overwritten in place: an in-flight traversal or a diagnostics dump may still hold
	// a reference to it, and this rewrite's whole point is that nothing is ever mutated once shared.
	Reference<GaussianSplatCachedGeom> geom = new GaussianSplatCachedGeom();
	geom->positions.resizeNoCopy(cloud.total_splats);
	std::memcpy(geom->positions.data(), cloud.positions.data(), cloud.total_splats * sizeof(Vec3f));

	// SESSION055: feature_size is maintained on the cloud itself (see SplatCloud::feature_size), so all we do here is
	// a single 4-byte-per-node memcpy - no scales copy, no per-node arithmetic loop.
	geom->feature_size.resizeNoCopy(cloud.total_splats);
	std::memcpy(geom->feature_size.data(), cloud.feature_size.data(), cloud.total_splats * sizeof(float));

	// SESSION059: cull_radius is maintained on the cloud the same way feature_size is - see SplatCloud::cull_radius.
	geom->cull_radius.resizeNoCopy(cloud.total_splats);
	std::memcpy(geom->cull_radius.data(), cloud.cull_radius.data(), cloud.total_splats * sizeof(float));

	// SESSION081 ETAP 4: pack the saturation gather's inputs - see GsSatPackedOccluder. This replaces the scales and
	// rotations memcpys and the alpha extract loop that used to be here, and it MOVES LESS MEMORY than they did (12B
	// written per node against 32B), so the arithmetic it adds is paid for out of the bandwidth it saves. Split across
	// the pool because this is the one place in the pipeline that touches every splat in the cloud, and it runs on the
	// main thread; serial fallback when there is no pool (the same shape every other split loop here uses).
	geom->sat_occl.resizeNoCopy(cloud.total_splats);
	{
		glare::TaskManager* const pack_task_manager = opengl_engine->getMainTaskManager();
		const size_t n = cloud.total_splats;
		const size_t num_chunks = (pack_task_manager != NULL) ?
			myMax<size_t>(1, myMin((size_t)myMax(1, (int)pack_task_manager->getConcurrency()) * 4, n / 16384)) : 1;
		glare::TaskGroupRef group = new glare::TaskGroup();
		for(size_t c=0; c<num_chunks; ++c)
		{
			Reference<GsSatPackTask> t = new GsSatPackTask();
			t->scales = cloud.scales.data(); t->rotations = cloud.rotations.data(); t->colours = cloud.colours.data();
			t->out = geom->sat_occl.data();
			t->i_begin = (n * c) / num_chunks;
			t->i_end   = (n * (c + 1)) / num_chunks;
			if(pack_task_manager != NULL) group->tasks.push_back(t); else t->run(0);
		}
		if(pack_task_manager != NULL) pack_task_manager->runTaskGroup(group);
	}

	cloud.cached_traversal_geom = geom;
	cloud.cached_traversal_geom_generation = cloud.topology_generation;
	return geom;
}


void GaussianSplatRenderer::fillTraversalScratch(SplatCloud& cloud, GaussianSplatLodTraversalScratch& scratch) const
{
	// SESSION058 diag: measures the cost below - see cpu_prof_log's declaration and the session058 snapshot. Before this
	// function's rewrite, this cost was a fixed ~23ms full-cloud memcpy on EVERY kick (session057 §3's mystery CPU load
	// when 0 splats were visible: an empty traversal returns near-instantly, so kicks - each paying this cost - queued
	// back to back with no gap). Now it's ~0 on a cache hit (the common case: camera moved/rotated, cloud unchanged) and
	// the same ~23ms memcpy only on a cache miss (the cloud's topology_generation changed since the cache was built).
	Timer prof_timer;
	const bool cache_hit = !cloud.cached_traversal_geom.isNull() && cloud.cached_traversal_geom_generation == cloud.topology_generation; // SESSION081: for the profiling print below only - see getOrBuildCachedGeom().
	scratch.geom = getOrBuildCachedGeom(cloud);

	scratch.members_snapshot.resize(cloud.members.size());
	for(size_t m=0; m<cloud.members.size(); ++m)
	{
		scratch.members_snapshot[m].splat_data = cloud.members[m].splat_data;
		scratch.members_snapshot[m].offset = cloud.members[m].offset;
		scratch.members_snapshot[m].count = cloud.members[m].count;
		scratch.members_snapshot[m].hidden = cloud.members[m].hidden; // SESSION059
	}

	if(cpu_prof_log)
	{
		const size_t bytes_copied = cache_hit ? 0 : cloud.total_splats * (sizeof(Vec3f) + sizeof(float) + sizeof(float) + sizeof(float) + sizeof(Vec3f) + sizeof(Vec4f)); // positions + feature_size + cull_radius (SESSION059) + alpha (SESSION074) + scales/rotations (SESSION077).
		conPrint("[gsr-prof] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms fillTraversalScratch cloud=" +
			toString(cloud.cloud_id) + " splats=" + toString(cloud.total_splats) +
			" " + (cache_hit ? std::string("HIT") : std::string("MISS")) +
			" bytes=" + toString(bytes_copied) +
			" took=" + doubleToStringNDecimalPlaces(prof_timer.elapsed() * 1000.0, 2) + "ms");
	}
}


// SESSION071: how faithfully the LoD nodes THAT ARE ACTUALLY BEING DRAWN stand in for the splats beneath them.
//
// Measured over the live draw list (SplatCloud::current_draw_indices), not over the whole tree. The first version of this
// walked every node, and the answer was dominated by the deepest levels simply because that is where almost all the nodes
// live (~3.6M of 4.1M sat at depth 19+) - while what the camera selects once it pulls back is a few thousand coarse nodes
// several levels up. The table was therefore describing nodes that were not in the frame at all.
//
// Three columns, because the previous round ruled out two candidate causes and the remaining suspect is geometric:
//
//   ratio  - node brightness over the alpha*area-weighted mean brightness of the original leaves beneath it. Each node is
//            scored against ITS OWN subtree, so the number is not confounded by which part of the scene sits at which
//            depth. 1.00 = the merge carried the colour up faithfully. Split dark/bright because the observed defect is
//            asymmetric - foliage came out clean, bright stone did not.
//   mean A - to be read against the leaves row below, not against 1: a merged node standing in for near-opaque leaves
//            should not itself be half transparent. The saturated column says where the alpha boost knob is inert by
//            construction, since clamp(1 * boost) is still 1.
//   cover  - the node's own drawn silhouette radius (3 sigma along its largest axis) over bounding_radius_os, the exact
//            enclosing radius of every leaf beneath it. A leaf scores exactly 1.00 by construction, which calibrates the
//            column. Below 1 means the node is drawn SMALLER than the descendants it replaces, so neighbouring nodes
//            leave gaps for the background through - which would show up as patches of lightening when opacity is raised
//            rather than a uniform lift, and would hurt bright surfaces against a dark background far more than foliage.
std::string GaussianSplatRenderer::mergeColourByDepthReport()
{
	const int max_depth_rows = 24;
	std::vector<size_t> nodes(max_depth_rows, 0), saturated(max_depth_rows, 0), dark_n(max_depth_rows, 0), bright_n(max_depth_rows, 0);
	std::vector<double> alpha_sum(max_depth_rows, 0.0), dark_ratio_sum(max_depth_rows, 0.0), bright_ratio_sum(max_depth_rows, 0.0), cover_sum(max_depth_rows, 0.0);
	size_t leaves_drawn = 0, total_drawn = 0;
	double leaves_alpha_sum = 0.0;
	bool any_tree = false;

	std::vector<uint32> depth;
	std::vector<double> leaf_w, leaf_lw;

	for(size_t c=0; c<clouds.size(); ++c)
	{
		const SplatCloud& cloud = *clouds[c];
		if(cloud.current_draw_indices.empty())
			continue;

		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			const CloudMember& member = cloud.members[m];
			const std::vector<GaussianSplatLodNode>& tree = member.splat_data->lod_tree;
			if(tree.empty())
				continue;

			const size_t n = tree.size();
			const size_t off = member.offset;

			// Is any of this member's range drawn at all?  Skip the O(tree) prep below entirely if not.
			bool member_drawn = false;
			for(size_t k=0; k<cloud.current_draw_indices.size(); ++k)
			{
				const uint32 idx = cloud.current_draw_indices[k];
				if(idx >= off && idx < off + n) { member_drawn = true; break; }
			}
			if(!member_drawn)
				continue;
			any_tree = true;

			// Depth, forward pass: a node's children are always linearised after it, so the parent's depth is known by the
			// time its children are reached.
			depth.assign(n, 0);
			for(size_t i=0; i<n; ++i)
				for(uint32 k=0; k<tree[i].child_count; ++k)
					depth[tree[i].child_start + k] = depth[i] + 1;

			// Leaf brightness beneath each node, reverse pass (= bottom-up, same ordering argument). Weighted by
			// alpha*area at LEAF level - the merge's own weights are re-derived per level, so this is an independent
			// reference rather than something the merge trivially reproduces.
			leaf_w .assign(n, 0.0);
			leaf_lw.assign(n, 0.0);
			for(size_t i = n; i-- > 0; )
			{
				if(tree[i].child_count == 0)
				{
					const Vec3f& sc = cloud.scales[off + i];
					float s0 = sc.x, s1 = sc.y, s2 = sc.z;
					if(s0 < s1) mySwap(s0, s1);
					if(s1 < s2) mySwap(s1, s2);
					if(s0 < s1) mySwap(s0, s1);
					const Vec4f& col = cloud.colours[off + i];
					const double w = (double)col.x[3] * (double)s0 * (double)s1; // Constant factors cancel in the ratio.
					leaf_w [i] = w;
					leaf_lw[i] = w * (0.2126 * col.x[0] + 0.7152 * col.x[1] + 0.0722 * col.x[2]); // Rec.709 luma.
				}
				else
					for(uint32 k=0; k<tree[i].child_count; ++k)
					{
						leaf_w [i] += leaf_w [tree[i].child_start + k];
						leaf_lw[i] += leaf_lw[tree[i].child_start + k];
					}
			}

			for(size_t k=0; k<cloud.current_draw_indices.size(); ++k)
			{
				const uint32 gidx = cloud.current_draw_indices[k];
				if(gidx < off || gidx >= off + n)
					continue; // Belongs to another member; that member's own pass picks it up.
				const size_t i = (size_t)gidx - off;
				total_drawn++;

				const Vec4f& col = cloud.colours[gidx];
				if(tree[i].child_count == 0)
				{
					// Leaves are ground truth - nothing was derived, so there is no drift to measure - but their mean
					// opacity is the reference the merged rows have to be read against.
					leaves_drawn++;
					leaves_alpha_sum += col.x[3];
					continue;
				}

				const int d = (int)myMin<uint32>(depth[i], (uint32)(max_depth_rows - 1));
				nodes[d]++;
				alpha_sum[d] += col.x[3];
				if(col.x[3] >= 0.99f) saturated[d]++;

				// Drawn silhouette against the true enclosing radius of the descendants. cull_radius is bounding_radius_os
				// already baked to world space, and for a leaf it is exactly 3 * max_scale - the same quantity the
				// numerator forms - which is what makes 1.00 the meaningful reference point here.
				const Vec3f& sc = cloud.scales[gidx];
				const float max_scale = myMax(sc.x, myMax(sc.y, sc.z));
				const float bound = cloud.cull_radius[gidx];
				if(bound > 1.0e-9f)
					cover_sum[d] += (double)(3.f * max_scale) / (double)bound;

				if(leaf_w[i] <= 0.0)
					continue;
				const double ref = leaf_lw[i] / leaf_w[i];
				if(ref < 1.0e-4) continue; // Reference is black; the ratio carries no information.
				const double ratio = (0.2126 * col.x[0] + 0.7152 * col.x[1] + 0.0722 * col.x[2]) / ref;
				if(ref < 0.5) { dark_n[d]++;   dark_ratio_sum[d]   += ratio; }
				else          { bright_n[d]++; bright_ratio_sum[d] += ratio; }
			}
		}
	}

	if(!any_tree)
		return "\nDrawn LoD nodes: no LoD-tree cloud has a live draw list to measure.\n";

	std::string s = "\nDrawn LoD nodes by tree depth (SESSION071 - depth 0 = root, i.e. the coarsest stand-in):\n";
	s += "  Measured over the LIVE DRAW LIST, so these are the nodes actually in the frame right now.\n";
	s += "  ratio = node brightness / alpha*area-weighted mean brightness of the original leaves beneath it (1.00 = faithful).\n";
	s += "  cover = drawn silhouette radius / exact enclosing radius of those leaves.  Below 1 = drawn smaller than what it\n";
	s += "          replaces, so neighbouring nodes leave gaps.  A leaf scores exactly 1.00, which calibrates the column.\n";
	s += "  depth      nodes    mean A   saturated     cover   ratio (dark ref)   ratio (bright ref)\n";
	for(int d=0; d<max_depth_rows; ++d)
	{
		if(nodes[d] == 0)
			continue;
		s += "  " + leftPad(toString(d), ' ', 5) +
			leftPad(uInt64ToStringCommaSeparated(nodes[d]), ' ', 11) +
			leftPad(doubleToStringNDecimalPlaces(alpha_sum[d] / (double)nodes[d], 3), ' ', 10) +
			leftPad(doubleToStringNDecimalPlaces(100.0 * (double)saturated[d] / (double)nodes[d], 1), ' ', 11) + "%" +
			leftPad(doubleToStringNDecimalPlaces(cover_sum[d] / (double)nodes[d], 3), ' ', 10) +
			leftPad(dark_n[d]   ? doubleToStringNDecimalPlaces(dark_ratio_sum[d]   / (double)dark_n[d],   3) : std::string("-"), ' ', 19) +
			leftPad(bright_n[d] ? doubleToStringNDecimalPlaces(bright_ratio_sum[d] / (double)bright_n[d], 3) : std::string("-"), ' ', 21) + "\n";
	}
	s += "  leaves" + leftPad(uInt64ToStringCommaSeparated(leaves_drawn), ' ', 11) +
		leftPad(leaves_drawn ? doubleToStringNDecimalPlaces(leaves_alpha_sum / (double)leaves_drawn, 3) : std::string("-"), ' ', 10) +
		"          -     1.000                  -                    -\n";
	s += "  Drawn nodes counted: " + uInt64ToStringCommaSeparated(total_drawn) + " (leaves row is the reference, not a defect).\n";
	return s;
}


std::string GaussianSplatRenderer::getFrustumStructureReport(float merge_colour_tol, float merge_angle_tol_deg)
{
	if(clouds.empty())
		return "No splat clouds registered.\n";

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Planef* const frustum_clip_planes = scene->frustum_clip_planes;
	const int num_frustum_clip_planes = scene->num_frustum_clip_planes;
	const Vec4f cam_pos_ws = scene->cam_to_world.getColumn(3);
	const Vec2i viewport_dims = opengl_engine->getViewportDims();
	const float focal_px = focalPxForScene(*scene, viewport_dims);

	Timer timer;

	std::string s = "==== Gaussian splat frustum structure report ====\n";
	s += "Camera at (" + doubleToStringNDecimalPlaces(cam_pos_ws[0], 2) + ", " + doubleToStringNDecimalPlaces(cam_pos_ws[1], 2) + ", " + doubleToStringNDecimalPlaces(cam_pos_ws[2], 2) + "), " +
		"viewport " + toString(viewport_dims.x) + " x " + toString(viewport_dims.y) + " (" + doubleToStringNDecimalPlaces((double)viewport_dims.x * viewport_dims.y * 1.0e-6, 2) + " Mpixel), " +
		"focal " + doubleToStringNDecimalPlaces(focal_px, 1) + " px\n";
	s += "Traversal params: pixel_scale_limit " + doubleToStringNDecimalPlaces(lod_pixel_scale_limit, 2) +
		", max_splats_budget " + uInt64ToStringCommaSeparated(lod_max_splats_budget) +
		", max_layer_density " + doubleToStringNDecimalPlaces(lod_max_layer_density, 1) + (lod_max_layer_density == 0 ? " (off)" : "") +
		", max_tree_depth " + toString(lod_max_tree_depth) + (lod_max_tree_depth == 0 ? " (off)" : "") + "\n";
	s += "Frustum test is on a node's centre: a node whose centre is just outside still rasterises, so the in-frustum\n"
		 "figures below slightly understate what is actually being shaded.\n";

	// What the pass actually cost on the frame before this button was pressed, and what state the cloud was in when it did.
	//
	// Both are here because a report without them is only half a record: everything below this line is a prediction of cost
	// from geometry, and the prediction is worth nothing unless the measured cost sits beside it, taken at the same camera,
	// on the same cloud.  Pairing the two up afterwards from a separately-copied diagnostics panel is exactly the step that
	// goes wrong - or gets skipped.
	//
	// Read the splat line, not the total: the total carries an extra full pass over the splats whenever the hide-overdraw
	// mask is being rebuilt, and it includes everything else in the frame besides.
	s += "\nSplat pass, GPU times measured on the last frame drawn:\n";
	if(opengl_engine->isProfilingEnabled())
	{
		s += "  draw splats      : " + doubleToStringNSigFigs(opengl_engine->getLastDrawSplatsGPUTime() * 1.0e3, 4) + " ms\n";
		s += "  splat depth blit : " + doubleToStringNSigFigs(opengl_engine->getLastSplatDepthBlitGPUTime() * 1.0e3, 4) + " ms\n";
		s += "  splat sat. mark  : " + doubleToStringNSigFigs(opengl_engine->getLastMarkSaturatedSplatsGPUTime() * 1.0e3, 4) + " ms\n";
		s += "  whole frame      : " + doubleToStringNSigFigs(opengl_engine->last_total_draw_GPU_time * 1.0e3, 4) + " ms\n";
		s += "  Splats drawn last frame: " + uInt64ToStringCommaSeparated(opengl_engine->last_num_splats_drawn) +
			" (before the vertex shader's own culling, which this count does not see)\n";
	}
	else
		s += "  Not measured - GPU profiling is off.  Tick \"Show frame time graphs\" before starting the client, and take\n"
			 "  this report again; the CPU-side frame time is vsync-bound and is not a substitute.\n";

	s += "Cloud state: " + (last_merge_description.empty() ? std::string("as loaded, not merged") : last_merge_description) + "\n";

	// Live draw-path settings that change what the numbers above mean.  Not the whole panel - only the ones that alter the
	// cost of the splat pass without altering the frontier the report describes, which are the ones that would otherwise
	// make two reports disagree for no visible reason.
	s += "Draw settings: alpha_cutoff " + doubleToStringNDecimalPlaces(splat_alpha_cutoff, 4) +
		// Said out loud whenever it is not the identity: every area and opacity figure below is of the adjusted cloud, so
		// two reports taken at different settings are not otherwise comparable.
		(((splat_alpha_gain != 1.f) || (splat_alpha_gamma != 1.f)) ?
			(", ALPHA ADJUSTED gain " + doubleToStringNDecimalPlaces(splat_alpha_gain, 2) + " gamma " + doubleToStringNDecimalPlaces(splat_alpha_gamma, 2)) : std::string()) +
		", draw slices " + toString(splat_num_draw_slices) + (splat_slice_growth != 1.f ? (" (growth " + doubleToStringNDecimalPlaces(splat_slice_growth, 2) + ")") : "") +
		// Loud, like the debug views below: the frame is incomplete, so neither the times nor anything read off the
		// picture is a statement about the scene.
		((splat_draw_slice_limit > 0) ? (", DRAW LIMIT " + toString(splat_draw_slice_limit) + " of " + toString(splat_num_draw_slices) + " slices (frame is incomplete)") : std::string()) +
		", saturation gate " + (splat_saturation_gate_enabled ? ("on at " + doubleToStringNDecimalPlaces(splat_saturation_threshold, 4) + ", mask 1/" + toString(splat_saturation_mask_downscale)) : std::string("off")) +
		// Loud for the same reason the layer cap is: it changes what the picture is allowed to contain, so a time taken
		// under it is not comparable with one taken without it.
		((splat_coverage_cap > 0.f) ? (", COVERAGE CAP at " + doubleToStringNDecimalPlaces(splat_coverage_cap, 4)) : std::string()) +
		", accum buffer " + (splat_accum_buffer_8bit ? "RGBA8" : "RGBA16F") +
		// Printed only when set, but loudly: it is the one setting here that changes what the picture is allowed to
		// contain, so a time taken under it is not comparable with one taken without it.
		// getHideMode() == 0, i.e. Clip is not running: there is one mask, and Clip owns it when it is on, which leaves a
		// set cap doing nothing.  (This read == 3, a value getHideMode() cannot return, so the line never printed at all.)
		(((splat_layer_cap > 0) && (getHideMode() == 0)) ? (", LAYER CAP " + toString(splat_layer_cap) + (splat_hide_test_conservative ? " (conservative)" : " (centre test)")) : std::string()) +
		(getShowOverdraw() ? (", OVERDRAW VIEW " + toString(splat_show_overdraw_mode) + " (blend is additive, times are not comparable)") : std::string()) +
		(getHideMode() != 0 ? (", HIDE MODE " + toString(getHideMode()) + " (a counting pass is in the frame)") : std::string()) + "\n";

	// World-wide roll-up, accumulated across the clouds below.
	size_t world_frontier = 0, world_frontier_in_frustum = 0, world_leaves_in_frustum = 0, world_frontier_leaves_in_frustum = 0;
	double world_quad_area = 0, world_ellipse_area = 0;
	double world_traversal_ms = 0; // SESSION072: sum of each cloud's own synchronous task.run() time - see below.
	size_t world_reached_rasteriser = 0, world_in_frustum_before_shader = 0; // Kept on the renderer afterwards, so the diagnostics panel can show what this press measured - see getDiagnostics().
	size_t world_reason_counts[FrontierStop_NumReasons];
	for(size_t i=0; i<FrontierStop_NumReasons; ++i)
		world_reason_counts[i] = 0;

	const size_t max_clouds_to_report = 8;
	for(size_t c=0; c<myMin(clouds.size(), max_clouds_to_report); ++c)
	{
		SplatCloud& cloud = *clouds[c];

		s += "\n--- cloud " + toString(cloud.cloud_id) + ": " + toString(cloud.members.size()) + (cloud.members.size() == 1 ? " member, " : " members, ") +
			uInt64ToStringCommaSeparated(cloud.total_splats) + " nodes uploaded ---\n";

		if(cloud.total_splats == 0)
		{
			s += "  (empty)\n";
			continue;
		}

		if(!cloudHasLodTree(cloud))
		{
			// Nothing to traverse, so every splat is drawn as-is.  Still worth a line: a cloud without a tree is invisible
			// to every LoD knob, and that is exactly the kind of thing a report like this exists to make obvious.
			size_t in_frustum = 0;
			for(size_t i=0; i<cloud.total_splats; ++i)
			{
				const Vec3f& p = cloud.positions[i];
				if(pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
					in_frustum++;
			}
			s += "  No LoD tree on any member - every splat is always drawn.  " + uInt64ToStringCommaSeparated(in_frustum) + " of them in frustum.\n";
			world_frontier += cloud.total_splats;
			world_frontier_in_frustum += in_frustum;
			world_reason_counts[FrontierStop_NoTree] += in_frustum;
			continue;
		}

		// Re-run this cloud's traversal here on the main thread, with the current camera and the current live parameters,
		// recording why it stopped at each selected node.  The per-frame traversal's frontier isn't kept on the CPU - it
		// goes straight into the instance index VBO and its scratch returns to the pool - so reading the live one would
		// mean holding a permanent copy of every selection, for a button pressed a handful of times per session.  Re-running
		// costs one traversal per click and gives the same answer, the task being a pure function of its inputs.
		Reference<GaussianSplatLodTraversalScratch> scratch;
		if(free_traversal_scratch.empty())
			scratch = new GaussianSplatLodTraversalScratch();
		else
		{
			scratch = free_traversal_scratch.back();
			free_traversal_scratch.pop_back();
		}

		fillTraversalScratch(cloud, *scratch);

		js::Vector<FrontierNodeRecord, 16> frontier;
		frontier.reserve(cloud.total_splats);

		// Null result queue: this frontier is for reading, not for drawing - see the task's own comment there.
		// SESSION055: report deliberately runs with frustum-cull off. It has to see the whole tree to answer "what would the
		// LoD hierarchy offer at this camera" - a frustum-culled traversal would misreport pruning ceilings and reason
		// classifications for anything behind the camera.
		GaussianSplatLodTraversalTask task(cloud.cloud_id, cloud.topology_generation, scratch, cam_pos_ws,
			lod_pixel_scale_limit, lod_max_splats_budget, lod_max_layer_density, lod_max_tree_depth, focal_px,
			/*frustum_clip_planes=*/NULL, /*num_frustum_clip_planes=*/0, /*frustum_cull_enabled=*/false,
			/*translation_dilation=*/NULL, /*rotation_dilation_rate=*/0.f,
			/*result_queue=*/NULL, &frontier);
		// SESSION072: isolates the traversal itself from the rest of this function's cost (stats accumulation, string
		// building below) - "Report took" at the bottom times the whole button press, this times just the thing the
		// owner actually wants to know the cost of. Same task, same call the per-frame async path makes (run() doesn't
		// know or care whether its caller is sync or a worker thread), just on the main thread and with frustum-cull
		// off - see the comment above.
		Timer cloud_traversal_timer;
		task.run(0);
		const double cloud_traversal_ms = cloud_traversal_timer.elapsed() * 1.0e3;
		world_traversal_ms += cloud_traversal_ms;
		s += "  Traversal: " + doubleToStringNSigFigs(cloud_traversal_ms, 4) + " ms (main thread, unculled - see report note above)\n";

		// Not returned to the pool yet: the pruning ceiling below walks scratch->selected_indices, which is the frontier in
		// the front-to-back order the draw actually uses, and which the frontier records deliberately don't preserve.

		//----------------------------- Structure of the trees themselves -----------------------------
		// Independent of the camera: what the hierarchy offers, against which the frontier below says what was taken.
		size_t tree_nodes = 0, tree_leaves = 0, tree_max_depth = 0, leaves_in_frustum = 0;
		js::Vector<uint16, 16> node_depth;

		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			const std::vector<GaussianSplatLodNode>& tree = cloud.members[m].splat_data->lod_tree;
			if(tree.empty())
				continue;
			const size_t member_offset = cloud.members[m].offset;

			// A node's children always sit later in the array than the node itself (see buildGaussianSplatLodTree()), so one
			// forward pass assigns every depth: a node's own depth is final by the time the pass reaches it.
			node_depth.resizeNoCopy(tree.size());
			node_depth[0] = 0;
			for(size_t i=0; i<tree.size(); ++i)
			{
				const GaussianSplatLodNode& node = tree[i];
				const uint16 depth = node_depth[i];
				tree_max_depth = myMax(tree_max_depth, (size_t)depth);

				if(node.child_count == 0)
				{
					tree_leaves++;
					const Vec3f& p = cloud.positions[member_offset + i];
					if(pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
						leaves_in_frustum++;
				}
				else
				{
					for(uint32 ch = node.child_start; ch < (uint32)node.child_start + node.child_count; ++ch)
						node_depth[ch] = (uint16)(depth + 1);
				}
			}
			tree_nodes += tree.size();
		}

		s += "  Trees: " + uInt64ToStringCommaSeparated(tree_nodes) + " nodes (" + uInt64ToStringCommaSeparated(tree_leaves) + " leaves, " +
			uInt64ToStringCommaSeparated(tree_nodes - tree_leaves) + " merged), " + toString(tree_max_depth + 1) + " levels deep\n";

		//----------------------------- The frontier the traversal selected -----------------------------
		size_t reason_counts[FrontierStop_NumReasons];
		for(size_t i=0; i<FrontierStop_NumReasons; ++i)
			reason_counts[i] = 0;

		size_t frontier_in_frustum = 0, frontier_leaves_in_frustum = 0;
		double depth_sum = 0;

		for(size_t i=0; i<frontier.size(); ++i)
		{
			const FrontierNodeRecord& rec = frontier[i];
			const Vec3f& p = cloud.positions[rec.cloud_idx];
			if(!pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
				continue;

			frontier_in_frustum++;
			reason_counts[myMin((size_t)rec.stop_reason, (size_t)FrontierStop_NumReasons - 1)]++;
			depth_sum += rec.depth;

			if(rec.stop_reason != FrontierStop_NoTree)
			{
				const GaussianSplatLodNode& node = cloud.members[rec.member_idx].splat_data->lod_tree[rec.tree_local_idx];
				if(node.child_count == 0)
					frontier_leaves_in_frustum++;
			}
		}

		s += "  Frontier: " + uInt64ToStringCommaSeparated(frontier.size()) + " nodes selected, " +
			uInt64ToStringCommaSeparated(frontier_in_frustum) + " with centre in frustum\n";

		// Self-check, so the report doesn't have to be believed on its own word: the traversal re-run here must produce the
		// same number of nodes as the one whose result is currently in the instance index VBO.  A mismatch means the two are
		// not the same computation, and every figure below is describing a frontier that is not the one on screen.
		//
		// One legitimate way to see a mismatch: the live selection is only refreshed once the camera has moved past this
		// cloud's threshold (or a parameter change forced it), so a report taken while the camera is still moving, or in the
		// frame or two after a parameter change, is compared against a frontier computed for a slightly different camera.
		// Stand still for a moment and take it again before treating a difference as a bug.
		{
			const size_t live_frontier = (size_t)myMax(0, cloud.ob->num_instances_to_draw);
			s += "  Live frontier currently in the index VBO: " + uInt64ToStringCommaSeparated(live_frontier);
			if(live_frontier == frontier.size())
				s += " - matches\n";
			else
				s += " - DIFFERS by " + uInt64ToStringCommaSeparated((live_frontier > frontier.size()) ? (live_frontier - frontier.size()) : (frontier.size() - live_frontier)) +
					(cloud.traversal_in_flight ? " (a traversal is in flight - the live figure is about to be replaced)\n" : " (camera moved since the live traversal, or the report is not reproducing it)\n");
		}

		// The headline number.  The frontier covers the leaves in view; how close the two counts are is how much of the
		// hierarchy's saving is actually being taken at this camera position.  At 1.0 the tree is fully unfolded in view and
		// no LoD parameter can remove a single drawn splat - only changing the cloud itself can.
		s += "  Leaves in frustum: " + uInt64ToStringCommaSeparated(leaves_in_frustum) + ", covered by " +
			uInt64ToStringCommaSeparated(frontier_in_frustum) + " frontier nodes = " +
			doubleToStringNDecimalPlaces(100.0 * (double)frontier_in_frustum / (double)myMax((size_t)1, leaves_in_frustum), 1) + "% unfolded\n";
		s += "  Of the in-frustum frontier, " + uInt64ToStringCommaSeparated(frontier_leaves_in_frustum) + " are leaves and " +
			uInt64ToStringCommaSeparated(frontier_in_frustum - frontier_leaves_in_frustum) + " are merged stand-ins\n";

		s += "\n  In-frustum frontier, by why the traversal stopped there:\n";
		{
			std::vector<std::string> labels(FrontierStop_NumReasons);
			std::vector<size_t> counts(FrontierStop_NumReasons);
			for(size_t i=0; i<FrontierStop_NumReasons; ++i)
			{
				labels[i] = stop_reason_labels[i];
				counts[i] = reason_counts[i];
				world_reason_counts[i] += reason_counts[i];
			}
			s += histogramLines("    ", labels, counts);
		}

		s += "  Mean frontier depth below the tree root: " +
			doubleToStringNDecimalPlaces(depth_sum / (double)myMax((size_t)1, frontier_in_frustum), 2) + "\n";

		// Four histograms used to stand here - frontier depth, whole-tree levels, branching factor, and layer_density both
		// of the frontier and of the nodes the traversal expanded.  All four have been read and answered, and none of them
		// moves any decision now:
		//
		// The depth, level and branching histograms described a tree that turns out to be almost entirely unfolded in
		// view (97-98% of drawn nodes are leaves on every viewpoint measured), so its shape above the leaves is not what
		// the cost depends on.  The stop-reason histogram above still says that in one line, which is all that is needed.
		//
		// layer_density answered a question that is now closed: whether max_layer_density has anything to bite on.  It
		// does not - 87.5% of expanded nodes sit in the 1-2 bucket and 0.3% above 16, so the metric does not separate the
		// crowded regions from the ordinary ones and the knob cannot work by construction rather than by tuning.
		//
		// Both are recoverable from history if the cloud or the tree builder changes enough to reopen them.

		//----------------------------- Where the fill actually comes from -----------------------------
		// The half of the report that matters once the hierarchy has been shown to be fully unfolded: the cost of this pass
		// is bytes blended, which is area, not splat count.  Every figure below is the vertex shader's own projection,
		// recomputed here - see splatFootprint().
		{
			const Vec2f focal_len_px((float)viewport_dims.x * scene->lens_sensor_dist / scene->use_sensor_width,
				(float)viewport_dims.y * scene->lens_sensor_dist / scene->use_sensor_height);
			const double viewport_px = (double)viewport_dims.x * (double)viewport_dims.y;

			double total_quad_area = 0, total_ellipse_area = 0, total_alpha_area = 0;
			size_t culled_by_shader = 0, culled_by_size_slice = 0, culled_by_dist_slice = 0, reached_rasteriser = 0;

			std::vector<size_t> area_counts(num_area_buckets, 0), opacity_counts(num_opacity_buckets, 0),
				flatness_counts(num_flatness_buckets, 0), distance_counts(num_distance_buckets, 0);
			std::vector<double> area_areas(num_area_buckets, 0), opacity_areas(num_opacity_buckets, 0),
				flatness_areas(num_flatness_buckets, 0), distance_areas(num_distance_buckets, 0);

			for(size_t i=0; i<frontier.size(); ++i)
			{
				const uint32 idx = frontier[i].cloud_idx;
				const Vec3f& p = cloud.positions[idx];
				if(!pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
					continue;

				const Vec3f& sc = cloud.scales[idx];
				// Adjusted, not stored: the report describes the splats the shader actually draws, and the shader applies
				// this same transform before it sizes the quad - see getAlphaGain(). Feeds the opacity histogram below too,
				// which is the place the effect of the gamma is meant to be read.
				const float opacity = adjustSplatAlpha(cloud.colours[idx][3], splat_alpha_gain, splat_alpha_gamma);

				// The two diagnostic slices, mirrored from gaussian_splat_vert_shader.glsl.  They are the only culls the
				// shader does that splatFootprint() below does not already reproduce, and without them this pass would
				// describe a cloud nobody is looking at whenever either slice is set - which is exactly when someone is
				// looking hardest.  Applied in the order the shader applies them, so the counts attribute the same way.
				const float feature_size = 2.f * myMax(sc.x, myMax(sc.y, sc.z));
				const bool size_clamp_active = (splat_size_clamp_min > 0.f) || (splat_size_clamp_max > 0.f);
				const bool outside_size_range = ((splat_size_clamp_min > 0.f) && (feature_size < splat_size_clamp_min)) ||
					((splat_size_clamp_max > 0.f) && (feature_size > splat_size_clamp_max));
				if(splat_size_clamp_invert ? (size_clamp_active && !outside_size_range) : outside_size_range)
				{
					culled_by_size_slice++;
					continue;
				}

				if(splat_dist_clamp_enabled)
				{
					const Vec4f pos_vs = scene->last_view_matrix * Vec4f(p.x, p.y, p.z, 1.f);
					const float dist_to_cam = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length(); // The shader takes length(pos_vs.xyz), so the w the transform leaves must not be in it.
					const bool inside_dist_range = (dist_to_cam >= splat_dist_clamp_min) && (dist_to_cam <= splat_dist_clamp_max);
					if(splat_dist_clamp_invert ? inside_dist_range : !inside_dist_range)
					{
						culled_by_dist_slice++;
						continue;
					}
				}

				const SplatFootprint fp = splatFootprint(p, sc, cloud.rotations[idx], opacity, scene->last_view_matrix,
					focal_len_px, viewport_dims, splat_alpha_cutoff, splat_ewa_fix_enabled, splat_near_fade_width);
				if(!fp.drawn)
				{
					culled_by_shader++;
					continue;
				}

				reached_rasteriser++;
				total_quad_area    += fp.quad_area_px;
				total_ellipse_area += fp.ellipse_area_px;
				total_alpha_area   += fp.alpha_integral_px;

				// Bucketed by the ellipse area, since that is the part that reaches the blender, and weighted by it too.
				const size_t ab = areaBucket(fp.ellipse_area_px);
				area_counts[ab]++; area_areas[ab] += fp.ellipse_area_px;

				const size_t ob = opacityBucket(opacity);
				opacity_counts[ob]++; opacity_areas[ob] += fp.ellipse_area_px;

				const float s_max = myMax(sc.x, myMax(sc.y, sc.z));
				const float s_min = myMin(sc.x, myMin(sc.y, sc.z));
				const size_t fb = flatnessBucket((s_max > 0.f) ? (s_min / s_max) : 1.f);
				flatness_counts[fb]++; flatness_areas[fb] += fp.ellipse_area_px;

				const size_t db = distanceBucket(cam_pos_ws.getDist(Vec4f(p.x, p.y, p.z, 1.f)));
				distance_counts[db]++; distance_areas[db] += fp.ellipse_area_px;
			}

			s += "\n  In-frustum frontier, screen coverage (vertex shader's own projection, clipped to the viewport):\n";
			s += "    Quads rasterised:      " + leftPad(uInt64ToStringCommaSeparated((uint64)total_quad_area), ' ', 16) + " px = " +
				leftPad(doubleToStringNDecimalPlaces(total_quad_area / viewport_px, 1), ' ', 8) + " layers per viewport pixel\n";
			s += "    Of that, blended:      " + leftPad(uInt64ToStringCommaSeparated((uint64)total_ellipse_area), ' ', 16) + " px = " +
				leftPad(doubleToStringNDecimalPlaces(total_ellipse_area / viewport_px, 1), ' ', 8) + " layers per viewport pixel  (the rest dies on the fragment shader's discard)\n";
			s += "    Sum of alpha:          " + leftPad(uInt64ToStringCommaSeparated((uint64)total_alpha_area), ' ', 16) + "    = " +
				leftPad(doubleToStringNDecimalPlaces(total_alpha_area / viewport_px, 2), ' ', 8) + " per viewport pixel       (compare the overdraw view's mode 2)\n";
			// The quantity the whole saturation argument turns on: a pixel is finished after ln(1 - threshold)/ln(1 - a)
			// layers, so this says how deep into a depth-sorted stack the composite stops being able to change.
			const double mean_alpha = (total_ellipse_area > 0) ? (total_alpha_area / total_ellipse_area) : 0.0;
			s += "    Mean alpha per blended fragment: " + doubleToStringNDecimalPlaces(mean_alpha, 4);
			if(mean_alpha > 0 && mean_alpha < 1)
				s += " - a pixel reaches coverage " + doubleToStringNDecimalPlaces(splat_saturation_threshold, 4) + " after " +
					doubleToStringNDecimalPlaces(std::log(1.0 - splat_saturation_threshold) / std::log(1.0 - mean_alpha), 0) + " layers, against " +
					doubleToStringNDecimalPlaces(total_ellipse_area / viewport_px, 0) + " drawn on an average pixel\n";
			else
				s += "\n";
			// What the panel's per-frame counters cannot say.  They are taken on the CPU, before the draw call, so they count
			// instances issued; everything below happens inside the vertex shader, where nothing on this side can observe
			// it.  This pass reproduces those tests splat by splat for the camera the report was taken at, which is the
			// only exact answer available - and the reason it lives on a button rather than in the frame.
			s += "    Reached the rasteriser: " + uInt64ToStringCommaSeparated(reached_rasteriser) + " of " +
				uInt64ToStringCommaSeparated(reached_rasteriser + culled_by_shader + culled_by_size_slice + culled_by_dist_slice) + " in frustum" +
				", size slice cut " + uInt64ToStringCommaSeparated(culled_by_size_slice) +
				", distance slice cut " + uInt64ToStringCommaSeparated(culled_by_dist_slice) + "\n";
			s += "    Culled by the shader before rasterising: " + uInt64ToStringCommaSeparated(culled_by_shader) +
				" (behind the near plane, degenerate, or opacity at or below alpha_cutoff " + doubleToStringNDecimalPlaces(splat_alpha_cutoff, 4) + ")\n";
			s += "    Note these are viewport-wide averages: the splats occupy part of the screen, so the dense regions the\n";
			s += "    overdraw view shows in red are several times these numbers.\n";

			//----------------------------- The biggest quads, in frustum or not -----------------------------
			// A second pass over the same frontier, deliberately without the in-frustum test the loop above starts with.
			//
			// That test is right for the coverage figures - they describe what the frame is made of - but it hides the case
			// this section exists for: the projection here is an *affine* approximation, valid near the view axis, and at a
			// large angle off it the Jacobian's focal*v/depth^2 term diverges.  Such a splat's centre lands off screen, so the
			// frustum test drops it, yet its quad is held only by the 2x viewport radius clamp and can reach back into the
			// frame as a screen-filling slab.  Listing the largest quads with their off-axis angle, and saying outright which
			// ones are sitting on the clamp, is what separates "legitimately large" from "the approximation gave up".
			{
				struct BigQuad
				{
					float radius1_px, radius2_px, bound_px, quad_area_px, off_axis_deg, dist_m, feature_size_m;
					bool at_clamp, centre_on_screen;
				};

				const size_t max_listed = 20;
				std::vector<BigQuad> biggest; // Kept sorted, largest first.  N is 20, so an insertion sort over it costs nothing against the projection work per splat.
				// Ranked by radius1 instead of by area, because the two find different things and only one of them is what an
				// eye picks out of the frame: a long thin streak, 900 px by 5, is among the most conspicuous quads on screen
				// and among the smallest by area, so the list above can never show one however far it is scrolled.
				std::vector<BigQuad> longest;
				size_t num_at_clamp = 0, num_offscreen_centre_reaching_frame = 0;

				for(size_t i=0; i<frontier.size(); ++i)
				{
					const uint32 idx = frontier[i].cloud_idx;
					const Vec3f& p = cloud.positions[idx];
					const Vec3f& sc = cloud.scales[idx];
					const float opacity = adjustSplatAlpha(cloud.colours[idx][3], splat_alpha_gain, splat_alpha_gamma);

					const SplatFootprint fp = splatFootprint(p, sc, cloud.rotations[idx], opacity, scene->last_view_matrix,
						focal_len_px, viewport_dims, splat_alpha_cutoff, splat_ewa_fix_enabled, splat_near_fade_width);
					if(!fp.drawn)
						continue;

					const Vec4f pos_vs = scene->last_view_matrix * Vec4f(p.x, p.y, p.z, 1.f);
					const float depth = -pos_vs[2];
					const float lateral = Vec4f(pos_vs[0], pos_vs[1], 0, 0).length();
					const float off_axis_deg = (float)(std::atan2((double)lateral, (double)myMax(depth, 1.0e-6f)) * 180.0 / Maths::pi<double>());

					// Where the quad's centre lands, in pixels from the middle of the viewport.  A centre off screen whose quad
					// still covers screen pixels is precisely the pathological case; counted separately from the clamp, since
					// either can happen without the other.
					const float centre_x_px = focal_len_px.x * pos_vs[0] / myMax(depth, 1.0e-6f);
					const float centre_y_px = focal_len_px.y * pos_vs[1] / myMax(depth, 1.0e-6f);
					const bool centre_on_screen = (std::fabs(centre_x_px) <= (float)viewport_dims.x * 0.5f) && (std::fabs(centre_y_px) <= (float)viewport_dims.y * 0.5f);

					const bool at_clamp = (fp.radius1_px >= fp.max_radius_px * 0.999f);
					if(at_clamp)
						num_at_clamp++;
					if(!centre_on_screen && (fp.quad_area_px > 0.f))
						num_offscreen_centre_reaching_frame++;

					const bool room_by_area   = !((biggest.size() >= max_listed) && (fp.quad_area_px <= biggest.back().quad_area_px));
					const bool room_by_length = !((longest.size() >= max_listed) && (fp.radius1_px  <= longest.back().radius1_px));
					if(!room_by_area && !room_by_length)
						continue;

					BigQuad q;
					q.radius1_px = fp.radius1_px;
					q.radius2_px = fp.radius2_px;
					q.bound_px = fp.max_radius_px;
					q.quad_area_px = fp.quad_area_px;
					q.off_axis_deg = off_axis_deg;
					q.dist_m = cam_pos_ws.getDist(Vec4f(p.x, p.y, p.z, 1.f));
					q.feature_size_m = 2.f * myMax(sc.x, myMax(sc.y, sc.z));
					q.at_clamp = at_clamp;
					q.centre_on_screen = centre_on_screen;

					if(room_by_area)
					{
						size_t ins = biggest.size();
						while((ins > 0) && (biggest[ins - 1].quad_area_px < q.quad_area_px))
							ins--;
						biggest.insert(biggest.begin() + ins, q);
						if(biggest.size() > max_listed)
							biggest.pop_back();
					}

					if(room_by_length)
					{
						size_t ins = longest.size();
						while((ins > 0) && (longest[ins - 1].radius1_px < q.radius1_px))
							ins--;
						longest.insert(longest.begin() + ins, q);
						if(longest.size() > max_listed)
							longest.pop_back();
					}
				}

				s += "\n  Worst quads on this frontier (screen-centre test deliberately not applied - see the code):\n";
				s += "    Sitting on the radius bound: " + uInt64ToStringCommaSeparated(num_at_clamp) +
					".  Centre off screen but still painting pixels: " + uInt64ToStringCommaSeparated(num_offscreen_centre_reaching_frame) + ".\n";

				const std::vector<BigQuad>* const lists[2] = { &biggest, &longest };
				const char* const list_names[2] = { "  By area:\n", "  By length (radius1) - a long thin streak is conspicuous on screen and tiny by area:\n" };
				for(int list_i=0; list_i<2; ++list_i)
				{
					s += std::string("\n  ") + list_names[list_i];
					s += "         radius1      radius2        bound   clipped area    off-axis    distance    own size\n";
					const std::vector<BigQuad>& list = *lists[list_i];
					for(size_t i=0; i<list.size(); ++i)
					{
						const BigQuad& q = list[i];
						s += "    " + leftPad(doubleToStringNDecimalPlaces(q.radius1_px, 0), ' ', 10) + " px" +
							leftPad(doubleToStringNDecimalPlaces(q.radius2_px, 0), ' ', 9) + " px" +
							leftPad(doubleToStringNDecimalPlaces(q.bound_px, 0), ' ', 9) + " px" +
							leftPad(uInt64ToStringCommaSeparated((uint64)q.quad_area_px), ' ', 14) + " px" +
							leftPad(doubleToStringNDecimalPlaces(q.off_axis_deg, 1), ' ', 10) + " deg" +
							leftPad(doubleToStringNDecimalPlaces(q.dist_m, 2), ' ', 10) + " m" +
							leftPad(doubleToStringNDecimalPlaces(q.feature_size_m, 3), ' ', 10) + " m" +
							(q.at_clamp ? "   AT CLAMP" : "") + (q.centre_on_screen ? "" : "   CENTRE OFF SCREEN") + "\n";
					}
				}
				s += "    AT CLAMP means radius1 is the bound rather than a projection, i.e. the affine approximation asked for\n";
				s += "    more than the splat's own size and distance can justify and was refused.  A row with radius1 well\n";
				s += "    under the bound is honest geometry: that splat really is that big on screen.\n";
			}

			s += "\n  Fill by splat size (blended area of one splat):\n";
			{
				std::vector<std::string> labels(num_area_buckets);
				for(size_t i=0; i<num_area_buckets; ++i) labels[i] = area_bucket_labels[i];
				s += weightedHistogramLines("    ", labels, area_counts, area_areas);
			}

			s += "\n  Fill by opacity (the blur-splat signature is fill concentrated in the low buckets):\n";
			{
				std::vector<std::string> labels(num_opacity_buckets);
				for(size_t i=0; i<num_opacity_buckets; ++i) labels[i] = opacity_bucket_labels[i];
				s += weightedHistogramLines("    ", labels, opacity_counts, opacity_areas);
			}

			s += "\n  Fill by flatness (smallest scale axis over largest):\n";
			{
				std::vector<std::string> labels(num_flatness_buckets);
				for(size_t i=0; i<num_flatness_buckets; ++i) labels[i] = flatness_bucket_labels[i];
				s += weightedHistogramLines("    ", labels, flatness_counts, flatness_areas);
			}

			s += "\n  Fill by distance from the camera:\n";
			{
				std::vector<std::string> labels(num_distance_buckets);
				for(size_t i=0; i<num_distance_buckets; ++i) labels[i] = distance_bucket_labels[i];
				s += weightedHistogramLines("    ", labels, distance_counts, distance_areas);
			}

			world_quad_area += total_quad_area;
			world_ellipse_area += total_ellipse_area;
			world_reached_rasteriser += reached_rasteriser;
			world_in_frustum_before_shader += reached_rasteriser + culled_by_shader + culled_by_size_slice + culled_by_dist_slice;

			// Set by the pruning block below and printed by the multi-view section after it - the two are separate blocks,
			// but this is what has to cross between them.
			std::string importance_discard_note, importance_state_note;

			//----------------------------- The ceiling on pruning by importance -----------------------------
			// The histograms above say where the fill is; this says how much of it is doing nothing.
			//
			// Composites the frontier in the order it is actually drawn - scratch->selected_indices, nearest first - into a
			// low-resolution transmittance buffer, and charges each splat with what it actually adds to the image:
			// sum over the pixels it covers of alpha * (transmittance still remaining in front of it).  A splat behind a
			// region the composite has already finished with scores zero however large and however opaque it is.
			//
			// This is the ceiling, not a plan: it is measured from one camera, and a splat invisible from here may be the
			// front layer from somewhere else, which is exactly why the published methods sample many viewpoints.  What it
			// answers is whether pruning by importance is worth building at all, before any of that machinery exists - and
			// it produces the per-splat importance that machinery would need, so the measurement is also the first piece of
			// it rather than scaffolding to be thrown away.
			//
			// A software rasteriser rather than a GPU pass because the numbers wanted are per splat, which on the GPU needs
			// atomics that WebGL2 does not have; at an eighth of the resolution the whole thing is a few tens of millions of
			// operations, and the answer wanted is a ratio, which the resolution does not move.
			{
				const int downscale = 8;
				const int lo_w = myMax(1, (viewport_dims.x + downscale - 1) / downscale);
				const int lo_h = myMax(1, (viewport_dims.y + downscale - 1) / downscale);
				const float lo_scale = 1.f / (float)downscale;

				js::Vector<float, 16> transmittance(lo_w * lo_h);
				for(size_t i=0; i<transmittance.size(); ++i)
					transmittance[i] = 1.f;

				// One entry per in-frustum selected splat.  Splats outside the frustum are composited too - they occlude
				// what is behind them - but are left out of the curve, since their score here says nothing about their
				// worth from a camera that can see them.
				struct PruneCandidate
				{
					float contribution; // Alpha it actually added to the image, in low-res pixel units.
					float fill;         // Full-res pixels it blends, i.e. what dropping it would save.
				};
				js::Vector<PruneCandidate, 16> candidates;
				candidates.reserve(frontier_in_frustum);

				double total_contribution = 0, hidden_fill = 0;
				size_t hidden_count = 0;

				// Fold this view into the cloud's running per-view record - see SplatCloud::importance_best_contribution.
				// Thrown away rather than reconciled when the cloud has been renumbered underneath it, since the indices
				// would silently refer to different splats.  Never silently, though: a discard costs the user however many
				// viewpoints they had already visited, and the first version of this said nothing at all when it happened.
				const uint64 layout = cloudLayoutFingerprint(cloud);

				// Printed unconditionally, not just when something looks wrong.  A run has already been seen where the
				// record went from one view to none between two reports without either the reset button or the discard
				// below being able to account for it, and reading the code did not explain it - so the state it depends
				// on is now stated outright every time rather than inferred from the run counter afterwards.
				importance_state_note = "    [state] views held on entry " + toString(cloud.importance_num_views) +
					", array " + toString(cloud.importance_best_contribution.size()) + " vs " + toString(cloud.total_splats) + " splats" +
					", layout " + toHexString(layout) + " vs held " + toHexString(cloud.importance_layout_fingerprint) +
					", cloud id " + toString(cloud.cloud_id) + ", topology gen " + toString(cloud.topology_generation) + "\n";

				if(cloud.importance_best_contribution.size() != cloud.total_splats || cloud.importance_layout_fingerprint != layout)
				{
					if(cloud.importance_num_views > 0)
						importance_discard_note = "    The cloud was renumbered since the last run, so the " + toString(cloud.importance_num_views) +
							" run(s) recorded before it had to be discarded - their indices no longer mean the same splats.\n";

					cloud.importance_best_contribution.resizeNoCopy(cloud.total_splats);
					cloud.importance_sum_fill.resizeNoCopy(cloud.total_splats);
					cloud.importance_views_drawn.resizeNoCopy(cloud.total_splats);
					for(size_t i=0; i<cloud.total_splats; ++i)
					{
						cloud.importance_best_contribution[i] = 0.f;
						cloud.importance_sum_fill[i] = 0.f;
						cloud.importance_views_drawn[i] = 0;
					}
					cloud.importance_layout_fingerprint = layout;
					cloud.importance_num_views = 0;
				}
				cloud.importance_num_views++;

				const js::Vector<uint32, 16>& draw_order = scratch->selected_indices;
				for(size_t i=0; i<draw_order.size(); ++i)
				{
					const uint32 idx = draw_order[i];
					const Vec3f& p = cloud.positions[idx];
					const SplatFootprint fp = splatFootprint(p, cloud.scales[idx], cloud.rotations[idx],
						adjustSplatAlpha(cloud.colours[idx][3], splat_alpha_gain, splat_alpha_gamma), // As drawn, not as stored - see getAlphaGain().
						scene->last_view_matrix, focal_len_px, viewport_dims, splat_alpha_cutoff, splat_ewa_fix_enabled, splat_near_fade_width);
					if(!fp.drawn)
						continue;

					// Axis-aligned bound of the ellipse, in low-res pixels.
					const float ext_x = std::fabs(fp.radius1_px * fp.axis1.x) + std::fabs(fp.radius2_px * fp.axis2.x);
					const float ext_y = std::fabs(fp.radius1_px * fp.axis1.y) + std::fabs(fp.radius2_px * fp.axis2.y);
					const int x0 = myMax(0, (int)std::floor((fp.centre_x_px - ext_x) * lo_scale));
					const int x1 = myMin(lo_w - 1, (int)std::ceil((fp.centre_x_px + ext_x) * lo_scale));
					const int y0 = myMax(0, (int)std::floor((fp.centre_y_px - ext_y) * lo_scale));
					const int y1 = myMin(lo_h - 1, (int)std::ceil((fp.centre_y_px + ext_y) * lo_scale));

					const float inv_s1_sq = 1.f / myMax(fp.sigma1_px * fp.sigma1_px, 1.0e-8f);
					const float inv_s2_sq = 1.f / myMax(fp.sigma2_px * fp.sigma2_px, 1.0e-8f);

					float contribution = 0;
					for(int y=y0; y<=y1; ++y)
					{
						const float py = ((float)y + 0.5f) * (float)downscale;
						for(int x=x0; x<=x1; ++x)
						{
							float& T = transmittance[y * lo_w + x];
							if(T < 1.0e-3f)
								continue; // Finished pixel - nothing here can change it, which is the whole point of the measurement.

							const float px = ((float)x + 0.5f) * (float)downscale;
							const float dx = px - fp.centre_x_px, dy = py - fp.centre_y_px;
							const float u = dx * fp.axis1.x + dy * fp.axis1.y;
							const float v = dx * fp.axis2.x + dy * fp.axis2.y;
							const float exponent = 0.5f * (u*u*inv_s1_sq + v*v*inv_s2_sq);
							if(exponent > 8.f)
								continue; // Beyond about 4 sigma; also past the quad the shader would have drawn.

							const float alpha = fp.opacity * std::exp(-exponent);
							if(alpha < splat_alpha_cutoff)
								continue; // The fragment shader discards these, so they cost no blend and add nothing.

							contribution += alpha * T;
							T *= (1.f - alpha);
						}
					}

					// Recorded before the in-frustum test below, not after: a splat whose centre is off screen can still
					// paint pixels, and "did it ever matter" has to count that as mattering.
					cloud.importance_best_contribution[idx] = myMax(cloud.importance_best_contribution[idx], contribution);
					cloud.importance_sum_fill[idx] += fp.ellipse_area_px;
					cloud.importance_views_drawn[idx]++;

					if(!pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
						continue;

					PruneCandidate cand;
					cand.contribution = contribution;
					cand.fill = fp.ellipse_area_px;
					candidates.push_back(cand);
					total_contribution += contribution;
					if(contribution <= 0.f)
					{
						hidden_count++;
						hidden_fill += fp.ellipse_area_px;
					}
				}

				s += "\n  Pruning ceiling, measured from this camera only (software composite at 1/" + toString(downscale) +
					" resolution, " + toString(lo_w) + " x " + toString(lo_h) + "):\n";

				// Self-check on the software rasteriser, without which none of the figures below may be believed.  Every
				// unit of contribution is one unit of coverage taken from a pixel that had it, so the total is the covered
				// area of the composite - which for a capture filling the view must come out near the pixel count.  A
				// rasteriser that is silently missing its splats, or placing them off screen, reports a few percent here
				// while every other number in this section still looks entirely reasonable.
				{
					double final_coverage = 0;
					for(size_t i=0; i<transmittance.size(); ++i)
						final_coverage += 1.0 - transmittance[i];
					s += "    Self-check: composite covered " + doubleToStringNDecimalPlaces(100.0 * final_coverage / (double)(lo_w * lo_h), 1) +
						"% of the low-res buffer (should be close to how much of the screen the cloud visibly fills; a few percent means this pass is broken)\n";
				}

				s += "    Splats that add nothing at all (entirely behind finished pixels): " + uInt64ToStringCommaSeparated(hidden_count) +
					" = " + doubleToStringNDecimalPlaces(100.0 * (double)hidden_count / (double)myMax((size_t)1, candidates.size()), 1) + "% of them, carrying " +
					doubleToStringNDecimalPlaces((total_ellipse_area > 0) ? (100.0 * hidden_fill / total_ellipse_area) : 0.0, 1) + "% of the fill\n";

				// Ordered by contribution per pixel of fill, not by contribution: the question is what to drop for the least
				// damage per unit of work saved, and a large faint splat and a small bright one are not comparable on
				// contribution alone.
				struct PruneCandidateLess
				{
					inline bool operator () (const PruneCandidate& a, const PruneCandidate& b) const
					{
						return (a.contribution * b.fill) < (b.contribution * a.fill); // a.contribution/a.fill < b.contribution/b.fill, without dividing by a fill that could be 0.
					}
				};
				std::sort(candidates.data(), candidates.data() + candidates.size(), PruneCandidateLess());

				const double energy_targets[] = { 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1 };
				const size_t num_energy_targets = staticArrayNumElems(energy_targets);

				s += "    Dropping the least useful splats first, by contribution per pixel of fill:\n";
				s += "      image energy lost |   splats dropped   |  fill removed\n";
				{
					size_t ci = 0;
					double lost = 0, fill_removed = 0;
					for(size_t t=0; t<num_energy_targets; ++t)
					{
						const double budget = energy_targets[t] * total_contribution;
						while(ci < candidates.size() && (lost + candidates[ci].contribution) <= budget)
						{
							lost += candidates[ci].contribution;
							fill_removed += candidates[ci].fill;
							ci++;
						}
						s += "      " + leftPad(doubleToStringNDecimalPlaces(energy_targets[t] * 100.0, 2), ' ', 16) + "% | " +
							leftPad(uInt64ToStringCommaSeparated(ci), ' ', 11) + " " +
							leftPad(doubleToStringNDecimalPlaces(100.0 * (double)ci / (double)myMax((size_t)1, candidates.size()), 1), ' ', 5) + "% | " +
							leftPad(doubleToStringNDecimalPlaces((total_ellipse_area > 0) ? (100.0 * fill_removed / total_ellipse_area) : 0.0, 1), ' ', 8) + "%\n";
					}
				}
				s += "    A splat scored here from one camera may be the front layer from another, so this bounds what\n";
				s += "    pruning could win, and does not license dropping these particular splats.\n";
			}

			//------------------------- The same ceiling, over every viewpoint recorded so far -------------------------
			// What the section above cannot answer on its own.  Each run of this report folds its composite into the
			// cloud's running record, so after visiting a handful of viewpoints the question becomes "which splats earned
			// nothing from *any* of them", which is the figure an offline pruner would actually act on.
			//
			// Kept separate from the single-camera section rather than replacing it: the single-camera numbers are what
			// the fill measurements above are consistent with, and they stay comparable between sessions.
			{
				s += "\n  Importance accumulated over " + toString(cloud.importance_num_views) +
					((cloud.importance_num_views == 1) ? " report run" : " report runs") + " (this one included; \"Reset importance\" clears it):\n";
				s += importance_state_note;
				s += importance_discard_note;

				if(cloud.importance_num_views < 2)
					s += "    One run only, so this says nothing the section above does not. Move the camera and run it again.\n";
				else
					s += "    Runs, not distinct viewpoints - the merge tolerances do not touch the composite, so a second run\n"
						 "    from the same spot to sweep them adds nothing here but is counted. One run per viewpoint is enough.\n";

				// Over leaves, not frontier nodes: merging is an operation on the cloud, and a merged stand-in is LoD
				// scaffolding rather than something an offline pass would ever delete.
				size_t num_leaves = 0, never_drawn = 0, drawn_but_dead = 0;
				double dead_mean_fill = 0, total_mean_fill = 0, total_best = 0;

				struct MultiViewCandidate
				{
					float best;      // Best contribution over the recorded views.
					float mean_fill; // Mean fill over the views that drew it, i.e. what dropping it saves on a typical one.
				};
				js::Vector<MultiViewCandidate, 16> mv_candidates;
				mv_candidates.reserve(cloud.total_splats);

				for(size_t m=0; m<cloud.members.size(); ++m)
				{
					const std::vector<GaussianSplatLodNode>& tree = cloud.members[m].splat_data->lod_tree;
					const size_t member_offset = cloud.members[m].offset;
					const size_t n = tree.empty() ? cloud.members[m].count : tree.size();
					for(size_t i=0; i<n; ++i)
					{
						if(!tree.empty() && tree[i].child_count != 0)
							continue; // Merged stand-in, not an original splat.

						const uint32 idx = (uint32)(member_offset + i);
						num_leaves++;

						const uint32 views = cloud.importance_views_drawn[idx];
						if(views == 0)
						{
							never_drawn++;
							continue; // No fill recorded either, so it contributes nothing to the mean-fill totals.
						}

						const float mean_fill = cloud.importance_sum_fill[idx] / (float)views;
						const float best = cloud.importance_best_contribution[idx];
						total_mean_fill += mean_fill;
						total_best += best;
						if(best <= 0.f)
						{
							drawn_but_dead++;
							dead_mean_fill += mean_fill;
						}

						MultiViewCandidate cand;
						cand.best = best;
						cand.mean_fill = mean_fill;
						mv_candidates.push_back(cand);
					}
				}

				const double inv_leaves = 100.0 / (double)myMax((size_t)1, num_leaves);
				s += "    Never selected by the traversal in any recorded view: " + leftPad(uInt64ToStringCommaSeparated(never_drawn), ' ', 11) +
					" = " + doubleToStringNDecimalPlaces((double)never_drawn * inv_leaves, 1) + "% of leaves\n";
				s += "    Drawn, but never added a pixel in any of them:        " + leftPad(uInt64ToStringCommaSeparated(drawn_but_dead), ' ', 11) +
					" = " + doubleToStringNDecimalPlaces((double)drawn_but_dead * inv_leaves, 1) + "%\n";
				s += "    Dead from every viewpoint recorded (the two above):   " + leftPad(uInt64ToStringCommaSeparated(never_drawn + drawn_but_dead), ' ', 11) +
					" = " + doubleToStringNDecimalPlaces((double)(never_drawn + drawn_but_dead) * inv_leaves, 1) + "%, carrying " +
					doubleToStringNDecimalPlaces((total_mean_fill > 0) ? (100.0 * dead_mean_fill / total_mean_fill) : 0.0, 1) + "% of the mean fill\n";
				s += "    A leaf standing behind a merged LoD ancestor in some view counts as not drawn there, so the first\n";
				s += "    line is an over-count by however much the tree substitutes - the frontier's leaf share above says\n";
				s += "    how much that is. Lower pixel_scale_limit to take it out of the picture entirely.\n";

				if(!mv_candidates.empty())
				{
					// Ordered by best contribution per mean pixel of fill, the multi-view form of the single-camera
					// ordering above.
					struct MultiViewLess
					{
						inline bool operator () (const MultiViewCandidate& a, const MultiViewCandidate& b) const
						{
							return (a.best * b.mean_fill) < (b.best * a.mean_fill);
						}
					};
					std::sort(mv_candidates.data(), mv_candidates.data() + mv_candidates.size(), MultiViewLess());

					const double energy_targets[] = { 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1 };
					const size_t num_energy_targets = staticArrayNumElems(energy_targets);

					s += "    Dropping the least useful splats first, by best contribution per mean pixel of fill:\n";
					s += "      best-view energy lost |   splats dropped   |  mean fill removed\n";
					size_t ci = 0;
					double lost = 0, fill_removed = 0;
					for(size_t t=0; t<num_energy_targets; ++t)
					{
						const double budget = energy_targets[t] * total_best;
						while(ci < mv_candidates.size() && (lost + mv_candidates[ci].best) <= budget)
						{
							lost += mv_candidates[ci].best;
							fill_removed += mv_candidates[ci].mean_fill;
							ci++;
						}
						s += "      " + leftPad(doubleToStringNDecimalPlaces(energy_targets[t] * 100.0, 2), ' ', 20) + "% | " +
							leftPad(uInt64ToStringCommaSeparated(ci), ' ', 11) + " " +
							leftPad(doubleToStringNDecimalPlaces(100.0 * (double)ci / (double)myMax((size_t)1, mv_candidates.size()), 1), ' ', 5) + "% | " +
							leftPad(doubleToStringNDecimalPlaces((total_mean_fill > 0) ? (100.0 * fill_removed / total_mean_fill) : 0.0, 1), ' ', 8) + "%\n";
					}
					s += "    \"Energy\" here is the sum over splats of their single best view, not any one image's energy:\n";
					s += "    a splat is judged by its best showing, which is the conservative way round for deleting it.\n";
					s += "    Still a lower bound on what matters - it only knows the viewpoints actually visited.\n";
				}
			}

		}

		free_traversal_scratch.push_back(scratch); // Everything that needed the traversal's own buffers is done with them.

		world_frontier += frontier.size();
		world_frontier_in_frustum += frontier_in_frustum;
		world_leaves_in_frustum += leaves_in_frustum;
		world_frontier_leaves_in_frustum += frontier_leaves_in_frustum;
	}

	if(clouds.size() > max_clouds_to_report)
		s += "\n(" + toString(clouds.size() - max_clouds_to_report) + " further clouds not reported)\n";

	if(clouds.size() > 1)
	{
		s += "\n--- world total ---\n";
		s += "  Frontier " + uInt64ToStringCommaSeparated(world_frontier) + " nodes, " + uInt64ToStringCommaSeparated(world_frontier_in_frustum) + " in frustum, of which " +
			uInt64ToStringCommaSeparated(world_frontier_leaves_in_frustum) + " leaves\n";
		s += "  Leaves in frustum " + uInt64ToStringCommaSeparated(world_leaves_in_frustum) + " = " +
			doubleToStringNDecimalPlaces(100.0 * (double)world_frontier_in_frustum / (double)myMax((size_t)1, world_leaves_in_frustum), 1) + "% unfolded\n";
		s += "  Fill: " + uInt64ToStringCommaSeparated((uint64)world_quad_area) + " px rasterised, " +
			uInt64ToStringCommaSeparated((uint64)world_ellipse_area) + " px blended\n";
		std::vector<std::string> labels(FrontierStop_NumReasons);
		std::vector<size_t> counts(FrontierStop_NumReasons);
		for(size_t i=0; i<FrontierStop_NumReasons; ++i)
		{
			labels[i] = stop_reason_labels[i];
			counts[i] = world_reason_counts[i];
		}
		s += histogramLines("    ", labels, counts);
	}

	s += mergeColourByDepthReport();

	s += "\nReport took " + doubleToStringNDecimalPlaces(timer.elapsed() * 1.0e3, 1) + " ms total (one full traversal per cloud, on the main thread), of which " +
		doubleToStringNDecimalPlaces(world_traversal_ms, 1) + " ms was traversal itself (see \"Traversal:\" per cloud above) - the rest is this report's own stats/string-building overhead, which the real per-frame path never pays.\n";
	// Kept for the diagnostics panel - see the members' comment.  Set from the world-wide roll-up whether or not the
	// per-world section above was printed, since that one is only printed when there is more than one cloud.
	last_report_reached_rasteriser = world_reached_rasteriser;
	last_report_in_frustum = world_in_frustum_before_shader;

	return s;
}


bool GaussianSplatRenderer::isValidHandle(Handle handle) const
{
	return handle_to_cloud.count(handle) != 0;
}


std::string GaussianSplatRenderer::getDiagnostics() const
{
	size_t num_merged_clouds = 0, largest_cloud_splats = 0, total_splats = 0;
	uint64 tex_bytes = 0, index_vbo_bytes = 0;
	for(size_t i=0; i<clouds.size(); ++i)
	{
		const SplatCloud& cloud = *clouds[i];
		total_splats += cloud.total_splats;
		largest_cloud_splats = myMax(largest_cloud_splats, cloud.total_splats);
		if(cloud.members.size() > 1)
			num_merged_clouds++;

		// Both are sized to the cloud's capacity, not its splat count: 4 RGBA32F texels and one uint32 index per splat.
		tex_bytes       += (uint64)cloud.gpu_capacity_splats * texels_per_splat * 4 * sizeof(float);
		index_vbo_bytes += (uint64)cloud.gpu_capacity_splats * sizeof(uint32);
	}

	uint64 scratch_bytes = 0;
	for(size_t i=0; i<free_scratch.size(); ++i)
	{
		const GaussianSplatSortScratch& s = *free_scratch[i];
		scratch_bytes += (uint64)(s.positions_snapshot.size() * sizeof(Vec3f) + s.items.size() * sizeof(GaussianSplatSortScratch::SortItem) +
			s.working_space.size() * sizeof(GaussianSplatSortScratch::SortItem) + s.coarse_indices.size() * sizeof(uint32) +
			s.precise_indices.size() * sizeof(uint32) + s.temp_counts.size() * sizeof(uint32));
	}

	uint64 traversal_scratch_bytes = 0;
	for(size_t i=0; i<free_traversal_scratch.size(); ++i)
	{
		const GaussianSplatLodTraversalScratch& s = *free_traversal_scratch[i];
		// SESSION058: s.geom is now a Reference<> into a per-cloud cache (see GaussianSplatCachedGeom) that may be shared
		// by more than one pooled scratch here, so this can overcount the geom bytes when that sharing happens - a
		// diagnostic-only imprecision, not a correctness issue (nothing here frees or double-counts an owned allocation).
		const uint64 geom_bytes = s.geom.isNull() ? 0 : (uint64)(s.geom->positions.size() * sizeof(Vec3f) + s.geom->feature_size.size() * sizeof(float) + s.geom->cull_radius.size() * sizeof(float)); // SESSION059: + cull_radius.
		traversal_scratch_bytes += geom_bytes + (uint64)(s.selected_indices.size() * sizeof(uint32));
	}

	// Total leaf count (pre-merge) vs total node count (leaves + merged, what's actually GPU-resident) across every
	// LoD-active cloud - the ratio is what the plan's session notes call out as an expected ~1.3x-1.8x, a sanity check
	// that the tree building/upload path is behaving, not a per-cloud figure.
	size_t total_leaves_lod_active = 0, total_nodes_lod_active = 0;
	for(size_t i=0; i<clouds.size(); ++i)
	{
		const SplatCloud& cloud = *clouds[i];
		if(!cloudHasLodTree(cloud))
			continue;
		total_nodes_lod_active += cloud.total_splats;
		for(size_t m=0; m<cloud.members.size(); ++m)
			total_leaves_lod_active += cloud.members[m].splat_data->numSplats();
	}

	std::string s;

	// Splats are fill-rate bound, so the pixel count is as much a part of "how much work is this" as the splat count -
	// and it is the first thing to check before comparing two clients' frame rates against each other.  The accumulation
	// buffer's own size is listed separately because it, not the viewport, is what the splats actually shade into: the
	// two agree today, and saying so out loud is how a future divergence gets noticed.
	const int viewport_w = opengl_engine->getViewPortWidth();
	const int viewport_h = opengl_engine->getViewPortHeight();
	s += "Viewport: " + toString(viewport_w) + " x " + toString(viewport_h) + " (" +
		doubleToStringNDecimalPlaces((double)viewport_w * viewport_h * 1.0e-6, 2) + " Mpixel)";
	if(opengl_engine->getCurrentScene()->splat_accum_renderbuffer.nonNull())
		s += ", accumulation buffer " + toString(opengl_engine->getCurrentScene()->splat_accum_renderbuffer->xRes()) + " x " +
			toString(opengl_engine->getCurrentScene()->splat_accum_renderbuffer->yRes());
	s += "\n";

	s += "Splat objects: " + toString(handle_to_cloud.size()) + "\n";
	s += "Drawable clouds: " + toString(clouds.size()) + " (" + toString(num_merged_clouds) + " merged)\n";
	s += "Clouds drawn last frame: " + toString(opengl_engine->last_num_splat_clouds_drawn) + "\n";
	s += "Splats: " + uInt64ToStringCommaSeparated(total_splats) + " total, " + uInt64ToStringCommaSeparated(largest_cloud_splats) + " in largest cloud\n";
	s += "Splats drawn last frame: " + uInt64ToStringCommaSeparated(opengl_engine->last_num_splats_drawn) +
		" (counted before the vertex shader, so culling that happens on GPU-side is not subtracted)\n";
	// The number the line above cannot give: what survived the shader's own culls - the distance and size slices, the near
	// plane, the alpha cutoff.  Nothing on the CPU can observe those while they happen, so this is what the last Frustum
	// report measured by repeating them splat by splat, and it describes the camera that report was taken at, not now.
	// Says which of the three things sharing the mask is actually using it, since only one can: a cap that is set while
	// Clip is ticked is doing nothing, and there is no way to tell that from the cap's own value.
	s += "Layer cap: " + ((splat_layer_cap > 0) ?
		(toString(splat_layer_cap) + " layers, " + (splat_hide_test_conservative ? "conservative test" : "centre test") +
			// getHideMode() == 0 is "Clip is off", i.e. the cap has the mask to itself.  (This read == 3, which getHideMode()
			// never returns, so every set cap was reported as not running - same slip as in getFrustumStructureReport().)
			((getHideMode() == 0) ? std::string() : std::string(" - NOT RUNNING: Clip owns the mask"))) :
		std::string("off")) + "\n";
	if(!splat_layer_estimate_result.empty())
		s += "Layer cap estimate: " + splat_layer_estimate_result + "\n"; // DIAGNOSTIC ONLY - what the last "Estimate" press measured, see OpenGLEngine::estimateSplatLayerCapSaving().
	s += "Reached the rasteriser: " + ((last_report_in_frustum == 0) ? std::string("not measured - press \"Frustum report\"") :
		(uInt64ToStringCommaSeparated(last_report_reached_rasteriser) + " of " + uInt64ToStringCommaSeparated(last_report_in_frustum) +
		" in frustum, as of the last Frustum report")) + "\n";
	// Draw calls rather than slices: a cloud with fewer splats than slices leaves some empty, so the two only agree
	// when there is something to draw in every one.
	s += "Draw slices: " + toString(splat_num_draw_slices) + " (" + toString(opengl_engine->last_num_splat_draw_calls) + " draw calls last frame)" +
		((splat_slice_growth == 1.f) ? std::string(", equal sizes") : (", growth " + doubleToStringNDecimalPlaces(splat_slice_growth, 2) + "x per slice")) + "\n";
	// Says why it isn't running, not just that it isn't: every reason below leaves the picture correct and the
	// optimisation silently absent, which is the hardest kind of thing to notice.
	// DIAGNOSTIC ONLY - see getCoverageCap().  Same "say why it is not running" treatment as the gate below, and for the
	// same reason: every way it can fail to run leaves a correct picture and a silently absent cut.
	std::string cov_cap_state;
	if(splat_coverage_cap <= 0.f)
		cov_cap_state = "off";
	else if(splat_layer_cap > 0)
		cov_cap_state = "set, but switched off by the layer cap above, which owns the same mask";
	else if(getHideMode() != 0)
		cov_cap_state = "set, but switched off by the hide-overdraw diagnostic, which owns the same mask";
	else if(splat_num_draw_slices <= 1)
		cov_cap_state = "set, but idle - needs more than one draw slice to cut between";
	else if(splat_show_overdraw_mode != 0)
		cov_cap_state = "set, but idle - not used in the overdraw views, where accumulated alpha counts layers rather than coverage";
	else if(!opengl_engine->splat_accum_gate_available)
		cov_cap_state = "set, but UNAVAILABLE - the saturation mask framebuffer could not be built";
	else if(saturation_mask_prog.isNull() || !saturation_mask_prog->isBuilt() || mask_reduce_prog.isNull() || !mask_reduce_prog->isBuilt())
		cov_cap_state = "set, but idle - mask shaders not built yet";
	else
		cov_cap_state = "on, cutting fragments where coverage has reached " + doubleToStringNDecimalPlaces(splat_coverage_cap, 4) +
			(splat_layer_cap_opaque ? " (capped pixels filled opaque)" : " (capped pixels left partly covered)");
	s += "Alpha saturation cap: " + cov_cap_state + "\n";
	// DIAGNOSTIC ONLY - loud, because every stage but 0 draws a deliberately incomplete picture, and a time read under one
	// of them is a statement about that stage rather than about the pass.
	if(splat_ablation_stage != 0)
		s += "!! ABLATION STAGE " + toString(splat_ablation_stage) +
			" is selected: the picture is deliberately incomplete and the frame time is that stage's, not the pass's.\n";
	if(splat_quad_radius_scale != 1.f)
	{
		s += "!! QUAD RADIUS SCALE " + doubleToStringNDecimalPlaces(splat_quad_radius_scale, 2) + ": splats are drawn at " +
			doubleToStringNDecimalPlaces(splat_quad_radius_scale * splat_quad_radius_scale * 100.0, 0) + "% of their real area";
		if(splat_area_scale_gamma != 1.f || splat_area_scale_ref_px != 20.f)
			s += " (area-weighted: gamma " + doubleToStringNDecimalPlaces(splat_area_scale_gamma, 2) + ", ref " +
				doubleToStringNDecimalPlaces(splat_area_scale_ref_px, 0) + "px - only splats at or above the reference get the full reduction)";
		s += ".\n";
	}
	if(splat_visible_slicing)
	{
		// What the placement actually found, not just that it is on: a cloud whose visible share is near 100% is one the
		// setting cannot do anything for, and that is worth being able to read rather than infer from the clock.
		size_t total_samples = 0, total_visible = 0;
		for(size_t i=0; i<clouds.size(); ++i)
			if(!clouds[i]->slice_visible_cdf.empty())
			{
				total_samples += clouds[i]->slice_visible_cdf.size();
				total_visible += (size_t)clouds[i]->slice_visible_cdf.back();
			}

		s += "Slice placement: by visible splats";
		if(total_samples > 0)
			s += " (" + doubleToStringNDecimalPlaces(100.0 * (double)total_visible / (double)total_samples, 1) + "% of the sampled draw order is in frustum)";
		else
			s += " (no sample yet - falling back to the plain index split)";
		s += "\n";
	}
	if(splat_coverage_shrink_strength > 0.f)
	{
		// Which formula the number feeds is not something the number itself says, and the two mean different things at
		// the same value - see getCoverageShrinkMode().
		const std::string shrink_desc = (splat_coverage_shrink_mode == 0) ?
			("shrink " + doubleToStringNDecimalPlaces(splat_coverage_shrink_strength, 2) + " by coverage") :
			("loss budget " + doubleToStringNDecimalPlaces(splat_coverage_shrink_strength, 2) + " of the light still getting through");
		if(!splat_saturation_gate_enabled || splat_num_draw_slices <= 1)
			s += "!! COVERAGE SHRINK (" + shrink_desc + ") is set but idle - needs the saturation gate on with more than one draw slice.\n";
		else
			s += "!! COVERAGE SHRINK (" + shrink_desc + "): splats over already-covered pixels are drawn smaller.\n";
	}

	std::string gate_state;
	if(!splat_saturation_gate_enabled)
		gate_state = "off";
	else if(splat_coverage_cap > 0.f)
		gate_state = "enabled, but stood down by the alpha saturation cap, which owns the same mask";
	else if(getHideMode() != 0)
		gate_state = "enabled, but switched off by the hide-overdraw diagnostic, which owns the same mask";
	else if(splat_num_draw_slices <= 1)
		gate_state = "enabled, but idle - needs more than one draw slice";
	else if(splat_show_overdraw_mode != 0)
		gate_state = "enabled, but idle - not used in the overdraw views, where accumulated alpha counts layers rather than coverage";
	else if(!opengl_engine->splat_accum_gate_available)
		gate_state = "enabled, but UNAVAILABLE - the saturation mask framebuffer could not be built";
	else if(saturation_mask_prog.isNull() || !saturation_mask_prog->isBuilt() || mask_reduce_prog.isNull() || !mask_reduce_prog->isBuilt())
		gate_state = "enabled, but idle - mask shaders not built yet";
	else
		gate_state = "on";

	if(getHideMode() != 0)
	{
		s += std::string("!! ") + ((getHideMode() == 1) ? "HIDE OVERDRAW" : "HIDE ALPHA") + " is on: splats whose centre lands where the " +
			((getHideMode() == 1) ? "layer count" : "summed alpha") + " reaches " + doubleToStringNDecimalPlaces(splat_overdraw_range_max, 2) +
			" are being thrown away in the vertex shader.\n";
		if(splat_hide_overdraw_enabled && splat_hide_alpha_enabled)
			s += "   Both boxes are ticked; there is only one mask, so layers win and 'Hide alpha' is doing nothing.\n";
		// The rebuild count is what says whether the total may be read: it only holds still once the counting pass has
		// stopped running, and it stops running only while nothing the mask depends on is changing.
		s += "   Mask rebuilds so far: " + uInt64ToStringCommaSeparated(splat_hide_overdraw_mask_rebuilds) +
			". While that number is holding still the frame carries no counting pass and the total is comparable;\n"
			"   while it is climbing (the camera is moving, or a setting just changed) the total carries a whole extra pass over the splats and 'draw splats' is the only fair number.\n";
	}

	s += "Saturation gate: " + gate_state + ", threshold " + doubleToStringNDecimalPlaces(splat_saturation_threshold, 4) +
		", mask 1/" + toString(splat_saturation_mask_downscale) + " res";
	if(opengl_engine->getCurrentScene()->splat_saturation_mask_texture.nonNull())
		// The pyramid level count is here because a mask with only one level silently stops the vertex-side test from
		// culling anything bigger than 2x2 texels, which looks exactly like the gate being switched off.
		s += " (" + toString(opengl_engine->getCurrentScene()->splat_saturation_mask_texture->xRes()) + " x " +
			toString(opengl_engine->getCurrentScene()->splat_saturation_mask_texture->yRes()) + ", " +
			toString(opengl_engine->getCurrentScene()->splat_saturation_mask_texture->getNumMipMapLevelsAllocated()) + " pyramid levels)";
	s += "\n";

	// Every live-tunable parameter, printed whether or not it is at its default.  Only the desktop client has a panel to
	// set these from, so the web runs on the hardcoded defaults, and the usual reason for reading this section at all is
	// to work out why the two clients look different on the same scene - which needs the values themselves, not just the
	// ones that happen to be interesting.  Not printed: the LoD tree's lod_base, which belongs to the client and is only
	// consumed when a .sog is loaded, so it isn't the renderer's to report.
	s += "Live params: pixel_scale_limit " + doubleToStringNDecimalPlaces(lod_pixel_scale_limit, 2) +
		", max_splats_budget " + uInt64ToStringCommaSeparated(lod_max_splats_budget) +
		", resort_move_threshold_ws " + doubleToStringNDecimalPlaces(lod_resort_move_threshold_ws, 3) + "\n";
	s += "             max_layer_density " + doubleToStringNDecimalPlaces(lod_max_layer_density, 1) + (lod_max_layer_density == 0 ? " (disabled)" : "") +
		", max_tree_depth " + toString(lod_max_tree_depth) + (lod_max_tree_depth == 0 ? " (disabled)" : "") + "\n";
	s += "             size_clamp [" + doubleToStringNDecimalPlaces(splat_size_clamp_min, 3) + ", " + doubleToStringNDecimalPlaces(splat_size_clamp_max, 3) + "]" +
		((splat_size_clamp_min == 0 && splat_size_clamp_max == 0) ? " (disabled)" : (splat_size_clamp_invert ? " inverted" : "")) +
		", alpha_cutoff " + doubleToStringNDecimalPlaces(splat_alpha_cutoff, 4) + "\n";
	// Printed whether or not it is doing anything, because a forgotten distance slice looks exactly like a broken scene.
	s += "             dist_clamp [" + doubleToStringNDecimalPlaces(splat_dist_clamp_min, 2) + ", " + doubleToStringNDecimalPlaces(splat_dist_clamp_max, 2) + "] m" +
		(!splat_dist_clamp_enabled ? " (disabled)" : (splat_dist_clamp_invert ? " INVERTED - this shell is hidden" : " - only this shell is shown")) + "\n"; // SESSION072: enable state now comes from its own checkbox, not from min/max sitting at the keep-everything default.
	// Says when the request and the reality differ, rather than only what was asked for: the overdraw views force half
	// float whatever this switch says, and a line reading RGBA8 next to a working overdraw ramp would be a puzzle.
	s += "             accumulation buffer " + std::string(splat_accum_buffer_8bit ? "RGBA8" : "RGBA16F") +
		((splat_accum_buffer_8bit && getShowOverdraw()) ? " requested, but RGBA16F in use - an 8-bit buffer saturates at one layer and cannot show overdraw" : "") + "\n";
	s += "Sorts in flight: " + toString(num_sorts_in_flight) + " / " + toString(max_concurrent_sorts) + "\n";
	s += "GPU mem: " + getMBSizeString((size_t)tex_bytes) + " data textures, " + getMBSizeString((size_t)index_vbo_bytes) + " index VBOs\n";
	s += "Sort scratch pooled: " + toString(free_scratch.size()) + " buffers, " + getMBSizeString((size_t)scratch_bytes) + "\n";
	s += "Splat shader prog built:   " + boolToString(shader_prog && shader_prog->isBuilt()) + "\n";
	s += "Resolve shader prog built: " + boolToString(resolve_prog && resolve_prog->isBuilt()) + "\n";

	if(total_nodes_lod_active > 0)
	{
		s += "LoD traversals in flight: " + toString(num_traversals_in_flight) + " / " + toString(max_concurrent_traversals) + "\n";
		s += "LoD traversal scratch pooled: " + toString(free_traversal_scratch.size()) + " buffers, " + getMBSizeString((size_t)traversal_scratch_bytes) + "\n";
		s += "LoD nodes (leaves+merged) vs leaves, LoD-active clouds only: " + uInt64ToStringCommaSeparated(total_nodes_lod_active) + " vs " +
			uInt64ToStringCommaSeparated(total_leaves_lod_active) + " (" + doubleToStringNDecimalPlaces((double)total_nodes_lod_active / myMax((size_t)1, total_leaves_lod_active), 2) + "x)\n";
	}

	// The per-cloud breakdown is what shows whether the partitioning is behaving - a world of separate captures should
	// show one member per cloud.  Capped, since a world could hold many.
	const size_t max_clouds_to_list = 8;
	for(size_t i=0; i<myMin(clouds.size(), max_clouds_to_list); ++i)
	{
		const SplatCloud& cloud = *clouds[i];
		s += "  cloud " + toString(cloud.cloud_id) + ": " + toString(cloud.members.size()) + (cloud.members.size() == 1 ? " member, " : " members, ") +
			uInt64ToStringCommaSeparated(cloud.total_splats) + " nodes uploaded" + (cloud.sort_in_flight ? ", sorting" : "");

		if(cloudHasLodTree(cloud))
		{
			s += ", " + uInt64ToStringCommaSeparated((uint64)cloud.ob->num_instances_to_draw) + " drawn (LoD)";
			if(cloud.traversal_in_flight)
				s += ", traversing";
			if(cloud.last_traversal_hit_budget_cap)
				s += " !WARNING! budget cap is limiting detail on this cloud";
			if(cloud.last_traversal_hit_density_cap)
				s += " !WARNING! density cap is limiting detail on this cloud";
			if(cloud.last_traversal_hit_depth_cap)
				s += " !WARNING! depth cap is limiting detail on this cloud";
			s += "\n";

			// TEMP diagnostic while calibrating max_layer_density (GaussianSplatLodNode::layer_density): root density and
			// its immediate children's range, per member, so a sane threshold can be read off directly instead of guessed.
			for(size_t m=0; m<cloud.members.size(); ++m)
			{
				const std::vector<GaussianSplatLodNode>& tree = cloud.members[m].splat_data->lod_tree;
				if(tree.empty())
					continue;
				s += "    member " + toString(m) + " tree: root density " + doubleToStringNDecimalPlaces(tree[0].layer_density, 1);
				if(tree[0].child_count > 0)
				{
					float min_d = tree[tree[0].child_start].layer_density;
					float max_d = min_d;
					for(uint32 c = tree[0].child_start + 1; c < (uint32)tree[0].child_start + tree[0].child_count; ++c)
					{
						min_d = myMin(min_d, tree[c].layer_density);
						max_d = myMax(max_d, tree[c].layer_density);
					}
					s += ", " + toString(tree[0].child_count) + " root children density [" + doubleToStringNDecimalPlaces(min_d, 1) + ", " + doubleToStringNDecimalPlaces(max_d, 1) + "]";
				}
				s += "\n";
			}
		}
		else
			s += "\n";
	}
	if(clouds.size() > max_clouds_to_list)
		s += "  (" + toString(clouds.size() - max_clouds_to_list) + " more)\n";

	return s;
}


Reference<SplatCloud> GaussianSplatRenderer::allocCloud()
{
	Reference<SplatCloud> cloud = new SplatCloud();
	cloud->cloud_id = next_cloud_id++;

	cloud->ob = new GLObject();
	cloud->ob->ob_to_world_matrix = Matrix4f::identity(); // Never changes: splat positions are baked into world space.
	cloud->ob->mesh_data = makeInstancedQuadMeshData(*opengl_engine->vert_buf_allocator);
	cloud->ob->num_instances_to_draw = 0;

	cloud->ob->materials.resize(1);
	OpenGLMaterial& mat = cloud->ob->materials[0];
	mat.shader_prog = shader_prog;
	mat.auto_assign_shader = false;

	// Route the object through OpenGLEngine::drawSplatClouds(), which orders whole clouds in depth order against each
	// other.  Neither of the engine's general-purpose passes will do: the transparent pass uses order-independent
	// transparency on native, which accumulates colour additively and so has no notion of one splat occluding another -
	// fine for a few glass surfaces, but splat clouds put hundreds of overlapping splats on every pixel, where it
	// saturates to white.  The alpha-blended pass does blend in depth order, but orders objects by distance to their
	// AABB, which isn't an exact ordering for clouds that overlap in screen space.
	mat.splat_cloud = true;

	// Also set alpha_blend, even though drawAlphaBlendedObjects() won't draw this object: the flag is what sets
	// MATERIAL_ALPHA_BLEND_BITFLAG on the batch, which is how the opaque pass, the depth pre-pass and the shadow passes
	// know to skip it.  OpenGLEngine::addObject() keys off splat_cloud to put the object in exactly one of the two sets.
	mat.alpha_blend = true;
	// One slot per uniform appended in buildShadersIfNeeded(), and it has to stay in step with that list: the draw path
	// walks the program's uniforms and indexes this array by the same i, so a slot short is an out-of-bounds read there
	// and an out-of-bounds write in think(). All but splat_tex_width below are set by think(), or by the draw path for the
	// saturation mask ones.
	mat.user_uniform_vals.resize(30); // SESSION066: +2 for splat_hide_count_active/threshold (indices 23, 24). SESSION067: +2 for splat_area_slice_mode/_px (25, 26). SESSION069: +1 for splat_jitter_px (27), +1 for splat_low_pass_variance (28). SESSION071: +1 for splat_point_size_px (29).
	mat.user_uniform_vals[2].intval = (int)splat_tex_width;

	// Build a real (if minimal) texture and VAO up front: adding the object to the engine before it has those would
	// leave the engine caching a stale, non-instanced VAO reference that a later VAO rebuild can't retroactively fix.
	ensureGpuCapacity(*cloud, /*needed_splats=*/0);

	// Deliberately not added to the engine here.  An empty cloud has no meaningful bounds, and handing the engine an
	// object whose aabb_os is still the placeholder would have it compute and cache a bogus aabb_ws.  The first member
	// to be baked in gives the cloud real bounds and an instance count, and adds it - see addCloudToEngineIfNeeded().
	clouds.push_back(cloud);
	return cloud;
}


void GaussianSplatRenderer::addCloudToEngineIfNeeded(SplatCloud& cloud)
{
	if(cloud.added_to_engine)
		return;

	assert(cloud.total_splats > 0); // Otherwise the bounds are still the placeholder.
	opengl_engine->addObject(cloud.ob); // Computes aabb_ws from the bounds set by rebuildCloudAABB().
	cloud.added_to_engine = true;
}


void GaussianSplatRenderer::destroyCloud(const Reference<SplatCloud>& cloud)
{
	if(cloud->added_to_engine)
		opengl_engine->removeObject(cloud->ob);

	for(size_t i=0; i<clouds.size(); ++i)
		if(clouds[i].ptr() == cloud.ptr())
		{
			clouds.erase(clouds.begin() + i);
			break;
		}

	// Any sort in flight for this cloud is left to run.  Its result carries the cloud id, which no longer matches a
	// live cloud, so drainSortResults() drops it and returns the scratch to the pool.
}


void GaussianSplatRenderer::rebuildVAO(SplatCloud& cloud)
{
	VertexSpec vertex_spec = cloud.ob->mesh_data->vertex_spec;
	vertex_spec.attributes[splat_index_attribute_loc].vbo = cloud.instance_index_vbo;
	vertex_spec.attributes[splat_index_attribute_loc].enabled = true;
#if DO_INDIVIDUAL_VAO_ALLOC
	cloud.ob->vert_vao = new VAO(cloud.ob->mesh_data->vbo_handle.vbo, cloud.ob->mesh_data->indices_vbo_handle.index_vbo, vertex_spec);
#else
	cloud.ob->vert_vao = new VAO(vertex_spec);
#endif
	// Bind the index VBO as the object's per-instance buffer.  Note that on the shared-VAO path (i.e. everywhere except
	// Mac and Emscripten) the VertexAttrib::vbo set above is ignored, and the engine instead binds instance_matrix_vbo
	// to binding point 1 at draw time - hence instance_vbo_stride_B, without which the engine would assume the stride of
	// an instance matrix rather than of our uint32 indices.
	cloud.ob->instance_matrix_vbo = cloud.instance_index_vbo;
	cloud.ob->instance_vbo_stride_B = (uint32)sizeof(uint32);
}


void GaussianSplatRenderer::uploadTexelRowsForSplatRange(SplatCloud& cloud, size_t first_splat, size_t num_splats_to_upload)
{
	if(num_splats_to_upload == 0)
		return;

	// Repack whole texture rows spanning the given splat range.  An arbitrary range doesn't align to row boundaries, so
	// this may re-pack a few splats belonging to a neighbouring member too.  That's harmless: their data is already
	// correct, and we just re-derive the same texels for them.
	const size_t first_texel     = first_splat * texels_per_splat;
	const size_t last_texel_excl = (first_splat + num_splats_to_upload) * texels_per_splat;
	const size_t start_row       = first_texel / splat_tex_width;
	const size_t end_row         = Maths::roundedUpDivide(last_texel_excl, splat_tex_width); // Exclusive.
	const size_t num_rows        = end_row - start_row;

	const size_t row_start_splat    = (start_row * splat_tex_width) / texels_per_splat;
	const size_t row_end_splat_excl = myMin(cloud.total_splats, (end_row * splat_tex_width) / texels_per_splat);

	js::Vector<float, 16> texel_data(splat_tex_width * num_rows * 4, 0.f);
	packSplatTexels(cloud.positions, cloud.scales, cloud.rotations, cloud.colours, row_start_splat, row_end_splat_excl, texel_data.data());

	cloud.ob->materials[0].albedo_texture->loadRegionIntoExistingTexture(/*mipmap_level=*/0, /*x=*/0, /*y=*/start_row, /*z=*/0,
		/*region_w=*/splat_tex_width, /*region_h=*/num_rows, /*region_d=*/1, /*src_row_stride_B=*/splat_tex_width * 4 * sizeof(float),
		ArrayRef<uint8>((const uint8*)texel_data.data(), texel_data.size() * sizeof(float)), /*bind_needed=*/true);
}


void GaussianSplatRenderer::writeIdentityIndices(SplatCloud& cloud, size_t first_splat, size_t num_splats)
{
	if(num_splats == 0)
		return;

	js::Vector<uint32, 16> indices(num_splats);
	for(size_t i=0; i<num_splats; ++i)
		indices[i] = (uint32)(first_splat + i);
	cloud.instance_index_vbo->updateData(first_splat * sizeof(uint32), indices.data(), indices.size() * sizeof(uint32));
}


void GaussianSplatRenderer::ensureGpuCapacity(SplatCloud& cloud, size_t needed_splats)
{
	// The albedo_texture check covers the case of a newly allocated cloud, where needed_splats (0) <=
	// gpu_capacity_splats (0) would otherwise skip creating a texture at all.
	if(needed_splats <= cloud.gpu_capacity_splats && cloud.ob->materials[0].albedo_texture.nonNull())
		return;

	size_t new_tex_h = texHeightForSplatCount(needed_splats);
	const size_t cur_tex_h = texHeightForSplatCount(myMax<size_t>(1, cloud.gpu_capacity_splats));
	new_tex_h = myMax(new_tex_h, cur_tex_h * 2); // Geometric growth, so repeated small appends don't reallocate every time.

	const size_t new_capacity_splats = (splat_tex_width * new_tex_h) / texels_per_splat;

	// Repack every splat into a fresh, bigger texture.  Growth is rare, so this cost isn't paid on every append.
	js::Vector<float, 16> texel_data(splat_tex_width * new_tex_h * 4, 0.f);
	packSplatTexels(cloud.positions, cloud.scales, cloud.rotations, cloud.colours, 0, cloud.total_splats, texel_data.data());

	cloud.ob->materials[0].albedo_texture = new OpenGLTexture(splat_tex_width, new_tex_h, opengl_engine,
		ArrayRef<uint8>((const uint8*)texel_data.data(), texel_data.size() * sizeof(float)),
		OpenGLTextureFormat::Format_RGBA_Linear_Float,
		OpenGLTexture::Filtering_Nearest, // Must be Nearest: this is a data texture, and filtering would blend unrelated splats' attributes together.
		OpenGLTexture::Wrapping_Clamp,
		/*has_mipmaps=*/false);

	// Grow the index VBO to match, in identity order.  Any in-flight sort's result will simply be reapplied, or dropped
	// if the cloud has also been renumbered, once it lands.
	js::Vector<uint32, 16> identity_indices(new_capacity_splats);
	for(size_t i=0; i<new_capacity_splats; ++i)
		identity_indices[i] = (uint32)i;
	cloud.instance_index_vbo = new VBO(identity_indices.data(), identity_indices.size() * sizeof(uint32), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	rebuildVAO(cloud);

	// vert_vao was just replaced.  If the object is already in the engine, the engine cached a draw-time VAO reference
	// when it was added that is now stale, and would keep drawing the old, freed VAO.  objectMaterialsUpdated()
	// recomputes it from the current vert_vao.  It must not be called before the object has been through
	// OpenGLEngine::addObject()'s buildObjectData(), which is why the flag is checked rather than assumed.
	if(cloud.added_to_engine)
		opengl_engine->objectMaterialsUpdated(*cloud.ob);

	cloud.gpu_capacity_splats = new_capacity_splats;
}


// SESSION071: widen a merged node's Gaussian so neighbouring merged nodes actually meet, instead of leaving a lattice of
// gaps between their peaks.
//
// The defect, isolated on the ablation ladder: stage 7 (flat alpha across the quad) keeps the bridge's brightness at
// pixel_scale_limit 15, and stage 8 - which differs by nothing except the Gaussian falloff exp(power) - immediately shows
// darkened regions with a visible blurred lattice on flat surfaces. At limit 1, where almost every drawn node is a leaf,
// the lattice is absent. So it is the merged nodes' Gaussians that fail to overlap.
//
// Why: the merge moment-matches its children, and a Gaussian matched to the spread of children uniformly filling a cell
// of width L has sigma = L/sqrt(12) = 0.289*L, while smooth coverage of a lattice of spacing d needs sigma/d >~ 0.5. The
// ripple amplitude of a sum of Gaussians on a lattice goes as 2*exp(-2*pi^2*sigma^2/d^2): 1.4% at 0.5, 8% at 0.4, ~30% at
// 0.289 - so the merge lands squarely in the bad regime. It reads as darkening rather than as ripple because compositing
// is concave (1-(1-a)^n): the peaks lose their excess to saturation while the troughs let the background through, so the
// mean brightness over the area is preserved (measured: mergeColourByDepthReport()'s ratio column sat at 1.00 throughout
// the hunt) while the visible brightness drops. It hits bright surfaces hardest because what shows through a trough is
// the dark forest behind them.
//
// The correction widens ONLY the spread part of the covariance, not the children's own size. Merge builds
// cov = sum(w * (child_cov + outer(d))) / sum(w), i.e. sigma^2 = sigma_own^2 + sigma_spread^2 per axis; the ripple comes
// from the spread term alone, so scaling the whole node isotropically would also inflate a flat surface's thickness and
// turn a wall into fog. Recovering the spread term needs no eigen-decomposition: the parent's covariance is diagonal in
// its own frame, so projecting sum(w * outer(d)) onto each parent axis gives that axis's sigma_spread^2 directly, and
// sigma_new^2 = sigma^2 + (k^2 - 1) * sigma_spread^2.
//
// k = sqrt(3) is analytic, not tuned: children uniformly filling a half-width h have sigma_spread = h/sqrt(3), the
// neighbouring node sits at d = 2h, and sigma >= 0.5*d = h gives k = sqrt(3). The owner independently found the lattice
// stops being visible at about 1.5, which is where the ripple formula puts it at ~5%, right at the threshold of
// visibility - so the measurement and the derivation agree.
static Vec3f widenedMergedScale(const std::vector<GaussianSplatLodNode>& tree, size_t node_idx, float k)
{
	const GaussianSplatLodNode& node = tree[node_idx];
	if((node.child_count == 0) || (k == 1.f))
		return node.scale; // A leaf is an original splat - its size is authored, never re-fitted, so there is nothing to correct.

	const Matrix4f R = Quat<float>(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]).toMatrix(); // Parent axes.

	// Same weights the merge itself used (opacity * volume - see mergeGaussianSplatLodNodes()), so the spread recovered
	// here is the spread that actually went into the parent's covariance, not a differently-weighted estimate of it.
	float spread_var[3] = { 0.f, 0.f, 0.f };
	float w_sum = 0.f;
	for(uint32 c=0; c<node.child_count; ++c)
	{
		const GaussianSplatLodNode& child = tree[node.child_start + c];
		const float w = child.colour.x[3] * (child.scale.x * child.scale.y * child.scale.z);
		const Vec3f dv = child.centre_os - node.centre_os;
		const Vec4f d(dv.x, dv.y, dv.z, 0.f);
		for(int a=0; a<3; ++a)
		{
			const float proj = dot(R.getColumn(a), d);
			spread_var[a] += w * proj * proj;
		}
		w_sum += w;
	}

	if(w_sum <= 1.0e-12f)
		return node.scale; // Degenerate group (every child ~transparent or ~zero volume); the merge fell back to equal weights and there is no meaningful spread to widen.

	const float kk = k * k - 1.f;
	const float inv_w = 1.f / w_sum;
	return Vec3f(
		std::sqrt(myMax(0.f, node.scale.x * node.scale.x + kk * spread_var[0] * inv_w)),
		std::sqrt(myMax(0.f, node.scale.y * node.scale.y + kk * spread_var[1] * inv_w)),
		std::sqrt(myMax(0.f, node.scale.z * node.scale.z + kk * spread_var[2] * inv_w)));
}


// Bakes member.splat_data into cloud's arrays at member.offset, using member's stored pose, and sets member.aabb_ws.
//
// The bounds are grown by each splat's own radius, rather than just holding the splat centres.  A splat is drawn as a
// quad extending well beyond its centre, so centre-only bounds would let two clouds whose bounds are marginally
// disjoint still have their fringe splats interpenetrating - and the partitioning would then leave them in separate
// clouds with no separating plane between them, which is the one failure that produces a wrong compositing order.
//
// SESSION071: merge_colour_mode re-derives the merged (non-leaf) nodes' colours after the pose bake - see
// recolorLodTree(). Done here rather than at the call sites because this is the only writer of cloud.colours, so a bake
// from any path (add, re-add, pose change) always leaves the colours matching the current settings.
//
// SESSION071: spread_widen is applied to MERGED nodes only (leaves are original splats and are never touched) - see
// widenedMergedScale() for the maths and the measurement behind it.
static void bakeMember(SplatCloud& cloud, CloudMember& member, const GaussianSplatMergeColourParams& merge_colour_params,
	float spread_widen)
{
	const GaussianSplatData& splat_data = *member.splat_data;
	const Vec4f translation_ws = member.translation_ws;
	const Quat<float>& rotation_ws = member.rotation_ws;
	const float uniform_scale_ws = member.uniform_scale_ws;

	js::AABBox aabb_ws = js::AABBox::emptyAABBox();

	if(splat_data.lod_tree.empty())
	{
		// No LoD tree built (yet, or ever, for this object) - bake every leaf splat, as before.
		const size_t num_splats = splat_data.numSplats();
		for(size_t i=0; i<num_splats; ++i)
		{
			const Vec3f& os_pos   = splat_data.positions[i];
			const Vec3f& os_scale = splat_data.scales[i];
			const Vec4f& os_rot   = splat_data.rotations[i]; // (x, y, z, w)

			const Vec4f rotated = rotation_ws.rotateVector(Vec4f(uniform_scale_ws * os_pos.x, uniform_scale_ws * os_pos.y, uniform_scale_ws * os_pos.z, 0.f));
			const Vec4f world_pos = translation_ws + rotated; // translation_ws.w == 1 and rotated.w == 0, so world_pos.w == 1, as a point should be.

			const Quat<float> os_quat(os_rot[0], os_rot[1], os_rot[2], os_rot[3]);
			const Quat<float> world_quat = rotation_ws * os_quat;

			const Vec3f world_scale = os_scale * uniform_scale_ws;

			const size_t dest = member.offset + i;
			cloud.positions[dest] = toVec3f(world_pos);
			cloud.scales   [dest] = world_scale;
			cloud.rotations[dest] = world_quat.v; // Quat::v is already (x, y, z, w), matching our storage convention.
			cloud.colours  [dest] = splat_data.colours[i]; // Colour and opacity aren't affected by the cloud's pose, but re-deriving them keeps this the single place a member's data is written.

			const float max_scale = myMax(world_scale.x, myMax(world_scale.y, world_scale.z));
			cloud.feature_size[dest] = 2.f * max_scale; // SESSION055 - see SplatCloud::feature_size.
			const float radius = splat_cutoff_sigmas * max_scale;
			cloud.cull_radius[dest] = radius; // SESSION059: a no-tree member's splats are never culled individually (see the traversal's NoTree branch), but keep this array correctly populated everywhere regardless - this is exactly the same 3-sigma cutoff radius as the tree-leaf case below.
			aabb_ws.enlargeToHoldPoint(world_pos - Vec4f(radius, radius, radius, 0.f));
			aabb_ws.enlargeToHoldPoint(world_pos + Vec4f(radius, radius, radius, 0.f));
		}
	}
	else
	{
		// A tree exists - bake every node (leaves and merged internal nodes alike), not just the leaves, since any node could end up in a drawn LoD selection later. A merged node transforms identically to a
		// leaf one under this rigid + uniform-scale bake, which is exactly why the tree is built in object space in the first place (see GaussianSplatLodTree.h).
		const std::vector<GaussianSplatLodNode>& tree = splat_data.lod_tree;
		for(size_t i=0; i<tree.size(); ++i)
		{
			const Vec3f& os_pos   = tree[i].centre_os;
			const Vec4f& os_rot   = tree[i].rotation; // (x, y, z, w)

			const Vec4f rotated = rotation_ws.rotateVector(Vec4f(uniform_scale_ws * os_pos.x, uniform_scale_ws * os_pos.y, uniform_scale_ws * os_pos.z, 0.f));
			const Vec4f world_pos = translation_ws + rotated;

			const Quat<float> os_quat(os_rot[0], os_rot[1], os_rot[2], os_rot[3]);
			const Quat<float> world_quat = rotation_ws * os_quat;

			// SESSION071: merged nodes only - a leaf is an original splat and must keep its authored size exactly.
			const Vec3f world_scale = widenedMergedScale(tree, i, spread_widen) * uniform_scale_ws;

			const size_t dest = member.offset + i;
			cloud.positions[dest] = toVec3f(world_pos);
			cloud.scales   [dest] = world_scale;
			cloud.rotations[dest] = world_quat.v;
			cloud.colours  [dest] = tree[i].colour;

			const float max_scale = myMax(world_scale.x, myMax(world_scale.y, world_scale.z));
			cloud.feature_size[dest] = 2.f * max_scale; // SESSION055 - see SplatCloud::feature_size.
			const float radius = splat_cutoff_sigmas * max_scale;
			// SESSION059: rigid + uniform-scale bake preserves lengths up to uniform_scale_ws, same reasoning as feature_size above - see GaussianSplatLodNode::bounding_radius_os's comment for why this (not feature_size) is what the traversal's frustum-cull margin needs.
			// SESSION071: max() with the node's own drawn radius, because spread_widen can widen a merged node past its
			// descendants' enclosing sphere - at which point the bound would no longer bound what is actually drawn, and the
			// cull would clip a node whose quad still reaches the screen. Identical to the old value at spread_widen 1.
			cloud.cull_radius[dest] = myMax(uniform_scale_ws * tree[i].bounding_radius_os, radius);
			aabb_ws.enlargeToHoldPoint(world_pos - Vec4f(radius, radius, radius, 0.f));
			aabb_ws.enlargeToHoldPoint(world_pos + Vec4f(radius, radius, radius, 0.f));
		}

		// SESSION071: re-derive the merged nodes' colours under the selected formulation, over the just-baked world-space
		// arrays (both formulations are ratio-based, so the uniform world scale cancels - see recolorLodTree()). Skipped
		// only when the result would be exactly what the tree already carries: Legacy with no widening.
		//
		// Running this AFTER the widening above is deliberate: the widened parent area feeds straight into the opacity
		// term, so a widened node spreads the same optical mass over more pixels instead of getting brighter.
		if(merge_colour_params.mode != GaussianSplatMergeColourMode_Legacy || spread_widen != 1.f)
			recolorLodTree(tree, &cloud.scales[member.offset], &cloud.colours[member.offset], merge_colour_params);
	}

	member.aabb_ws = aabb_ws;
}


// SESSION079: same reasoning as setSatPrefilterMode() just above - the verdict is baked into U(P) at traversal time,
// so a live change needs a fresh traversal forced. See getSatPrefilterThreshold().
void GaussianSplatRenderer::setSatPrefilterThreshold(float v)
{
	if(v == sat_prefilter_threshold)
		return;
	sat_prefilter_threshold = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION076: the "coarse" checkbox now changes what a traversal PRODUCES, not just what the filter is allowed to draw
// from it - with the layer undrawn, the traversal skips capturing grid-useless coarse nodes and evicts the rest once the
// grid is built. So a cached U(P) built while this was off physically has no coarse layer left in it, and turning the
// checkbox back on could not restore the edge-filling patch until something else happened to force a fresh traversal -
// during pure rotation, nothing does. Dropping the cached frontiers here forces one, exactly as setSatPrefilterMode() does.
void GaussianSplatRenderer::setCoarseFloorEnabled(bool v)
{
	if(v == split_coarse_floor_enabled)
		return;
	split_coarse_floor_enabled = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION076: same cached-frontier drop as setSatPrefilterMode() above, for the same reason - the sat_diag_* counters
// are filled when a frontier is BUILT, so without forcing a fresh traversal the checkbox would appear to do nothing
// until the camera happened to move.
void GaussianSplatRenderer::setSatDiagLog(bool v)
{
	if(v == sat_diag_log)
		return;
	sat_diag_log = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION078: see getSatDebugOverlayMode(). Same cache-invalidation shape as setSatDiagLog() just above - this changes
// what a traversal computes (Off vs. non-Off decides whether sat_accum_t/sat_amp_sum are built at all, and whether the
// grid is force-built with the saturation filter itself off), so a cached frontier from before the change cannot be
// reinterpreted under the new value.
void GaussianSplatRenderer::setSatDebugOverlayMode(GaussianSplatSatDebugOverlayMode v)
{
	if(v == sat_debug_overlay_mode)
		return;
	sat_debug_overlay_mode = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
		clouds[i]->sat_grid_debug_tex = NULL; // SESSION076 §9: stale overlay texture from before the toggle - not drawn either way (gated at the call site), but no reason to keep the GPU memory.
		clouds[i]->sat_grid_debug_ramp_tex = NULL; // SESSION077: its ramp-view companion.
		clouds[i]->sat_grid_debug_maskfix_tex = NULL; // SESSION081: its mask_fix-view companion.
	}
}


// SESSION076 CALIBRATION: rebuilds the saturation grid from scratch, so like setSatPrefilterMode() it must force a
// fresh traversal - otherwise a live change sits invisible until the camera happens to move, which during pure rotation
// is never. See getSatGridSubdiv().
void GaussianSplatRenderer::setSatGridSubdiv(float v)
{
	if(v == sat_grid_subdiv)
		return;
	sat_grid_subdiv = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION088 FIX: was `split_coarse_pixel_scale = v;` inline in the header, invalidating nothing. That was correct for
// the knob's original coarse-floor-cut role, which kickOffTraversals() reads live at kick time - but session076 gave it
// a second job, feeding gsSatGridResForFocal() alongside sat_grid_subdiv, and nothing was ever told that had happened.
// Owner-visible as the knob doing nothing whatsoever on a static camera: no fresh traversal, and no barrier rebuild
// either, since the gate in kickOffSaturationBuilds() did not know this parameter existed. Both halves are fixed - here,
// and by coarse_pixel_scale_used in that gate.
//
// Mirrors setSatGridSubdiv() above deliberately, including what it does NOT clear: a grid built at a different res has a
// different sat_depth SIZE, which gsSatBarrierDisagreement() already treats as total disagreement, so the barrier does
// not need dropping by hand the way setSatRegionRadius()'s same-size depth shift does.
void GaussianSplatRenderer::setCoarsePixelScale(float v)
{
	if(v == split_coarse_pixel_scale)
		return;
	split_coarse_pixel_scale = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION078: alpha gain/gamma now feed the CPU saturation prefilter as well as the draw path - see getAlphaGain()'s
// comment - so a change here has to drop cached frontiers, same reasoning as setSatGridSubdiv() just above. Cheap while
// the "ignore" alpha-adjust switch is on, its default: MainWindow.cpp always passes 1/1 in that case, so these are
// no-ops (the early-out below fires) unless the owner has deliberately turned adjustment on.
void GaussianSplatRenderer::setAlphaGain(float v)
{
	if(v == splat_alpha_gain)
		return;
	splat_alpha_gain = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


void GaussianSplatRenderer::setAlphaGamma(float v)
{
	if(v == splat_alpha_gamma)
		return;
	splat_alpha_gamma = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION078: same reasoning as setSatGridSubdiv() just above - this changes what the grid ASSERTS (a ball of camera
// positions rather than the single one it was built from), so an existing frontier's sat_depth was derived under the old
// value and cannot be reinterpreted under the new one. Drop the caches and force a fresh traversal. See
// getSatRegionRadius().
// SESSION088 FIX: the barrier and the inherited far block go too. The barrier itself always rebuilt correctly here -
// region_radius_used is in kickOffSaturationBuilds()'s staleness gate - but the far-block reuse gate then ABSORBED the
// result: its barrier-disagreement test is deliberately flip-only (saturated/unsaturated state changes, not depth
// changes) so that organic depth churn between rebuilds does not force needless rebuilds, and R's effect is a
// near-uniform depth shift that flips very few tiles. So the new barrier arrived, was judged "close enough", and the
// far block kept its old LoD until camera motion invalidated it some other way. Dropping both here bypasses that
// tolerance for a deliberate, manual change, exactly as setFrontierReuseSplitDist() already does; the tolerance itself
// (gsSatBarrierDisagreement(), btol) is untouched and still governs camera-driven reuse.
void GaussianSplatRenderer::setSatRegionRadius(float v)
{
	if(v == sat_region_radius)
		return;
	sat_region_radius = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->cached_sat_barrier = NULL;
		clouds[i]->last_unpruned_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION081: same reasoning as setSatRegionRadius() above - changes sat_depth, so cached frontiers built under the
// old value are not valid under the new one.
void GaussianSplatRenderer::setSatRegionClosingTiles(int v)
{
	if(v == sat_region_closing_tiles)
		return;
	sat_region_closing_tiles = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION085 - see getSatMinRatio(). Deliberately NOT like the four setters above: this changes only what the APPLY
// decides, so no barrier and no frontier is invalidated. What it must do instead is clear the apply throttle, which
// skips a kick whose (source, barrier) pair is the one the last kick already used - without this the knob would
// appear dead until the camera happened to move, which is exactly the sort of thing that gets mistaken for the
// mechanism not working.
// SESSION085 ETAP 3 - see getSatBiasCeiling(). Unlike setSatMinRatio() this changes what a TRAVERSAL produces (which
// nodes the walk stops at), not merely what the apply decides - so cached frontiers built under the old value are not
// valid under the new one, exactly like the four setters above it.
// SESSION088 FIX: last_unpruned_ufrontier as well. Clearing only cached_ufrontier forces a fresh traversal but does not
// stop that traversal INHERITING its far block from the previous one (see the reuse gate in the task's ctor) - and the
// reuse key is topology/pixel scale/budget, none of which the ceiling touches, so the inherited block keeps the LoD it
// was selected with under the old ceiling. The near field then re-biases and the far field does not, which is
// owner-visible as "changing bias does nothing until I move".
void GaussianSplatRenderer::setSatBiasCeiling(float v)
{
	if(v == sat_bias_ceiling)
		return;
	sat_bias_ceiling = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->last_unpruned_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION088 - see getSatBiasExponent(). Same class of knob as setSatBiasCeiling() above: it changes what a TRAVERSAL
// produces, never what the barrier measures, so cached frontiers go and cached_sat_barrier deliberately stays.
//
// last_unpruned_ufrontier is dropped as well, which the ceiling's setter does not do. Without it the forced re-kick
// below still INHERITS the far block the previous traversal selected under the old exponent - the reuse key sees a
// matching topology and pixel scale and has no idea the curve moved underneath it - so the far field, which is the
// half this knob exists to change, keeps its old LoD until camera motion happens to rebuild that block. The knob then
// reads as working on the near field only, or as not working at all on a stationary camera.
void GaussianSplatRenderer::setSatBiasExponent(float v)
{
	if(v == sat_bias_exponent)
		return;
	sat_bias_exponent = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->last_unpruned_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION080 STEP B: same reasoning as the two setters above - this changes what a traversal PRODUCES (a frontier
// partly inherited from its predecessor rather than walked from scratch), so cached frontiers built under the old value
// are not valid inputs under the new one. Dropping last_unpruned_ufrontier too is what makes turning the knob back to 0
// exact rather than approximately exact: the next traversal then has no predecessor to inherit from and walks the whole
// tree, which is the pre-session080 behaviour bit-for-bit. See getFrontierReuseSplitDist().
void GaussianSplatRenderer::setFrontierReuseSplitDist(float v)
{
	if(v == frontier_reuse_split_dist)
		return;
	frontier_reuse_split_dist = v;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		clouds[i]->cached_ufrontier = NULL;
		clouds[i]->last_unpruned_ufrontier = NULL;
		clouds[i]->have_last_traversal_cam_pos = false;
	}
}


// SESSION086 CALIBRATION: no cache drop. Unlike the three setters above this does not change what a traversal
// PRODUCES - the far block it accepts or rejects is the same object either way, and the only thing that moves is HOW
// FAR the camera may get from that block's anchor before the next traversal walks the tree instead. A frontier already
// in hand stays a valid input under the new value; the new bound simply applies from the next traversal on. See
// getFrontierReuseDriftFraction().
void GaussianSplatRenderer::setFrontierReuseDriftFraction(float v)
{
	frontier_reuse_drift_fraction = v;
}


// SESSION086 - see the declaration for why the seed count is measured rather than chosen.
void GaussianSplatRenderer::updateExpandSeedTarget(double seed_max_ms, double task_sum_ms, double prologue_ms, double expand_ms)
{
	if(seed_max_ms <= 0.0 || task_sum_ms <= 0.0)
		return; // Serial path, or a traversal that never reached expandParallel() - nothing measured, so nothing to learn from.

	glare::TaskManager* const task_manager = opengl_engine->getMainTaskManager();
	const size_t concurrency = myMax<size_t>(1, task_manager != NULL ? (size_t)task_manager->getConcurrency() : 1);
	const double fair_share_ms = task_sum_ms / (double)concurrency; // What one thread would carry if the split were perfect.

	// Bounds. The floor is the smallest split that could keep the pool fed at all; the ceiling is what stops a
	// pathological scene from driving the serial split loop somewhere expensive - the loop is one node expansion per
	// iteration (measured 0.21ms at 4354 seeds), so the ceiling is generous rather than tight.
	const size_t min_target = concurrency * 4;
	const size_t max_target = concurrency * 4096;

	size_t target = expand_seed_target > 0 ? expand_seed_target : concurrency * 16;
	if(seed_max_ms > fair_share_ms * 1.5)
		target *= 2;                                   // The worst seed still carries well over its share: split finer.
	else if(prologue_ms > expand_ms * 0.05)
		target /= 2;                                   // The split is now a real cost of its own: back off. See the declaration.

	expand_seed_target = myClamp(target, min_target, max_target);
}


void GaussianSplatRenderer::setMergeColourMode(GaussianSplatMergeColourMode v)
{
	if(v == splat_merge_colour_params.mode)
		return;
	splat_merge_colour_params.mode = v;
	recolourAllClouds();
}


void GaussianSplatRenderer::setMergeSpreadWiden(float v)
{
	if(v == splat_merge_spread_widen)
		return;
	splat_merge_spread_widen = v;
	rebakeAllClouds();
}


// SESSION071: geometry changed, so unlike recolourAllClouds() this has to go back through bakeMember() - scale drives
// feature_size, cull_radius and the member bounds as well as the colour re-derivation. Still cheap next to a traversal:
// a flat pass over the nodes with no tree walk and no sort, and the draw list is untouched, so what is on screen is the
// same selection drawn at a different size.
void GaussianSplatRenderer::rebakeAllClouds()
{
	for(size_t c=0; c<clouds.size(); ++c)
	{
		SplatCloud& cloud = *clouds[c];
		if(cloud.members.empty())
			continue;
		for(size_t m=0; m<cloud.members.size(); ++m)
			bakeMember(cloud, cloud.members[m], splat_merge_colour_params, splat_merge_spread_widen);
		uploadTexelRowsForSplatRange(cloud, 0, cloud.total_splats);
		rebuildCloudAABB(cloud);
	}
}


// SESSION071: re-derive every loaded cloud's merged-node colours under the current formulation + alpha boost, in place.
//
// Only colours change, so this deliberately does NOT re-bake: pose data (positions/scales/rotations/bounds) is unaffected
// by the formulation, and re-baking 17M splats to change a colour array would make the A/B toggle unusable. A leaf's
// colour is ground truth and is never rewritten, and every merged node is fully re-derived from its children rather than
// accumulated onto its current value, so switching settings back and forth is exactly reversible - covered by test 10 in
// GaussianSplatLodTreeTests.
//
// Skips the LoD traversal entirely: the draw list selects WHICH nodes are drawn, which is unchanged; only the colour each
// selected node carries is different, and that lives in the splat texture the upload below refreshes.
void GaussianSplatRenderer::recolourAllClouds()
{
	for(size_t c=0; c<clouds.size(); ++c)
	{
		SplatCloud& cloud = *clouds[c];
		bool any_recoloured = false;
		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			CloudMember& member = cloud.members[m];
			const std::vector<GaussianSplatLodNode>& tree = member.splat_data->lod_tree;
			if(tree.empty())
				continue; // No tree: every splat is an original leaf, so there are no merged colours to derive.

			// Legacy is what the tree itself carries, so restoring it means re-deriving from the (untouched) leaves rather
			// than reading tree[i].colour back - which would be equivalent here, but only for as long as Legacy stays the
			// formulation the build path bakes in. Going through recolorLodTree() keeps that assumption out of this code.
			recolorLodTree(tree, &cloud.scales[member.offset], &cloud.colours[member.offset], splat_merge_colour_params);
			any_recoloured = true;
		}

		if(any_recoloured)
			uploadTexelRowsForSplatRange(cloud, 0, cloud.total_splats);
	}
}


void GaussianSplatRenderer::rebuildCloudAABB(SplatCloud& cloud)
{
	js::AABBox aabb = js::AABBox::emptyAABBox();
	for(size_t m=0; m<cloud.members.size(); ++m)
		aabb.enlargeToHoldAABBox(cloud.members[m].aabb_ws);

	cloud.aabb_ws = aabb;
	cloud.ob->mesh_data->aabb_os = aabb; // ob_to_world_matrix is identity, so object space is world space here.

	// Refreshes ob->aabb_ws, which drawSplatClouds() culls and orders on.  Skipped for a cloud that hasn't been added to
	// the engine yet: OpenGLEngine::addObject() derives aabb_ws itself, from the bounds just set above.
	if(cloud.added_to_engine)
		opengl_engine->updateObjectTransformData(*cloud.ob);
}


// Draw index that sample 'j' of 'num_samples' was taken from, out of a draw order of 'draw_count' instances.  Written
// once and used from both ends - taking the samples and mapping a sample back to a draw index - so the two cannot drift
// apart.  See GaussianSplatRenderer::getVisibleSlicingEnabled().
static inline int sliceSampleDrawIndex(int j, int num_samples, int draw_count)
{
	return (int)(((int64)j * (int64)draw_count) / (int64)num_samples);
}


// Takes the evenly spaced sample of a cloud's draw order that frustum-aware slicing works from - see
// getVisibleSlicingEnabled().  Called from every place that writes the instance index VBO, so the sample always
// describes the order actually being drawn.
//
// The positions are copied out rather than the indices kept, so that a later renumbering of the cloud cannot turn a
// sample into an out-of-range lookup; a sample that is merely out of date misplaces a boundary, which the slicing
// cannot turn into a visible difference.
//
// Unconditional, rather than skipped while the feature is off: it is a few thousand copies against a VBO upload of the
// whole draw order that has just happened anyway, and having the sample always present is what lets the checkbox be
// switched on mid-session and take effect on the next frame rather than the next traversal.
void GaussianSplatRenderer::noteDrawOrderForSlicing(SplatCloud& cloud, const uint32* draw_indices, size_t count)
{
	cloud.slice_visible_cdf.clear(); // Built against the previous sample, so it does not describe this one.  think() rebuilds it.

	// SESSION066: retain the full draw order so the "Count in frustum" button can report the really-drawn count (draw list
	// tested against the current frustum + slices). A single memcpy of the same indices just uploaded to the VBO; happens
	// only on a draw-order write (traversal/filter land), not per frame.
	// SESSION073: briefly gated behind a live checkbox (draw_order_diag_enabled) to avoid this on rotation once survivors
	// reached a few million - reverted at the owner's request: the report should always reflect what's on screen without
	// having to remember to arm a toggle first.
	cloud.current_draw_indices.resizeNoCopy(count);
	if(count > 0)
		std::memcpy(cloud.current_draw_indices.data(), draw_indices, count * sizeof(uint32));

	const int draw_count = (int)count;
	const int num_samples = myMin(draw_count, max_slice_samples);

	cloud.slice_sample_draw_count = draw_count;
	cloud.slice_sample_positions.resizeNoCopy(num_samples);

	for(int j=0; j<num_samples; ++j)
	{
		const uint32 splat_index = draw_indices[sliceSampleDrawIndex(j, num_samples, draw_count)];
		cloud.slice_sample_positions[j] = (splat_index < cloud.positions.size()) ? cloud.positions[splat_index] : Vec3f(0.f);
	}
}


// Synchronous placeholder frontier, written immediately after a structural change (append or rebuild), so the cloud
// isn't left showing a stale or garbage selection for the one-to-a-few-frame gap before the next background traversal
// completes.  Root-only for a member with a built tree (the cheapest non-empty frontier); every splat for a member
// without one, since there's nothing to choose between without a tree.  Degrades to exactly the pre-LoD "draw
// everything" behaviour when no member in the cloud has a tree at all.
void GaussianSplatRenderer::writePlaceholderSelection(SplatCloud& cloud)
{
	js::Vector<uint32, 16> selection;
	selection.reserve(cloud.members.size());

	for(size_t m=0; m<cloud.members.size(); ++m)
	{
		const CloudMember& member = cloud.members[m];
		if(member.hidden) // SESSION059 - see GaussianSplatRenderer::setObjectHidden().
			continue;
		if(member.splat_data->lod_tree.empty())
			for(size_t i=0; i<member.count; ++i)
				selection.push_back((uint32)(member.offset + i));
		else
			selection.push_back((uint32)member.offset); // Root is always node 0 of a member's tree - see buildGaussianSplatLodTree().
	}

	cloud.instance_index_vbo->updateData(0, selection.data(), selection.size() * sizeof(uint32));
	cloud.ob->num_instances_to_draw = (int)selection.size();
	noteDrawOrderForSlicing(cloud, selection.data(), selection.size());
}


void GaussianSplatRenderer::appendMemberToCloud(SplatCloud& cloud, const CloudMember& member_in)
{
	const size_t old_total = cloud.total_splats;
	const size_t new_total = old_total + member_in.count;

	cloud.positions.resize(new_total);
	cloud.scales   .resize(new_total);
	cloud.rotations.resize(new_total);
	cloud.colours  .resize(new_total);
	cloud.feature_size.resize(new_total); // SESSION055 - see SplatCloud::feature_size.
	cloud.cull_radius.resize(new_total); // SESSION059 - see SplatCloud::cull_radius.

	cloud.members.push_back(member_in);
	CloudMember& member = cloud.members.back();
	member.offset = old_total;
	bakeMember(cloud, member, splat_merge_colour_params, splat_merge_spread_widen);

	cloud.total_splats = new_total;

	ensureGpuCapacity(cloud, new_total);

	uploadTexelRowsForSplatRange(cloud, old_total, member_in.count); // If ensureGpuCapacity() just repacked everything, this re-uploads the same correct data, which is harmless.
	writeIdentityIndices(cloud, old_total, member_in.count); // Again redundant but harmless if ensureGpuCapacity() just rebuilt the whole VBO.

	rebuildCloudAABB(cloud);
	addCloudToEngineIfNeeded(cloud); // For the first member: only now does the cloud have real bounds and an instance count for the engine to cache.

	// The appended splats are in identity order relative to the rest, so the cloud needs a fresh sort.  This doesn't
	// bump structure_generation: a pure append leaves existing indices meaningful, so an in-flight sort's result still
	// applies to the prefix it covers.
	cloud.have_last_sort_cam_pos = false;

	// A new member changes what this cloud's LoD frontier should be, and unlike the sort above, an in-flight traversal's
	// result genuinely can't be trusted afterwards - it was computed against a member set that didn't include this one.
	// topology_generation is a stricter counter than structure_generation for exactly this reason (see its declaration).
	// writePlaceholderSelection() stands in with a cheap synchronous frontier until the next background traversal - kicked
	// off because of the generation bump below, even if the camera hasn't moved - lands.
	cloud.topology_generation++;
	writePlaceholderSelection(cloud);
}


void GaussianSplatRenderer::rebuildCloud(SplatCloud& cloud)
{
	size_t total = 0;
	for(size_t m=0; m<cloud.members.size(); ++m)
	{
		cloud.members[m].offset = total;
		total += cloud.members[m].count;
	}

	cloud.total_splats = total;
	cloud.positions.resize(total);
	cloud.scales   .resize(total);
	cloud.rotations.resize(total);
	cloud.colours  .resize(total);
	cloud.feature_size.resize(total); // SESSION055 - see SplatCloud::feature_size.
	cloud.cull_radius.resize(total); // SESSION059 - see SplatCloud::cull_radius.

	for(size_t m=0; m<cloud.members.size(); ++m)
		bakeMember(cloud, cloud.members[m], splat_merge_colour_params, splat_merge_spread_widen);

	ensureGpuCapacity(cloud, total);

	uploadTexelRowsForSplatRange(cloud, 0, total);
	writeIdentityIndices(cloud, 0, total);

	rebuildCloudAABB(cloud);
	addCloudToEngineIfNeeded(cloud);

	cloud.structure_generation++; // Every member was renumbered, so any in-flight sort's indices no longer mean the same splats.
	cloud.have_last_sort_cam_pos = false;

	// Every member's offset just moved, so an in-flight traversal's result (indices computed against the old offsets)
	// no longer means the same nodes - topology_generation++ (a stricter counter than structure_generation, see its
	// declaration) makes kickOffTraversals() treat this cloud as unconditionally overdue for a fresh one.
	cloud.topology_generation++;
	writePlaceholderSelection(cloud);
}


void GaussianSplatRenderer::mergeIntersectingClouds(SplatCloud& seed_cloud)
{
	const size_t max_splats = maxSplatsPerCloud();

	size_t merged_total = seed_cloud.total_splats;
	bool merged_any = false;
	bool changed = true;
	while(changed)
	{
		changed = false;

		// Absorbing a cloud grows seed_cloud's bounds, which can bring a cloud that was previously disjoint - including
		// one already passed over in this scan - into contact.  Hence the outer loop: this runs to a fixpoint.
		for(size_t i=0; i<clouds.size(); )
		{
			SplatCloud* const other = clouds[i].ptr();
			if(other == &seed_cloud || !seed_cloud.aabb_ws.intersectsAABB(other->aabb_ws))
			{
				++i;
				continue;
			}

			if(merged_total + other->total_splats > max_splats)
			{
				// The merged cloud wouldn't fit in one data texture.  Leave the two separate: their relative draw order
				// becomes approximate, which is a far better failure than not rendering one of them at all.
				conPrint("GaussianSplatRenderer: not merging two intersecting splat clouds, as the result would exceed the " +
					toString(max_splats) + " splat per-cloud limit.  Their relative draw order will be approximate.");
				++i;
				continue;
			}

			for(size_t m=0; m<other->members.size(); ++m)
			{
				seed_cloud.members.push_back(other->members[m]);
				handle_to_cloud[other->members[m].handle] = &seed_cloud;
			}
			merged_total += other->total_splats;

			// Union the bounds now rather than waiting for the rebuild below, since the fixpoint test above needs them.
			seed_cloud.aabb_ws.enlargeToHoldAABBox(other->aabb_ws);

			const Reference<SplatCloud> other_ref = clouds[i]; // Keep alive across the erase inside destroyCloud().
			destroyCloud(other_ref);

			merged_any = true;
			changed = true;
			// Don't advance i: clouds[] shifted down over the erased entry.
		}
	}

	// One rebuild for the whole merge, rather than one per absorbed cloud.
	if(merged_any)
		rebuildCloud(seed_cloud);
}


// Builds the merged copy of one member's loaded splats, LoD tree and all.  Object space throughout: the tolerances have
// already been converted by the caller, and the tree is built in the same space the source's was, so the result bakes to
// world space by exactly the path an unmerged member takes (see bakeMember()).
static GaussianSplatDataRef mergedSplatData(const GaussianSplatData& src, const GaussianSplatCoplanarMergeParams& params_os, float lod_base,
	GaussianSplatCoplanarMergeStats& stats_out)
{
	Reference<GaussianSplatData> merged = new GaussianSplatData();
	merged->positions = src.positions;
	merged->scales    = src.scales;
	merged->rotations = src.rotations;
	merged->colours   = src.colours;

	stats_out = coplanarMergeSplats(merged->positions, merged->scales, merged->rotations, merged->colours, params_os);

	// Recomputed rather than copied: merging moves centres, so the source's bounds are no longer this cloud's.  Same
	// convention as the decoder's - the splat centres only, not their extent (see GaussianSplatData::aabb_os).
	merged->aabb_os = js::AABBox::emptyAABBox();
	for(size_t i=0; i<merged->positions.size(); ++i)
		merged->aabb_os.enlargeToHoldPoint(Vec4f(merged->positions[i].x, merged->positions[i].y, merged->positions[i].z, 1.f));

	// The tree describes the splats it was built from, so a merged cloud needs a new one - keeping the old tree would
	// leave its leaves pointing at splats that no longer exist.
	try
	{
		if(merged->numSplats() > 0)
			merged->lod_tree = buildGaussianSplatLodTree(merged->positions.data(), merged->scales.data(), merged->rotations.data(), merged->colours.data(),
				merged->numSplats(), lod_base);
	}
	catch(std::exception&)
	{
		// As at load time: the tree is a nice-to-have, and an empty one means "no LoD, draw every splat" rather than a
		// broken cloud (see GaussianSplatData::lod_tree).
		merged->lod_tree.clear();
	}

	return merged;
}


std::string GaussianSplatRenderer::applyCoplanarMerge(const GaussianSplatCoplanarMergeParams& params_ws, float lod_base)
{
	Timer timer;

	// One source cloud can be registered several times over (GUIClient's splat_data_cache serves repeat loads of a URL to
	// every object using it), and the merge is object-space work that doesn't depend on where the object stands - so it is
	// done once per (source, scale) pair and the result shared.  The scale is part of the key because it is what converts
	// the world-space tolerances into object space: two objects at different scales are genuinely different merges.
	struct MergedSource
	{
		GaussianSplatDataRef data;
		GaussianSplatCoplanarMergeStats stats;
	};
	std::map<std::pair<const GaussianSplatData*, float>, MergedSource> merged_for_source;

	size_t total_in = 0, total_out = 0, groups_merged = 0, groups_refused = 0, groups_total = 0, largest_merged = 0, largest_found = 0, members_merged = 0;
	double area_in = 0, area_out = 0;

	for(size_t c=0; c<clouds.size(); ++c)
	{
		SplatCloud& cloud = *clouds[c];

		size_t merged_here = 0;
		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			CloudMember& member = cloud.members[m];
			const GaussianSplatDataRef src = member.unmerged_splat_data.nonNull() ? member.unmerged_splat_data : member.splat_data;
			if(src->numSplats() < 2)
				continue;

			const float scale = myMax(member.uniform_scale_ws, 1.0e-6f); // A zero-scaled object would divide by zero below; it draws as nothing anyway.

			const std::pair<const GaussianSplatData*, float> key(src.ptr(), scale);
			std::map<std::pair<const GaussianSplatData*, float>, MergedSource>::iterator res = merged_for_source.find(key);
			if(res == merged_for_source.end())
			{
				GaussianSplatCoplanarMergeParams params_os = params_ws;
				params_os.across  = params_ws.across  / scale;
				params_os.through = params_ws.through / scale;
				params_os.alpha_cutoff = splat_alpha_cutoff; // Not the caller's to choose: it is where this renderer cuts a quad off, and the merge's "does this group pay?" test has to use the same one - see the field's comment.
				params_os.alpha_gain  = splat_alpha_gain;   // Same reason: the quad is sized from the adjusted alpha, so the areas the test compares have to be too - see getAlphaGain().
				params_os.alpha_gamma = splat_alpha_gamma;

				MergedSource entry;
				entry.data = mergedSplatData(*src, params_os, lod_base, entry.stats);
				res = merged_for_source.insert(std::make_pair(key, entry)).first;
			}

			// Accumulated per member rather than per distinct source: two objects sharing one decoded cloud are two
			// clouds' worth of splats on the screen, and these figures are about the world, not about the file.
			const GaussianSplatCoplanarMergeStats& stats = res->second.stats;
			total_in += stats.splats_in;
			total_out += stats.splats_out;
			groups_merged += stats.groups_merged;
			groups_refused += stats.groups_refused;
			groups_total += stats.groups_total;
			largest_merged = myMax(largest_merged, stats.largest_group_merged);
			largest_found = myMax(largest_found, stats.largest_group_found);
			area_in += stats.area_in * (double)scale * (double)scale; // Object-space areas, reported in world units so that a world of differently-scaled objects sums to something meaningful.
			area_out += stats.area_out * (double)scale * (double)scale;

			member.unmerged_splat_data = src;
			member.splat_data = res->second.data;
			member.count = res->second.data->numNodes();
			merged_here++;
		}

		// Member counts have changed, so every offset in this cloud has moved: the full rebuild is the path that already
		// handles that (re-bake, re-upload, generation bumps, placeholder frontier).  The importance record is dropped by
		// its own layout fingerprint, which is the right outcome - it was accumulated against splats that no longer exist.
		if(merged_here > 0)
		{
			rebuildCloud(cloud);
			members_merged += merged_here;
		}
	}

	if(members_merged == 0)
		return "Coplanar merge: no splat cloud registered, nothing done.";

	// A replacement is a moment match over its group, so it can reach a little past the members it stood in for - which
	// means a cloud's bounds can grow by a few centimetres, and two clouds that were disjoint before might now touch.
	// Under-merging is the direction that breaks the cross-cloud draw order (see the Partitioning note in the header), so
	// the test is re-run rather than assumed.  Restarting on any change because absorbing a cloud shifts clouds[] under
	// the index; the list is a handful of entries, so the cost of being crude here is nil.
	for(size_t i=0; i<clouds.size(); )
	{
		const size_t num_before = clouds.size();
		mergeIntersectingClouds(*clouds[i]);
		if(clouds.size() == num_before)
			++i;
		else
			i = 0;
	}

	// Kept as well as returned: the frustum report prints it, so every report taken from here on says which merge it is
	// describing without that having to be remembered from further up the log.
	last_merge_description = "merged at across " + doubleToStringNDecimalPlaces(params_ws.across * 100.0, 1) + " cm, through " +
		doubleToStringNDecimalPlaces(params_ws.through * 100.0, 1) + " cm, colour " + doubleToStringNDecimalPlaces(params_ws.colour_tol, 3) +
		", angle " + doubleToStringNDecimalPlaces(params_ws.angle_tol_deg, 0) + " deg, " +
		(params_ws.flatten_onto_surface ? "flattened onto the surface" : "NOT flattened (replacement fitted around the group's depth)") + " - " +
		uInt64ToStringCommaSeparated(total_in) + " splats -> " + uInt64ToStringCommaSeparated(total_out);

	std::string s = "Coplanar merge at across " + doubleToStringNDecimalPlaces(params_ws.across * 100.0, 1) + " cm, through " +
		doubleToStringNDecimalPlaces(params_ws.through * 100.0, 1) + " cm, colour " + doubleToStringNDecimalPlaces(params_ws.colour_tol, 3) +
		", angle " + doubleToStringNDecimalPlaces(params_ws.angle_tol_deg, 0) + " deg, " +
		(params_ws.flatten_onto_surface ? "flattened" : "not flattened") + ":\n";
	s += "  Splats:  " + uInt64ToStringCommaSeparated(total_in) + " -> " + uInt64ToStringCommaSeparated(total_out) + "  (" +
		doubleToStringNDecimalPlaces((total_in > 0) ? (100.0 * (1.0 - (double)total_out / (double)total_in)) : 0.0, 1) + "% removed)\n";
	s += "  Area drawn, face-on: " + doubleToStringNDecimalPlaces(area_in, 1) + " -> " + doubleToStringNDecimalPlaces(area_out, 1) + " m^2  (" +
		doubleToStringNDecimalPlaces((area_in > 0) ? (100.0 * (1.0 - area_out / area_in)) : 0.0, 1) + "% saved)\n";
	s += "    The ellipse the shader really rasterises, cutoff radius and all, but seen face-on and at unit distance - so\n";
	s += "    it is the view-independent half of the answer, weighting a wall the camera never looks at as heavily as the\n";
	s += "    one in front of it.  What the pass is bound by is on-screen fill: read the frustum report's \"blended\" line\n";
	s += "    and its layers-to-saturation figure, from the same spot, before and after.\n";
	s += "  Groups collapsed: " + uInt64ToStringCommaSeparated(groups_merged) + ", largest " + uInt64ToStringCommaSeparated(largest_merged) +
		"; refused for painting more than they replaced: " + uInt64ToStringCommaSeparated(groups_refused) + "\n";
	// Without this line, "nothing was merged" has two readings that want opposite next steps - the grouping reached
	// nothing, or it reached plenty and the area test threw it out.  Session 42 spent itself on that ambiguity.
	s += "  What the grouping found before judging any of it: " + uInt64ToStringCommaSeparated(groups_total) + " groups, " +
		doubleToStringNDecimalPlaces((groups_total > 0) ? ((double)total_in / (double)groups_total) : 0.0, 2) + " splats each on average, largest " +
		uInt64ToStringCommaSeparated(largest_found) + ".\n";
	s += "    A largest of 1 there means the reach or the tolerances found nothing to consider; a large one beside few\n";
	s += "    collapses means it found plenty and the area test refused it, which is the arithmetic, not a bug.\n";
	s += "  Took " + doubleToStringNDecimalPlaces(timer.elapsed(), 2) + " s, LoD trees rebuilt.";
	return s;
}


std::string GaussianSplatRenderer::restoreUnmergedSplats()
{
	Timer timer;

	size_t members_restored = 0, total_splats = 0;

	for(size_t c=0; c<clouds.size(); ++c)
	{
		SplatCloud& cloud = *clouds[c];

		size_t restored_here = 0;
		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			CloudMember& member = cloud.members[m];
			if(member.unmerged_splat_data.isNull())
				continue;

			member.splat_data = member.unmerged_splat_data;
			member.unmerged_splat_data = GaussianSplatDataRef();
			member.count = member.splat_data->numNodes();
			total_splats += member.splat_data->numSplats();
			restored_here++;
		}

		if(restored_here > 0)
		{
			rebuildCloud(cloud);
			members_restored += restored_here;
		}
	}

	if(members_restored > 0)
		last_merge_description.clear(); // So the next report describes itself as taken on the loaded cloud.

	if(members_restored == 0)
		return "Nothing to restore: no cloud is merged.";

	return "Restored " + uInt64ToStringCommaSeparated(members_restored) + " splat object(s) to their loaded splats (" +
		uInt64ToStringCommaSeparated(total_splats) + " splats), in " + doubleToStringNDecimalPlaces(timer.elapsed(), 2) + " s.";
}


// SESSION073: see the header's comment. Builds a fresh GaussianSplatData per distinct source pointer (never mutates one
// in place - an in-flight traversal/filter task may hold its own Reference to the object being rebuilt, mid-read, on a
// worker thread; swapping member.splat_data to a new object is what keeps that read safe, same as applyCoplanarMerge()
// above), then hands rebuildCloud() the swapped-in members exactly as applyCoplanarMerge() does.
std::string GaussianSplatRenderer::rebuildAllLodTrees(float lod_base)
{
	Timer timer;

	std::map<const GaussianSplatData*, GaussianSplatDataRef> rebuilt_for_source;

	size_t members_rebuilt = 0, total_splats = 0, total_nodes = 0;

	for(size_t c=0; c<clouds.size(); ++c)
	{
		SplatCloud& cloud = *clouds[c];

		size_t rebuilt_here = 0;
		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			CloudMember& member = cloud.members[m];
			const GaussianSplatDataRef& src = member.splat_data;
			if(src.isNull() || src->numSplats() < 2)
				continue;

			std::map<const GaussianSplatData*, GaussianSplatDataRef>::iterator res = rebuilt_for_source.find(src.ptr());
			if(res == rebuilt_for_source.end())
			{
				Reference<GaussianSplatData> rebuilt = new GaussianSplatData();
				rebuilt->positions = src->positions;
				rebuilt->scales    = src->scales;
				rebuilt->rotations = src->rotations;
				rebuilt->colours   = src->colours;
				rebuilt->aabb_os   = src->aabb_os;

				try
				{
					rebuilt->lod_tree = buildGaussianSplatLodTree(rebuilt->positions.data(), rebuilt->scales.data(), rebuilt->rotations.data(), rebuilt->colours.data(),
						rebuilt->numSplats(), lod_base);
				}
				catch(std::exception&)
				{
					// As at load time: the tree is a nice-to-have, and an empty one means "no LoD, draw every splat" rather
					// than a broken cloud - see mergedSplatData()'s own catch and GaussianSplatData::lod_tree's comment.
					rebuilt->lod_tree.clear();
				}

				res = rebuilt_for_source.insert(std::make_pair(src.ptr(), rebuilt)).first;
			}

			member.splat_data = res->second;
			member.count = res->second->numNodes();
			total_splats += res->second->numSplats();
			total_nodes += res->second->numNodes();
			rebuilt_here++;
		}

		if(rebuilt_here > 0)
		{
			rebuildCloud(cloud);
			members_rebuilt += rebuilt_here;
		}
	}

	if(members_rebuilt == 0)
		return "Rebuild LoDs: no splat cloud registered, nothing done.";

	return "Rebuilt " + uInt64ToStringCommaSeparated(members_rebuilt) + " splat object(s) at lod_base " + doubleToStringNDecimalPlaces(lod_base, 3) + " (" +
		uInt64ToStringCommaSeparated(total_splats) + " splats, " + uInt64ToStringCommaSeparated(total_nodes) + " tree nodes total), in " +
		doubleToStringNDecimalPlaces(timer.elapsed(), 2) + " s.";
}


GaussianSplatRenderer::Handle GaussianSplatRenderer::addObject(const GaussianSplatDataRef& splat_data, const Vec4f& translation_ws,
	const Quat<float>& rotation_ws, float uniform_scale_ws)
{
	buildShadersIfNeeded();

	const size_t max_splats = maxSplatsPerCloud();
	if(splat_data->numNodes() > max_splats) // Node count if an LoD tree has been built (wider than leaf count, since it includes merged internal nodes too), else leaf count - see GaussianSplatData::numNodes().
		throw glare::Exception("Can't render a splat cloud with " + toString(splat_data->numNodes()) + " nodes: the per-cloud limit is " + toString(max_splats) + ".");

	CloudMember member;
	member.handle = next_handle++;
	member.splat_data = splat_data;
	member.offset = 0; // Assigned by appendMemberToCloud().
	member.count = splat_data->numNodes();
	member.translation_ws = translation_ws;
	member.rotation_ws = rotation_ws;
	member.uniform_scale_ws = uniform_scale_ws;
	member.hidden = false; // SESSION059 - every object starts visible; see CloudMember::hidden.

	// Start the member in a cloud of its own and then let the partitioning merge it, rather than deciding up front
	// which cloud it belongs in.  Baking is what produces the member's bounds, and the bounds are what the merge test
	// needs, so this way the common case - a capture that intersects nothing - bakes exactly once.
	Reference<SplatCloud> cloud = allocCloud();
	appendMemberToCloud(*cloud, member);
	handle_to_cloud[member.handle] = cloud.ptr();

	mergeIntersectingClouds(*cloud); // Absorbs any cloud this one now touches.  cloud itself always survives.

	return member.handle;
}


bool GaussianSplatRenderer::updateObjectTransform(Handle handle, const Vec4f& translation_ws, const Quat<float>& rotation_ws, float uniform_scale_ws)
{
	const std::map<Handle, SplatCloud*>::iterator res = handle_to_cloud.find(handle);
	if(res == handle_to_cloud.end())
		return false;

	SplatCloud& cloud = *res->second;

	for(size_t m=0; m<cloud.members.size(); ++m)
	{
		CloudMember& member = cloud.members[m];
		if(member.handle == handle)
		{
			member.translation_ws = translation_ws;
			member.rotation_ws = rotation_ws;
			member.uniform_scale_ws = uniform_scale_ws;

			bakeMember(cloud, member, splat_merge_colour_params, splat_merge_spread_widen); // Re-bakes in place: offsets and counts are unchanged.
			uploadTexelRowsForSplatRange(cloud, member.offset, member.count);
			rebuildCloudAABB(cloud);

			// This member's splats may now be in the wrong depth order relative to the rest of the cloud.  This doesn't
			// bump structure_generation: offsets and counts are unchanged, so an in-flight sort's indices stay meaningful.
			// Still needed for a cloud with no LoD tree at all - see kickOffSorts().
			cloud.have_last_sort_cam_pos = false;

			// SESSION057: for an LoD-active cloud, kickOffSorts() above does nothing - it unconditionally skips any cloud
			// with an LoD tree, because sorting is folded into the traversal's selection instead (see kickOffSorts()'s
			// cloudHasLodTree check and comment). So moving/rotating/rescaling a member here previously left the cloud's
			// draw order and LoD frontier stale until the *camera* moved far enough to naturally re-kick a traversal -
			// visibly, moving one splat object inside another (e.g. an avatar posed inside a building capture) did not
			// re-sort. Bumping topology_generation makes kickOffTraversals() treat the cloud as unconditionally overdue,
			// the same mechanism appendMemberToCloud()/rebuildCloud() already use for their own structural changes - a
			// bake here is exactly that "any add or rebuild" this counter is defined to catch. It also makes
			// drainTraversalResults() drop any in-flight traversal computed against this member's old world-space pose.
			//
			// Deliberately NOT calling writePlaceholderSelection() here, unlike those two: this is a live per-drag-frame
			// path (a gizmo drag or a UI transform spinbox calls updateObjectTransform() every frame while editing), and
			// the placeholder collapses the *whole cloud* to one root node per member - visibly flashing the entire
			// merged cloud down to a handful of splats every frame during a drag, not just the member being moved. That
			// collapse is unnecessary here anyway: offsets/counts are unchanged, so the previous selection's indices
			// still point at valid data - bakeMember() already overwrote it in place with the new pose - just possibly
			// not the ideal LoD choice for the new pose until the forced-overdue traversal above lands.
			cloud.topology_generation++;

			// The cloud's bounds have moved, so it may now touch clouds it didn't before.  Note that the reverse is not
			// checked: a cloud is never split back apart once merged.  An over-merged cloud draws correctly, just with
			// coarser culling, whereas splitting one that shouldn't be split is what breaks the ordering.
			mergeIntersectingClouds(cloud);

			return true;
		}
	}

	assert(0); // handle_to_cloud pointed at a cloud that doesn't hold this member.
	return false;
}


bool GaussianSplatRenderer::setObjectHidden(Handle handle, bool hidden)
{
	const std::map<Handle, SplatCloud*>::iterator res = handle_to_cloud.find(handle);
	if(res == handle_to_cloud.end())
		return false;

	SplatCloud& cloud = *res->second;
	for(size_t m=0; m<cloud.members.size(); ++m)
		if(cloud.members[m].handle == handle)
		{
			if(cloud.members[m].hidden == hidden)
				return true; // No-op - avoid the synchronous placeholder flash below for a toggle that didn't change anything.

			cloud.members[m].hidden = hidden;

			// Immediate synchronous feedback (same tool writeIdentityIndices()... no, writePlaceholderSelection() uses
			// for a structural change): the toggle is a debug action a person is watching happen, not a per-frame drag,
			// so - unlike updateObjectTransform()'s live-drag path just above, which deliberately skips this exact call
			// to avoid a flash - collapsing to root-only immediately here is the right trade: instant confirmation the
			// checkbox did something, at the cost of one frame at coarse LoD until the traversal below lands with full
			// detail (minus the now-hidden member).
			writePlaceholderSelection(cloud);

			// Force a fresh traversal/sort so the change also takes effect for whichever path is live for this cloud
			// (cloudHasLodTree() decides which - see kickOffTraversals()/kickOffSorts()), the same "unconditionally
			// overdue" mechanism forceTraversalRefresh() uses, just scoped to this one cloud rather than the whole world.
			cloud.have_last_traversal_cam_pos = false;
			cloud.have_last_sort_cam_pos = false;

			return true;
		}

	assert(0); // handle_to_cloud pointed at a cloud that doesn't hold this member.
	return false;
}


bool GaussianSplatRenderer::getObjectHidden(Handle handle) const
{
	const std::map<Handle, SplatCloud*>::const_iterator res = handle_to_cloud.find(handle);
	if(res == handle_to_cloud.end())
		return false;

	const SplatCloud& cloud = *res->second;
	for(size_t m=0; m<cloud.members.size(); ++m)
		if(cloud.members[m].handle == handle)
			return cloud.members[m].hidden;

	assert(0);
	return false;
}


bool GaussianSplatRenderer::removeObject(Handle handle)
{
	const std::map<Handle, SplatCloud*>::iterator res = handle_to_cloud.find(handle);
	if(res == handle_to_cloud.end())
		return false;

	SplatCloud* const cloud = res->second;
	handle_to_cloud.erase(res);

	for(size_t m=0; m<cloud->members.size(); ++m)
		if(cloud->members[m].handle == handle)
		{
			cloud->members.erase(cloud->members.begin() + m);
			break;
		}

	if(cloud->members.empty())
	{
		// The common case, since most objects end up in a cloud of their own.
		Reference<SplatCloud> cloud_ref = cloud; // Keep alive across the erase inside destroyCloud().
		destroyCloud(cloud_ref);
	}
	else
		rebuildCloud(*cloud); // Removing a member renumbers everything after it, so the whole cloud is re-baked and re-uploaded.

	return true;
}


void GaussianSplatRenderer::removeAllObjects()
{
	for(size_t i=0; i<clouds.size(); ++i)
		opengl_engine->removeObject(clouds[i]->ob);

	clouds.clear();
	handle_to_cloud.clear();

	// Any sorts still in flight are left to run: their results carry cloud ids that no longer match a live cloud, so
	// drainSortResults() drops them and returns their scratch to the pool.
}


void GaussianSplatRenderer::drainSortResults()
{
	sort_result_queue.dequeueAnyQueuedItems(completed_msgs);

	// SESSION069 - any completed sort changed at least one cloud's draw order this frame, so the TAA history is stale
	// for whichever pixels those splats moved through and must be reset.  A spurious reset (all msgs referenced dropped
	// clouds) costs 4 frames back to full quality, cheap vs. the visible smear a missed reset would cause.
	if(!completed_msgs.empty()) noteContentApplied();

	for(size_t i=0; i<completed_msgs.size(); ++i)
	{
		const GaussianSplatSortResultMsg* const msg = static_cast<const GaussianSplatSortResultMsg*>(completed_msgs[i].ptr());

		SplatCloud* cloud = NULL;
		for(size_t c=0; c<clouds.size(); ++c)
			if(clouds[c]->cloud_id == msg->cloud_id)
			{
				cloud = clouds[c].ptr();
				break;
			}

		if(msg->stage == GaussianSplatSortResultMsg::Stage_Precise)
		{
			// Bookkeeping first, and unconditionally: the sort has finished and its scratch is free to reuse whether or
			// not the cloud it was for still exists.
			num_sorts_in_flight--;
			free_scratch.push_back(msg->scratch);
			if(cloud)
				cloud->sort_in_flight = false;
		}

		// Drop results for a cloud that has since been merged away or removed, and results computed before a
		// renumbering, whose indices no longer mean the same splats.
		if(!cloud || msg->generation != cloud->structure_generation)
			continue;

		// If a later message in this same batch is for the same cloud and generation, that one supersedes this one.
		// Both stages can land in the same frame on a fast-sorting cloud, and uploading the coarse order only to
		// overwrite it with the precise order in the same frame is a pointless buffer upload.
		bool superseded_this_frame = false;
		for(size_t j=i + 1; j<completed_msgs.size(); ++j)
		{
			const GaussianSplatSortResultMsg* const later = static_cast<const GaussianSplatSortResultMsg*>(completed_msgs[j].ptr());
			if(later->cloud_id == msg->cloud_id && later->generation == msg->generation)
			{
				superseded_this_frame = true;
				break;
			}
		}

		if(!superseded_this_frame)
		{
			// The snapshot this was computed from may be a strict prefix of the current, possibly since-grown cloud.
			// Only write as many bytes as the result actually covers: any appended tail beyond it already holds valid
			// identity-order indices written by appendMemberToCloud().
			// Sampled over exactly what this result covers (see noteDrawOrderForSlicing() below).  On a cloud that grew
			// since the sort was kicked off that is a prefix of the draw order, leaving the appended tail unsampled for
			// the frame or two until the next sort lands - which skews where the boundaries fall slightly and can do
			// nothing else, since slicing cannot change the picture.
			const js::Vector<uint32, 16>& sorted_indices = msg->sortedIndices();

			// SESSION059: this path (a cloud with no LoD-active member - see kickOffSorts()'s cloudHasLodTree() guard)
			// has no per-member awareness inside GaussianSplatSortTask itself, unlike the traversal path, which skips a
			// hidden member's nodes during expand (see GaussianSplatLodTraversalTask::run()). Filtering here instead, on
			// every landed sort, is what keeps a hidden member hidden across re-sorts - the one-off filter
			// writePlaceholderSelection() applies at toggle time (see setObjectHidden()) would otherwise be undone by
			// the very next sort completing with the full, unfiltered order. Common case (nothing hidden) skips the scan
			// entirely - this is a debug tool, not a path worth a permanent per-cloud "any hidden" cache.
			bool any_hidden = false;
			for(size_t m=0; m<cloud->members.size(); ++m)
				if(cloud->members[m].hidden) { any_hidden = true; break; }

			if(!any_hidden)
			{
				cloud->instance_index_vbo->updateData(0, sorted_indices.data(), sorted_indices.size() * sizeof(uint32));
				noteDrawOrderForSlicing(*cloud, sorted_indices.data(), sorted_indices.size());
			}
			else
			{
				js::Vector<uint32, 16> filtered;
				filtered.reserve(sorted_indices.size());
				for(size_t k=0; k<sorted_indices.size(); ++k)
				{
					const uint32 idx = sorted_indices[k];
					bool hidden = false;
					for(size_t m=0; m<cloud->members.size(); ++m)
					{
						const CloudMember& member = cloud->members[m];
						if(idx >= member.offset && idx < member.offset + member.count) { hidden = member.hidden; break; }
					}
					if(!hidden)
						filtered.push_back(idx);
				}
				cloud->instance_index_vbo->updateData(0, filtered.data(), filtered.size() * sizeof(uint32));
				cloud->ob->num_instances_to_draw = (int)filtered.size();
				noteDrawOrderForSlicing(*cloud, filtered.data(), filtered.size());
			}
		}
	}

	completed_msgs.clear(); // Drop the references, so a scratch just returned to the pool isn't kept alive by a stale message.
}


void GaussianSplatRenderer::kickOffSorts()
{
	glare::TaskManager* const task_manager = opengl_engine->getMainTaskManager();
	if(task_manager == NULL)
		return;

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Vec4f cam_pos_ws = scene->cam_to_world.getColumn(3);

	while(num_sorts_in_flight < max_concurrent_sorts)
	{
		// Pick the cloud most overdue for a sort, measured as how far the camera has moved relative to that cloud's own
		// threshold.  With a cap on concurrent sorts, a world of many clouds would otherwise service them in an
		// arbitrary order, and the nearby cloud whose order actually looks wrong could be starved by distant ones.
		SplatCloud* best_cloud = NULL;
		float best_ratio = 1.f; // A cloud has to be past its own threshold to be worth sorting at all.
		for(size_t i=0; i<clouds.size(); ++i)
		{
			SplatCloud* const cloud = clouds[i].ptr();
			if(cloud->total_splats == 0 || cloud->sort_in_flight)
				continue;

			// This sort targets the cloud's whole [0, total_splats) range, which for an LoD-active cloud is wider than
			// what's actually being drawn (num_instances_to_draw is the traversal selection's size, not total_splats -
			// see drainTraversalResults()).  Sorting the whole range would overwrite the selection at the front of
			// instance_index_vbo with an unrelated full-cloud order.  An LoD-active cloud doesn't need this sort anyway:
			// GaussianSplatLodTraversalTask::run() already sorts the selection itself, front-to-back, on the same worker
			// thread that computed it - see its decorate-sort at the end of run() - so this whole-cloud sort is left to
			// clouds with no LoD tree at all, which still draw every splat and still need it.
			if(cloudHasLodTree(*cloud))
				continue;

			float ratio;
			if(!cloud->have_last_sort_cam_pos)
				ratio = std::numeric_limits<float>::max(); // Never sorted, or invalidated by a change to the cloud.
			else
			{
				const float threshold = myMax(min_resort_move_threshold_ws, cloud->aabb_ws.distanceToPoint(cam_pos_ws) * resort_threshold_dist_fraction);
				ratio = cam_pos_ws.getDist(cloud->last_sort_cam_pos_ws) / threshold;
			}

			if(ratio > best_ratio)
			{
				best_ratio = ratio;
				best_cloud = cloud;
			}
		}

		if(best_cloud == NULL)
			break;

		Reference<GaussianSplatSortScratch> scratch;
		if(free_scratch.empty())
			scratch = new GaussianSplatSortScratch();
		else
		{
			scratch = free_scratch.back();
			free_scratch.pop_back();
		}

		// Freeze a snapshot of the cloud's current positions on the main thread before handing off to the worker, so the
		// worker never touches the live, growable arrays.
		scratch->positions_snapshot.resizeNoCopy(best_cloud->total_splats);
		std::memcpy(scratch->positions_snapshot.data(), best_cloud->positions.data(), best_cloud->total_splats * sizeof(Vec3f));

		best_cloud->sort_in_flight = true;
		best_cloud->have_last_sort_cam_pos = true;
		best_cloud->last_sort_cam_pos_ws = cam_pos_ws;
		num_sorts_in_flight++;

		Matrix4f world_to_cam;
		scene->cam_to_world.getInverseForAffine3Matrix(world_to_cam);

		task_manager->addTask(new GaussianSplatSortTask(best_cloud->cloud_id, best_cloud->structure_generation, scratch, world_to_cam, &sort_result_queue));
	}
}


// SESSION076 §9 / SESSION077 / SESSION081: (re)build cloud's overlay textures from the barrier's diagnostic
// accumulators - see drainSaturationBuildResults()'s call site and GaussianSplatRenderer::getSatGridDebugInfo().
// Always allocates fresh textures rather than updating in place: this only runs once per BUILD while the overlay is
// requested, so the extra allocation cost is not worth avoiding, and fresh textures mean a resolution change
// (sat_grid_subdiv) can never be applied to a texture sized for the old resolution.
void GaussianSplatRenderer::updateSatGridDebugTexture(SplatCloud& cloud, const GaussianSplatSaturationBarrier& barrier)
{
	const size_t res = (size_t)barrier.sat_grid_res;

	// The barrier may predate the overlay being switched on, in which case there is nothing to show.
	if(barrier.sat_accum_t.size() != res * res || barrier.sat_amp_sum.size() != res * res)
	{
		cloud.sat_grid_debug_tex = NULL;
		cloud.sat_grid_debug_ramp_tex = NULL;
		cloud.sat_grid_debug_maskfix_tex = NULL;
		return;
	}

	cloud.sat_grid_debug_tex = new OpenGLTexture(res, res, opengl_engine,
		ArrayRef<uint8>((const uint8*)barrier.sat_accum_t.data(), barrier.sat_accum_t.size() * sizeof(float)),
		OpenGLTextureFormat::Format_Greyscale_Float,
		OpenGLTexture::Filtering_Nearest, // Nearest: tiles are discrete decisions, not a field to interpolate.
		OpenGLTexture::Wrapping_Clamp,
		/*has_mipmaps=*/false);

	cloud.sat_grid_debug_ramp_tex = new OpenGLTexture(res, res, opengl_engine,
		ArrayRef<uint8>((const uint8*)barrier.sat_amp_sum.data(), barrier.sat_amp_sum.size() * sizeof(float)),
		OpenGLTextureFormat::Format_Greyscale_Float,
		OpenGLTexture::Filtering_Nearest,
		OpenGLTexture::Wrapping_Clamp,
		/*has_mipmaps=*/false);

	// SESSION081: "mask_fix" - the FINAL sat_depth, already post closing+erosion (both run inside
	// gsBuildSaturationGrid[Parallel]() before the barrier is handed back - see gsSatApplyRegionErosion()). No new
	// capture needed, unlike accum_t/amp_sum above: sat_depth is always populated whenever the grid is built at all,
	// overlay or not. Encoded as a plain 0/1 mask (finite = saturated = 1) rather than reusing sat_depth's raw
	// distances, so the shader can test it the same simple way as the Mask view - see the frag shader's mode 2.
	if(barrier.sat_depth.size() == res * res)
	{
		js::Vector<float, 16> mask_fix(res * res);
		for(size_t i=0; i<res * res; ++i)
			mask_fix[i] = std::isfinite(barrier.sat_depth[i]) ? 1.f : 0.f;

		cloud.sat_grid_debug_maskfix_tex = new OpenGLTexture(res, res, opengl_engine,
			ArrayRef<uint8>((const uint8*)mask_fix.data(), mask_fix.size() * sizeof(float)),
			OpenGLTextureFormat::Format_Greyscale_Float,
			OpenGLTexture::Filtering_Nearest,
			OpenGLTexture::Wrapping_Clamp,
			/*has_mipmaps=*/false);
	}
	else
		cloud.sat_grid_debug_maskfix_tex = NULL;

	cloud.sat_grid_debug_anchor_ws = barrier.anchor_pos_ws;
}


void GaussianSplatRenderer::getSatGridDebugInfo(std::vector<SatGridDebugInfo>& out) const
{
	out.clear();
	for(size_t i=0; i<clouds.size(); ++i)
		if(clouds[i]->sat_grid_debug_tex.nonNull() && clouds[i]->sat_grid_debug_ramp_tex.nonNull() && clouds[i]->sat_grid_debug_maskfix_tex.nonNull())
		{
			SatGridDebugInfo info;
			info.tex = clouds[i]->sat_grid_debug_tex;
			info.ramp_tex = clouds[i]->sat_grid_debug_ramp_tex;
			info.maskfix_tex = clouds[i]->sat_grid_debug_maskfix_tex;
			info.anchor_ws = clouds[i]->sat_grid_debug_anchor_ws;
			out.push_back(info);
		}
}


// SESSION081: applies a finished saturation barrier build to its cloud - see GaussianSplatSaturationBarrier and
// kickOffSaturationBuilds(). Deliberately unconditional about installing the result (unlike drainTraversalResults()'s
// staleness-guarded follow-up): a barrier build has no "derived from" frontier to go stale against - it is a fresh,
// self-contained fact about the world as of anchor_pos_ws, correct on arrival regardless of what has happened to any
// particular traversal since it was kicked. The only thing that can invalidate it is a structural change to the
// cloud, checked below via topology_generation.
void GaussianSplatRenderer::drainSaturationBuildResults()
{
	sat_build_result_queue.dequeueAnyQueuedItems(completed_sat_build_msgs);

	for(size_t i=0; i<completed_sat_build_msgs.size(); ++i)
	{
		const GaussianSplatSaturationBuildResultMsg* const msg = static_cast<const GaussianSplatSaturationBuildResultMsg*>(completed_sat_build_msgs[i].ptr());

		SplatCloud* cloud = NULL;
		for(size_t c=0; c<clouds.size(); ++c)
			if(clouds[c]->cloud_id == msg->cloud_id)
			{
				cloud = clouds[c].ptr();
				break;
			}
		if(!cloud)
			continue; // Cloud merged away or removed while the build ran - nothing to update.

		cloud->sat_build_in_flight = false; // Unconditional - the slot is free whether or not the result below is still usable, same "bookkeeping first" rule drainTraversalResults() follows.

		// SESSION088: feed the predictive anchor's lead - see GsMeasuredLatency and getSatPredictGain(). Taken here,
		// before the topology guard below, and unconditionally: a build that gets DISCARDED still measured how long a
		// build takes on this machine and this scene, which is the only thing this number is used for. Placed with the
		// other bookkeeping for the same reason that one is unconditional.
		sat_build_latency_measured.addSample(diag_timer.elapsed() - cloud->sat_build_kick_time_s);

		// A structural change invalidated the frontier this was built from - see SplatCloud::topology_generation.
		// Drop the result; kickOffSaturationBuilds() will see cached_sat_barrier is stale (or null) against the fresh
		// last_unpruned_ufrontier the next traversal produces, and re-kick.
		if(msg->topology_generation != cloud->topology_generation)
			continue;

		cloud->cached_sat_barrier = msg->barrier;

		// SESSION085 ETAP 3: with the LoD bias on, a traversal's OUTPUT is a function of the barrier as well as of the
		// camera and the knobs - so a new barrier invalidates the standing traversal exactly as a camera move does. This is
		// a missing cache invalidation, not a refinement heuristic: without it the barrier updates and nothing re-walks, so
		// the coarsening chosen under the PREVIOUS barrier stands until some unrelated event happens to kick a traversal.
		//
		// Owner-visible as the one complaint left after etap 3's sweep: stopping the camera left regions at the LoD the old
		// barrier had justified, and they stayed there. Under the prune the same staleness showed as holes and the apply
		// pipeline re-ran on a barrier change (kickOffSaturationApplies() keys on it), so the traversal never had to care.
		//
		// Same "unconditionally overdue" mechanism forceTraversalRefresh() uses, scoped to this cloud. No feedback loop:
		// kickOffSaturationBuilds() decides to rebuild from the camera position and the knobs, never from the frontier, so
		// the traversal this provokes cannot provoke another barrier.
		if(sat_bias_ceiling > 1.f)
			cloud->have_last_traversal_cam_pos = false;
		const GaussianSplatSaturationBarrier& b = *msg->barrier;

		// SESSION074/076/079/080/081 DIAGNOSTIC: the build's own numbers, printed once per build rather than once per
		// traversal now that the two are decoupled - a build is a much rarer event (once per R-ball exit, not once
		// per kick), so this line's cadence directly shows the win: compare its frequency against [gsr-traversal]'s.
		if(filter_debug_log)
		{
			if(b.sat_grid_res > 0)
			{
				const size_t num_tiles = (size_t)b.sat_grid_res * (size_t)b.sat_grid_res;
				size_t sat_tiles = 0;
				for(size_t t=0; t<b.sat_depth.size(); ++t)
					if(std::isfinite(b.sat_depth[t]))
						++sat_tiles;
				conPrint("[gsr-sat-build] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms " +
					// SESSION085: the walk is the only occluder source now, so this is unconditional. occl= below counts
					// what the walk yielded, against frontier_n (the render frontier it is spent instead of).
					"visited=" + uInt64ToStringCommaSeparated(b.walk_visited) +
					" walk_ms=" + doubleToStringNDecimalPlaces(b.walk_ms, 2) +
					" walk_sort_ms=" + doubleToStringNDecimalPlaces(b.walk_sort_ms, 2) + " " +
					"thr=" + doubleToStringNDecimalPlaces(b.threshold_used, 3) + // SESSION080: the knobs THIS barrier was built with, not the live settings - see the field's old comment on GaussianSplatUnculledFrontier.
					" sub=" + doubleToStringNDecimalPlaces(b.subdiv_used, 3) +
					" R=" + doubleToStringNDecimalPlaces(b.region_radius_used, 3) +
					" close=" + toString(b.closing_tiles_used) + // SESSION081
					" occl=" + uInt64ToStringCommaSeparated(b.num_occluders) +
					" tiles=" + uInt64ToStringCommaSeparated(num_tiles) +
					" sat_tiles=" + doubleToStringNDecimalPlaces(num_tiles > 0 ? (100.0 * (double)sat_tiles / (double)num_tiles) : 0.0, 1) + "%" +
					" cl=" + uInt64ToStringCommaSeparated(b.erode_stats[3]) + // SESSION081: pinholes the closing pass filled BEFORE the erosion below ran - see gsSatApplyClosing().
					" er_ceil=" + uInt64ToStringCommaSeparated(b.erode_stats[0]) +
					" er_win=" + uInt64ToStringCommaSeparated(b.erode_stats[1]) +
					" (" + doubleToStringNDecimalPlaces(num_tiles > 0 ? (100.0 * (double)(b.erode_stats[0] + b.erode_stats[1]) / (double)num_tiles) : 0.0, 1) + "%)" +
					" er_radmax=" + uInt64ToStringCommaSeparated(b.erode_stats[2]) +
					" gather_ms=" + doubleToStringNDecimalPlaces(b.gather_ms, 2) +
					" grid_ms=" + doubleToStringNDecimalPlaces(b.grid_build_ms, 2) +
					// SESSION088 PREDICTIVE ANCHOR - see getSatPredictGain(). Three numbers that have to be read together:
					//   lat_meas : the EMA of THIS stage's kick->drain round trip, which is the lead's whole basis.
					//   lead     : how far ahead of the camera the kick that produced this barrier was aimed. 0 = the knob
					//              is off, or the camera was not moving - both are correct and indistinguishable here.
					//   lead/R   : the lead against sat_region_radius, the ball this barrier is valid over. Above ~1 the
					//              anchor is a whole ball ahead of the camera, so the prediction is doing more than the
					//              guarantee's own slack - that is the ratio to watch if the far field starts leading
					//              rather than lagging.
					" lat_meas=" + doubleToStringNDecimalPlaces(sat_build_latency_measured.get(0.f) * 1000.0, 1) + "ms" +
					" lead=" + doubleToStringNDecimalPlaces(diag_sat_predict_lead_m, 2) + "m" +
					" (" + doubleToStringNDecimalPlaces(b.region_radius_used > 0.f ? (diag_sat_predict_lead_m / b.region_radius_used) : 0.0, 2) + "R)");

				// SESSION081 PLAN ETAP 0 DIAGNOSTIC: grid_ms broken into its measured sub-phases, plus the "sum of parts
				// vs whole" check (misc_ms) - see [[reconcile_parts_with_the_whole]]. A significant misc_ms means a
				// phase inside grid_ms is still unaccounted for.
				if(sat_diag_log)
				{
					const double misc_ms = b.grid_build_ms - (b.rec_ms + b.dep_ms + b.close_ms + b.erode_ms);
					conPrint("[gsr-sat-build-phases] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms " +
						"n=" + uInt64ToStringCommaSeparated(b.frontier_n) +
						" rec_ms=" + doubleToStringNDecimalPlaces(b.rec_ms, 2) +
						" dep_ms=" + doubleToStringNDecimalPlaces(b.dep_ms, 2) +
						" close_ms=" + doubleToStringNDecimalPlaces(b.close_ms, 2) +
						" erode_ms=" + doubleToStringNDecimalPlaces(b.erode_ms, 2) +
						" misc_ms=" + doubleToStringNDecimalPlaces(misc_ms, 2) +
						" (" + doubleToStringNDecimalPlaces(b.grid_build_ms > 0.0 ? (100.0 * misc_ms / b.grid_build_ms) : 0.0, 1) + "% of grid_ms)" +
						" num_recs=" + uInt64ToStringCommaSeparated(b.num_recs) +
						" gate1=" + uInt64ToStringCommaSeparated(b.gate1_survivors) + // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY - see gsBuildSaturationGridParallel()'s out_gate1_survivors.
						" wasted=" + uInt64ToStringCommaSeparated(b.gate1_survivors - b.num_recs) +
						" (" + doubleToStringNDecimalPlaces(b.gate1_survivors > 0 ? (100.0 * (double)(b.gate1_survivors - b.num_recs) / (double)b.gate1_survivors) : 0.0, 1) + "% of gate1)" +
						" rec_task_ms min=" + doubleToStringNDecimalPlaces(b.rec_task_ms_min, 2) + // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY - is one chunk (the near, front-of-list one) a straggler the whole task group waits on?
						" mean=" + doubleToStringNDecimalPlaces(b.rec_task_ms_mean, 2) +
						" max=" + doubleToStringNDecimalPlaces(b.rec_task_ms_max, 2) +
						" num_strips=" + toString(b.num_strips) +
						" concurrency=" + toString(b.concurrency_used) +
						" strip_reads=" + uInt64ToStringCommaSeparated((uint64)b.num_strips * (uint64)b.num_recs) +
						" num_blocks=" + uInt64ToStringCommaSeparated(b.num_blocks)); // SESSION081 PLAN, ETAP 0 follow-up DIAGNOSTIC.

					// SESSION082 PLAN §1.1, TEMPORARY DIAGNOSTIC: the composition of the unsaturated (+inf) tiles, which
					// is what er_win kills on. empty= is "no occluder ever deposited here" (an honestly open direction);
					// the four q<n> buckets are quarters of the way to the saturation threshold, so q4 is "mass is there
					// and nearly made it". A large q3+q4 says the threshold/coarse cut is the thing to fix, not erosion.
					{
						const size_t inf_total = b.inf_hist[0] + b.inf_hist[1] + b.inf_hist[2] + b.inf_hist[3] + b.inf_hist[4];
						conPrint("[gsr-sat-inf] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms " +
							"inf=" + uInt64ToStringCommaSeparated(inf_total) +
							" (" + doubleToStringNDecimalPlaces(num_tiles > 0 ? (100.0 * (double)inf_total / (double)num_tiles) : 0.0, 1) + "% of tiles)" +
							" | empty=" + uInt64ToStringCommaSeparated(b.inf_hist[0]) +
							" (" + doubleToStringNDecimalPlaces(inf_total > 0 ? (100.0 * (double)b.inf_hist[0] / (double)inf_total) : 0.0, 1) + "%)" +
							" q1=" + uInt64ToStringCommaSeparated(b.inf_hist[1]) +
							" q2=" + uInt64ToStringCommaSeparated(b.inf_hist[2]) +
							" q3=" + uInt64ToStringCommaSeparated(b.inf_hist[3]) +
							" q4=" + uInt64ToStringCommaSeparated(b.inf_hist[4]) +
							" (partial=" + doubleToStringNDecimalPlaces(inf_total > 0 ? (100.0 * (double)(inf_total - b.inf_hist[0]) / (double)inf_total) : 0.0, 1) + "%)");
					}

					// SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC. rec_par/dep_par are busy-sum over group wall -
					// the thread count each phase ACTUALLY got, against the concurrency it was promised. rec_start_max
					// is the discriminator: near zero with a long wall means the chunks ran but were preempted; a large
					// fraction of the wall means they sat queued behind other pool work. move_ms is the serial
					// compaction that has been hiding inside rec_ms all session.
					const double rec_wall = b.sched_stats[3], dep_wall = b.dep_ms;
					conPrint("[gsr-sat-sched] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms " +
						"conc=" + doubleToStringNDecimalPlaces(b.sched_stats[7], 0) +
						// SESSION085: the gather busy/par pair went with GsSatGatherTask. The walk path reports its two
						// costs on [gsr-sat-build] itself (walk_ms/walk_sort_ms), which together with gather_ms account
						// for the whole of this stage.
						" | gather wall=" + doubleToStringNDecimalPlaces(b.gather_ms, 2) +
						" | rec wall=" + doubleToStringNDecimalPlaces(rec_wall, 2) +
						" busy=" + doubleToStringNDecimalPlaces(b.sched_stats[2], 1) +
						" par=" + doubleToStringNDecimalPlaces(rec_wall > 0.0 ? (b.sched_stats[2] / rec_wall) : 0.0, 2) +
						" start_max=" + doubleToStringNDecimalPlaces(b.sched_stats[0], 2) +
						" start_mean=" + doubleToStringNDecimalPlaces(b.sched_stats[1], 2) +
						" | move_ms=" + doubleToStringNDecimalPlaces(b.sched_stats[4], 2) +
						" bin_ms=" + doubleToStringNDecimalPlaces(b.sched_stats[8], 2) + // SESSION081 PHASE 1c.
						" | dep wall=" + doubleToStringNDecimalPlaces(dep_wall, 2) +
						" busy=" + doubleToStringNDecimalPlaces(b.sched_stats[6], 1) +
						" par=" + doubleToStringNDecimalPlaces(dep_wall > 0.0 ? (b.sched_stats[6] / dep_wall) : 0.0, 2) +
						" start_max=" + doubleToStringNDecimalPlaces(b.sched_stats[5], 2));
				}

				if(sat_diag_log)
				{
					const double occl = (double)b.num_occluders;
					conPrint("[gsr-sat-build-diag] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms occl=" + uInt64ToStringCommaSeparated(b.num_occluders) +
						" writers=" + uInt64ToStringCommaSeparated(b.diag_writers) +
						" (" + doubleToStringNDecimalPlaces(occl > 0 ? (100.0 * (double)b.diag_writers / occl) : 0.0, 1) + "%)" +
						" tile_writes=" + uInt64ToStringCommaSeparated(b.diag_tile_writes) +
						" per_writer=" + doubleToStringNDecimalPlaces(b.diag_writers > 0 ? ((double)b.diag_tile_writes / (double)b.diag_writers) : 0.0, 1) +
						" tile_iters=" + uInt64ToStringCommaSeparated(b.diag_tile_stats[0]) +
						" tail_rej=" + uInt64ToStringCommaSeparated(b.diag_tile_stats[1]) +
						" sat_skip=" + uInt64ToStringCommaSeparated(b.diag_tile_stats[2]) +
						" per_writer_iters=" + doubleToStringNDecimalPlaces(b.diag_writers > 0 ? ((double)b.diag_tile_stats[0] / (double)b.diag_writers) : 0.0, 1));
					// SESSION081 ETAP 4: [gsr-sat-aniso] is gone. It reported max(scale)/min(scale) over the occluder
					// population, read straight from geom->scales in the gather; that array no longer exists (see
					// GsSatPackedOccluder), and the packed record cannot reconstruct the ratio. The measurement it
					// existed to make was session077's and is recorded there.
				}
			}
			else
				conPrint("[gsr-sat-build] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms no occluders - empty barrier");
		}

		// SESSION076 §9, SESSION078, SESSION081: rebuild the sat_depth debug texture alongside the barrier, gated on
		// getSatDebugOverlayMode() != Off - see updateSatGridDebugTexture(). Runs on the main thread with a valid GL
		// context, same as the VBO updates elsewhere in this function.
		if(sat_debug_overlay_mode != GaussianSplatSatDebugOverlayMode_Off && b.sat_grid_res > 0)
			updateSatGridDebugTexture(*cloud, b);
		else
		{
			cloud->sat_grid_debug_tex = NULL;
			cloud->sat_grid_debug_ramp_tex = NULL;
			cloud->sat_grid_debug_maskfix_tex = NULL;
		}
	}
}




void GaussianSplatRenderer::drainTraversalResults()
{
	traversal_result_queue.dequeueAnyQueuedItems(completed_traversal_msgs);

	// SESSION069 - see drainSortResults() above.  Traversal results change which splats are selected for drawing; the
	// LoD-settle mechanism (session066) applies exactly one of these ~0.3s after the camera stops, i.e. right in the
	// middle of a TAA accumulation window - and without this reset the two LoD sets would smear together for the
	// remaining frames of the window ("top layer peeled" from session053).
	if(!completed_traversal_msgs.empty()) noteContentApplied();

	for(size_t i=0; i<completed_traversal_msgs.size(); ++i)
	{
		const GaussianSplatLodTraversalResultMsg* const msg = static_cast<const GaussianSplatLodTraversalResultMsg*>(completed_traversal_msgs[i].ptr());

		SplatCloud* cloud = NULL;
		for(size_t c=0; c<clouds.size(); ++c)
			if(clouds[c]->cloud_id == msg->cloud_id)
			{
				cloud = clouds[c].ptr();
				break;
			}

		// Bookkeeping first, and unconditionally: the traversal has finished and its scratch is free to reuse whether or
		// not the cloud it was for still exists.
		num_traversals_in_flight--;
		free_traversal_scratch.push_back(msg->scratch);
		if(cloud)
			cloud->traversal_in_flight = false;

		// Drop results for a cloud that has since been merged away or removed, and results computed against a member set
		// the cloud no longer has - see SplatCloud::topology_generation's comment.
		if(!cloud || msg->topology_generation != cloud->topology_generation)
			continue;

		// Already sorted front-to-back by GaussianSplatLodTraversalTask::run()'s decorate-sort - nothing left to do here
		// but write it.  kickOffSorts() never touches an LoD-active cloud (see its cloudHasLodTree() guard), so there's no
		// separate sort state on the cloud to invalidate here the way a structural change invalidates the old sort's.
		const js::Vector<uint32, 16>& selected = msg->scratch->selected_indices;

		// SESSION063: split filter path. When split_filter_enabled the traversal ran cull-off, so `selected` is the whole
		// unculled frontier U(P) and the msg carries the SoA copy. Adopt it as this cloud's cache and mark it for filtering;
		// kickOffFilters() then derives the draw list S(P,R) asynchronously and drainFilterResults() uploads THAT (never
		// U(P), which is the ~7.5M unculled set). The VBO is deliberately left untouched here: the last filter's S(P,R)
		// keeps drawing until the new one lands (~13ms), so a U(P) rebuild costs no main-thread hitch. The cap flags below
		// are still copied out - the scratch is about to return to the pool.
		if(split_filter_enabled && msg->unculled_frontier.nonNull())
		{
			// SESSION085 ETAP 6: unconditional. This used to defer installing a fresh frontier when a saturation APPLY was
			// about to replace it with a pruned copy - the "draw unpruned" question. With the prune gone there is no
			// follow-up to wait for: what a traversal produces IS what gets drawn, already biased by the barrier during the
			// walk itself. That is the latency win the plan's etap 0 identified, arriving as a structural simplification
			// rather than as an optimisation.
			cloud->cached_ufrontier = msg->unculled_frontier;
			cloud->ufrontier_needs_filter = true; // Produce the first S(P,R) for this fresh U(P) even with no rotation.

			// SESSION080: this branch is the only place the UNPRUNED frontier is ever seen - a saturation apply
			// overwrites cached_ufrontier with its pruned copy once it lands - so it is the only place the "previous
			// frontier" that STEP A's staleness comparison and STEP B's reuse both need can be captured. See the
			// field's comment for why the pruned one will not do for either. Assigned NULL when nothing wants it, so
			// the extra frontier is released rather than pinned.
			// SESSION081: kept unconditionally whenever the saturation stage is wanted at all - this is now BOTH
			// kickOffSaturationBuilds()'s and kickOffSaturationApplies()'s only source (neither reads cached_ufrontier
			// any more), not just a diagnostic nicety. Without this, a session with diag off and reuse=0 would starve
			// every barrier build and apply of input and neither would ever run.
			// SESSION085 ETAP 3: the bias is a third consumer - see kickOffSaturationBuilds(). The barrier build still
			// requires a source frontier to exist (it reports occl= against frontier_n), so without this the bias-only
			// configuration would drop the source here and then never get a barrier built at all.
			const bool sat_wants_source = (sat_debug_overlay_mode != GaussianSplatSatDebugOverlayMode_Off) || (sat_bias_ceiling > 1.f);
			cloud->last_unpruned_ufrontier = (sat_diag_log || frontier_reuse_split_dist > 0.f || sat_wants_source) ?
				msg->unculled_frontier : Reference<GaussianSplatUnculledFrontier>();

			cloud->last_traversal_hit_budget_cap = msg->scratch->hit_budget_cap;
			cloud->last_traversal_hit_density_cap = msg->scratch->hit_density_cap;
			cloud->last_traversal_hit_depth_cap = msg->scratch->hit_depth_cap;
			cloud->last_traversal_sat_bias_stops = msg->scratch->num_sat_bias_stops; // SESSION085 ETAP 3
			cloud->last_traversal_sat_bias_ceiling = msg->scratch->sat_bias_ceiling_used; // SESSION085 ETAP 6
			cloud->last_traversal_sat_bias_tested = msg->scratch->num_sat_bias_tested;

			// SESSION086: close the loop on how finely the next traversal splits its seeds - see updateExpandSeedTarget().
			if(msg->unculled_frontier.nonNull())
				updateExpandSeedTarget(msg->unculled_frontier->expand_seed_max_ms, msg->unculled_frontier->expand_task_sum_ms,
					msg->unculled_frontier->expand_prologue_ms, msg->unculled_frontier->expand_ms);

			// SESSION076/081: no saturation numbers to print here any more. A traversal's first (and now only) message
			// carries the UNPRUNED frontier and never builds or applies a grid itself - APPLY prints from
			// drainSaturationApplyResults() ([gsr-sat-apply]); BUILD prints from drainSaturationBuildResults()
			// ([gsr-sat-build]) - both entirely separate, independently-paced pipelines now.
			//
			// SESSION080: the traversal's OWN cost is the opposite case and belongs here rather than there. It is fully
			// known by message 1 - the walk, the sort and the SoA build are all finished before this message is sent -
			// and, unlike the saturation numbers, it exists whether or not the saturation stage runs at all. Printing it
			// from the follow-up branch (where it originally sat) tied it to a message that is never sent with that stage
			// off, which is exactly the configuration the stage-by-stage measurement model wants it in. Still once per
			// traversal either way: message 1 is sent unconditionally, exactly once, by every traversal.
			{
				const GaussianSplatUnculledFrontier& uf = *msg->unculled_frontier;

				// SESSION080 STEP B: read once here for reuse_drift= below. A null scene (no current scene at drain time)
				// reads as the origin, which only makes that one log field meaningless.
				const OpenGLScene* const drift_scene = opengl_engine->getCurrentScene();
				const Vec4f cam_pos_now_for_drift = (drift_scene != NULL) ? drift_scene->cam_to_world.getColumn(3) : Vec4f(0.f);

				// SESSION080 DIAGNOSTIC: the traversal task's own cost, split into tree walk/selection vs. sort - see
				// GaussianSplatUnculledFrontier::expand_ms/sort_ms. Answers whether session054's "sort was ~76% of
				// traversal time" figure - measured against std::sort, before the switch to radix - is still the shape of
				// the cost today.
				if(filter_debug_log)
					conPrint("[gsr-traversal] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms " +
						"expand_ms=" + doubleToStringNDecimalPlaces(uf.expand_ms, 2) +
						" sort_ms=" + doubleToStringNDecimalPlaces(uf.sort_ms, 2) +
						// SESSION087: sort_ms split into the sort proper and the unpack that follows it - see the fields.
						" radix_ms=" + doubleToStringNDecimalPlaces(uf.sort_radix_ms, 2) +
						" unpack_ms=" + doubleToStringNDecimalPlaces(uf.sort_unpack_ms, 2) +
						" total_ms=" + doubleToStringNDecimalPlaces(uf.expand_ms + uf.sort_ms, 2) +
						// SESSION080: traversal_output_n rather than uf.indices.size(). They are equal on THIS (unpruned)
						// frontier, but the field is the one that stays right if this line is ever read off a pruned copy,
						// where indices.size() has shrunk to pool_after. See the field's comment.
						" n=" + uInt64ToStringCommaSeparated(uf.traversal_output_n) +
						// SESSION080: whether this traversal's DFS was truncated rather than run to convergence - see
						// GaussianSplatLodTraversalScratch::hit_budget_cap and session054's note (in the sort block's
						// comment) that expand's DFS-order budget-clip policy assumed this never fires. Read off the cloud,
						// which the three lines just above set from this very message's scratch.
						" budget_cap=" + boolToString(cloud->last_traversal_hit_budget_cap) +
						" density_cap=" + boolToString(cloud->last_traversal_hit_density_cap) +
						" depth_cap=" + boolToString(cloud->last_traversal_hit_depth_cap) +
						" bias=" + doubleToStringNDecimalPlaces(cloud->last_traversal_sat_bias_ceiling, 2) + // SESSION085 ETAP 6: the ceiling this traversal ran with. 1 = the bias was off (or no barrier yet).
						" sat_bias=" + uInt64ToStringCommaSeparated(cloud->last_traversal_sat_bias_stops) + "/" + uInt64ToStringCommaSeparated(cloud->last_traversal_sat_bias_tested) + // SESSION085 ETAP 3: nodes the bias STOPPED, over nodes it was asked about. A big denominator with a small numerator means the occlusion test is refusing, not that the bias is idle.
						// SESSION080 DIAGNOSTIC: the parallel expand's own shape - see the fields' comment. par= is
						// task_sum/task_max, the parallelism the seed split actually offers; compare it against how much
						// of expand_ms the longest task accounts for.
						" seeds=" + uInt64ToStringCommaSeparated(uf.expand_seeds) +
						" tasks=" + uInt64ToStringCommaSeparated(uf.expand_num_tasks) +
						" prologue_ms=" + doubleToStringNDecimalPlaces(uf.expand_prologue_ms, 2) +
						" task_max_ms=" + doubleToStringNDecimalPlaces(uf.expand_task_max_ms, 2) +
						" workers=" + uInt64ToStringCommaSeparated(uf.expand_workers) + // SESSION086 DIAGNOSTIC - see the field.
						" seed_max_ms=" + doubleToStringNDecimalPlaces(uf.expand_seed_max_ms, 2) +
						" seed_target=" + uInt64ToStringCommaSeparated(uf.expand_seed_target) +
						" task_sum_ms=" + doubleToStringNDecimalPlaces(uf.expand_task_sum_ms, 2) +
						// SESSION080 DIAGNOSTIC (plan2 §4.1): the serial concatenation after runTaskGroup() returns, split
						// into the destination's allocation and the copy loop itself - see expand_splice_reserve_ms. Both
						// 0 on the serial path (expandParallel() never ran). sort_alloc is the same question asked of the
						// sort's own scratch, and is a SUBSET of sort_ms, not an addition to it.
						" splice_res_ms=" + doubleToStringNDecimalPlaces(uf.expand_splice_reserve_ms, 2) +
						" splice_cpy_ms=" + doubleToStringNDecimalPlaces(uf.expand_splice_copy_ms, 2) +
						" sort_alloc_ms=" + doubleToStringNDecimalPlaces(uf.sort_alloc_ms, 2) +
						" soa_ms=" + doubleToStringNDecimalPlaces(uf.soa_ms, 2) + // SESSION080: was untimed before - see the field.
						// SESSION080: node distance distribution, metres - sizes the far-tail-reuse question. See the field.
						" d10=" + doubleToStringNDecimalPlaces(uf.dist_pctile[0], 1) +
						" d25=" + doubleToStringNDecimalPlaces(uf.dist_pctile[1], 1) +
						" d50=" + doubleToStringNDecimalPlaces(uf.dist_pctile[2], 1) +
						" d75=" + doubleToStringNDecimalPlaces(uf.dist_pctile[3], 1) +
						" d90=" + doubleToStringNDecimalPlaces(uf.dist_pctile[4], 1) +
						" par=" + doubleToStringNDecimalPlaces(uf.expand_task_max_ms > 0.0 ? (uf.expand_task_sum_ms / uf.expand_task_max_ms) : 0.0, 2) +
						// SESSION080 DIAGNOSTIC (plan doc STEP A): sort staleness vs. whatever this cloud was drawing off
						// just before this kick - see GaussianSplatUnculledFrontier::sort_staleness_*. All zero when
						// sort_staleness_diag_enabled was off, there was no previous frontier, or topology changed.
						" delta_ws=" + doubleToStringNDecimalPlaces(uf.sort_staleness_delta_ws, 3) +
						" stale_common=" + uInt64ToStringCommaSeparated(uf.sort_staleness_common_n) +
						"/" + uInt64ToStringCommaSeparated(uf.sort_staleness_prev_n) +
						" stale_max=" + uInt64ToStringCommaSeparated(uf.sort_staleness_max_disp) +
						" stale_mean=" + doubleToStringNDecimalPlaces(uf.sort_staleness_mean_disp, 1) +
						" stale_gt1k=" + uInt64ToStringCommaSeparated(uf.sort_staleness_gt1k) +
						" (" + doubleToStringNDecimalPlaces(uf.sort_staleness_common_n > 0 ? (100.0 * (double)uf.sort_staleness_gt1k / (double)uf.sort_staleness_common_n) : 0.0, 2) + "%)" +
						" stale_gt10k=" + uInt64ToStringCommaSeparated(uf.sort_staleness_gt10k) +
						" (" + doubleToStringNDecimalPlaces(uf.sort_staleness_common_n > 0 ? (100.0 * (double)uf.sort_staleness_gt10k / (double)uf.sort_staleness_common_n) : 0.0, 2) + "%)" +
						" stale_ms=" + doubleToStringNDecimalPlaces(uf.sort_staleness_ms, 2) +
						// SESSION080 STEP B: what frontier reuse did - see GaussianSplatUnculledFrontier::reuse_roots. split=0
						// means the knob was off OR a precondition declined it (see the task's reuse_enabled), so a non-zero
						// split with reuse_n=0 is the interesting failure: the scheme ran and inherited nothing.
						" reuse_split=" + doubleToStringNDecimalPlaces(uf.reuse_split_dist_used, 2) +
						// SESSION088: the two knobs that decide whether a block survives, printed so a capture identifies
						// which arm it is without being taken on trust. Added after a two-arm comparison was run, and only
						// caught afterwards, on a region radius that had not been put back - neither knob appeared anywhere
						// in the log, so nothing contradicted the assumption.
						" btol=" + doubleToStringNDecimalPlaces(sat_barrier_agree_tol, 3) +
						" rdrift=" + doubleToStringNDecimalPlaces(frontier_reuse_drift_fraction, 3) +
						" satR=" + doubleToStringNDecimalPlaces(sat_region_radius, 3) +
						" reuse_roots=" + uInt64ToStringCommaSeparated(uf.reuse_roots) +
						" reuse_n=" + uInt64ToStringCommaSeparated(uf.reuse_n) +
						" (" + doubleToStringNDecimalPlaces(uf.indices.size() + uf.reuse_n > 0 ? (100.0 * (double)uf.reuse_n / (double)(uf.indices.size() + uf.reuse_n)) : 0.0, 1) + "%)" + // SESSION080 §4.3: indices.size() is this frontier's NEAR segment; the far block is reuse_n. traversal_output_n counts the whole walk, which double-counts on the traversal that builds a block.
						" reuse_ms=" + doubleToStringNDecimalPlaces(uf.reuse_ms, 2) +
						// How far this frontier's OLDEST inherited data's anchor is from where the camera is now - the quantity
						// the full-rebuild safety trigger watches. Rising towards split*0.25 and then reuse_n dropping to 0 for
						// one traversal is the trigger firing, which is working as intended, not a fault.
						" reuse_drift=" + doubleToStringNDecimalPlaces(cam_pos_now_for_drift.getDist(uf.reuse_base_anchor_ws), 2) +
					// SESSION088: the LoD half of the reuse bound, measured whatever the tolerance is set to - see
					// gsSatBarrierDisagreement(). Read it against reuse_drift= above: the two are the two independent
					// reasons a block can be too old, and the point of this line is to show how differently they move.
					// SESSION088: n/a, not 0, when nothing was compared - a not-measured row and a perfect-agreement row are
					// different facts and the first cut of this printed both as 0.00%.
					(uf.barrier_measured ?
						(" barrier_dis=" + doubleToStringNDecimalPlaces(uf.barrier_disagreement * 100.0, 2) + "%" +
						// barrier_dis is what the gate reads, which since the third cut is bd_flip alone. The other two stay
						// printed so that choice remains checkable rather than assumed - see gsSatBarrierDisagreement().
						// NOTE bd_reld is a MEAN over both-saturated tiles, and a handful of near-zero depths dominate it
						// (it read 483% on the interior against a 10% tolerance): treat it as "there is a heavy tail",
						// never as a typical move.
						" bd_flip=" + doubleToStringNDecimalPlaces(uf.barrier_flip * 100.0, 2) + "%" +
						" bd_depth=" + doubleToStringNDecimalPlaces(uf.barrier_depth * 100.0, 2) + "%" +
						" bd_reld=" + doubleToStringNDecimalPlaces(uf.barrier_mean_rel_depth * 100.0, 1) + "%") :
						std::string(" barrier_dis=n/a")) +
					// SESSION088 DIAGNOSTIC, TEMPORARY - see GaussianSplatUnculledFrontier::diag_live_barrier_time. The gap
					// between the two barrier builds the comparison was made across. NOTE both are KICK-time facts while
					// this line prints at DRAIN, so a gap of about one barrier cadence is the normal steady state and says
					// nothing about correctness - reading it otherwise cost session088 a wrong bug report.
					" bt_delta=" + doubleToStringNDecimalPlaces((uf.diag_live_barrier_time - uf.diag_block_barrier_time) * 1000.0, 1) + "ms");
			}

			continue;
		}

		// SESSION058 diag: measures the synchronous GL upload cost on drain, the second suspected contributor to the
		// session057 §3 CPU cost alongside fillTraversalScratch() (see that function's [gsr-prof] instrumentation).
		Timer prof_timer;
		cloud->instance_index_vbo->updateData(0, selected.data(), selected.size() * sizeof(uint32));
		if(cpu_prof_log)
			conPrint("[gsr-prof] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms drainVBOUpload cloud=" +
				toString(cloud->cloud_id) + " indices=" + toString(selected.size()) +
				" bytes=" + toString(selected.size() * sizeof(uint32)) +
				" took=" + doubleToStringNDecimalPlaces(prof_timer.elapsed() * 1000.0, 2) + "ms");

		cloud->ob->num_instances_to_draw = (int)selected.size();
		noteDrawOrderForSlicing(*cloud, selected.data(), selected.size());
		cloud->last_traversal_hit_budget_cap = msg->scratch->hit_budget_cap; // Copied out here since the scratch itself goes back to the pool below and may be reused by a different cloud's traversal next.
		cloud->last_traversal_hit_density_cap = msg->scratch->hit_density_cap;
		cloud->last_traversal_hit_depth_cap = msg->scratch->hit_depth_cap;
	}

	completed_traversal_msgs.clear(); // Drop the references, so a scratch just returned to the pool isn't kept alive by a stale message.
}


// SESSION063: the cheap half of the split architecture's per-frame work. For each cloud that has a cached U(P), kick an
// async filter (GaussianSplatFilterTask) when either a fresh U(P) just landed (ufrontier_needs_filter) or the camera has
// rotated past a small threshold since the last filter. A filter is ~13ms on the worker (measured session063) vs a
// ~450ms traversal, so this is what lets a pure rotation refresh the draw list.
//
// K3 (session063): the filter carries the same anisotropic dilation the old cull-traversal used, so the frustum edge is
// pushed out in the direction the camera is moving/turning and doesn't trail during the filter's own latency window (the
// edge lag the split path had before this). Computed here, once per kick, from the per-frame motion trackers (think()):
// translation from cam_velocity_ema_ws, rotation from max(cam_angular_speed_ema, cam_angular_speed_peak). The window is
// filter_latency_estimate - far shorter than the traversal's 0.3s, since a filter lands in ~13ms - so the margins (and
// the over-inclusion they cost) are correspondingly small. Baseline floors cover the static->moving transition, where
// both trackers still read ~zero.
void GaussianSplatRenderer::kickOffFilters()
{
	if(!split_filter_enabled)
		return;
	glare::TaskManager* const task_manager = opengl_engine->getMainTaskManager();
	if(task_manager == NULL)
		return;

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Vec4f cam_pos_ws = scene->cam_to_world.getColumn(3);
	const Vec4f cam_forward_ws = normalise(scene->cam_to_world.getColumn(1)); // SESSION055: col 1 is forward - see kickOffTraversals().
	const float rotation_cos_threshold = 0.99939f; // cos(2deg) - tighter than the traversal's 5deg re-kick since a filter is ~35x cheaper.

	// Motion dilation, shared by every cloud kicked this pass (depends only on camera motion, not on the cloud). Knobs are
	// live-tunable - see getFilterDilationLatency() etc.
	// SESSION088: the window is the stage's own MEASURED kick->drain round trip when that is switched on, and the fixed
	// filter_dilation_latency otherwise - see getFilterLatencyMeasuredEnabled(). One value substituted at one place: every
	// use below (translation floor, rotation baseline, swept_fine) reads this, so the fine layer's whole dilation moves
	// together and the coarse tail keeps its own separate, deliberately wider window untouched.
	//
	// The x2 is not padding, it is the quantity the window is actually supposed to cover, and leaving it out was the
	// first version of this code's mistake. What must be covered is not "how long until THIS result lands" but "how long
	// this result stays on screen", and a list is displayed from its own kick until the NEXT drain replaces it - two
	// round trips, not one. That is exactly why session079 raised the constant to 0.3 against an ~85ms measured round
	// trip ("the envelope is ~2x"), and session088's own captures agree from the other side: filter age= reads
	// med 127ms / p90 257ms / max 307ms on the interior, i.e. the 0.3 constant is not an over-estimate of the round trip,
	// it is a correct estimate of the ENVELOPE. Substituting a 1x typical-case EMA for a 2x worst-case envelope would
	// have narrowed the band ~1.8x and shown up as edge holes under rotation - the one thing this knob must not do.
	//
	// So what the measurement buys is not a smaller number, it is a number that follows the machine and the scene
	// instead of being frozen at the one they were measured on: 2 x 165ms interior and 2 x 163ms exterior both land
	// within 10% of the 0.3 the owner arrived at by hand, which is the measurement confirming the constant rather than
	// replacing it.
	const float filter_latency_envelope_factor = 2.f;
	const float dilation_latency_eff = filter_latency_measured_enabled ?
		(filter_latency_measured.get(filter_dilation_latency * 0.5f) * filter_latency_envelope_factor) : filter_dilation_latency; // Cold-start fallback is halved so that x2 reproduces the constant exactly until the first sample lands.
	const Vec4f ema_expected = cam_velocity_ema_ws * dilation_latency_eff;
	const float min_trans_dilation_m = filter_min_trans_rate_m_per_s * dilation_latency_eff;
	float trans_dilation[6] = { 0,0,0,0,0,0 };
	for(int i=0; i<scene->num_frustum_clip_planes && i<6; ++i)
		trans_dilation[i] = myMax(min_trans_dilation_m, myMax(0.f, dot(scene->frustum_clip_planes[i].getNormal(), ema_expected)));
	const float min_rot_rate_rad_s = filter_min_rot_rate_deg_per_s * (3.14159265f / 180.f);

	// SESSION064: the per-frame re-filter trigger follows THIS frame's raw rotation (cam_inst_angular_speed), which is 0
	// the moment the camera stops - NOT the smoothed ema/peak, which coast down over ~2s. Using the decaying trackers
	// here re-filtered a *static* camera every frame with an ever-shrinking dilation band (rate_fine_baseline below), so S(P,R)
	// churned by ~1.4% for ~2s after every stop; invisible without the saturation gate, but the gate amplified the
	// membership churn into visible "boiling" in dense regions - see session064 snapshot. Now: rotating stops with the
	// camera, we stop re-filtering, and the last (in-motion-band) selection is simply held.
	const bool rotating = cam_inst_angular_speed > min_rot_rate_rad_s;

	// Band WIDTH: always predictive max(ema, peak) - a kick issued during motion must dilate wide enough to cover where the
	// edge will be by the time this async filter lands (~13ms), and peak keeps that margin up through a burst. This must NOT
	// collapse on the instantaneous stop signal: mouse rotation is jittery and cam_inst_angular_speed dips below the floor
	// between frames mid-drag, so a floor-width kick on such a frame would trail the moving edge and open a hole (SESSION066
	// regression, reverted here). Tightening happens only via the decayed-peak "settle" below, never by narrowing the band a
	// live kick uses.
	// SESSION071: measurement cap - see getFilterMaxRotRateDegPerS(). Clamps only the WIDTH the dilation band uses; does
	// not affect `rotating`/`motion_calmed` (still driven by the raw, uncapped trackers), so capping cannot suppress a
	// re-filter or the settle transition - it only bounds how wide a kick's band gets.
	const float max_rot_rate_rad_s = filter_max_rot_rate_deg_per_s * (3.14159265f / 180.f);
	const float w_effective = myMin(myMax(cam_angular_speed_ema, cam_angular_speed_peak), max_rot_rate_rad_s);
	// SESSION066: the camera has come to rest once even the slow-decay peak has fallen back to the floor (~2s after a stop) -
	// distinct from `rotating` (this frame's raw speed), which goes false instantly. A cloud still carrying an above-floor
	// band at that point (filter_dilation_elevated) gets exactly one "settle" re-filter, whose width is now ~floor, to swap
	// the held wide selection for a tight one. Fires once (the kick clears the flag) and only after motion genuinely calmed,
	// so it neither boils (session064: that was the per-frame re-filter off decaying trackers, gated out by `rotating` above)
	// nor holes (the in-motion band stays predictive until this point). This is session059's traversal "settle" ported to
	// the filter - see SplatCloud::last_traversal_dilation_elevated's comment.
	const bool motion_calmed = w_effective <= min_rot_rate_rad_s;
	// SESSION072: anisotropic rotational dilation (session067 §8/§15 plan A). Previously rate_fine/rate_coarse were
	// isotropic - w_floored (= max(w_effective, min_rot_rate_rad_s)) applied identically to all 6 planes via rate*dist
	// inside filterUnculledFrontier(), so a violent flick multiplied the whole draw list regardless of which edge it
	// actually threatened. Split into two additive-by-max components instead:
	//   - baseline (isotropic): min_rot_rate_rad_s has no direction to be anisotropic about - it exists purely to
	//     cover the static->moving transition, where nothing has rotated yet in ANY direction (see min_rot_rate_rad_s's
	//     own history above). Kept as a plain rate*dist floor, same as before, just no longer mixed with the measured
	//     term before the split.
	//   - measured (anisotropic): w_effective paired with cam_angular_axis_ema_ws (SESSION072 - see its own comment)
	//     gives an actual swept-angle VECTOR. filterUnculledFrontier() turns this into a per-plane, per-node pad via
	//     dot(plane_normal, rotation_swept x (node_pos - cam_pos)) - by Cauchy-Schwarz this is never more than the old
	//     isotropic rate*dist (equality only when the node sits exactly perpendicular to the rotation axis), so this
	//     is a strict reduction in padding, never an increase, over every plane and every node.
	// The two are combined with max(), not sum, at the per-node/per-plane site in filterUnculledFrontier() - mirrors
	// exactly how translation_dilation[] above already floors its per-plane EMA/empirical shift against
	// min_trans_dilation_m.
	const float rate_fine_baseline   = min_rot_rate_rad_s * dilation_latency_eff;           // SESSION063 K4 tight window, SESSION072: baseline-only now. SESSION088: measured window when enabled - see dilation_latency_eff.
	const float rate_coarse_baseline = min_rot_rate_rad_s * filter_coarse_dilation_latency;  // wider coarse window, baseline-only.
	// SESSION072: axis and swept magnitudes passed separately rather than as two pre-scaled vectors - the two swept
	// vectors are parallel (same axis, different latency), so the filter can do one cross product per plane instead of
	// two and hoist the fine/coarse blend out of its plane loop. See filterUnculledFrontier().
	const float swept_fine   = w_effective * dilation_latency_eff; // SESSION088 - see dilation_latency_eff.
	const float swept_coarse = w_effective * filter_coarse_dilation_latency;

	const char* filter_kick_reason = "?"; // SESSION064 DIAG
	while(num_filters_in_flight < max_concurrent_filters)
	{
		SplatCloud* best_cloud = NULL;
		for(size_t i=0; i<clouds.size(); ++i)
		{
			SplatCloud* const cloud = clouds[i].ptr();
			if(cloud->cached_ufrontier.isNull() || cloud->filter_in_flight)
				continue;

			// Want a filter when: a fresh U(P) needs its first one; the view rotated past the threshold since the last kick;
			// the camera is turning THIS frame (re-filter every frame so the edge stays fresh under the dilation); or motion
			// has calmed and this cloud is still carrying an over-wide band (SESSION066 settle - see motion_calmed above).
			bool want = cloud->ufrontier_needs_filter || !cloud->have_last_filter_cam_forward;
			const char* reason = (cloud->ufrontier_needs_filter || !cloud->have_last_filter_cam_forward) ? "first/U(P)" : "?"; // SESSION064 DIAG
			if(!want)
			{
				const bool rotated = dot(cam_forward_ws, cloud->last_filter_cam_forward_ws) < rotation_cos_threshold;
				const bool settle = cloud->filter_dilation_elevated && motion_calmed; // SESSION066: one tight re-filter to release the held wide band once peak has decayed.
				want = rotated || rotating || settle;
				reason = rotating ? "rotating" : (rotated ? "rotated" : (settle ? "settle" : "?")); // SESSION064/066 DIAG
			}

			if(want) { best_cloud = cloud; filter_kick_reason = reason; break; } // First eligible - filters are cheap, no need to rank like traversals.
		}
		if(best_cloud == NULL)
			break;

		best_cloud->filter_in_flight = true;
		best_cloud->ufrontier_needs_filter = false;
		best_cloud->filter_dilation_elevated = (w_effective > min_rot_rate_rad_s); // SESSION066: remember if this kick used an above-floor band, so the settle above can later tighten it once peak decays (and the settle kick itself, at ~floor width, clears it).
		best_cloud->have_last_filter_cam_forward = true;
		best_cloud->last_filter_cam_forward_ws = cam_forward_ws;
		best_cloud->diag_filter_kick_time_s = diag_timer.elapsed();   // SESSION073 DIAGNOSTIC - see the member's comment.
		best_cloud->diag_filter_band_fine_rad = swept_fine;           // SESSION073 DIAGNOSTIC: what this kick's fine layer actually dilated by, to compare against how far the camera turns before the result is replaced.
		num_filters_in_flight++;

		// SESSION074: frustum-plane culling and the saturation pre-filter are independent mechanisms that happen to share
		// this one streaming pass over U(P). Passing zero planes disables only the frustum half - the plane loop inside
		// filterUnculledFrontier() then does nothing, every node is "inside", and the saturation test still runs on each
		// one - so the two can be measured separately on the same scene. Note the coarse floor drops out entirely in that
		// mode, which is correct rather than incidental: it exists only to plug frustum-edge gaps revealed by motion, and
		// with no frustum edge there is nothing for it to plug (the band restriction below rejects it on its own).
		const int filter_num_planes = filter_frustum_planes_enabled ? scene->num_frustum_clip_planes : 0;

		task_manager->addTask(new GaussianSplatFilterTask(best_cloud->cloud_id, best_cloud->cached_ufrontier,
			scene->frustum_clip_planes, filter_num_planes, cam_pos_ws, rate_fine_baseline, rate_coarse_baseline,
			cam_angular_axis_ema_ws, swept_fine, swept_coarse, trans_dilation, coarse_layer_debug,
			split_coarse_floor_enabled, &filter_result_queue)); // SESSION075: "coarse" checkbox now controls only whether the (possibly sat-only-captured) coarse layer is allowed to draw - see filterUnculledFrontier()'s draw_coarse_layer.

		if(filter_debug_log) // SESSION064 DIAG: how long do filter kicks continue after the camera stops, and with what band?
			conPrint("[gsr-filter-kick] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms reason=" + std::string(filter_kick_reason) +
				" ema=" + doubleToStringNDecimalPlaces(cam_angular_speed_ema * (180.0 / 3.14159265), 1) + "deg/s" +
				" peak=" + doubleToStringNDecimalPlaces(cam_angular_speed_peak * (180.0 / 3.14159265), 1) + "deg/s" +
				" w_eff=" + doubleToStringNDecimalPlaces(w_effective * (180.0 / 3.14159265), 1) + "deg/s" + // SESSION072: the capped magnitude actually paired with the axis below.
				" axis=(" + doubleToStringNDecimalPlaces(cam_angular_axis_ema_ws.x[0], 2) + "," + doubleToStringNDecimalPlaces(cam_angular_axis_ema_ws.x[1], 2) + "," + doubleToStringNDecimalPlaces(cam_angular_axis_ema_ws.x[2], 2) + ")" + // SESSION072
				" rate_fine_baseline=" + doubleToStringNDecimalPlaces(rate_fine_baseline, 3));
	}
}


// SESSION063: apply completed filter results. Uploads the survivor draw list S(P,R) to the instance VBO - the only GL
// call in the filter pipeline, hence on the main thread. Drops a result whose U(P) has since been replaced by a fresh
// traversal (pointer identity against the cloud's current cached_ufrontier).
void GaussianSplatRenderer::drainFilterResults()
{
	filter_result_queue.dequeueAnyQueuedItems(completed_filter_msgs);

	// SESSION069 - see drainSortResults() above.  Per-orientation filter (session063 split architecture) also revises
	// the draw list, so TAA must reset when it lands.
	if(!completed_filter_msgs.empty()) noteContentApplied();

	for(size_t i=0; i<completed_filter_msgs.size(); ++i)
	{
		const GaussianSplatFilterResultMsg* const msg = static_cast<const GaussianSplatFilterResultMsg*>(completed_filter_msgs[i].ptr());

		SplatCloud* cloud = NULL;
		for(size_t c=0; c<clouds.size(); ++c)
			if(clouds[c]->cloud_id == msg->cloud_id) { cloud = clouds[c].ptr(); break; }

		if(cloud)
		{
			num_filters_in_flight--;
			cloud->filter_in_flight = false;
			// SESSION088: the round trip session073's age= has been printing all along, now read back rather than only
			// looked at - see GsMeasuredLatency and getFilterLatencyMeasuredEnabled(). With the bookkeeping above and
			// unconditional for the same two reasons: it costs a subtraction, and a result the staleness guard below
			// DISCARDS still measured how long this pipeline takes, which is the only thing this number is for.
			filter_latency_measured.addSample(diag_timer.elapsed() - cloud->diag_filter_kick_time_s);
		}
		else
		{
			num_filters_in_flight--; // Cloud gone; still account for the slot.
			continue;
		}

		// SESSION081: accept whenever this result's SOURCE U(P) is not older than whatever produced what is currently
		// on screen - not only when it is an exact identity match against cloud->cached_ufrontier. The identity-only
		// version used to throw the whole result away the instant ANY newer traversal or saturation apply landed
		// while this filter was computing, which cost nothing when that was rare - but once saturation applies started
		// landing far more often (see the session081 apply decoupling/throttling), cached_ufrontier started changing
		// often enough that a filter's own ~13-60ms round trip regularly lost the race, and DISCARDING (rather than
		// applying) meant the on-screen selection could go stale across several consecutive lost races before one
		// finally landed - directly measured: stale=44deg against a 40deg band, from 2 of 3 kicks being thrown away
		// in a row during combined translation+rotation. That is a frustum-edge hole, on top of - not the same
		// mechanism as - the saturation-side one this session started from.
		//
		// Why relaxing this is still safe even though a filter CHOOSES content (not just prunes it, unlike the
		// saturation apply guard this mirrors): a superseded frontier's successor is ALREADY installed as
		// cached_ufrontier and has ALREADY set ufrontier_needs_filter, so its own filter kick is already queued (or in
		// flight) independently of what this guard decides here. Accepting an intermediate, not-literally-latest
		// result therefore never delays or replaces the eventual newest one - it only ever replaces something OLDER
		// currently on screen with something less old, while the truly latest is still on its way regardless.
		if(msg->frontier->built_time_real_s < cloud->last_applied_filter_frontier_time_s)
			continue;
		cloud->last_applied_filter_frontier_time_s = msg->frontier->built_time_real_s;

		// SESSION063 K4: survivors is the globally sorted draw list (fine + coarse interleaved by depth, or coarse-only in
		// the debug view - the filter already applied that). Upload as-is.
		const js::Vector<uint32, 16>& survivors = msg->survivors;
		cloud->instance_index_vbo->updateData(0, survivors.data(), survivors.size() * sizeof(uint32));
		cloud->ob->num_instances_to_draw = (int)survivors.size();
		noteDrawOrderForSlicing(*cloud, survivors.data(), survivors.size());

		// SESSION073 DIAGNOSTIC: band adequacy in degrees - see SplatCloud::diag_filter_kick_time_s's comment.
		//   age    : real kick->drain round-trip (what the fixed filter_dilation_latency knob is supposed to stand in for).
		//   swept  : how far the camera actually turned over that window.
		//   band   : the angular margin this result was dilated by (swept_fine, from its own kick).
		//   stale  : total angle the PREVIOUS list went stale by before this one replaced it - the worst case, since a list
		//            stays on screen from its own kick until the next drain, not just until its own.
		// A hole is expected exactly when stale > band; the excess is its angular size at the frustum edge.
		// The two state writes are unconditional (a couple of assignments) so that toggling the log on mid-flight doesn't
		// report its first line against a stale reference pose; only the trigonometry below is gated - session072 measured
		// that diagnostic work in this path costs real frame time.
		const Vec4f prev_applied_forward = cloud->diag_applied_kick_forward_ws;
		const bool had_applied = cloud->diag_have_applied_filter;
		cloud->diag_applied_kick_forward_ws = cloud->last_filter_cam_forward_ws; // This result is now the list on screen; the next drain reports how stale it got.
		cloud->diag_have_applied_filter = true;

		if(filter_debug_log) // SESSION064 DIAG: does the applied survivor count keep changing after the camera stops? That churn is what the gate turns into boiling.
		{
			const double now_s_diag = diag_timer.elapsed();
			const Vec4f cur_forward_diag = normalise(opengl_engine->getCurrentScene()->cam_to_world.getColumn(1)); // Column 1 is forward; normalise because acos(dot()) silently under-reads small angles otherwise.
			const float swept_deg = std::acos(myClamp(dot(cur_forward_diag, cloud->last_filter_cam_forward_ws), -1.f, 1.f)) * (180.f / 3.14159265f);
			const float stale_deg = had_applied ?
				(std::acos(myClamp(dot(cur_forward_diag, prev_applied_forward), -1.f, 1.f)) * (180.f / 3.14159265f)) : 0.f;
			const float band_deg = cloud->diag_filter_band_fine_rad * (180.f / 3.14159265f);
			conPrint("[gsr-filter-drain] t" + doubleToStringNDecimalPlaces(now_s_diag * 1000.0, 0) + "ms surv=" + uInt64ToStringCommaSeparated(survivors.size()) +
				" coarse=" + uInt64ToStringCommaSeparated(msg->num_coarse_survivors) + // SESSION072 DIAGNOSTIC
				" fine=" + uInt64ToStringCommaSeparated(survivors.size() - msg->num_coarse_survivors) +
				" pool=" + uInt64ToStringCommaSeparated(msg->frontier->indices.size() +
					(msg->frontier->far_block.nonNull() ? msg->frontier->far_block->indices.size() : 0)) + // SESSION080 §4.3: both segments - indices.size() alone is just the near one, which read as pool < surv.

				" compute=" + doubleToStringNDecimalPlaces(msg->filter_compute_ms, 2) + "ms" + // SESSION072 DIAGNOSTIC: pure filterUnculledFrontier() cost, no scheduling.
				// SESSION073 DIAGNOSTIC - see the block above.
				" age=" + doubleToStringNDecimalPlaces((now_s_diag - cloud->diag_filter_kick_time_s) * 1000.0, 1) + "ms" +
				// SESSION088: the EMA over that same age, and the window kickOffFilters() actually used. Printed
				// unconditionally of the knob so the measurement can be checked against the constant it would replace
				// BEFORE it is switched on - lat_used= says which of the two is currently in the loop. See
				// getFilterLatencyMeasuredEnabled().
				" lat_meas=" + doubleToStringNDecimalPlaces(filter_latency_measured.get(0.f) * 1000.0, 1) + "ms" +
				" lat_used=" + doubleToStringNDecimalPlaces((filter_latency_measured_enabled ? (filter_latency_measured.get(filter_dilation_latency * 0.5f) * 2.f) : filter_dilation_latency) * 1000.0, 1) + "ms" + // The x2 envelope, mirroring kickOffFilters()'s dilation_latency_eff - see there.
				" swept=" + doubleToStringNDecimalPlaces(swept_deg, 2) + "deg" +
				" band=" + doubleToStringNDecimalPlaces(band_deg, 2) + "deg" +
				" stale=" + doubleToStringNDecimalPlaces(stale_deg, 2) + "deg" +
				" deficit=" + doubleToStringNDecimalPlaces(stale_deg - band_deg, 2) + "deg");

		}
	}

	completed_filter_msgs.clear();
}


// SESSION055 diag: one shared Timer for kickOffTraversals logging; timestamps are ms-since-first-log so
// pauses and streaks in the traversal pipeline read easily against each other. Toggle: getKickDebugLog()/
// setKickDebugLog() (SESSION072 - was a build-time const here, see its declaration's comment in the header).
// Only two events actually print, both signal-only:
//   [gsr-kick]        each successful traversal kick, with reason (rot / trans / topo / first / settle - SESSION059).
//   [gsr-rot-blocked] when the camera is rotating fast enough to trigger the 5deg re-kick BUT no kick went out
//                     (either the target cloud's slot is in flight, or all concurrent slots are full).
// Both are throttled per-event-type - see rot_blocked_min_gap_ms below - so a held-down rotation logs a heartbeat,
// not a flood.
static double last_rot_blocked_log_ms = -1e9;
static const double rot_blocked_min_gap_ms = 250.0;

// SESSION081: kicks a GaussianSplatSaturationBuildTask for any cloud whose saturation barrier is missing, stale by
// the ball guarantee (camera has left the R-ball its anchor was built at), or built under knobs that no longer match
// the live settings. Independent of kickOffTraversals() entirely - see GaussianSplatSaturationBarrier for why the
// two no longer need to move in lockstep.
//
// NOTE: setSatRegionRadius()/setSatGridSubdiv()/setSatPrefilterThreshold()/setSatRegionClosingTiles() still drop
// cached_ufrontier and force a fresh TRAVERSAL when dragged, which is more than strictly needed now - the "knobs
// changed" check below would catch a mismatch and rebuild just the BARRIER from the cloud's existing
// last_unpruned_ufrontier, no fresh traversal required. Left as is: those are rare, owner-driven spinbox edits, not a
// hot path, and changing four setters' invalidation scope is outside this decoupling's actual goal.
void GaussianSplatRenderer::kickOffSaturationBuilds()
{
	glare::TaskManager* const task_manager = opengl_engine->getMainTaskManager();
	if(task_manager == NULL)
		return;

	// SESSION085 ETAP 3: the barrier now has THREE consumers, not one - the prune, the debug overlay, and the LoD bias -
	// so it is built when ANY of them wants it. Gating it on the prune checkbox alone meant turning the prune off to
	// measure the bias on its own silently starved the bias of its only input, and the bias would have read as "does
	// nothing" rather than as "was never given a barrier". That is exactly the failure session075 hit with the coarse
	// floor (one flag conflating "capture it" with "draw it"), and it cost a wasted arm of the session085 occluder
	// comparison when the same shape appeared again there.
	const bool sat_wanted = (sat_debug_overlay_mode != GaussianSplatSatDebugOverlayMode_Off) ||
		(sat_bias_ceiling > 1.f); // SESSION085 ETAP 3 - see getSatBiasCeiling().
	if(!sat_wanted)
		return;

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Vec4f cam_pos_ws = scene->cam_to_world.getColumn(3);

	// SESSION088 PREDICTIVE ANCHOR - see getSatPredictGain() for the full argument. A build kicked now lands one
	// measured round trip from now, by which time the camera is elsewhere; aim the anchor at where it will be instead of
	// where it is. Gain 0 makes lead exactly zero, so predicted_cam_pos_ws is cam_pos_ws and everything below is
	// bit-for-bit the previous behaviour.
	//
	// The lead is the MEASURED barrier round trip (getSatPredictGain()'s last paragraph on why it must not be a
	// constant), and it is zero until the first build has been drained - a cold start therefore anchors at the camera,
	// which is the safe direction to be wrong in.
	//
	// SESSION088, SECOND CUT: the velocity is measured KICK TO KICK here rather than read from cam_velocity_ema_ws,
	// and that is the whole difference between this working and not. The shared EMA is a per-FRAME differencer whose dt
	// guard in think() admits intervals down to 0.1ms, so an ordinary few-millimetre step on a very short frame reports
	// a triple-digit m/s, and alpha=0.15 then carries that spike for ~7 frames. Harmless where it was built for -
	// dilation, where over-prediction is just padding - but this is an ANCHOR, where over-prediction does not widen a
	// margin, it builds the barrier in the wrong place. Measured session088 with the first cut running: real camera
	// speed (recovered from reuse_drift's slope, which is differenced over whole traversals) never left 3 m/s interior
	// or 15 m/s exterior, while the lead the EMA produced reached 10.45m against a 103ms round trip - an implied
	// ~100 m/s, and 70% of exterior builds anchored further than a whole region radius out.
	//
	// Differencing over the kick-to-kick interval instead averages over exactly the window the prediction is about
	// (~160ms interior, ~300ms exterior), so a single short frame cannot move it, and no clamp or magic speed cap is
	// needed - the estimator is simply measuring the right thing over the right span. The interval guard mirrors
	// think()'s: too short and it is the same small-dt amplifier again, too long and the camera's path between the two
	// samples is not a straight line any more, so a chord speed says nothing useful about where it is heading.
	// The reference is the last ACTUAL kick, not the last time this function ran: it runs every frame and mostly decides
	// to do nothing, so sampling per call would difference over a frame again - the very thing this replaces - and the
	// interval guard would then reject nearly every sample. The state is therefore advanced at the kick site below.
	const double sat_kick_now_s = diag_timer.elapsed();
	const double sat_kick_dt = sat_kick_now_s - sat_predict_prev_kick_time_s;
	Vec4f sat_predict_lead_ws(0.f);
	if(have_sat_predict_prev_kick && sat_kick_dt > 0.03 && sat_kick_dt < 1.0)
		sat_predict_lead_ws = (cam_pos_ws - sat_predict_prev_kick_pos_ws) * (float)((sat_build_latency_measured.get(0.f) * sat_predict_gain) / sat_kick_dt);
	const Vec4f predicted_cam_pos_ws = cam_pos_ws + sat_predict_lead_ws;
	diag_sat_predict_lead_m = sat_predict_lead_ws.length(); // DIAGNOSTIC only - printed on [gsr-sat-build], see there.

	const Vec2i viewport_dims = opengl_engine->getViewportDims();
	const float focal_x = (float)viewport_dims.x * scene->lens_sensor_dist / scene->use_sensor_width;
	const float focal_y = (float)viewport_dims.y * scene->lens_sensor_dist / scene->use_sensor_height;
	const float focal_px = (focal_x + focal_y) * 0.5f; // Same average-of-axes approximation kickOffTraversals() uses - see there for why.

	for(size_t i=0; i<clouds.size(); ++i)
	{
		SplatCloud& cloud = *clouds[i];
		if(cloud.total_splats == 0 || cloud.sat_build_in_flight)
			continue;
		if(cloud.last_unpruned_ufrontier.isNull() || cloud.last_unpruned_ufrontier->topology_generation != cloud.topology_generation)
			continue; // Nothing to build from yet, or what we have predates a structural change - a fresher one is presumably already in flight from the traversal that changed this.

		const Reference<GaussianSplatSaturationBarrier>& barrier = cloud.cached_sat_barrier;
		bool needs_build = barrier.isNull();
		if(!needs_build)
		{
			// A live-tuned knob has moved since this barrier was built - it must take effect on the next opportunity,
			// not silently keep serving a stale barrier indefinitely.
			if(barrier->threshold_used != sat_prefilter_threshold || barrier->subdiv_used != sat_grid_subdiv ||
				barrier->coarse_pixel_scale_used != split_coarse_pixel_scale || // SESSION088: "px" sizes the grid exactly like "sub" does but was missing here, so the knob was completely inert on a static camera - see setCoarsePixelScale().
				barrier->region_radius_used != sat_region_radius || barrier->closing_tiles_used != sat_region_closing_tiles)
				needs_build = true;
			// SESSION081: the ball guarantee itself - see GaussianSplatSaturationBarrier. > not >=: at R=0 (point-anchored)
			// any nonzero movement invalidates, but standing still does not.
			// SESSION088: tested from the PREDICTED point, not the raw camera, because that is the point the anchor was
			// placed at - the two must agree or the mechanism fights itself. Testing the raw camera against a lead anchor
			// would report a standing offset of |lead| that no amount of rebuilding could reduce, so any lead longer than
			// sat_region_radius would rebuild the barrier every single frame. Kept consistent, the cadence is unchanged:
			// under steady motion anchor and test point translate together, so the ball is still left after exactly
			// sat_region_radius of travel. See getSatPredictGain().
			else if(predicted_cam_pos_ws.getDist(barrier->anchor_pos_ws) > sat_region_radius)
				needs_build = true;
		}
		if(!needs_build)
			continue;

		Reference<GaussianSplatSaturationBuildTask> t = new GaussianSplatSaturationBuildTask();
		t->cloud_id = cloud.cloud_id;
		t->topology_generation = cloud.topology_generation;
		t->source_frontier = cloud.last_unpruned_ufrontier;
		t->geom = getOrBuildCachedGeom(cloud);
		// SESSION081: allocated once per cloud, on its first build, then reused by every later one - see GsSatBuildScratch.
		if(cloud.sat_build_scratch.isNull())
			cloud.sat_build_scratch = new GaussianSplatSaturationBuildScratch();
		t->build_scratch = cloud.sat_build_scratch;
		t->anchor_pos_ws = predicted_cam_pos_ws; // SESSION088: the predicted camera position, = cam_pos_ws when the gain is 0 - see the lead's computation above.
		t->saturation_threshold = sat_prefilter_threshold;
		t->grid_subdiv = sat_grid_subdiv;
		t->region_radius = sat_region_radius;
		t->closing_tiles = sat_region_closing_tiles;
		t->focal_px = focal_px;
		t->coarse_pixel_scale = split_coarse_pixel_scale;
		t->alpha_gain = splat_alpha_gain; t->alpha_gamma = splat_alpha_gamma;
		t->diag_log = sat_diag_log;
		t->overlay_requested = sat_debug_overlay_mode != GaussianSplatSatDebugOverlayMode_Off;
		// SESSION082: snapshot the members for the tree walk - fresh, not from the geom cache, see the field's comment.
		t->walk_members.resize(cloud.members.size());
		for(size_t m=0; m<cloud.members.size(); ++m)
		{
			t->walk_members[m].splat_data = cloud.members[m].splat_data;
			t->walk_members[m].offset = cloud.members[m].offset;
			t->walk_members[m].hidden = cloud.members[m].hidden;
		}
		t->task_manager = task_manager;
		t->result_queue = &sat_build_result_queue;

		cloud.sat_build_in_flight = true;
		cloud.sat_build_kick_time_s = sat_kick_now_s; // SESSION088: unconditional (one assignment) so the latency tracker is fed whether or not any log is on - see the member and drainSaturationBuildResults().
		// SESSION088: advance the predictive velocity's reference to THIS kick - see sat_predict_lead_ws above for why it
		// has to be the last real kick rather than the last call. Assigned per kick rather than once after the loop
		// because both writes are the same values for every cloud in this pass (one camera, one clock), so a second
		// cloud simply re-writes what the first did.
		sat_predict_prev_kick_pos_ws = cam_pos_ws;
		sat_predict_prev_kick_time_s = sat_kick_now_s;
		have_sat_predict_prev_kick = true;
		task_manager->addTask(t);
	}
}




void GaussianSplatRenderer::kickOffTraversals()
{
	glare::TaskManager* const task_manager = opengl_engine->getMainTaskManager();
	if(task_manager == NULL)
		return;

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Vec4f cam_pos_ws = scene->cam_to_world.getColumn(3);
	// SESSION055: Substrata camera basis convention is right=col0, forward=col1, up=col2 (see OpenGLEngine.cpp uses of
	// cam_to_world.getColumn). Column 2 (up) is INSENSITIVE to pure-yaw rotation at zero pitch, which is exactly the case
	// (keyboard turn) where re-kicks were silently not firing during session055 diagnosis. Read col 1 (forward), and
	// normalise: a small non-unit drift in the matrix makes dot > 1 at small angles, myClamp truncates to 1, acos returns
	// 0, and small rotations vanish from the trigger and from cam_angular_speed_ema in think().
	const Vec4f cam_forward_ws = normalise(scene->cam_to_world.getColumn(1));

	const Vec2i viewport_dims = opengl_engine->getViewportDims();
	const float focal_x = (float)viewport_dims.x * scene->lens_sensor_dist / scene->use_sensor_width;
	const float focal_y = (float)viewport_dims.y * scene->lens_sensor_dist / scene->use_sensor_height;
	const float focal_px = (focal_x + focal_y) * 0.5f; // Average of the two axes - a splat's on-screen size only differs meaningfully per axis with a non-square viewport/sensor, close enough for the traversal's coarse pixel_scale budget.

	// SESSION055: with frustum-cull on the traversal is view-dependent, so a rotation in place has to trigger a re-kick even
	// though the camera position hasn't changed. 5deg is the coarse threshold - a full 360deg pan then costs 72 traversals,
	// which is far under the concurrency cap and negligible per traversal after the session054 speed-up. Kept an internal
	// constant rather than a live knob: the trade is not scene-dependent, it's about how large a stale frustum edge is
	// tolerable, and 5deg was chosen to be well under what a first-person turn perceives as a hitch. cos(5deg) ~= 0.99619.
	const float rotation_cos_threshold = 0.99619f;
	// SESSION063: in split_filter mode the traversal is deliberately run cull-off (it builds the orientation-independent
	// U(P)), so cull_active goes false here - which also turns off the rotation/settle re-kick triggers and the dilation
	// block below, because a rotation no longer needs a fresh traversal: the async filter re-derives S(P,R) from the
	// cached U(P) instead (see drainTraversalResults()/GaussianSplatFilterTask). Position/topology triggers still fire.
	const bool cull_active = lod_frustum_cull_enabled && !split_filter_enabled;

	// SESSION055: anisotropic frustum-cull dilation. Async traversal takes ~150-500ms per session054; during that window
	// the camera keeps moving/rotating, so nodes that were outside at kick time can be inside by the time the result
	// applies - visible as holes along the screen edge on turns and back-motion.
	// Rate model: max(EMA, empirical). EMA is the smoothed motion tracker (rise-fast, fall-slow); empirical is the
	// actual (translation, rotation) that has happened between the previous kick for THIS cloud and now. Empirical
	// catches burst motion (mouse flicks) that EMA underestimates once the flick ends and the value decays before the
	// next kick; EMA catches motion that just started (empirical from last kick is stale then). The max covers both.
	// traversal_latency_estimate is a conservative constant matching session054's measured 165ms interior / 450ms
	// bridge - one number for both because the dilation is a safety margin, not a precise correction.
	const float traversal_latency_estimate = 0.3f; // seconds. Conservative; raising it costs a little cull, lowering it risks holes.
	const double now_s = diag_timer.elapsed();

	// SESSION055 baseline (see its own comment further down, where it's applied) - hoisted here too so the SESSION059
	// "settle" check just below can compare the *current* w_effective against the same floor every kick already
	// guarantees, without duplicating the derivation. min_rot_dilation_rad is the amount of dilation baseline alone
	// contributes; min_rot_rate_rad_s is the underlying rate, which is what "has w_effective calmed back down" needs.
	//
	// SESSION059 tried giving the settle-triggered kick extra margin here, to cover a mouse flick starting during the
	// settle kick's own in-flight window (the "first flick after calm" hole session055 already documented as generally
	// unavoidable - previously seen once at app startup, now recurring on every pause because the "stuck wide" bug this
	// session fixed used to accidentally leave a wide margin in place forever instead of correctly tightening it).
	// Two attempts, both reverted:
	//   - Raising this constant itself fixed the hole but made EVERY kick pay for mouse-flick-sized margin permanently
	//     (measured: pinned CPU at ~19ms with no drop, regardless of motion).
	//   - Scoping the extra margin to just the settle kick avoided that, but on this session's km-scale test scene,
	//     rotation_dilation_rate's distance-proportional term means even a scoped ~60deg margin balloons the settle
	//     kick's own selection back up near the stuck-bug's size (~7.8M vs ~2.4M splats) - so "settle" stopped visibly
	//     dropping GPU at all, defeating its purpose.
	// Root cause turned out to be latency, not margin: the first kick after resuming motion already computes a large,
	// correct dilation (session059 measured 80.9deg from a fresh flick) but still takes the full ~500ms
	// traversal_latency_estimate to land, and the tight, undialted settle selection is what's on screen for that whole
	// window regardless of how generous the *next* kick's margin will be once it arrives. No margin tuning here can
	// shorten that window - the real fix (parallel expand, session054 §2A.4 - splitting the DFS expand across worker
	// threads to cut traversal latency directly, ~3x expected) is next session's work, see the session059 snapshot.
	const float min_rot_rate_deg_per_s = 45.f;   // half of typical keyboard turn rate
	const float min_rot_rate_rad_s = min_rot_rate_deg_per_s * (3.14159265f / 180.f);
	const float min_rot_dilation_rad = min_rot_rate_rad_s * traversal_latency_estimate;

	while(num_traversals_in_flight < max_concurrent_traversals)
	{
		// Pick the cloud most overdue for a traversal, same "how far the camera has moved relative to this cloud's own
		// threshold" idea kickOffSorts() uses, plus a structural check that has no sort equivalent: a cloud whose topology
		// changed since its last kicked-off traversal is unconditionally overdue, even with zero camera movement, because
		// an append that isn't in the traversal yet means a whole member is missing from the frontier - see
		// SplatCloud::topology_generation's comment.
		SplatCloud* best_cloud = NULL;
		float best_ratio = 1.f;
		const char* best_reason = "none"; // SESSION055 diag - see kick_debug_log.
		for(size_t i=0; i<clouds.size(); ++i)
		{
			SplatCloud* const cloud = clouds[i].ptr();
			if(cloud->total_splats == 0 || cloud->traversal_in_flight || !cloudHasLodTree(*cloud))
				continue;

			float ratio;
			const char* reason;
			if(cloud->last_traversal_kicked_topology_generation != cloud->topology_generation)
			{
				ratio = std::numeric_limits<float>::max();
				reason = "topo";
			}
			else if(!cloud->have_last_traversal_cam_pos)
			{
				ratio = std::numeric_limits<float>::max();
				reason = "first";
			}
			else
			{
				const float threshold = myMax(lod_resort_move_threshold_ws, cloud->aabb_ws.distanceToPoint(cam_pos_ws) * resort_threshold_dist_fraction);
				ratio = cam_pos_ws.getDist(cloud->last_traversal_cam_pos_ws) / threshold;
				reason = "trans";

				// SESSION055: only meaningful while cull is on; without it the traversal is rotation-invariant as before.
				if(cull_active)
				{
					const float rot_dot = dot(cam_forward_ws, cloud->last_traversal_cam_forward_ws);
					if(rot_dot < rotation_cos_threshold)
					{
						ratio = std::numeric_limits<float>::max();
						reason = "rot";
					}
					// SESSION059: "settle" re-kick - see SplatCloud::last_traversal_dilation_elevated's comment. Only
					// reached when rotation itself didn't already trigger above: this cloud's applied selection was
					// picked with an elevated margin, but nothing has rotated since, so nothing else would ever notice
					// that margin is now stale. Once w_effective has decayed back to (or below) the baseline every kick
					// already floors to, one more kick captures a tight selection at the current, calmed-down pose.
					else if(cloud->last_traversal_dilation_elevated && (myMax(cam_angular_speed_ema, cam_angular_speed_peak) <= min_rot_rate_rad_s))
					{
						ratio = std::numeric_limits<float>::max();
						reason = "settle";
					}
				}
			}

			if(ratio > best_ratio)
			{
				best_ratio = ratio;
				best_cloud = cloud;
				best_reason = reason;
			}
		}

		if(best_cloud == NULL)
			break;

		Reference<GaussianSplatLodTraversalScratch> scratch;
		if(free_traversal_scratch.empty())
			scratch = new GaussianSplatLodTraversalScratch();
		else
		{
			scratch = free_traversal_scratch.back();
			free_traversal_scratch.pop_back();
		}

		fillTraversalScratch(*best_cloud, *scratch);

		// SESSION055: per-cloud dilation. Empirical rates use the delta since THIS cloud's previous kick (each cloud may
		// have been kicked at a different moment). First-kick fallback: no empirical, EMA only.
		float translation_dilation[6] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		float rotation_dilation_rate = 0.f;
		float empirical_ang_rate_dbg = 0.f;
		if(cull_active)
		{
			// SESSION055: per-plane max of EMA-vs-empirical directional shift. Doing max on the *shift* rather than on
			// the velocity vector lets each plane pick the estimate that actually threatens it: a stale-east EMA gives
			// dot(west_plane_n, ema*L)<0 (rejected by max with 0), while empirical west gives dot(west_plane_n, emp*L)>0
			// (kept). Handles direction reversals without needing to reason about "which velocity direction is real".
			const Vec4f ema_expected = cam_velocity_ema_ws * traversal_latency_estimate;
			Vec4f emp_expected(0.f);
			float w_effective = myMax(cam_angular_speed_ema, cam_angular_speed_peak); // SESSION055: peak covers mouse-flick-in-past for the next ~1s of kicks - see the peak update in think().
			if(best_cloud->have_last_traversal_cam_pos)
			{
				const double dt = myMax(1.0e-3, now_s - best_cloud->last_traversal_kick_time_s); // Floor guards a degenerate 0 dt (would explode empirical rate); 1ms is well below any realistic kick cadence.
				const Vec4f empirical_v = (cam_pos_ws - best_cloud->last_traversal_cam_pos_ws) * (float)(1.0 / dt);
				emp_expected = empirical_v * traversal_latency_estimate;
				const float rot_dot = myClamp(dot(cam_forward_ws, best_cloud->last_traversal_cam_forward_ws), -1.f, 1.f);
				const float empirical_w = std::acos(rot_dot) / (float)dt;
				empirical_ang_rate_dbg = empirical_w;
				w_effective = myMax(w_effective, empirical_w);
			}
			for(int i=0; i<scene->num_frustum_clip_planes && i<6; ++i)
			{
				const Vec4f n = scene->frustum_clip_planes[i].getNormal();
				const float shift_ema = dot(n, ema_expected);
				const float shift_emp = dot(n, emp_expected);
				translation_dilation[i] = myMax(0.f, myMax(shift_ema, shift_emp));
			}
			rotation_dilation_rate = w_effective * traversal_latency_estimate;

			// SESSION055: baseline minimum dilation covers the "static->moving" transition. Both EMA and empirical
			// read zero at that transition (nothing has moved yet since last kick), so the traversal that lands next
			// has no rotation margin at all - and by the time the next kick captures the new motion, another 120ms
			// of holes have shown. Baseline says "even if pose looked frozen at kick time, allow for X m/s and Y
			// deg/s of motion possibly starting during this traversal". Tunable; roll back this block if the picture
			// doesn't improve, since it costs a modest amount of over-inclusion in genuinely static scenes.
			// min_rot_rate_deg_per_s/min_rot_dilation_rad are hoisted above the while-loop now - see SESSION059's comment
			// there - so the "settle" check can share the exact same floor this block applies.
			const float min_trans_rate_m_per_s = 2.0f;   // ~walking pace
			const float min_trans_dilation_m  = min_trans_rate_m_per_s * traversal_latency_estimate;
			for(int i=0; i<scene->num_frustum_clip_planes && i<6; ++i)
				translation_dilation[i] = myMax(translation_dilation[i], min_trans_dilation_m);
			rotation_dilation_rate = myMax(rotation_dilation_rate, min_rot_dilation_rad);
		}

		// SESSION059: remember whether this kick used more than baseline dilation, so the "settle" check above can catch
		// this cloud once w_effective decays back down with no further rotation to naturally trigger a fresh kick - see
		// SplatCloud::last_traversal_dilation_elevated's comment. A tiny epsilon avoids flagging float noise right at
		// the floor as "elevated".
		best_cloud->last_traversal_dilation_elevated = cull_active && (rotation_dilation_rate > min_rot_dilation_rad + 1.0e-6f);

		best_cloud->traversal_in_flight = true;
		best_cloud->have_last_traversal_cam_pos = true;
		best_cloud->last_traversal_cam_pos_ws = cam_pos_ws;
		best_cloud->last_traversal_cam_forward_ws = cam_forward_ws; // SESSION055 - see rotation_cos_threshold above.
		best_cloud->last_traversal_kick_time_s = now_s;
		best_cloud->last_traversal_kicked_topology_generation = best_cloud->topology_generation;
		num_traversals_in_flight++;

		if(kick_debug_log)
			conPrint("[gsr-kick] t" + doubleToStringNDecimalPlaces(now_s * 1000.0, 0) + "ms cloud=" +
				toString(best_cloud->cloud_id) + " reason=" + std::string(best_reason) +
				" ratio=" + (best_ratio == std::numeric_limits<float>::max() ? std::string("inf") : doubleToStringNDecimalPlaces(best_ratio, 2)) +
				" in_flight=" + toString(num_traversals_in_flight) + "/" + toString(max_concurrent_traversals) +
				" splats=" + toString(best_cloud->total_splats) +
				" w_ema=" + doubleToStringNDecimalPlaces(cam_angular_speed_ema * (180.0 / 3.14159265), 1) +
				" w_emp=" + doubleToStringNDecimalPlaces(empirical_ang_rate_dbg * (180.0 / 3.14159265), 1) +
				" rot_dil=" + doubleToStringNDecimalPlaces(rotation_dilation_rate * (180.0 / 3.14159265), 1) + "deg");

		// SESSION055: pass frustum planes (copied into task, see its ctor) and the anisotropic dilation numbers computed
		// once above per kickOffTraversals() call.
		// SESSION075: coarse-floor DATA capture (feeds both the edge-filling draw layer AND the saturation grid) is now
		// requested by EITHER consumer wanting it - previously a single flag conflated "capture the coarse floor" with
		// "draw it as edge-filling", so turning "coarse" off (wanted only to skip the draw) also starved the saturation
		// grid of its only data source. The "coarse" checkbox (split_coarse_floor_enabled) now controls solely whether
		// the captured layer is later allowed to survive the filter - see kickOffFilters()'s draw_coarse_layer.
		// SESSION076: the saturation grid no longer needs the coarse floor - it accumulates the fine frontier instead
		// (see the traversal task's saturation phase). So capture is back to being requested by its one real consumer,
		// the drawn edge-filling patch. With the "coarse" checkbox off, the ~1.7M coarse nodes that session075 started
		// capturing for the grid's sake are simply never produced: no push_backs in the DFS, no ~56% growth of the radix
		// sort's array, no SoA gather, no inflated pool for the filter to stream.
		const bool coarse_capture_needed = split_coarse_floor_enabled;
		task_manager->addTask(new GaussianSplatLodTraversalTask(best_cloud->cloud_id, best_cloud->topology_generation, scratch, cam_pos_ws,
			lod_pixel_scale_limit, lod_max_splats_budget, lod_max_layer_density, lod_max_tree_depth, focal_px,
			scene->frustum_clip_planes, scene->num_frustum_clip_planes, cull_active,
			translation_dilation, rotation_dilation_rate,
			&traversal_result_queue,
			/*frontier_record=*/NULL, /*build_unculled_frontier=*/split_filter_enabled, // SESSION063: cull-off traversal builds U(P) for the split filter.
			/*coarse_floor_enabled=*/split_filter_enabled && coarse_capture_needed, /*coarse_pixel_scale=*/split_coarse_pixel_scale, // SESSION063 K4, SESSION075.
			/*dist_clamp_enabled=*/splat_dist_clamp_enabled, splat_dist_clamp_min, splat_dist_clamp_max, splat_dist_clamp_invert, // SESSION072.
			/*sat_diag_log=*/sat_diag_log, // SESSION076 DIAGNOSTIC.
			/*coarse_layer_drawn=*/split_coarse_floor_enabled, // SESSION076: lets the task skip/evict coarse nodes when the layer is captured only to feed the saturation grid - see its ctor param.
			/*task_manager=*/task_manager, // SESSION079
			/*prev_frontier=*/best_cloud->last_unpruned_ufrontier, // SESSION080: the previous traversal's UNPRUNED output, for STEP A's diagnostic and STEP B's reuse - NOT cached_ufrontier, see that field's comment.
			/*sort_staleness_diag_enabled=*/sat_diag_log, // SESSION080: the "diag" checkbox, NOT the filter log that prints the line.
			// The two have to be separate because this measurement costs 185-848ms of single-threaded, memory-hungry work
			// and visibly inflates the NEXT traversal's timings through contention (soa_ms, a pure map, was seen to swing
			// 72 -> 529ms). Sharing the filter log's toggle made "print the timings" and "corrupt the timings" the same
			// switch, so the pipeline could not be timed and traced at once. This checkbox already means exactly that
			// trade for the saturation counters - see getSatDiagLog() - so it is the right home for it.
			/*reuse_split_dist=*/frontier_reuse_split_dist, // SESSION080 STEP B - see getFrontierReuseSplitDist(). 0 = walk the whole tree, as before.
			// SESSION085 ETAP 3: the cloud's current barrier, for the LoD bias. One build behind by construction - the same
			// staleness the apply stage already lives with - and null on a cloud's first kick, which leaves the bias off.
			/*sat_barrier=*/best_cloud->cached_sat_barrier,
			/*sat_bias_ceiling=*/sat_bias_ceiling, // SESSION085 ETAP 3 - see getSatBiasCeiling(). 1 = off.
			/*reuse_drift_fraction=*/frontier_reuse_drift_fraction, // SESSION086 - see getFrontierReuseDriftFraction().
			/*expand_seed_target=*/expand_seed_target, // SESSION086 - see updateExpandSeedTarget().
			/*sat_barrier_agree_tol=*/sat_barrier_agree_tol, // SESSION088 - see getSatBarrierAgreeTol(). 1 = accept any barrier change, as before.
			/*sat_bias_exponent=*/sat_bias_exponent)); // SESSION088 - see getSatBiasExponent(). 1 = session085's original curve.
	}

	// SESSION055 diag: after the while-loop, detect *unmet* rotation demand - a cloud whose forward has shifted past the
	// re-kick threshold since its last kick, but that couldn't be kicked this pass because its slot is in flight or all
	// concurrent slots are full. Throttled: at most one line every rot_blocked_min_gap_ms, so a held-down rotation logs a
	// heartbeat rather than a per-frame stream.
	if(kick_debug_log && cull_active)
	{
		int blocked_inflight = 0, blocked_no_slot = 0;
		float worst_deg_over_threshold = 0.f;
		for(size_t i=0; i<clouds.size(); ++i)
		{
			const SplatCloud* const cloud = clouds[i].ptr();
			if(cloud->total_splats == 0 || !cloudHasLodTree(*cloud) || !cloud->have_last_traversal_cam_pos)
				continue;
			const float rot_dot = myClamp(dot(cam_forward_ws, cloud->last_traversal_cam_forward_ws), -1.f, 1.f);
			if(rot_dot >= rotation_cos_threshold)
				continue; // Within tolerance, no demand.
			const float deg = std::acos(rot_dot) * (180.f / 3.14159265f);
			if(deg > worst_deg_over_threshold)
				worst_deg_over_threshold = deg;
			if(cloud->traversal_in_flight)
				blocked_inflight++;
			else if(num_traversals_in_flight >= max_concurrent_traversals) // Slot exhaustion by *other* clouds' work.
				blocked_no_slot++;
		}
		if(blocked_inflight > 0 || blocked_no_slot > 0)
		{
			const double now_ms = diag_timer.elapsed() * 1000.0;
			if(now_ms - last_rot_blocked_log_ms >= rot_blocked_min_gap_ms)
			{
				last_rot_blocked_log_ms = now_ms;
				conPrint("[gsr-rot-blocked] t" + doubleToStringNDecimalPlaces(now_ms, 0) + "ms worst=" +
					doubleToStringNDecimalPlaces(worst_deg_over_threshold, 1) + "deg (thr 5deg) inflight_blocked=" +
					toString(blocked_inflight) + " no_slot_blocked=" + toString(blocked_no_slot) +
					" w=" + doubleToStringNDecimalPlaces(cam_angular_speed_ema * (180.0 / 3.14159265), 1) + "deg/s");
			}
		}
	}
}


// SESSION079: hold a frame-time target by trading LoD detail for speed - see getAdaptivePixelScale() for why a fixed
// pixel_scale_limit cannot be right for both an interior and an open flyover.
//
// The band is what makes this work under vsync. Frame time cannot report HEADROOM - a frame that could have been drawn
// in 4ms and one that took 16 both read as 16 - so a controller aiming at a single number would sit in its own
// deadband and never move, and a narrow deadband would ratchet one way. Instead there are two thresholds with real
// distance between them, taken from the owner's own zones: at or above the target rate there is room, so refine;
// below the floor (0.8x the target - 40fps against a 50fps target) it is too slow, so coarsen; the band between is
// where it is meant to sit and nothing happens. Quantisation does not hurt this: vsync forces individual frames to
// 1/60, 1/30 and so on, but the MEAN over a second of mixed frames moves continuously across the band.
//
// The signal is wall frame time rather than the GL splat-pass timer query, which is gated behind
// query_profiling_enabled and is unreliable-to-absent under WebGL2 - and the web client is the target, so a controller
// built on it would not ship.
//
// It deliberately does NOT force a traversal for every adjustment. The value is picked up by whatever traversal
// happens next, and traversals happen naturally whenever the camera moves - which is when performance matters. A
// forced refresh is only for the standing-still case, where nothing else would ever apply the new value, and is rate
// limited so a static camera cannot turn this into a traversal treadmill (that failure mode is session064's).
void GaussianSplatRenderer::updateAdaptivePixelScale()
{
	const double now_s = adaptive_timer.elapsed();
	const double prev_s = adaptive_last_frame_time_s;
	adaptive_last_frame_time_s = now_s;
	if(!adaptive_pixel_scale_enabled || prev_s < 0.0)
		return; // First frame after enabling has no period to measure.

	// Every frame contributes a sample; the DECISION is taken on a slow cadence (see gs_adaptive_eval_interval_s).
	// Sampling every frame is not the expensive part it looks like - the clock read below is one this function needs
	// anyway, leaving an add and an increment - and sampling less often would be strictly worse data: one frame in
	// fifty, picked arbitrarily, is as likely to land on a hitch as on a fast frame, whereas the mean over the whole
	// window is stable.
	//
	// Clamped so a hitch, a breakpoint or a minimised window cannot drag the average to a limit.
	adaptive_frame_ms_sum += myClamp((now_s - prev_s) * 1000.0, 1.0, 200.0);
	++adaptive_frame_samples;
	if(now_s - adaptive_last_eval_time_s < gs_adaptive_eval_interval_s)
		return;
	adaptive_last_eval_time_s = now_s;

	const double window_ms = adaptive_frame_ms_sum / (double)myMax<size_t>(1, adaptive_frame_samples);
	adaptive_frame_ms_sum = 0.0;
	adaptive_frame_samples = 0;
	adaptive_frame_ms_ema = window_ms; // What the readout shows, and what the step below acts on.

	// The owner's zones: target 50fps, and 40 is where this mechanism is supposed to start working. Above the target
	// there is room to spend; below the floor there is not; between them is the band it should live in.
	const double refine_ms  = 1000.0 / (double)myMax(1.f, adaptive_target_fps);                              // >= target rate: room to spend.
	const double coarsen_ms = 1000.0 / (double)myMax(1.f, adaptive_target_fps * gs_adaptive_floor_fraction); // < floor rate: too slow.

	// Asymmetric on purpose. Backing off has to outrun the drop that caused it; creeping forward has to be slow enough
	// that it does not read as detail pumping. ~6s from 2 to the ceiling, ~22s back - a drop is fixed quickly and the
	// recovery is meant to go unnoticed.
	float v = adaptive_pixel_scale;
	if(window_ms > coarsen_ms)
		v *= 1.30f; // Too slow: one firm step coarser.
	else if(window_ms <= refine_ms)
		v *= 0.93f; // Room to spare: edge back towards detail.

	// The clamp is the honest statement of what this can do. Below the floor the splat count explodes for detail no
	// display resolves; above the ceiling the scene is visibly blocky and giving up more buys little.
	adaptive_pixel_scale = myClamp(v, gs_adaptive_pixel_scale_min, gs_adaptive_pixel_scale_max);
	lod_pixel_scale_limit = adaptive_pixel_scale;

	// Standing-still case only - see the note above. Compared against what the frontier ON SCREEN was actually built
	// with, so a natural traversal that already picked the value up costs nothing here.
	if(adaptive_last_force_time_s >= 0.0 && now_s - adaptive_last_force_time_s < 2.0)
		return;
	for(size_t i=0; i<clouds.size(); ++i)
		if(clouds[i]->cached_ufrontier.nonNull())
		{
			const float applied = clouds[i]->cached_ufrontier->pixel_scale_limit;
			if(applied > 0.f && (adaptive_pixel_scale / applied > 1.15f || applied / adaptive_pixel_scale > 1.15f))
			{
				forceTraversalRefresh();
				adaptive_last_force_time_s = now_s;
				break;
			}
		}
}


void GaussianSplatRenderer::think()
{
	updateAdaptivePixelScale(); // SESSION079 - must precede kickOffTraversals() below so a kick this frame uses the new value.
	// Applying a completed sort or traversal is the only GL call in either background pipeline, which is why both happen
	// here on the main thread rather than in the worker tasks.
	drainSortResults();
	drainSaturationBuildResults(); // SESSION081: independent of the traversal pipeline - see kickOffSaturationBuilds().
	drainTraversalResults();
	drainFilterResults(); // SESSION063: apply completed per-orientation filters (split architecture).

	if(clouds.empty())
		return;

	const Vec2i viewport_dims = opengl_engine->getViewportDims();
	const OpenGLScene* const scene = opengl_engine->getCurrentScene();

	// SESSION067 - the splat vertex shader works entirely in accumulation-buffer pixels: it sizes the quad in them, maps
	// it back to clip space by dividing by viewport_dims_px, and indexes the saturation mask through them.  So when the
	// accumulation buffer is smaller than the frame, these two uniforms have to describe *it*, not the window - the
	// shader is self-consistent either way, and every pixel-space quantity in it then scales together.
	//
	// This is the only place the scale touches the projection, and deliberately so.  The LoD traversal keeps its own
	// full-viewport focal length (see kickOffTraversals()'s focalPxForScene() call), so the selection is unchanged and
	// only the buffer the selected splats land in gets smaller - see getAccumBufferScale() for the measurement that says
	// letting the selection coarsen too would cost detail and buy nothing.
	const Vec2i accum_dims = accumBufferDimsForViewport(viewport_dims);

	// Focal length in pixels, derived the same way as the engine's own screen-space projections:
	// focal_px = viewport_px * (lens_sensor_dist / sensor_size).
	const float focal_x = (float)accum_dims.x * scene->lens_sensor_dist / scene->use_sensor_width;
	const float focal_y = (float)accum_dims.y * scene->lens_sensor_dist / scene->use_sensor_height;

	// SESSION069 - Pick this frame's TAA jitter offset (or reset the accumulation if anything invalidated it) BEFORE
	// the per-cloud loop below writes user_uniform_vals[27], so all clouds share one consistent offset for the frame.
	updateTAAState(accum_dims);

	// SESSION069 fix - the vertex shader's per-splat anti-alias low-pass (+0.3, historically hardcoded) is sized in
	// ACCUM-BUFFER texels, which band-limits every single frame's render to the accum grid's own Nyquist limit -
	// exactly the sub-texel structure TAA's jittered accumulation is supposed to recover.  A single frame can never
	// contain more than the low-pass allows, no matter how many differently-phased frames are later averaged.  While
	// TAA is running, shrink the low-pass to what a FULL-FRAME render would use (0.3 full-res-pixels², expressed in
	// accum-pixels² by the scale-squared factor below) - the temporal average supplies the anti-aliasing instead, the
	// same trade every jittered-supersampling scheme makes.  Floored well above 0 so the covariance never gets close
	// enough to singular to matter (see the "shouldn't normally happen" comment at the 2D covariance's own inversion).
	{
		const bool taa_active_now = isTAAActiveForResolve();
		const float accum_scale = (viewport_dims.x > 0) ? ((float)accum_dims.x / (float)viewport_dims.x) : 1.f;
		splat_low_pass_variance_current = taa_active_now ? myMax(0.3f * accum_scale * accum_scale, 0.02f) : 0.3f;
	}

	for(size_t i=0; i<clouds.size(); ++i)
	{
		OpenGLMaterial& mat = clouds[i]->ob->materials[0];
		mat.user_uniform_vals[0].vec2 = Vec2f((float)accum_dims.x, (float)accum_dims.y);
		mat.user_uniform_vals[1].vec2 = Vec2f(focal_x, focal_y);
		// user_uniform_vals[2] (splat_tex_width) is constant, and was set in allocCloud().
		mat.user_uniform_vals[3].vec2 = Vec2f(splat_size_clamp_min, splat_size_clamp_max);
		mat.user_uniform_vals[4].intval = splat_size_clamp_invert ? 1 : 0;
		mat.user_uniform_vals[5].floatval = splat_alpha_cutoff;
		mat.user_uniform_vals[6].intval = splat_show_overdraw_mode;
		// 7 and 8 (splat_saturation_mask_block / _max_level) belong to the draw path, which decides per frame whether the
		// gate actually runs - see setSplatMaskBlockSize().
		// SESSION072: no shader-side enabled flag needed - when the checkbox is off, upload a wide-open range instead
		// (matches the "min 0 / max 1000 keeps everything" no-op the shader already relied on before the checkbox existed).
		mat.user_uniform_vals[9].vec2 = splat_dist_clamp_enabled ? Vec2f(splat_dist_clamp_min, splat_dist_clamp_max) : Vec2f(0.f, std::numeric_limits<float>::max());
		mat.user_uniform_vals[10].intval = (splat_dist_clamp_enabled && splat_dist_clamp_invert) ? 1 : 0;
		// 11 (splat_mask_centre_test) also belongs to the draw path - see setSplatMaskBlockSize().
		mat.user_uniform_vals[12].vec2 = Vec2f(splat_alpha_gain, splat_alpha_gamma); // See getAlphaGain().
		// 13 (splat_frag_mask_block) belongs to the draw path - see setSplatFragMaskBlockSize().
		mat.user_uniform_vals[14].intval = splat_ablation_stage; // DIAGNOSTIC ONLY - see getAblationStage().
		mat.user_uniform_vals[15].floatval = splat_quad_radius_scale; // DIAGNOSTIC ONLY - see getQuadRadiusScale().
		mat.user_uniform_vals[16].floatval = splat_area_scale_gamma; // DIAGNOSTIC ONLY - see getAreaScaleGamma().
		mat.user_uniform_vals[17].floatval = splat_area_scale_ref_px; // DIAGNOSTIC ONLY - see getAreaScaleRefPx().
		mat.user_uniform_vals[18].floatval = splat_coverage_shrink_strength; // DIAGNOSTIC ONLY - see getCoverageShrinkStrength().
		mat.user_uniform_vals[19].intval = splat_ewa_fix_enabled ? 1 : 0; // See getEWAProjectionFixEnabled().
		mat.user_uniform_vals[20].floatval = splat_near_fade_width; // See getNearFadeWidth().
		mat.user_uniform_vals[21].intval = splat_coverage_shrink_mode; // DIAGNOSTIC ONLY - see getCoverageShrinkMode().
		mat.user_uniform_vals[22].floatval = splat_near_epsilon; // See getNearEpsilon().

		// SESSION067 DIAGNOSTIC - the projected-area slice, see getAreaSliceMode().  The threshold is typed in frame
		// pixels (that is the unit getFrustumStructureReport()'s "Fill by splat size" histogram is in, which is where
		// the number comes from), but the shader measures area in accumulation-buffer pixels, since think() scaled its
		// focal length to that buffer just above.  Area goes as the square of a linear scale, so the conversion is the
		// ratio of the two pixel counts - and with it the same setting selects the same splats at any buffer scale,
		// which is the whole point of being able to A/B the two halves at different resolutions.
		const double frame_px = (double)viewport_dims.x * (double)viewport_dims.y;
		const double buffer_px = (double)accum_dims.x * (double)accum_dims.y;
		mat.user_uniform_vals[25].intval = splat_area_slice_mode;
		mat.user_uniform_vals[26].floatval = (frame_px > 0) ? (float)(splat_area_slice_px * (buffer_px / frame_px)) : splat_area_slice_px;
		// SESSION069 - subpixel jitter picked by updateTAAState() before this loop.  Zero when TAA is inert (off or
		// scale 1); the vertex shader's shift is then a no-op.
		mat.user_uniform_vals[27].vec2 = splat_taa_current_jitter_px;
		mat.user_uniform_vals[28].floatval = splat_low_pass_variance_current; // SESSION069 fix - see above.
		mat.user_uniform_vals[29].floatval = splat_point_size_px; // SESSION071 DIAGNOSTIC - see getPointSizePx().
	}

	buildVisibleSliceCDFs();

	// SESSION055: camera-motion tracker for anisotropic frustum-cull dilation - see the cull block in
	// GaussianSplatLodTraversalTask::run() and the members' comments in GaussianSplatRenderer.h. Filter rule is
	// max(instant, ema): rises the moment motion starts (so kickOffTraversals() picks up on the first-motion frame,
	// not three frames in) and decays exponentially on stop. alpha 0.15 = ~7-frame time constant at 60 Hz.
	{
		const Vec4f cur_cam_pos = scene->cam_to_world.getColumn(3);
		const Vec4f cur_cam_forward = normalise(scene->cam_to_world.getColumn(1)); // SESSION055: col 1 is forward in Substrata, col 2 is up - see the parallel note in kickOffTraversals(). Normalise so tiny non-unit drift can't clamp acos() to zero at small angles.
		const double dt_raw = prev_think_timer.elapsed();
		prev_think_timer.reset();
		cam_inst_angular_speed = 0.f; // SESSION064: default to "not rotating this frame" - overwritten below when a valid dt lets us measure it. A guarded-out frame (first post-load, tab-out) must not leave a stale non-zero value re-triggering the filter.
		if(have_prev_think_cam_state && dt_raw > 1.0e-4 && dt_raw < 0.5) // Guard against zero/huge dt (first frame post-load, breakpoint, tab-out).
		{
			const float dt = (float)dt_raw;
			const Vec4f inst_vel = (cur_cam_pos - prev_think_cam_pos_ws) * (1.f / dt); // Vec4f arithmetic gives .w = 0 (positions differ, w cancels), correct for a velocity vector.
			const float dot_fw = myClamp(dot(cur_cam_forward, prev_think_cam_forward_ws), -1.f, 1.f);
			const float inst_ang = std::acos(dot_fw) / dt;
			cam_inst_angular_speed = inst_ang; // SESSION064: raw, un-smoothed - see the member's comment and kickOffFilters().
			const float alpha = 0.15f;
			// SESSION072: captured before cam_angular_speed_ema is overwritten below - see the axis snap further down.
			const bool was_calm = cam_angular_speed_ema <= 1.0e-4f;
			// SESSION055: plain EMA for velocity vector - the per-axis abs-max used earlier kept the STALE direction on
			// reversal (e.g. long east motion followed by short west step held ema pointing east, so kickOff dilated
			// the east plane instead of west and left holes on the west edge). max(EMA, empirical) is applied per
			// plane at kick time instead, where it can compare directional shifts against a specific plane normal.
			// For angular speed the value is a scalar magnitude (non-negative), so a simple max(inst, blended) is
			// safe and useful - it holds bursts through the EMA decay tail.
			const Vec4f blended_vel = cam_velocity_ema_ws * (1.f - alpha) + inst_vel * alpha;
			const float blended_ang = cam_angular_speed_ema * (1.f - alpha) + inst_ang * alpha;
			cam_velocity_ema_ws = blended_vel;
			cam_velocity_ema_ws.x[3] = 0.f;
			cam_angular_speed_ema = myMax(inst_ang, blended_ang);
			// SESSION055: slow-decay peak (attack instant, half-life ~35 frames @ 60Hz ~ 580ms). After a mouse flick,
			// dilation stays elevated for ~1s of subsequent kicks - covers the case where the FLICK happens between
			// two kicks and the currently-in-flight traversal (kicked before the flick with low w) can't help, but
			// the NEXT kick, informed by peak, is over-dilated so a follow-up flick is already covered.
			cam_angular_speed_peak = myMax(inst_ang, cam_angular_speed_peak * 0.98f);

			// SESSION072: rotation axis for the split filter's anisotropic dilation - see cam_angular_axis_ema_ws's
			// comment and kickOffFilters(). cross(prev_fwd, cur_fwd)'s length is sin(angle) between the two unit
			// vectors, so it doubles as the reliability signal for the direction it also gives: near a near-zero
			// rotation the cross product is noise-dominated, and normalising it would inject a near-random axis into
			// the EMA - so skip the update entirely on those frames and hold the last good direction, the same way
			// the magnitude EMA holds its value while decaying rather than resetting to a fresh (here: meaningless)
			// sample.
			//
			// SESSION072: uses its own, much faster blend rate (alpha_axis) than the magnitude EMA's
			// alpha above. A typical mouse flick is over in 3-6 frames - far under alpha=0.15's ~7-frame (117ms) time
			// constant - so the axis was still mostly pointing at whatever it held before the flick for MOST of a short
			// flick's duration, systematically under-dilating the actually-threatened plane (magnitude was never the
			// problem: cam_angular_speed_ema/_peak already respond instantly via max(inst,blended)/attack-instant peak
			// above, so a correctly-SIZED dilation vector was being paired with a WRONGLY-DIRECTED one). alpha_axis
			// converges to ~94% of the true direction within 3 frames instead of ~24%.
			const float alpha_axis = 0.5f;
			const Vec4f cross_v = crossProduct(prev_think_cam_forward_ws, cur_cam_forward); // Rotating prev_fwd by inst_ang about this axis (right-hand rule) reaches cur_fwd.
			float cross_len;
			const Vec4f inst_axis = normalise(cross_v, cross_len); // NaN if cross_len==0 - fine, never read unless the guard below passes, which it can't from exactly 0.
			if(cross_len > 1.0e-5f)
			{
				// SESSION072: snap rather than blend when starting from a standstill (cam_angular_speed_ema was ~0 the
				// frame before this one) - the held axis could be from an arbitrarily long-ago rotation in a totally
				// different direction, and blending it in at all (even at alpha_axis) would dilute the one useful
				// sample available with irrelevant history. Once a rotation is already under way, blending is right -
				// it's what rejects frame-to-frame axis jitter from mouse-input noise during a steady turn.
				const Vec4f blended_axis = was_calm ? inst_axis : (cam_angular_axis_ema_ws * (1.f - alpha_axis) + inst_axis * alpha_axis);
				float blended_len;
				const Vec4f blended_axis_n = normalise(blended_axis, blended_len);
				cam_angular_axis_ema_ws = (blended_len > 1.0e-5f) ? blended_axis_n : inst_axis; // Guards the blend cancelling near-exactly (axis flipped between samples).
			}
		}
		prev_think_cam_pos_ws = cur_cam_pos;
		prev_think_cam_forward_ws = cur_cam_forward;
		have_prev_think_cam_state = true;
	}

	// SESSION088 DIAGNOSTIC - [gsr-sat-probe]. See getSatProbeLog() for what question this exists to answer. Entirely
	// read-only: it samples the barrier the bias is already reading and prints, touching no cached state, so unlike the
	// overlay toggles it cannot perturb what it measures.
	//
	// Off the FRAME rather than off a build, deliberately - a build only happens once the camera leaves the R-ball, so
	// every build-tied trace goes silent exactly when the camera is held still on the geometry under investigation.
	if(sat_probe_log)
	{
		const double probe_now_s = diag_timer.elapsed();
		if(probe_now_s - sat_probe_last_print_s > 0.5) // Twice a second: enough to read while holding a view, cheap enough that the sort inside gsSatGridStats() never shows.
		{
			sat_probe_last_print_s = probe_now_s;

			for(size_t i=0; i<clouds.size(); ++i)
			{
				const SplatCloud& cloud = *clouds[i];
				if(cloud.cached_sat_barrier.isNull() || cloud.cached_sat_barrier->sat_grid_res == 0)
					continue;
				const GaussianSplatSaturationBarrier& b = *cloud.cached_sat_barrier;
				const int res = b.sat_grid_res;

				GsSatGridStats gs;
				gsSatGridStats(b.sat_depth, res, gs);

				const Vec4f probe_cam_pos = scene->cam_to_world.getColumn(3);
				const Vec4f probe_fwd = normalise(scene->cam_to_world.getColumn(1)); // Column 1 is forward - see the tracker block above.

				// The direction is taken from the BARRIER'S ANCHOR, not the camera, because that is the origin sat_depth is
				// measured from and the one satBiasFor() encodes against. At R = 1 the two are within a metre, but the probe
				// must not quietly answer a different question than the bias asks.
				const Vec4f anchor_to_cam = probe_cam_pos - b.anchor_pos_ws;
				const int centre_tile = gsSatGridTileForDir(probe_fwd, res);
				const float centre_depth = b.sat_depth[centre_tile];

				// The 3x3 neighbourhood of the aimed-at tile, in grid order. This IS the artifact, printed: if the field is
				// smooth these nine numbers are close, and if it is not, the spread here is the LoD step the owner is looking at.
				const int cu = centre_tile % res, cv = centre_tile / res;
				std::string nbr_str;
				for(int dv=-1; dv<=1; ++dv)
					for(int du=-1; du<=1; ++du)
					{
						const int u = myClamp(cu + du, 0, res - 1), v = myClamp(cv + dv, 0, res - 1);
						const float d = b.sat_depth[(size_t)v * (size_t)res + (size_t)u];
						nbr_str += (std::isfinite(d) ? doubleToStringNDecimalPlaces(d, 1) : std::string("inf"));
						nbr_str += (du == 1) ? ((dv == 1) ? "" : " | ") : " ";
					}

				// What a node sitting at each of these distances STRAIGHT AHEAD would actually be multiplied by. This is the
				// whole bias arithmetic reproduced - gsSatBuriedRatioSq() then satBiasFor()'s curve - so the number printed is
				// the number the walk would use, not an approximation of it.
				std::string bias_str;
				static const float probe_dists_m[4] = { 50.f, 100.f, 200.f, 400.f };
				for(int k=0; k<4; ++k)
				{
					const Vec4f node_offset = anchor_to_cam + probe_fwd * probe_dists_m[k]; // Node at that distance ahead of the CAMERA, expressed from the anchor.
					const float dist_sq = node_offset[0]*node_offset[0] + node_offset[1]*node_offset[1] + node_offset[2]*node_offset[2]; // Same form satBiasFor() uses, so the probe's arithmetic matches the walk's bit for bit.
					const float ratio_sq = gsSatBuriedRatioSq(node_offset, dist_sq, b.sat_depth, res);
					// SESSION088: the same denominator satBiasFor() forms, exponent included - the probe must not keep
					// reporting session085's curve once the walk is running a different one.
					const float denom = (ratio_sq <= 1.f) ? 1.f : ((sat_bias_exponent == 1.f) ? ratio_sq : std::pow(ratio_sq, sat_bias_exponent));
					const float bias = (ratio_sq <= 1.f) ? 1.f : (1.f + (sat_bias_ceiling - 1.f) * (1.f - 1.f / denom));
					bias_str += doubleToStringNDecimalPlaces(probe_dists_m[k], 0) + "m=x" + doubleToStringNDecimalPlaces(bias, 2) + " ";
				}

				conPrint("[gsr-sat-probe] t" + doubleToStringNDecimalPlaces(probe_now_s * 1000.0, 0) + "ms " +
					"res=" + toString(res) +
					" R=" + doubleToStringNDecimalPlaces(b.region_radius_used, 3) +
					" ceil=" + doubleToStringNDecimalPlaces(sat_bias_ceiling, 1) +
					" exp=" + doubleToStringNDecimalPlaces(sat_bias_exponent, 2) + // SESSION088 - see getSatBiasExponent(). The bias= readouts below are computed with it.
					" anchor_d=" + doubleToStringNDecimalPlaces(probe_cam_pos.getDist(b.anchor_pos_ws), 2) + "m" +
					" | sat=" + doubleToStringNDecimalPlaces(gs.num_tiles > 0 ? (100.0 * (double)gs.num_finite / (double)gs.num_tiles) : 0.0, 1) + "%" +
					" depth p10=" + doubleToStringNDecimalPlaces(gs.d_p10, 1) +
					" med=" + doubleToStringNDecimalPlaces(gs.d_med, 1) +
					" p90=" + doubleToStringNDecimalPlaces(gs.d_p90, 1) +
					" max=" + doubleToStringNDecimalPlaces(gs.d_max, 1) +
					// The contrast block - the field's shape. See GsSatGridStats for how to read adj_* and flat=.
					" | adj p50=" + doubleToStringNDecimalPlaces(gs.adj_p50, 2) +
					" p90=" + doubleToStringNDecimalPlaces(gs.adj_p90, 2) +
					" max=" + doubleToStringNDecimalPlaces(gs.adj_max, 1) +
					" >2x=" + doubleToStringNDecimalPlaces(gs.num_both_finite > 0 ? (100.0 * (double)gs.num_adj_gt2 / (double)gs.num_both_finite) : 0.0, 1) + "%" +
					" >4x=" + doubleToStringNDecimalPlaces(gs.num_both_finite > 0 ? (100.0 * (double)gs.num_adj_gt4 / (double)gs.num_both_finite) : 0.0, 1) + "%" +
					" flat=" + doubleToStringNDecimalPlaces(100.0 * gs.max_window_frac, 1) + "%" +
					" edge=" + doubleToStringNDecimalPlaces(gs.num_pairs > 0 ? (100.0 * (double)gs.num_edge_pairs / (double)gs.num_pairs) : 0.0, 1) + "%" +
					// What the camera is actually pointed at.
					" | fwd tile=" + toString(centre_tile) +
					" d=" + (std::isfinite(centre_depth) ? doubleToStringNDecimalPlaces(centre_depth, 1) : std::string("inf")) +
					" bias " + bias_str +
					"| 3x3 " + nbr_str);
			}
		}
	}

	kickOffSorts();
	kickOffSaturationBuilds(); // SESSION081: independent cadence - see the function's own comment. Before apply so a build that just landed this frame can be picked up immediately.
	kickOffTraversals();
	kickOffFilters(); // SESSION063: derive S(P,R) from cached U(P) on rotation / after a fresh U(P) (split architecture).
}


/*
Rebuilds each cloud's cumulative count of in-frustum samples, which is what visibleFractionToDrawIndex() inverts to place
the draw slice boundaries - see getVisibleSlicingEnabled().

Here, per frame, rather than in the background traversal, because it is the one thing about the draw order that depends
on where the camera is *looking*: the traversal and the sort are both by distance and so survive a rotation untouched
(see GaussianSplatLodTraversalTask's own comment on why that is deliberate), while what is in frustum changes with every
turn of the head.  Affordable at that rate only because the sample is a fixed budget - max_slice_samples point-in-frustum
tests per cloud, whatever the cloud's size.

The test is on a splat's centre, so a splat whose centre is just off screen still rasterises and is counted as invisible
here.  That is a statistical estimate being used to place a boundary, not a culling decision: slicing tiles the whole
draw order whatever the boundaries are, so an error here can move where a saturation check happens and nothing else.
*/
void GaussianSplatRenderer::buildVisibleSliceCDFs()
{
	if(!splat_visible_slicing)
	{
		for(size_t i=0; i<clouds.size(); ++i)
			clouds[i]->slice_visible_cdf.clear(); // So a frame with it off cannot leave a CDF behind for the draw path to pick up.
		return;
	}

	const OpenGLScene* const scene = opengl_engine->getCurrentScene();
	const Planef* const frustum_clip_planes = scene->frustum_clip_planes;
	const int num_frustum_clip_planes = scene->num_frustum_clip_planes;

	for(size_t i=0; i<clouds.size(); ++i)
	{
		SplatCloud& cloud = *clouds[i];
		const js::Vector<Vec3f, 16>& samples = cloud.slice_sample_positions;

		cloud.slice_visible_cdf.resizeNoCopy(samples.size());

		int running = 0;
		for(size_t j=0; j<samples.size(); ++j)
		{
			const Vec3f& p = samples[j];
			if(pointInFrustum(frustum_clip_planes, num_frustum_clip_planes, Vec4f(p.x, p.y, p.z, 1.f)))
				running++;
			cloud.slice_visible_cdf[j] = running;
		}
	}
}


int GaussianSplatRenderer::visibleFractionToDrawIndex(const GLObject* ob, double fraction, int draw_count) const
{
	if(!splat_visible_slicing)
		return -1;

	// The ends are answered without consulting the samples at all, so that the slices still tile the whole draw order
	// exactly: everything is drawn either way, and only the boundaries between the slices move.
	if(fraction <= 0.0)
		return 0;
	if(fraction >= 1.0)
		return draw_count;

	const SplatCloud* cloud = NULL;
	for(size_t i=0; i<clouds.size(); ++i)
		if(clouds[i]->ob.ptr() == ob)
		{
			cloud = clouds[i].ptr();
			break;
		}

	if(cloud == NULL || cloud->slice_visible_cdf.empty())
		return -1;

	// The sample describes a draw order of a different length than the one being drawn - a traversal result landed
	// between the sample being taken and this frame.  Nothing here is wrong enough to be worth using, and the caller's
	// fallback is the plain index split, so say so rather than guess.
	if(cloud->slice_sample_draw_count != draw_count)
		return -1;

	const int num_samples = (int)cloud->slice_visible_cdf.size();
	const int total_visible = cloud->slice_visible_cdf[num_samples - 1];
	if(total_visible <= 0)
		return -1; // Nothing of this cloud is in view, so there is no visible order to place boundaries along.

	// Smallest sample at which this fraction of everything visible has been passed.  Binary search rather than a walk:
	// the CDF is non-decreasing by construction, and this is called twice per slice per cloud per frame.
	const int target = (int)(fraction * (double)total_visible + 0.5);
	int lo = 0, hi = num_samples - 1;
	while(lo < hi)
	{
		const int mid = lo + (hi - lo) / 2;
		if(cloud->slice_visible_cdf[mid] >= target)
			hi = mid;
		else
			lo = mid + 1;
	}

	return sliceSampleDrawIndex(lo, num_samples, draw_count);
}
