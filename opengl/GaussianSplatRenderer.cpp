/*=====================================================================
GaussianSplatRenderer.cpp
-------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatRenderer.h"


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
#include "../utils/BitUtils.h"
#include "../utils/ConPrint.h"
#include "../utils/Exception.h"
#include "../utils/RefCounted.h"
#include "../utils/StringUtils.h"
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

	// Key this frontier was built for; drainTraversalResults() checks these before reusing it, so a settings change that
	// alters the selection can't be answered from a stale U(P).  Orientation is deliberately NOT here - that's the point.
	uint64 topology_generation;
	Vec4f anchor_pos_ws;
	float pixel_scale_limit;
	size_t max_splats_budget;
	float max_layer_density;
	int max_tree_depth;
	float focal_px;
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
		last_traversal_kicked_topology_generation(0), last_traversal_hit_budget_cap(false), last_traversal_hit_density_cap(false), last_traversal_hit_depth_cap(false), last_traversal_dilation_elevated(false),
		cached_traversal_geom_generation(0), importance_num_views(0),
		filter_in_flight(false), ufrontier_needs_filter(false), have_last_filter_cam_forward(false), last_filter_cam_forward_ws(0.f), // SESSION063
		filter_dilation_elevated(false) // SESSION066
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

	// SESSION063: the cached unculled frontier U(P) for the split filter architecture, when split_filter_enabled. Set by
	// drainTraversalResults() from a cull-off traversal; a rotation re-filters this instead of re-traversing. Null until
	// the first split-mode traversal for this cloud lands. See GaussianSplatUnculledFrontier.
	Reference<GaussianSplatUnculledFrontier> cached_ufrontier;
	bool filter_in_flight;                 // True from a filter kick until its result is applied/dropped - see kickOffFilters().
	bool ufrontier_needs_filter;           // Set when a fresh U(P) lands, so kickOffFilters() produces the first S(P,R) for it even with no rotation.
	bool have_last_filter_cam_forward;
	Vec4f last_filter_cam_forward_ws;      // Camera forward at the last filter kick, so a rotation past threshold re-filters - see kickOffFilters().
	bool filter_dilation_elevated;         // SESSION066: true if the last filter kick for this cloud used an above-floor dilation band, so kickOffFilters() knows to fire one tight "settle" re-filter once the slow-decay peak falls back to the floor - the filter analogue of last_traversal_dilation_elevated. Without it a wide band from an in-motion kick would stay applied after the camera stops (the observed stall).

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
};


namespace
{


const size_t texels_per_splat = 4; // See the texel layout in gaussian_splat_vert_shader.glsl.
const size_t splat_tex_width = 4096; // Gives ~16.7M splat capacity where GL_MAX_TEXTURE_SIZE >= 16384, which is common.
const int splat_index_attribute_loc = 1; // Forced in buildShadersIfNeeded().  Slot 1 is otherwise "normal_in", which splats have no use for.

const int coarse_key_bits = 16; // How many high bits of the sort key the coarse stage buckets on, i.e. 65536 evenly spaced depth slices.

const int max_concurrent_traversals = 2; // Mirrors max_concurrent_sorts above, for the same reason: caps traversal scratch memory at roughly this many clouds' worth rather than letting it scale with the world.


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
	Reference<GaussianSplatUnculledFrontier> unculled_frontier;
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
	FrontierStop_OutOfFrustum, // SESSION055: the node's centre is outside the frustum (dilated by 1.5*feature_size to keep large nodes whose centre is just past a plane), so it and its subtree were skipped. Only produced when frustum-cull is on (see GaussianSplatRenderer::setFrustumCullEnabled). getFrustumStructureReport() disables cull, so this bucket stays 0 there - it exists so the runtime path can bucket cheaply and so the count matches what the fast path actually did.
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
class GaussianSplatLodTraversalTask : public glare::Task
{
public:
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
		bool coarse_floor_enabled_ = false, float coarse_pixel_scale_ = 20.f) // SESSION063 K4: also capture a coarse floor into U(P) - see GaussianSplatUnculledFrontier::is_coarse.
	:	cloud_id(cloud_id_), topology_generation(topology_generation_), scratch(scratch_), cam_pos_ws(cam_pos_ws_),
		pixel_scale_limit(pixel_scale_limit_), max_splats_budget(max_splats_budget_), max_layer_density(max_layer_density_), max_tree_depth(max_tree_depth_), focal_px(focal_px_),
		num_frustum_clip_planes(num_frustum_clip_planes_), frustum_cull_enabled(frustum_cull_enabled_),
		rotation_dilation_rate(rotation_dilation_rate_),
		result_queue(result_queue_),
		frontier_record(frontier_record_),
		build_unculled_frontier(build_unculled_frontier_),
		coarse_floor_enabled(coarse_floor_enabled_), coarse_pixel_scale(coarse_pixel_scale_)
	{
		if(num_frustum_clip_planes < 0)
			num_frustum_clip_planes = 0;
		if(num_frustum_clip_planes > (int)staticArrayNumElems(frustum_clip_planes))
			num_frustum_clip_planes = (int)staticArrayNumElems(frustum_clip_planes); // Bounds guard against a scene ever growing past 6 planes; the cull just misses planes past the 6th, cannot false-cull.
		for(int i=0; i<num_frustum_clip_planes; ++i)
			frustum_clip_planes[i] = frustum_clip_planes_[i];
		for(int i=0; i<(int)staticArrayNumElems(translation_dilation); ++i)
			translation_dilation[i] = translation_dilation_ ? translation_dilation_[i] : 0.f;
	}

	virtual void run(size_t /*thread_index*/) override
	{
		const js::Vector<Vec3f, 16>& positions = scratch->geom->positions; // SESSION058: shared cached snapshot, never the live cloud arrays - see GaussianSplatCachedGeom.
		const js::Vector<float, 16>& feature_sizes = scratch->geom->feature_size; // SESSION054: replaces per-push Vec3f scales[] lookup + 3-way max in makeHeapItem.
		const js::Vector<float, 16>& cull_radii = scratch->geom->cull_radius; // SESSION059: enclosing-sphere bound for the frustum-cull margin below - NOT the same quantity as feature_size, see GaussianSplatLodNode::bounding_radius_os's comment.

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

		// SESSION054: build (dist_sq, idx) pairs directly during traversal, so the sort phase reuses the distance already
		// computed for pixel_scale rather than re-scanning positions[] a second time. Squared distance is a monotone key,
		// preserves ascending sort order, and skips 3-8M sqrt calls that the previous getDist()-per-item scan cost.
		struct DistIdx { float dist_sq; uint32 idx; };
		struct DistIdxLess { inline bool operator () (const DistIdx& a, const DistIdx& b) const { return a.dist_sq < b.dist_sq; } }; // Nearest first (matches GaussianSplatSortResultMsg's convention for the front-to-back "under" blend). Used only by the small-N std::sort fallback inside floatKeyAscendingSort.
		struct DistIdxKey  { inline float operator () (const DistIdx& x) const { return x.dist_sq; } }; // Sort::floatKeyAscendingSort keys on this float; squared distance is non-negative so FloatFlip's positive-branch monotone mapping applies.

		// SESSION063 K4: fine frontier nodes (is_coarse bit 0) and coarse-floor nodes (bit 31 of idx set) go into the SAME
		// list and are sorted together by distance, so the draw is globally front-to-back across both layers - a near coarse
		// node correctly occludes a far fine one. The bit is packed into the top of idx (cloud indices are well under 2^31)
		// so the radix sort, which keys only on dist_sq, carries it for free; it's unpacked when the output is built.
		js::Vector<DistIdx, 16> decorated;
		js::Vector<float, 16> coarse_flags; // SESSION063 K4: parallel to output after the sort - 1 per coarse-floor node, 0 per fine node.

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
		while(!stack.empty())
		{
			const HeapItem top = stack.back();
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
			if(frustum_cull_enabled)
			{
				const Vec3f& p = positions[cloud_idx_u32];
				const float base_margin = cull_radii[cloud_idx_u32];
				const float dist_to_node = std::sqrt(top.dist_sq); // dist_sq is already computed in makeHeapItem (session054); one sqrt per pop.
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
					(decorated.size() + stack.size() + node.child_count > max_splats_budget);
				if(top.pixel_scale <= coarse_pixel_scale || terminal)
				{
					DistIdx cd; cd.dist_sq = top.dist_sq; cd.idx = cloud_idx_u32 | 0x80000000u; // Bit 31 marks a coarse-floor node.
					decorated.push_back(cd);
					captured_here = true;
				}
			}
			const bool child_coarse_captured = top.coarse_captured || captured_here;

			// Converged - already fine enough, no need to expand further.
			if(top.pixel_scale <= pixel_scale_limit)
			{
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32;
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_Converged);
				continue;
			}

			if(node.child_count == 0)
			{
				// A leaf can't be expanded regardless of pixel_scale - it's already the finest detail this tree has.
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32;
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_Leaf);
				continue;
			}

			// Density-capped: expanding this node would recurse into a region estimated to be this dense with overdraw
			// (see GaussianSplatLodNode::layer_density) - stop here and use this node's own merged approximation
			// instead, regardless of how coarse its pixel_scale still looks. max_layer_density <= 0 disables this check
			// (the "0 = unlimited" convention GaussianSplatSettingsWidget's other debug knobs use).
			if(max_layer_density > 0.f && node.layer_density > max_layer_density)
			{
				hit_density_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32;
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_DensityCap);
				continue;
			}

			// Depth-capped: a hard, global ceiling on how many levels traversal may unfold, independent of pixel_scale/
			// density/budget - "the tree will never unfold finer than this, anywhere". max_tree_depth <= 0 disables it.
			if(max_tree_depth > 0 && top.depth >= (uint32)max_tree_depth)
			{
				hit_depth_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32;
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_DepthCap);
				continue;
			}

			// Budget cap: expanding would push us over the ceiling. Keep this node's own merged representation instead.
			// Current node is already popped, so the projected total after pushing children is decorated.size() + stack.size() + child_count.
			if(decorated.size() + stack.size() + node.child_count > max_splats_budget)
			{
				hit_budget_cap = true;
				DistIdx d; d.dist_sq = top.dist_sq; d.idx = cloud_idx_u32;
				decorated.push_back(d);
				recordFrontierNode(cloud_idx_u32, top.member_idx, top.tree_local_idx, top.depth, FrontierStop_BudgetCap);
				continue;
			}

			for(uint32 c = node.child_start; c < (uint32)node.child_start + node.child_count; ++c)
			{
				HeapItem child = makeHeapItem(top.member_idx, c, m.offset, top.depth + 1, positions, feature_sizes);
				child.coarse_captured = child_coarse_captured; // SESSION063 K4: propagate the once-per-branch capture flag.
				stack.push_back(child);
			}
		}

		// Sort the selection front-to-back by camera distance. Each node's dist_sq was computed once in makeHeapItem (or
		// inline in the NoTree branch) and carried through, so this pass is pure sort - no positions[] lookup or sqrt.
		// SESSION054: uses glare-core's serial 11-bit-chunk radix sort (Sort::floatKeyAscendingSort), which was ~76% of
		// traversal time as std::sort on 3-8M elements. floatKeyAscendingSort falls back to std::sort itself for N<320.
		// Two-stage (coarse+precise) GaussianSplatSortTask isn't reused: that machinery exists to keep an unbounded
		// whole-cloud sort off the main thread's critical path via a fast approximate first pass; a traversal's output is
		// already budget-bounded, so one exact radix pass here is both simpler and fast enough - see kickOffSorts()'s
		// cloudHasLodTree() guard, which leaves an LoD-active cloud to this sort instead of the old one.
		// SESSION063 K4: one global front-to-back sort over fine + coarse nodes together (see the decorated declaration).
		js::Vector<DistIdx, 16> sort_scratch(decorated.size());
		Sort::floatKeyAscendingSort(decorated.data(), decorated.size(), DistIdxLess(), DistIdxKey(), sort_scratch.data(), /*put_result_in_working_space=*/false);

		output.resizeNoCopy(decorated.size());
		coarse_flags.resizeNoCopy(decorated.size());
		for(size_t i=0; i<decorated.size(); ++i)
		{
			const uint32 packed = decorated[i].idx;
			output[i] = packed & 0x7FFFFFFFu;                  // Real cloud index (bit 31 stripped).
			coarse_flags[i] = (packed >> 31) ? 1.f : 0.f;      // 1 = coarse-floor node, for the filter's per-node dilation.
		}

		scratch->hit_budget_cap = hit_budget_cap;
		scratch->hit_density_cap = hit_density_cap;
		scratch->hit_depth_cap = hit_depth_cap;

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
			if(build_unculled_frontier)
			{
				Reference<GaussianSplatUnculledFrontier> uf = new GaussianSplatUnculledFrontier();
				const size_t n = output.size();
				uf->indices.resizeNoCopy(n);
				uf->px.resizeNoCopy(n); uf->py.resizeNoCopy(n); uf->pz.resizeNoCopy(n); uf->radius.resizeNoCopy(n);
				uf->is_coarse.resizeNoCopy(n);
				for(size_t i=0; i<n; ++i)
				{
					const uint32 idx = output[i];
					const Vec3f& p = positions[idx];
					uf->indices[i] = idx;
					uf->px[i] = p.x; uf->py[i] = p.y; uf->pz[i] = p.z;
					uf->radius[i] = cull_radii[idx];
					uf->is_coarse[i] = coarse_flags[i];
				}
				uf->topology_generation = topology_generation;
				uf->anchor_pos_ws = cam_pos_ws;
				uf->pixel_scale_limit = pixel_scale_limit;
				uf->max_splats_budget = max_splats_budget;
				uf->max_layer_density = max_layer_density;
				uf->max_tree_depth = max_tree_depth;
				uf->focal_px = focal_px;
				msg->unculled_frontier = uf;
			}

			result_queue->enqueue(msg);
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
	};

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
		return item;
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
};


} // end anonymous namespace


GaussianSplatRenderer::GaussianSplatRenderer(OpenGLEngine& opengl_engine_)
:	opengl_engine(&opengl_engine_), next_handle(1), next_cloud_id(1), num_sorts_in_flight(0),
	num_traversals_in_flight(0), num_filters_in_flight(0), lod_pixel_scale_limit(1.0f), lod_max_splats_budget(10000000), lod_resort_move_threshold_ws(0.1f),
	lod_max_layer_density(0.0f), lod_max_tree_depth(0), lod_frustum_cull_enabled(true), split_filter_enabled(false),
	filter_dilation_latency(0.17f), filter_min_rot_rate_deg_per_s(45.f), filter_max_rot_rate_deg_per_s(40.f), filter_min_trans_rate_m_per_s(2.0f), // SESSION063 K3 (K4 defaults: coarse floor covers the edge, so the fine dilation can be tight/cheap); SESSION071 max: 40deg/s default, owner-confirmed no visible holes at the canonical test scene; SESSION072: 0.06->0.17 - measured kick-to-drain round trip is 136-166ms during a fast flick (owner's [gsr-filter-kick]/[gsr-filter-drain] log), not the ~13ms the filter task itself takes - the gap is real frame time (millions of survivors to issue/draw), not queueing, and existed under the old isotropic dilation too, just masked by its uniform over-padding on every plane.
	split_coarse_floor_enabled(true), split_coarse_pixel_scale(30.f), filter_coarse_dilation_latency(0.9f), coarse_layer_debug(false), // SESSION063 K4
	filter_debug_log(false), kick_debug_log(false), cpu_prof_log(false), // SESSION072: default off - see getFilterDebugLog()'s comment.
	splat_point_size_px(1.f),
	splat_merge_spread_widen(3.0f), // SESSION071: analytic minimum is sqrt(3) (see widenedMergedScale()); owner default set higher for extra margin.
	// SESSION071: GaussianSplatMergeColourParams defaults to Energy - owner-confirmed better at every pixel scale limit tested; Legacy is kept only as the A/B comparison.
	have_prev_think_cam_state(false), prev_think_cam_pos_ws(0.f), prev_think_cam_forward_ws(0.f), cam_velocity_ema_ws(0.f), cam_angular_speed_ema(0.f), cam_angular_speed_peak(0.f), cam_inst_angular_speed(0.f), cam_angular_axis_ema_ws(0.f), // SESSION072
	splat_size_clamp_min(0.0f), splat_size_clamp_max(0.0f), splat_size_clamp_invert(false),
	splat_dist_clamp_min(0.0f), splat_dist_clamp_max(1000.0f), splat_dist_clamp_invert(false), splat_alpha_cutoff(1.0f / 255.0f),
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
static size_t filterUnculledFrontier(const GaussianSplatUnculledFrontier& uf, const Planef* planes, int num_planes,
	const Vec4f& cam_pos_ws, float rate_fine_baseline, float rate_coarse_baseline,
	const Vec4f& rotation_axis, float swept_fine, float swept_coarse, // SESSION072: unit rotation axis + the angle swept during each layer's latency window - see above.
	const float* trans_dilation, bool coarse_only_debug,
	js::Vector<uint32, 16>& out_indices,
	size_t* out_num_coarse = NULL) // SESSION072 DIAGNOSTIC: if non-null, receives how many of the survivors were coarse-floor nodes - see [gsr-filter-drain].
{
	size_t num_coarse_out = 0;
	const size_t n = uf.indices.size();
	out_indices.resizeNoCopy(n); // Worst case every node survives.
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
	uint32* const out = out_indices.data();
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
		if(!coarse_only_debug && is_coarse && inside_tight) continue; // Band restriction: coarse only survives beyond the tight frustum.
		out[num_out++] = idx[i];
		if(is_coarse) ++num_coarse_out; // SESSION072 DIAGNOSTIC counting - see out_num_coarse.
	}
	out_indices.resize(num_out); // Shrink to survivor count (keeps prefix, no realloc) so .size() is authoritative.
	if(out_num_coarse)
		*out_num_coarse = num_coarse_out;
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
		const float* trans_dilation_, bool coarse_only_debug_, ThreadSafeQueue<Reference<ThreadMessage> >* result_queue_)
	:	cloud_id(cloud_id_), frontier(frontier_), num_planes(num_planes_), cam_pos_ws(cam_pos_ws_),
		rate_fine_baseline(rate_fine_baseline_), rate_coarse_baseline(rate_coarse_baseline_),
		rotation_axis(rotation_axis_), swept_fine(swept_fine_), swept_coarse(swept_coarse_),
		coarse_only_debug(coarse_only_debug_), result_queue(result_queue_)
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
			rotation_axis, swept_fine, swept_coarse, trans_dilation, coarse_only_debug, msg->survivors, &msg->num_coarse_survivors);
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

			const Vec4f pos_vs = scene->last_view_matrix * Vec4f(pos.x, pos.y, pos.z, 1.f);
			const float dist_to_cam = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length();
			const bool inside_dist_range = (dist_to_cam >= splat_dist_clamp_min) && (dist_to_cam <= splat_dist_clamp_max);
			if(splat_dist_clamp_invert ? inside_dist_range : !inside_dist_range)
				return false;

			return true;
		};

		result.total += cloud.total_splats; // Leaves + merged internal nodes - matches what the traversal iterates over.
		result.frontier += cloud.cached_ufrontier.isNull() ? 0 : cloud.cached_ufrontier->indices.size(); // SESSION072: U(P) as last cached by the traversal stage (split_filter_enabled) - 0 if none cached yet.
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

	const float vx = pos_vs[0], vy = pos_vs[1];
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
	"out of frustum" // SESSION055 - only produced by the fast path with cull enabled; getFrustumStructureReport() disables cull so this stays 0 there.
};


void GaussianSplatRenderer::fillTraversalScratch(SplatCloud& cloud, GaussianSplatLodTraversalScratch& scratch) const
{
	// SESSION058 diag: measures the cost below - see cpu_prof_log's declaration and the session058 snapshot. Before this
	// function's rewrite, this cost was a fixed ~23ms full-cloud memcpy on EVERY kick (session057 §3's mystery CPU load
	// when 0 splats were visible: an empty traversal returns near-instantly, so kicks - each paying this cost - queued
	// back to back with no gap). Now it's ~0 on a cache hit (the common case: camera moved/rotated, cloud unchanged) and
	// the same ~23ms memcpy only on a cache miss (the cloud's topology_generation changed since the cache was built).
	Timer prof_timer;
	bool cache_hit;

	// SESSION058: reuse the cached snapshot if the cloud hasn't structurally changed since it was built - see
	// GaussianSplatCachedGeom's comment for why topology_generation is a safe invalidation signal (every writer of
	// cloud.positions/cloud.feature_size bumps it in the same call). A hit is just a Reference<> copy: no allocation,
	// no memcpy, regardless of cloud size.
	if(!cloud.cached_traversal_geom.isNull() && cloud.cached_traversal_geom_generation == cloud.topology_generation)
	{
		cache_hit = true;
	}
	else
	{
		cache_hit = false;

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

		cloud.cached_traversal_geom = geom;
		cloud.cached_traversal_geom_generation = cloud.topology_generation;
	}
	scratch.geom = cloud.cached_traversal_geom;

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
		const size_t bytes_copied = cache_hit ? 0 : cloud.total_splats * (sizeof(Vec3f) + sizeof(float) + sizeof(float)); // positions + feature_size + cull_radius (SESSION059).
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

				const Vec4f pos_vs = scene->last_view_matrix * Vec4f(p.x, p.y, p.z, 1.f);
				const float dist_to_cam = Vec4f(pos_vs[0], pos_vs[1], pos_vs[2], 0.f).length(); // The shader takes length(pos_vs.xyz), so the w the transform leaves must not be in it.
				const bool inside_dist_range = (dist_to_cam >= splat_dist_clamp_min) && (dist_to_cam <= splat_dist_clamp_max);
				if(splat_dist_clamp_invert ? inside_dist_range : !inside_dist_range)
				{
					culled_by_dist_slice++;
					continue;
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
		((splat_dist_clamp_min <= 0.f && splat_dist_clamp_max >= 1000.f && !splat_dist_clamp_invert) ? " (keeps everything)" : (splat_dist_clamp_invert ? " INVERTED - this shell is hidden" : " - only this shell is shown")) + "\n";
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
			cloud->cached_ufrontier = msg->unculled_frontier;
			cloud->ufrontier_needs_filter = true; // Produce the first S(P,R) for this fresh U(P) even with no rotation.
			cloud->last_traversal_hit_budget_cap = msg->scratch->hit_budget_cap;
			cloud->last_traversal_hit_density_cap = msg->scratch->hit_density_cap;
			cloud->last_traversal_hit_depth_cap = msg->scratch->hit_depth_cap;
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
	const Vec4f ema_expected = cam_velocity_ema_ws * filter_dilation_latency;
	const float min_trans_dilation_m = filter_min_trans_rate_m_per_s * filter_dilation_latency;
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
	const float rate_fine_baseline   = min_rot_rate_rad_s * filter_dilation_latency;        // SESSION063 K4 tight window, SESSION072: baseline-only now.
	const float rate_coarse_baseline = min_rot_rate_rad_s * filter_coarse_dilation_latency;  // wider coarse window, baseline-only.
	// SESSION072: axis and swept magnitudes passed separately rather than as two pre-scaled vectors - the two swept
	// vectors are parallel (same axis, different latency), so the filter can do one cross product per plane instead of
	// two and hoist the fine/coarse blend out of its plane loop. See filterUnculledFrontier().
	const float swept_fine   = w_effective * filter_dilation_latency;
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
		num_filters_in_flight++;

		task_manager->addTask(new GaussianSplatFilterTask(best_cloud->cloud_id, best_cloud->cached_ufrontier,
			scene->frustum_clip_planes, scene->num_frustum_clip_planes, cam_pos_ws, rate_fine_baseline, rate_coarse_baseline,
			cam_angular_axis_ema_ws, swept_fine, swept_coarse, trans_dilation, coarse_layer_debug, &filter_result_queue));

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
		}
		else
		{
			num_filters_in_flight--; // Cloud gone; still account for the slot.
			continue;
		}

		// Drop if the U(P) this was filtered from is no longer the cloud's current one - a newer traversal has landed and
		// its own filter is or will be in flight. Applying the stale survivors would flash an older selection.
		if(msg->frontier.ptr() != cloud->cached_ufrontier.ptr())
			continue;

		// SESSION063 K4: survivors is the globally sorted draw list (fine + coarse interleaved by depth, or coarse-only in
		// the debug view - the filter already applied that). Upload as-is.
		const js::Vector<uint32, 16>& survivors = msg->survivors;
		cloud->instance_index_vbo->updateData(0, survivors.data(), survivors.size() * sizeof(uint32));
		cloud->ob->num_instances_to_draw = (int)survivors.size();
		noteDrawOrderForSlicing(*cloud, survivors.data(), survivors.size());

		if(filter_debug_log) // SESSION064 DIAG: does the applied survivor count keep changing after the camera stops? That churn is what the gate turns into boiling.
			conPrint("[gsr-filter-drain] t" + doubleToStringNDecimalPlaces(diag_timer.elapsed() * 1000.0, 0) + "ms surv=" + uInt64ToStringCommaSeparated(survivors.size()) +
				" coarse=" + uInt64ToStringCommaSeparated(msg->num_coarse_survivors) + // SESSION072 DIAGNOSTIC
				" fine=" + uInt64ToStringCommaSeparated(survivors.size() - msg->num_coarse_survivors) +
				" pool=" + uInt64ToStringCommaSeparated(msg->frontier->indices.size()) +
				" compute=" + doubleToStringNDecimalPlaces(msg->filter_compute_ms, 2) + "ms"); // SESSION072 DIAGNOSTIC: pure filterUnculledFrontier() cost, no scheduling.
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
		task_manager->addTask(new GaussianSplatLodTraversalTask(best_cloud->cloud_id, best_cloud->topology_generation, scratch, cam_pos_ws,
			lod_pixel_scale_limit, lod_max_splats_budget, lod_max_layer_density, lod_max_tree_depth, focal_px,
			scene->frustum_clip_planes, scene->num_frustum_clip_planes, cull_active,
			translation_dilation, rotation_dilation_rate,
			&traversal_result_queue,
			/*frontier_record=*/NULL, /*build_unculled_frontier=*/split_filter_enabled, // SESSION063: cull-off traversal builds U(P) for the split filter.
			/*coarse_floor_enabled=*/split_filter_enabled && split_coarse_floor_enabled, /*coarse_pixel_scale=*/split_coarse_pixel_scale)); // SESSION063 K4.
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


void GaussianSplatRenderer::think()
{
	// Applying a completed sort or traversal is the only GL call in either background pipeline, which is why both happen
	// here on the main thread rather than in the worker tasks.
	drainSortResults();
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
		mat.user_uniform_vals[9].vec2 = Vec2f(splat_dist_clamp_min, splat_dist_clamp_max);
		mat.user_uniform_vals[10].intval = splat_dist_clamp_invert ? 1 : 0;
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

	kickOffSorts();
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
