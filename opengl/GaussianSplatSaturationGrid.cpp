/*=====================================================================
GaussianSplatSaturationGrid.cpp
---------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatSaturationGrid.h"


#include "../maths/mathstypes.h"
#include "../maths/SSE.h" // SESSION081 PHASE 2 PREFETCH: _mm_prefetch. Routed through sse2neon on ARM by this header, so the hint is real on mobile too, not silently desktop-only.
#include "../utils/TaskManager.h" // SESSION079: gsBuildSaturationGridParallel().
#include "../utils/Reference.h"
#include "../utils/Timer.h" // SESSION081 PLAN ETAP 0: per-phase diagnostic timers - see gsSatApplyRegionErosion() and gsBuildSaturationGridParallel().
#include <cstring> // SESSION079: memmove() in gsBuildSaturationGridParallel().
#include <cmath>
#include <limits>
#include <vector>    // SESSION088 DIAGNOSTIC: gsSatGridStats()'s percentile scratch.
#include <algorithm> // SESSION088 DIAGNOSTIC: std::sort() in gsSatGridStats().


// SESSION074: fixed margin for the octahedral mapping's area distortion (not equal-area; the literature bounds the
// stretch under 2x, worst near the square's corners). Applied to the tile's nominal angular size wherever a footprint
// is compared against it, so every such comparison errs towards "this node covers fewer tiles than the ideal geometry
// says" - the safe direction for this stage, since under-covering only weakens the cull, never breaks the picture.
static const float gs_sat_grid_oct_margin = 1.5f;


// SESSION079 PERF: gsDirToOct()'s body with the L1 norm passed in rather than recomputed. gsSatGridFootprint() below
// needs that same L1 for the local tile angle, and used to compute it twice - once here, once in the tile-angle helper -
// on every one of the millions of nodes each of the two passes walks. Split so the two can share it; gsDirToOct() keeps
// its own one-argument form for callers outside this file.
static inline Vec2f gsDirToOctWithL1(const Vec4f& dir, float abs_sum)
{
	const float inv = abs_sum > 1.0e-12f ? (1.f / abs_sum) : 0.f;
	const float px = dir.x[0] * inv;
	const float py = dir.x[1] * inv;
	if(dir.x[2] <= 0.f)
	{
		const float sx = (px >= 0.f) ? 1.f : -1.f;
		const float sy = (py >= 0.f) ? 1.f : -1.f;
		return Vec2f((1.f - std::fabs(py)) * sx, (1.f - std::fabs(px)) * sy);
	}
	else
		return Vec2f(px, py);
}


Vec2f gsDirToOct(const Vec4f& dir)
{
	// SESSION074: identical formula to float32x3_to_oct() in opengl/shaders/frag_utils.glsl (Cigolle et al.) - project
	// the sphere onto the octahedron, then onto the xy plane, then reflect the lower hemisphere's folds over the
	// diagonals so the whole sphere maps into one [-1, 1]^2 square.
	//
	// Note this divides by the vector's own L1 norm, so it is invariant under positive scaling - callers pass raw
	// offsets, never normalised directions. See the header.
	return gsDirToOctWithL1(dir, std::fabs(dir.x[0]) + std::fabs(dir.x[1]) + std::fabs(dir.x[2]));
}


float gsSatGridTileAngle(int res)
{
	// SESSION074: res*res tiles cover the whole sphere (4*pi steradians). Treating a tile as approximately square in
	// angle, tile_ang^2 ~= 4*pi / res^2, hence tile_ang = sqrt(4*pi) / res. Inflated by the octahedral distortion
	// margin so every caller's footprint comparison stays on the conservative side - see that constant's comment.
	return (std::sqrt(4.f * 3.14159265f) / myMax((float)res, 1.f)) * gs_sat_grid_oct_margin;
}


int gsSatGridResForFocal(float focal_px, float tile_px)
{
	// SESSION088: one knob, in pixels. This used to be coarse_pixel_scale/(focal * tile_subdiv) - two knobs that never
	// appeared anywhere except as that ratio, so only their quotient was ever real. That quotient has units of PIXELS
	// (it is coarse_pixel_scale/tile_subdiv), i.e. "how many screen pixels one grid tile spans", which is what the pair
	// was always jointly expressing and is the same unit the rest of the LoD stage is stated in (pixel_scale_limit).
	// The owner's calibrated 30/0.3 is exactly tile_px = 100.
	//
	// Keeping it in PIXELS rather than in angle (or fixing res outright) is what keeps the grid adaptive: focal_px is
	// linear in viewport width, so a tile holds a constant SCREEN size and the resolution follows the display - a phone
	// builds a proportionally cheaper grid, a 4K display a finer one, both with the same visual coarseness. Stating it
	// as an angle or as res would invert that: the tile's pixel size would then grow with the display, giving the
	// largest screen the coarsest barrier while a phone paid desktop cost for accuracy it cannot show.
	const float safe_focal = myMax(focal_px, 1.f);
	const float tile_ang = myMax(tile_px, 1.0e-3f) / safe_focal;

	// Inverse of gsSatGridTileAngle(), including its distortion margin, so that the res chosen here and the tile size
	// reported there are consistent with each other rather than two independent approximations.
	const float res_raw = (std::sqrt(4.f * 3.14159265f) * gs_sat_grid_oct_margin) / tile_ang;

	int res = (int)std::ceil(res_raw);
	res = myMax(res, gsSatGridMinRes);
	const int max_res = (int)std::sqrt((float)gsSatGridMaxTiles); // SESSION074: hard ceiling - see gsSatGridMaxTiles's comment.
	res = myMin(res, max_res);
	return res;
}


float gsSatProjectedRadius(const Vec3f& scales, const Vec4f& rot, const Vec4f& unit_dir)
{
	// SESSION077: see the header for the derivation and the sanity checks. Guarded against a degenerate axis: a splat
	// with a zero (or denormal) extent would otherwise divide by zero below, and 3DGS captures do contain
	// effectively-planar splats - the owner's interior measured extents down to 1/63000 of the splat's longest axis.
	const float s0 = myMax(scales.x, 1.0e-8f);
	const float s1 = myMax(scales.y, 1.0e-8f);
	const float s2 = myMax(scales.z, 1.0e-8f);

	// Rotation matrix columns = the splat's own principal axes. Written out rather than built as a Matrix3f: only the
	// three dot products below are wanted, and this runs once per occluder over millions of them.
	const float x = rot[0], y = rot[1], z = rot[2], w = rot[3];
	const float xx = x*x, yy = y*y, zz = z*z;
	const float xy = x*y, xz = x*z, yz = y*z, wx = w*x, wy = w*y, wz = w*z;

	const float dx = unit_dir[0], dy = unit_dir[1], dz = unit_dir[2];

	const float d0 = dx * (1.f - 2.f*(yy + zz)) + dy * (2.f*(xy + wz))        + dz * (2.f*(xz - wy));
	const float d1 = dx * (2.f*(xy - wz))       + dy * (1.f - 2.f*(xx + zz))  + dz * (2.f*(yz + wx));
	const float d2 = dx * (2.f*(xz + wy))       + dy * (2.f*(yz - wx))        + dz * (1.f - 2.f*(xx + yy));

	const float q = (d0*d0) / (s0*s0) + (d1*d1) / (s1*s1) + (d2*d2) / (s2*s2); // d^T C^-1 d.

	return std::sqrt(s0 * s1 * s2 * std::sqrt(q));
}


int gsSatGridTileForDir(const Vec4f& dir, int res)
{
	const Vec2f oct = gsDirToOct(dir);
	int u = (int)((oct.x * 0.5f + 0.5f) * (float)res);
	int v = (int)((oct.y * 0.5f + 0.5f) * (float)res);
	u = myClamp(u, 0, res - 1);
	v = myClamp(v, 0, res - 1);
	return v * res + u;
}


// SESSION074: the tile span a node's angular footprint covers, as inclusive [u0, u1] x [v0, v1] grid coordinates.
//
// The node's footprint centre and radius, in grid-tile units - the common half of both sides' span logic.
//
// The footprint is a disc of angular radius ang_radius around the node's direction. Rather than projecting that disc
// through the octahedral mapping exactly (which is neither linear nor equal-area, and would need the fold topology
// handled), it is bounded in oct space by scaling the angular radius into tile units.
//
// SESSION076: both sides now use the TOUCH span. They used to differ - the write side demanded full coverage of a
// tile, on the reasoning that "this tile is behind opaque coverage" is a claim a partly-covering node has not
// established. That reasoning was sound only while the contribution was binary. Now a partly-covering node contributes
// a correspondingly small alpha instead of a full one, so the claim it makes is already proportionate to what it
// covers, and demanding full coverage would simply discard it. See the write loop for the model.
//
// SESSION078 FIX: the scale used to be the GLOBAL mean tile angle, sqrt(4*pi)/res, times a flat 1.5 fudge standing in
// for "the octahedral mapping's area distortion". That distortion is not a constant, and treating it as one is what put
// a camera-anchored clear patch on plainly opaque surfaces.
//
// The oct map is a radial projection sphere <-> octahedron. For a unit direction d write s = |dx| + |dy| + |dz| (its L1
// norm, which is 1/r_oct for the octahedron point it lands on). A surface element dA there subtends dA*cos(phi)/r_oct^2
// with cos(phi) = 1/(sqrt(3)*r_oct), and dA is itself sqrt(3) times its own footprint in the oct plane, so the two
// sqrt(3)s cancel and
//
//   d(solid angle) / d(oct area) = s^3
//
// exactly. s runs from 1 along the world axes to sqrt(3) along the body diagonals, so the solid angle behind a tile of
// fixed oct area varies by 3*sqrt(3) = 5.2x across the sphere. A tile is (2/res)^2 in oct area, hence a local angular
// size of (2/res) * s^1.5.
//
// The old constant, 1.5*sqrt(4*pi)/res = 5.32/res, overstated that by 2.7x along the axes and 1.17x along the
// diagonals. Overstating the tile angle understates sigma_tiles, and the write loop's deposit goes as sigma^2, so every
// occluder laid down between 7.1x (axes) and 1.36x (diagonals) too little occlusion. The shortfall is worst along
// exactly +-x, +-y, +-z - which is where a surface's nearest point sits when you stand square-on to it - so an opaque
// wall or ceiling came out clear in a patch around its perpendicular foot and painted away from it, the patch riding
// along with the anchor while staying welded to the geometry under pure rotation. Both are what the owner reported.
//
// It is also why sweeping the threshold never helped: the error is a 5x GRADIENT across the sphere, not a gain, so a
// threshold can only slide the contour along it, never flatten it.
//
// SESSION079 PERF: returns the RECIPROCAL of that angle, and takes l1 and inv_dist rather than deriving them. The value
// it used to return was only ever divided into an angular radius, and the two quantities it used to recompute -
// l1 = |dx|+|dy|+|dz| and 1/sqrt(l2_sq) - are both already in the hands of both call sites (the oct mapping needs the
// first, the angular radius needs the second). So s = l1 * inv_dist costs a multiply here instead of a sqrt and a
// divide, and returning the reciprocal folds the caller's divide into a multiply as well. Same formula, same result to
// within float reassociation; three divides and two sqrts per node become one of each.
static inline float gsSatGridInvLocalTileAngle(float l1, float inv_dist, int res)
{
	// s = L1/L2 of the offset, i.e. the L1 norm of the unit direction it points along. Scale-invariant, so callers pass
	// raw offsets here exactly as they already do to gsDirToOct(). The degenerate-length guard this used to carry lives
	// at both call sites instead, as a stricter dist_sq < 1e-12 test made before inv_dist is formed at all.
	const float s = l1 * inv_dist;
	return (myMax((float)res, 1.f) * 0.5f) / (s * std::sqrt(myMax(s, 1.f))); // 1 / ((2/res) * s^1.5).
}


// SESSION079 PERF: inv_dist = 1/sqrt(dist_sq) is passed in - see gsSatGridInvLocalTileAngle(). Callers must have
// rejected degenerate offsets (dist_sq < 1e-12) before calling.
static inline float gsSatGridFootprint(const Vec4f& offset, float ang_radius, int res, float inv_dist, float& cu, float& cv)
{
	const float l1 = std::fabs(offset.x[0]) + std::fabs(offset.x[1]) + std::fabs(offset.x[2]);
	const Vec2f oct = gsDirToOctWithL1(offset, l1);
	cu = (oct.x * 0.5f + 0.5f) * (float)res;
	cv = (oct.y * 0.5f + 0.5f) * (float)res;
	return ang_radius * gsSatGridInvLocalTileAngle(l1, inv_dist, res);
}


// Half the diagonal of a 1x1 tile, in tile units: the distance from a tile's centre to its furthest corner. A tile is
// fully inside a disc iff its centre is within (R - this) of the disc's centre, and touched by it iff within
// (R + this) - the two roundings the write and read sides respectively need.
static const float gs_sat_tile_half_diag = 0.70710678f;


// SESSION076: an occluder whose peak contribution to any tile is below this is skipped outright. It is a
// negligible-contribution cutoff, not a tuning knob: 0.2% of a tile's transmittance is an order of magnitude below the
// resolution of any saturation threshold the panel can express, so nothing that clears it can change a verdict. Its
// only job is to keep the millions of ~2px fine-scale nodes - whose integrated occlusion is genuinely nil - out of the
// write loop.
static const float gs_sat_min_occluder_amp = 0.002f;


// SESSION079: floor on how thin a strip the row-parallel FILTER passes (closing, erosion) will cut. Each of those reads
// a window around every row it writes, so a strip's halo is redundant work shared with its neighbours, and thinner
// strips mean proportionally more of it. 8 rows also keeps a strip's accumulators (8*res*4 bytes) inside L1 at any res
// this stage produces.
//
// SESSION081: the deposit no longer uses this - see gs_sat_deposit_strip_rows. It shared the constant while both passes
// had the same "every strip pays a fixed per-strip cost" shape; the deposit does not any more, and the filter passes
// were never measured at any other value, so they keep the 8 they were calibrated with.
static const int gs_sat_min_strip_rows = 8;


// SESSION081: absolute floor on how thin a strip the phase-2 DEPOSIT will cut - split from gs_sat_min_strip_rows above.
// This is a hard limit on strip THICKNESS; the strip COUNT is derived from the thread count, not from this - see
// gsSatDepositStripCount().
//
// The deposit's original reason for 8 was that "every strip walks the WHOLE occluder array - that is the price of not
// reordering anything - so thinner strips buy less tile work each while paying the same walk". That walk is gone: each
// strip now gets a list of exactly the records reaching its rows (see GsSatBinTask), so a thinner strip pays strictly
// less work, and the argument for a floor of 8 went with it. The L1 half of the argument points the other way - 4 rows
// is half the working set of 8 - so it was never what set the number.
//
// What thinning DOES cost is list duplication: a record straddling a strip boundary is deposited by both strips, and
// thinner strips mean more boundaries. At 8 rows / 12 strips the measured writers/num_recs was 1.110 - the average
// record already reaches 1.11 strips. That ratio is the thing to watch when moving this; it is printed as
// writers/num_recs in [gsr-sat-build-diag] and is a direct per-record multiplier on phase 2's real work.
//
// SWEPT, not assumed (owner's reference scene, R=0.1 sub=0.3, ~9 builds each, teleporting between viewpoints):
//
//   rows  strips  dep_ms+bin_ms (mean/median)  dep_par   writers/rec   tile_iters/rec
//     8     12          37.8                     6.40       1.110          4.31
//     4     25          34.3 / 32.4             10.64       1.243          4.34
//     2     50          32.5 / 32.8             12.28       1.468          4.08
//
// 25 strips wins. 50 buys no more time - mean -1.8ms, median +0.4ms, against a +-5.1ms standard error on the
// difference, i.e. nothing - while writers/rec climbs to 1.468, so nearly half of all records get deposited twice. The
// duplication cost grows linearly with the number of boundaries while the parallelism gain has flattened (10.64 ->
// 12.28 of 17), and the knee sits between these two rows.
//
// This value is only the FLOOR on thickness, not the operating point: 2 rows is where duplication starts to run away,
// so nothing thinner is ever cut. What actually sets the strip count is the thread count - see gsSatDepositStripCount().
//
// Read tile_iters/rec, not dep_busy, when judging a change here: dep_busy is each strip task's WALL time, so with more
// strips than threads it inflates with contention even when the work is identical - it rose 48% between the first two
// rows while the actual per-record tile work did not move at all. That mistake was made once already; the counters
// above are the honest ones.
static const int gs_sat_deposit_min_strip_rows = 2;


// SESSION081: how many strips phase 2 cuts, derived from the thread count rather than fixed.
//
// The sweep above found 25 strips optimal - but that was measured on the owner's 17-thread desktop, and a number tuned
// there is the wrong shape of answer for this project. The web client is the product, and its pool is not the desktop's:
// SDLClient.cpp clamps main_task_manager to 8 threads under EMSCRIPTEN (against 512 native), and the worker threads are
// preallocated by -sPTHREAD_POOL_SIZE=30, so a phone typically runs this stage with 4-8. Fixing the strip count at 25
// there would pay the full duplication cost - every strip boundary is a record deposited twice - to feed threads that
// do not exist.
//
// So the count targets ~1.5 strips per thread. The 0.5 is deliberate oversubscription: strips carry unequal work (the
// deposit is heaviest across the middle rows of the octahedral grid, lightest at the poles), and a spare half-strip per
// thread lets runTaskGroup()'s pool steal the tail instead of idling behind the heaviest one. Above that the extra
// parallelism flattens while duplication keeps climbing, which is exactly what the 50-strip row measured.
//
// It reproduces the measured optimum where it was measured - 17 threads -> 25 strips - and degrades correctly
// elsewhere: 8 threads -> 12, a phone's 4 -> 6, single-core -> 1 (which takes the no-binning path in GsSatStripTask).
//
// Two things still bound it: the grid cannot give strips thinner than the floor above, and the count cannot exceed 255
// because GsSatBinTask packs a record's first and last strip into one uint16, a byte each. 255 is unreachable from a
// realistic thread count, but it is enforced rather than assumed - silent list corruption is not an acceptable failure
// mode on hardware we do not own.
static inline int gsSatDepositStripCount(int res, int concurrency)
{
	const int strips_for_threads = myMax(1, (concurrency * 3) / 2);
	const int strips_for_res     = myMax(1, res / gs_sat_deposit_min_strip_rows);
	return myClamp(myMin(strips_for_res, strips_for_threads), 1, 255);
}


// SESSION079: memory ceiling on the parallel build's record scratch. The block size is derived from this rather than
// fixed in nodes, because both things this trades off are sizes in bytes, not counts.
//
// Without any ceiling the scratch is one record per occluder - ~71MB on the owner's reference scene, on top of the ~60MB
// of occluder arrays already live, which more than doubles this stage's peak footprint. The web client runs on phones,
// so that is not acceptable as an unbounded quantity.
//
// But blocking is not free either: each block is a synchronisation barrier where every strip waits for the slowest, and
// at 262144 nodes per block (12 blocks, 23 task-group launches) that measured 33.5 -> 39.6ms, an 18% loss. A byte budget
// gets both: on a large desktop scene it yields two or three blocks, where the barrier cost is negligible, and on a
// smaller scene - a phone's, where the LoD budget produces far fewer occluders in the first place - it yields one block,
// i.e. no blocking at all.
// SESSION081: the 32MB this carried since session079 was calibrated against a ~3M-occluder scene ("two or three
// blocks", above). The owner's reference scene is now ~13.2M occluders, which the same 32MB/24B math turns into
// 10 blocks / 20 task-group launches per build. Measured cost of that (owner, 4-teleport averages, R=1): forcing
// one block cut rec_ms 115.2->46.1ms and dep_ms 124.2->39.6ms. Both phases moved, not just the deposit, so the
// cost is less "N sync barriers" than N cold restarts of the scratch buffer plus N task-group launch pairs
// contending with traversal/apply/filter on the shared pool.
//
// Split by platform rather than picking one flat number, because the two ends are not really in tension: the
// scratch is sized to min(n, budget/24B) (see block_n below), so raising the ceiling costs nothing on any scene
// whose n already sits under it. Mobile scenes produce far fewer occluders (the same LoD-budget difference the
// comment above already relied on), so the desktop number moving does not affect them; and a scene large enough
// to exceed this budget would already be dominated by the occluder SoA gather that runs before this stage, not
// by this buffer. EMSCRIPTEN is the project's existing platform gate - see maths/SSE.h.
#if defined(EMSCRIPTEN)
static const size_t gs_sat_rec_scratch_bytes = 32 * 1024 * 1024; // Web/mobile: unchanged from session079.
#else
static const size_t gs_sat_rec_scratch_bytes = 512 * 1024 * 1024; // Desktop: one block for the reference scene (13.2M*24B ~= 317MB), with headroom.
#endif


// SESSION076: exp(-x) over x in [0, gs_sat_exp_lut_max), sampled at bin centres. The write loop below evaluates one
// Gaussian per tile touched - tens of millions per grid build - and expf() at ~10-20ns each would put that cost back on
// the same order as the tree walk we just moved it off. Nearest-bin lookup is ~1.5% relative error, which is far below
// this stage's own resolution (see gs_sat_min_occluder_amp) and biases nothing systematically.
//
// Built at static-init time by the constructor rather than lazily, so the worker threads that read it never race to
// initialise it.
static const int gs_sat_exp_lut_n = 256;
static const float gs_sat_exp_lut_max = 8.f; // exp(-8) = 3.4e-4: past here the contribution is below the cutoff above.

struct GsSatExpLut
{
	GsSatExpLut()
	{
		for(int i=0; i<gs_sat_exp_lut_n; ++i)
			v[i] = std::exp(-(((float)i + 0.5f) * (gs_sat_exp_lut_max / (float)gs_sat_exp_lut_n)));
	}
	float v[gs_sat_exp_lut_n];
};
static const GsSatExpLut gs_sat_exp_lut;

static inline float gsSatExpNeg(float x) // x >= 0; caller has already rejected x >= gs_sat_exp_lut_max.
{
	const int bin = (int)(x * ((float)gs_sat_exp_lut_n / gs_sat_exp_lut_max));
	return gs_sat_exp_lut.v[myClamp(bin, 0, gs_sat_exp_lut_n - 1)];
}


// SESSION079 PERF: floor/ceil without a library call.
//
// This project builds MSVC x64 with no /arch: flag (cmake/shared_cxx_settings.cmake defines -D__SSE4_1__ as a
// PREPROCESSOR symbol only, and -D__NO_AVX__ besides), so the codegen baseline is SSE2 and std::floor/std::ceil are CRT
// calls, not a single roundss. This loop makes 4 of them per writer and gsSatOccluded() makes 4 per node - 4.2M and
// 11.9M calls respectively on the owner's reference viewpoint - which the session079 cost decomposition placed in the
// two most expensive slices of both passes.
//
// Every one of these results is immediately clamped to [0, res-1], and that is what makes the replacement exact rather
// than approximate:
//
//  - floor vs truncation differ ONLY for negative arguments, and every negative argument clamps to 0 either way. So
//    myClamp((int)x, 0, res-1) == myClamp((int)std::floor(x), 0, res-1) for all x. No helper needed - just drop the call.
//  - ceil needs the real thing, but t + ((float)t < x) is exactly ceil for every finite x: for x > 0 it adds one iff x
//    has a fractional part, and for x <= 0 truncation already IS ceil.
//
// Out-of-range inputs behave as before: cvttss2si returns INT_MIN, which clamps the same way the old (int)std::floor()
// of the same value did.
static inline int gsSatCeilToInt(float x)
{
	const int t = (int)x;
	return t + (((float)t < x) ? 1 : 0);
}


// SESSION079: the accumulation over one horizontal STRIP of the grid's rows, [v_lo, v_hi).
//
// Splitting the pass by rows is what makes it both local and parallel, without approximating anything:
//
//  - Every tile belongs to exactly one strip, so two strips never touch the same accum/sat_depth entry. No locks, no
//    atomics, and no false sharing beyond the single cache line a strip boundary may straddle.
//  - Within a strip the occluders are still walked in the caller's front-to-back order, so each tile sees exactly the
//    sequence it saw before, and its barrier is still the first crossing's far edge. Results are BIT-IDENTICAL to the
//    single-strip case. This is deliberately NOT the plan's stage-7 log reformulation, which quantised depth into
//    buckets and gave up exactness to get the same parallelism.
//  - A strip's working set is (v_hi-v_lo)*res*4 bytes for each of accum and sat_depth. At res=101 over 8 strips that is
//    about 10KB, which lives in L1. That matters: the session079 locality probe reordered the occluders by tile and
//    measured the tile loop 24-50% faster on identical work, which is what put this structure ahead of everything else
//    left in the plan.
//
// accum and amp_sum are STRIP-LOCAL (row v lives at (v - v_lo)*res); sat_depth is the full grid, which the caller owns.
struct GsSatStripStats
{
	GsSatStripStats() : writers(0), tile_writes(0), tile_iters(0), tail_rej(0), sat_skip(0) {}
	size_t writers, tile_writes, tile_iters, tail_rej, sat_skip;
};


// SESSION079: one occluder's finished geometry - everything the tile loop needs and nothing it does not.
//
// This exists because measurement said it had to. The first cut of the strip-parallel build had every strip re-walk the
// whole occluder array and redo the geometry, keeping only the nodes falling in its own rows. Cutting that run short
// right after the strip test measured 80.10ms of a total 88ms: 91% of the parallel build was work every strip repeated,
// and only ~8ms was work the split actually divided. That is why the strip count made no difference between 3 and 12 -
// there was almost nothing left to divide.
//
// So the geometry is computed ONCE into these records, and the strips read them. 24 bytes each, ~1.6M of them.
// GsSatOccluderRec itself lives in the header - see there.


// The per-tile deposit for one occluder, restricted to rows [v_lo, v_hi). accum and amp_sum are strip-local (row v is at
// (v - v_lo)*res); sat_depth is the full grid.
//
// Shared by the serial and the parallel builds so the arithmetic has exactly one definition - the two must agree to the
// bit, and the only honest way to guarantee that is for there to be one copy.
static inline void gsSatDepositRec(const GsSatOccluderRec& rec, int res, float remaining_threshold,
	float* const sat_depth, float* const accum, float* const amp_sum, bool keep_accumulating_past_saturation,
	int v_lo, int v_hi, GsSatStripStats* stats)
{
	const float cu = rec.cu, cv = rec.cv, span = rec.span, amp = rec.amp;

	const int u0 = myClamp(gsSatCeilToInt(cu - span - 0.5f), 0, res - 1); // SESSION079 PERF: exact, call-free floor/ceil - see gsSatCeilToInt().
	const int u1 = myClamp((int)         (cu + span - 0.5f), 0, res - 1);
	const int v0_grid = myClamp(gsSatCeilToInt(cv - span - 0.5f), 0, res - 1);
	const int v1_grid = myClamp((int)         (cv + span - 0.5f), 0, res - 1);
	if(u1 < u0 || v1_grid < v0_grid)
		return;

	// SESSION079: the strip is INTERSECTED with the grid-clamped span, never substituted for it. Clamping straight to
	// [v_lo, v_hi-1] looks equivalent and is not: a node whose whole footprint sits outside the strip has both ends
	// collapse onto the strip's edge row, so it would deposit onto a row it does not actually touch. That is real - it
	// showed up as the parallel build reporting 4,815,255 tile writes against the serial build's 4,759,321.
	//
	// The grid clamp above stays exactly as the serial path had it, including its own edge behaviour, so a full-grid strip
	// reproduces the serial result bit for bit.
	const int v0 = myMax(v0_grid, v_lo);
	const int v1 = myMin(v1_grid, v_hi - 1);
	if(v1 < v0)
		return; // Footprint does not reach this strip.

	if(stats) ++stats->writers; // SESSION076 DIAGNOSTIC: this node contributes something.

	for(int v=v0; v<=v1; ++v)
	{
		float* const accum_row = accum     + (size_t)(v - v_lo) * (size_t)res;
		float* const depth_row = sat_depth + (size_t)v          * (size_t)res;
		float* const sum_row = amp_sum ? (amp_sum + (size_t)(v - v_lo) * (size_t)res) : NULL;
		const float dv = ((float)v + 0.5f) - cv;
		const float dv_sq = dv * dv;

		// SESSION081 PLAN ETAP 1: TEMPORARILY REVERTED to plain scalar for an apples-to-apples A/B measurement - the
		// SSE4.1 version (same shape as this loop, 4 tiles/iteration) measured SLOWER on a clean 4-teleport dep_ms
		// average (145.4ms) than the single old scalar sample it was first compared against (78.3ms). Before
		// concluding anything, need the OLD code's dep_ms averaged over the SAME clean protocol, not one sample
		// against four - see the session081 plan doc. Nothing here was ever committed, so the vectorised version
		// is not in git history - it is saved verbatim in tmp/session081_etap1_simd_deposit_backup.cpp pending the
		// fair comparison, in case it needs to come back.
		for(int u=u0; u<=u1; ++u)
		{
			if(stats) ++stats->tile_iters; // SESSION079 DIAGNOSTIC.

			const float du = ((float)u + 0.5f) - cu;
			const float x = (du * du + dv_sq) * rec.inv_2var;
			if(x >= gs_sat_exp_lut_max)
			{
				if(stats) ++stats->tail_rej; // SESSION079 DIAGNOSTIC: fell off the Gaussian's tail.
				continue; // Contribution below the lookup's tail - see gs_sat_exp_lut_max.
			}

			const bool already_saturated = depth_row[u] != std::numeric_limits<float>::infinity();

			// SESSION077: the fast path (skip a tile once its barrier is set) applies unless a diagnostic accumulator wants
			// the running totals past saturation. depth_row[u] is only ever set on the FIRST crossing either way, so the
			// barrier this grid prunes by is bit-identical whether or not the overlays are on.
			if(already_saturated && !keep_accumulating_past_saturation)
			{
				if(stats) ++stats->sat_skip; // SESSION079 DIAGNOSTIC: barrier already set, nothing left to do for this tile.
				continue;
			}

			// SESSION079 PERF: the lookup sits below the two rejects above. It used to precede them, so every tile that fell
			// off the tail or was already saturated - a third of all iterations on the reference viewpoint - paid for a
			// multiply, a float-to-int conversion, a clamp and a load it then discarded.
			const float w = gsSatExpNeg(x);

			if(stats) ++stats->tile_writes; // SESSION076 DIAGNOSTIC: write amplification.

			accum_row[u] *= (1.f - amp * w);
			if(!already_saturated && accum_row[u] <= remaining_threshold)
				depth_row[u] = rec.far_edge;

			if(sum_row) sum_row[u] += amp * w;
		}
	}
}


static void gsSatBuildStrip(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float remaining_threshold, float region_radius,
	float* const sat_depth, float* const accum, float* const amp_sum, bool keep_accumulating_past_saturation,
	int v_lo, int v_hi, GsSatStripStats* stats)
{
	// When the strip is the whole grid there is nothing to reject, so the test below is compiled out of the way and the
	// serial path costs exactly what it did before this split.
	const bool strip_is_partial = !(v_lo == 0 && v_hi == res);
	const float strip_lo_f = (float)v_lo;
	const float strip_hi_f = (float)v_hi;

	// SESSION079 PERF: coefficient for the cheap early-reject bound below. Derivation, from the write loop's own amp
	// formula further down:
	//
	//   tile_angle = (2/res) * s^1.5,  s in [1, sqrt(3)]           => tile_angle is MINIMAL at s=1 (worst case, largest footprint)
	//   radius_tiles = ang_radius / tile_angle <= (r/dist) * (res/2)
	//   sigma = radius_tiles/3  =>  sigma^2 <= (r/dist)^2 * res^2/36
	//   amp = alpha * 2*pi*sigma^2 / max(1, 2*pi*var) <= alpha * 2*pi*sigma^2      (denominator is always >= 1)
	//   amp <= alpha * (r^2/dist^2) * (pi*res^2/18)
	//
	// i.e. amp <= alpha * r^2 * amp_reject_k / dist_sq. This is a genuine upper bound (s=1 is the worst case across the
	// whole sphere), so a node whose bound already clears gs_sat_min_occluder_amp would have its true amp clear it too -
	// see the reject test itself for what that buys.
	const float amp_reject_k = 3.14159265f * (float)res * (float)res / 18.f;

	// SESSION074: nodes are assumed already front-to-back (a filtered subsequence of the traversal's globally sorted
	// output - see GaussianSplatUnculledFrontier / GaussianSplatLodTraversalTask::run()). This is the ONLY sequential,
	// order-dependent pass in the whole mechanism - see the header's file comment - so it must not be reordered or
	// parallelised without re-deriving correctness.
	for(size_t i=0; i<n; ++i)
	{
		const float dx = px[i] - anchor_pos_ws.x[0];
		const float dy = py[i] - anchor_pos_ws.x[1];
		const float dz = pz[i] - anchor_pos_ws.x[2];
		const float dist_sq = dx*dx + dy*dy + dz*dz;
		if(dist_sq < 1.0e-12f) // Degenerate: node sits at the anchor. No meaningful direction; skip.
			continue;

		const float r = radius[i];

		// SESSION079 PERF: cheap early reject, BEFORE the sqrt. Measured on the owner's reference viewpoint, only 35.3%
		// of the occluders handed to this loop ever contribute anything - the other 64.7% are thrown away by the
		// amp < gs_sat_min_occluder_amp test far below, AFTER paying for the sqrt, the oct unwrap, the local tile angle
		// and the whole amplitude derivation. That is nearly two million nodes per build billed in full for nothing.
		//
		// amp_reject_k above bounds the true amp from above using only dist_sq, r and alpha, in three multiplies and no
		// sqrt. dist_sq is moved to the right-hand side so there is not even a divide. Because it is an upper bound, a
		// node rejected here would certainly have been rejected below - so `writers` must come out BIT-IDENTICAL. That
		// is the whole correctness check for this stage, and it is completely objective.
		//
		// The real amp < gs_sat_min_occluder_amp test below stays exactly where it is: this bound is deliberately loose
		// (it assumes the most favourable direction on the sphere, s=1), so it only catches the clear-cut bulk and the
		// exact test still has to catch the remainder.
		const float a = myClamp(alpha[i], 0.f, 1.f); // Hoisted from the amplitude block below, which used to declare it - same value, needed here first.
		if(a * r * r * amp_reject_k < gs_sat_min_occluder_amp * dist_sq)
			continue;


		// SESSION074: angular radius without trigonometry. The exact value is asin(r/dist); for the small angles this
		// pass deals with, r/dist is within a fraction of a percent of it, and it is only ever compared against tile
		// sizes that already carry a 1.5x safety margin. Kept as the ratio to avoid a per-node asin AND the sqrt that
		// feeding it would require - both measured as a large part of this pass's first-cut cost. Where a node is
		// close enough for the approximation to drift (r comparable to dist), it drifts towards a LARGER angle, i.e.
		// towards covering more tiles - so the comparison below is bounded by the exact one, not looser than it.
		const float inv_dist = 1.f / std::sqrt(dist_sq); // One sqrt per node here is unavoidable: sat_depth is stored as a real distance, and the span needs a real angle.
		// SESSION078 CORRECTION: this is deliberately NOT eroded by region_radius, though the first cut of region pruning
		// did erode it (angular radius (r - R)/d, dropping any occluder with r <= R outright, on the argument that an
		// occluder smaller than the region can be stepped around). That argument holds for an ISOLATED occluder and is
		// wrong for the thing this grid is actually made of. A wall is a dense sheet of thousands of small overlapping
		// splats; you cannot step around the WALL by moving 2cm, only around any one of its atoms. Eroding every atom
		// shrinks the aggregate's interior coverage, not just its silhouette, which is not what the parallax does.
		//
		// It failed loudly in practice. The LoD holds nodes near a fixed screen size, so near-field occluders - exactly
		// the ones doing the occluding when you stand near a wall - have 3-sigma world radii of order 1-2cm. R = 0.02
		// therefore deleted most of a near wall's occluder population outright, its tiles stopped saturating, and large
		// islands of geometry behind it (owner's report: a far wall and a sofa, plainly behind a double thickness of
		// opaque wall) came back into the draw list. R = 0 was correct and R = 0.02 was not, for a 2cm region.
		//
		// The ball conservatism belongs on the READ side instead, where it acts on the finished AGGREGATE mask rather
		// than on each atom: a candidate node's footprint is dilated by R/d and every tile it then touches must agree it
		// is occluded (see gsSatOccluded()). That is the same silhouette-edge guard, applied to the surface as a whole -
		// a candidate near the edge of a saturated region reaches an unsaturated tile and is kept, while one deep inside
		// a genuinely opaque wall is still dropped. The build side keeps only the depth half of the conservatism below.
		const float ang_radius = r * inv_dist;

		// SESSION079: gsSatGridFootprint() unrolled into its two halves - textually the same arithmetic - so the strip test
		// can sit between them. cv falls out of the oct unwrap alone, while the local tile angle below is the expensive
		// half; a node that cannot reach this strip should not pay for it.
		const float l1 = std::fabs(dx) + std::fabs(dy) + std::fabs(dz);
		const Vec2f oct = gsDirToOctWithL1(Vec4f(dx, dy, dz, 0.f), l1);
		const float cu = (oct.x * 0.5f + 0.5f) * (float)res;
		const float cv = (oct.y * 0.5f + 0.5f) * (float)res;

		if(strip_is_partial)
		{
			// Same s=1 upper bound the early amp reject rests on: radius_tiles <= ang_radius * res/2, since the local tile
			// angle is smallest along the world axes. Half a tile's diagonal is added for the same reason the real span adds
			// it. Bounding from above means a node is only ever skipped when it provably cannot touch this strip.
			const float span_max = ang_radius * ((float)res * 0.5f) + gs_sat_tile_half_diag;
			if(cv + span_max < strip_lo_f || cv - span_max >= strip_hi_f)
				continue;
		}

		const float radius_tiles = ang_radius * gsSatGridInvLocalTileAngle(l1, inv_dist, res);

		// SESSION076: an occluder is a GAUSSIAN, not a uniformly opaque disc, and the whole of this pass's accuracy
		// turns on that distinction. Both earlier cuts modelled it as a disc and failed in opposite directions:
		//
		//  - at the shader's 3-sigma draw cutoff, the disc stamped near-opaque coverage out to where the node is
		//    actually ~1% of peak, so every occluder cast a shadow 3x too wide in angle. Owner-visible as plainly
		//    unoccluded objects on a table disappearing, and as which object disappeared changing under a 10-20cm
		//    camera step.
		//  - shrunk to 1 sigma to fix that, the disc became exactly one tile across at the coarse scale, so the
		//    full-coverage write rule could never be satisfied by a coarse-scale node at all - the mid-field lost its
		//    occluders entirely and a chair three metres away stopped occluding anything behind it.
		//
		// There is no radius that is right, because the error is in the disc, not its size. So: contribute to every
		// tile the footprint TOUCHES, weighted by the node's own Gaussian falloff to that tile.
		//
		// The weight is not the Gaussian sampled at the tile centre, which would be a point sample of a continuous
		// quantity and would make a sub-tile node either count fully (centre hit) or not at all (centre missed). It is
		// the Gaussian CONVOLVED WITH THE TILE - the tile's own box approximated by its variance, 1/12 per axis. That
		// makes the contribution conserve the node's integrated occlusion in both regimes, which is the same energy
		// argument session071 used for merged colours (GaussianSplatMergeColourMode_Energy):
		//
		//   var  = sigma^2 + 1/12                     (Gaussian variance widened by the tile's own)
		//   a_eff(d) = amp * exp(-d^2 / (2*var))
		//
		// with amp set so that the node's whole contribution comes to its true integrated occlusion, alpha*2*pi*sigma^2
		// (the integral of a 2D Gaussian of peak alpha), in tile-area units.
		//
		// SESSION077 FIX: amp used to be alpha * sigma^2 / var, which normalises against the kernel's INTEGRAL,
		// 2*pi*var. The write loop below does not integrate: it sums a_eff over the DISCRETE lattice of tile centres,
		// and the two only agree while the kernel is wide compared to a tile. At the small end they diverge badly - with
		// sigma << 1, var -> 1/12, the kernel's width is sqrt(1/12) = 0.29 tiles, so the discrete sum collapses to ~1.0
		// (essentially the centre tile alone) against an integral of 2*pi/12 = 0.52. Every such node therefore deposited
		// 12/(2*pi) = 1.9x its own occlusion.
		//
		// The error is confined to small sigma, which makes it a DISTANCE-dependent bias rather than a constant one: the
		// LoD holds nodes at a roughly fixed screen size, so sigma_tiles is small everywhere EXCEPT near the camera,
		// where the tree runs out of depth and hands back leaf splats larger than the pixel limit. Far field
		// over-saturated by ~1.9x while the near field was exact, so an opaque surface came out painted beyond a
		// camera-relative radius and clear inside it - the moving patch the owner tracked across a wall and a ceiling in
		// session077.
		//
		// Normalising against the discrete sum instead removes it. That sum is separable, and its 1D factor
		// (a theta function) is within a few percent of max(1, sqrt(2*pi*var)) across the whole range - 1 when the
		// kernel fits inside one tile, sqrt(2*pi*var) once it spans several - so S = max(1, 2*pi*var) in 2D:
		//
		//   amp = alpha * 2*pi*sigma^2 / max(1, 2*pi*var)
		//
		// Both limits come out right, and neither is special-cased: sigma >> 1 gives amp -> alpha (the node blankets the
		// tile with its own opacity), sigma << 1 gives amp -> alpha*2*pi*sigma^2 deposited into essentially one tile,
		// which is exactly that node's integrated occlusion.
		const float sigma_tiles = radius_tiles * (1.f / gs_sat_occluder_sigmas);
		const float sigma_sq = sigma_tiles * sigma_tiles;
		const float var = sigma_sq + (1.f / 12.f);
		const float two_pi = 6.28318531f;
		const float amp = a * (two_pi * sigma_sq) / myMax(1.f, two_pi * var);
		if(amp < gs_sat_min_occluder_amp)
			continue; // Integrated occlusion is nil - see gs_sat_min_occluder_amp. Skips the fine-scale bulk cheaply.

		// SESSION074: recorded at the node's FAR edge (dist + radius), not its centre or near edge - deliberately
		// conservative. The true saturation point lies somewhere within this node's footprint (we don't know exactly
		// where), and placing the cut past its entire extent guarantees this stage only ever drops fine geometry
		// unambiguously behind the coarse mass that caused saturation - never something that might still be in front
		// of, or interleaved with, the very geometry that saturated the tile.
		// SESSION078: + region_radius because from the ball point furthest from this occluder its far edge sits a further
		// R away, and the barrier has to be past it from every position in the ball. This is the build side's whole share
		// of the region conservatism - see the footprint comment above for why the angular half is not applied here.
		const float far_edge = (dist_sq * inv_dist) + r + region_radius; // SESSION079 PERF: dist_sq/sqrt(dist_sq) is dist - a multiply where 1/inv_dist was a divide.

		// SESSION076: every tile the footprint reaches, weighted - no entitlement test.
		//
		// Entitlement ("this node may only claim tiles it fully blankets") existed to stop a coarse blob from asserting
		// occlusion across directions its silhouette never covered. Feeding the grid the fine frontier removes the
		// premise: a fine node's footprint IS the geometry's footprint, so a node that half-covers a tile makes exactly
		// a half-covered tile's worth of claim, and a tile straddling a silhouette accumulates the mixture it actually
		// contains rather than being arbitrated one way or the other by a rule.
		//
		// This is what makes the ambiguous cell behave correctly on its own: a tile that is half opaque table and half
		// open space accumulates towards half-opacity and does not reach the threshold, so nothing behind it is dropped.
		// Under the coarse source that same cell was claimed outright by whichever blob's disc happened to reach it.
		GsSatOccluderRec rec;
		rec.cu = cu;
		rec.cv = cv;
		rec.span = radius_tiles + gs_sat_tile_half_diag;
		rec.amp = amp;
		rec.far_edge = far_edge;
		rec.inv_2var = 0.5f / var;

		gsSatDepositRec(rec, res, remaining_threshold, sat_depth, accum, amp_sum, keep_accumulating_past_saturation, v_lo, v_hi, stats);
	}
}


// SESSION080 - REGION EROSION. One max-filter over the finished grid, radius set per tile by R/sat_depth. This is the
// angular half of the ball guarantee, and it belongs HERE - on the finished aggregate mask - for the reason the build's
// footprint comment spells out: eroding individual occluder atoms destroys a dense surface's interior (session078's
// failure), while eroding the aggregate only eats into it from its silhouettes, which is exactly what parallax does.
//
// Why a max-filter is the whole operation. Unsaturated tiles are +inf, so a single max over the window does both halves
// at once:
//   - mask erosion: any +inf in the window makes the result +inf, i.e. a tile within reach of open sky stops being
//     trusted. This is the silhouette band, and its width is the parallax swing.
//   - depth worst case: among tiles that all agree, the barrier taken is the furthest any of them claims.
//
// The consequence worth stating plainly, because it is what makes a large R useful rather than useless: an ENCLOSING
// occluder (the inside of a room, a cave, an interior inside a much larger outdoor scene) has no +inf anywhere in the
// directions it covers, so no amount of erosion removes anything, and everything outside the shell stays prunable at
// any R. The cost falls entirely on silhouettes - a doorway, a window, the edge of a wall you can step around - which
// is precisely where a moving camera does reveal new geometry and where pruning therefore must stop.
//
// Boundary handling matches gsSatOccluded()'s: the window is clamped to the grid rather than wrapped across the
// octahedral seam. Consistent with the read side, and the under-coverage is confined to the map's edge rows.


// SESSION080 DIAGNOSTIC: the erosion's two kill mechanisms, kept apart because they say opposite things about what to
// do next. by_ceiling means R is simply too large next to that direction's occluder (an honest refusal - the ball
// reaches past the occluding mass); by_window means the tile lost to a nearby unsaturated tile, which is the intended
// silhouette behaviour but is ALSO how a noisy, pinholed mask destroys itself once max_radius reaches 1. Which of the
// two dominates decides whether the answer is a different R or a different erosion rule.
struct GsSatErodeStats
{
	GsSatErodeStats() : by_ceiling(0), by_window(0), max_radius(0) {}
	size_t by_ceiling, by_window, max_radius;
};
static void gsSatErodeRegionRows(const float* const src, float* const dst, int res,
	float region_radius, int v_begin, int v_end, GsSatErodeStats* out_stats)
{
	const float inf = std::numeric_limits<float>::infinity();
	GsSatErodeStats st;
	for(int v=v_begin; v<v_end; ++v)
		for(int u=0; u<res; ++u)
		{
			const size_t idx = (size_t)v * (size_t)res + (size_t)u;
			const float here = src[idx];
			if(here == inf)
			{
				dst[idx] = inf; // Already unusable; nothing a max could do to it.
				continue;
			}

			const float rad_f = gsSatRegionErosionTiles(region_radius, here, res); // `here` IS the barrier - see gsSatRegionErosionTiles().
			if(rad_f > (float)gs_sat_region_max_erosion_tiles)
			{
				dst[idx] = inf; // Past the ceiling: R is not small next to this occluder - see the constant.
				++st.by_ceiling;
				continue;
			}

			const int rad = (int)rad_f;
			if((size_t)rad > st.max_radius) st.max_radius = (size_t)rad; // SESSION080 DIAGNOSTIC: over tiles that were saturated, so it reports the radius actually in use, not one derived from an empty direction.
			if(rad <= 0)
			{
				dst[idx] = here; // Swing is under a tile - the grid cannot express anything finer.
				continue;
			}

			const int u0 = myMax(u - rad, 0), u1 = myMin(u + rad, res - 1);
			const int v0 = myMax(v - rad, 0), v1 = myMin(v + rad, res - 1);
			float worst = here;
			for(int vv=v0; vv<=v1 && worst != inf; ++vv)
			{
				const float* const row = src + (size_t)vv * (size_t)res;
				for(int uu=u0; uu<=u1; ++uu)
					if(row[uu] > worst)
					{
						worst = row[uu];
						if(worst == inf)
							break; // Cannot get worse - the rest of the window is irrelevant.
					}
			}
			dst[idx] = worst;
			if(worst == inf)
				++st.by_window;
		}

	if(out_stats)
		*out_stats = st;
}


// SESSION081 - HOLE CLOSING. Runs BEFORE the region erosion above, to fix an interaction the erosion's own doc comment
// already flagged as a risk ("how a noisy, pinholed mask destroys itself once max_radius reaches 1"): the erosion is a
// pure MAX-window (grayscale dilation) over the finished mask, so a single tile that missed the saturation threshold
// by noise - not a real silhouette - spreads its "unusable" (+inf) verdict across the WHOLE erosion window around it,
// up to gs_sat_region_max_erosion_tiles wide. Measured (owner's interior, R=0.9): er_win alone erased 62.0% of an
// otherwise 65.1%-saturated mask, collapsing dropped% from an 86.6% R=0 baseline to 7.9%, with er_ceil=0 in every
// capture - the honest "R is too large for this occluder" refusal was never the cause.
//
// Closing removes exactly this class of defect: a hole narrower than 2*rc+1 tiles is filled solid before the erosion
// ever sees it, while a hole at least that wide - a real doorway, a wall's corner, the edge of the geometry - passes
// through unchanged, because no window inside it is ever fully surrounded by finite neighbours. Same "erode the
// aggregate, not the atoms" principle the build side already uses (gsSatBuildStrip()'s SESSION078 CORRECTION),
// applied one level up: here it is the erosion's OWN silhouette detector being protected from individual atoms
// (single noisy tiles), not the occluders.
//
// Two full-grid passes, values only - no separate binary mask is kept, because +inf already IS this domain's
// "background" and behaves correctly as the identity element for min:
//  1. tmp[t] = MIN over a (2*rc+1)^2 window of src (t included). This is binary dilation of the "saturated" indicator,
//     expressed on real values: tmp[t] comes out finite iff at least one tile in the window is finite, and the value
//     picked (the nearest local barrier) is provisional scratch for step 2, not a final answer.
//  2. For a tile whose ORIGINAL value was +inf: dst[t] = MAX over the same window of tmp. This is binary erosion of
//     the now-dilated indicator - finite only if EVERY tile in the window was filled by step 1, i.e. the hole is
//     fully surrounded within rc tiles on every side. MAX also picks the FURTHEST of the locally-available barriers,
//     the same "never drop more than necessary" direction the R-erosion below and the build side's +region_radius
//     already commit to.
// A tile whose original value was already finite is never touched (dst[t] = src[t] verbatim) - closing only ever ADDS
// conservatism to a hole, it never revises a real measurement.
//
// closing_radius_tiles <= 0 reproduces the pre-session081 behaviour bit-for-bit (this whole pass is skipped).
struct GsSatCloseStats
{
	GsSatCloseStats() : closed(0) {}
	size_t closed; // Tiles that were +inf in the input and came out finite - i.e. pinholes actually filled.
};


static void gsSatCloseMinRows(const float* const src, float* const tmp_min, int res, int rc, int v_begin, int v_end)
{
	const float inf = std::numeric_limits<float>::infinity();
	for(int v=v_begin; v<v_end; ++v)
	{
		const int v0 = myMax(v - rc, 0), v1 = myMin(v + rc, res - 1);
		for(int u=0; u<res; ++u)
		{
			const int u0 = myMax(u - rc, 0), u1 = myMin(u + rc, res - 1);
			float m = inf;
			for(int vv=v0; vv<=v1; ++vv)
			{
				const float* const row = src + (size_t)vv * (size_t)res;
				for(int uu=u0; uu<=u1; ++uu)
					if(row[uu] < m)
						m = row[uu];
			}
			tmp_min[(size_t)v * (size_t)res + (size_t)u] = m;
		}
	}
}


static void gsSatCloseFinishRows(const float* const src, const float* const tmp_min, float* const dst, int res, int rc,
	int v_begin, int v_end, GsSatCloseStats* out_stats)
{
	const float inf = std::numeric_limits<float>::infinity();
	GsSatCloseStats st;
	for(int v=v_begin; v<v_end; ++v)
	{
		const int v0 = myMax(v - rc, 0), v1 = myMin(v + rc, res - 1);
		for(int u=0; u<res; ++u)
		{
			const size_t idx = (size_t)v * (size_t)res + (size_t)u;
			if(src[idx] != inf) { dst[idx] = src[idx]; continue; } // Real measurement - closing never revises it.

			const int u0 = myMax(u - rc, 0), u1 = myMin(u + rc, res - 1);
			float worst = -inf;
			bool all_finite = true;
			for(int vv=v0; vv<=v1 && all_finite; ++vv)
			{
				const float* const row = tmp_min + (size_t)vv * (size_t)res;
				for(int uu=u0; uu<=u1; ++uu)
					if(row[uu] == inf) { all_finite = false; break; }
					else if(row[uu] > worst) worst = row[uu];
			}
			dst[idx] = all_finite ? worst : inf;
			if(all_finite) ++st.closed;
		}
	}
	if(out_stats) *out_stats = st;
}


// SESSION081: row-parallel workers for the two passes above - same disjoint-write/full-read shape as GsSatErodeTask
// below. Two SEPARATE task classes (and, in gsSatApplyClosing(), two separate task groups) because pass 2 reads
// tmp_min across the WHOLE grid - a window can straddle any strip boundary - so every strip's pass 1 must finish
// before any strip starts pass 2.
class GsSatCloseMinTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/) { gsSatCloseMinRows(src, tmp_min, res, rc, v_begin, v_end); }
	const float* src; float* tmp_min; int res, rc, v_begin, v_end;
};
class GsSatCloseFinishTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/) { gsSatCloseFinishRows(src, tmp_min, dst, res, rc, v_begin, v_end, &stats); }
	const float* src; const float* tmp_min; float* dst; int res, rc, v_begin, v_end;
	GsSatCloseStats stats;
};


// SESSION081: safety ceiling on the closing radius. Cost is O((2*rc+1)^2) per tile for two full-grid passes - the same
// shape as the erosion window below, but in practice far smaller (a handful of tiles fixes pinhole noise; nothing
// about closing needs anywhere near gs_sat_region_max_erosion_tiles). This ceiling exists only to stop a mistyped
// huge value from being expensive, not because larger values would be unsound.
static const int gs_sat_closing_max_radius_tiles = 8;


// SESSION081: applies closing in place into `sat_depth` (swaps a scratch buffer into it, same pattern
// gsSatApplyRegionErosion below uses for its own result). No-op when closing_radius_tiles <= 0.
static void gsSatApplyClosing(js::Vector<float, 16>& sat_depth, int res, int closing_radius_tiles,
	glare::TaskManager* task_manager, size_t* out_closed)
{
	if(out_closed) *out_closed = 0;
	if(closing_radius_tiles <= 0 || res == 0)
		return;
	const int rc = myMin(closing_radius_tiles, gs_sat_closing_max_radius_tiles);

	js::Vector<float, 16> tmp_min(sat_depth.size());
	js::Vector<float, 16> closed(sat_depth.size());

	const int concurrency = task_manager ? myMax(1, (int)task_manager->getConcurrency()) : 1;
	const int num_strips = myClamp(res / gs_sat_min_strip_rows, 1, concurrency);

	GsSatCloseStats total;
	if(num_strips <= 1)
	{
		gsSatCloseMinRows(sat_depth.data(), tmp_min.data(), res, rc, 0, res);
		gsSatCloseFinishRows(sat_depth.data(), tmp_min.data(), closed.data(), res, rc, 0, res, &total);
	}
	else
	{
		{
			glare::TaskGroupRef group = new glare::TaskGroup();
			js::Vector<Reference<GsSatCloseMinTask>, 16> tasks(num_strips);
			for(int t=0; t<num_strips; ++t)
			{
				Reference<GsSatCloseMinTask> task = new GsSatCloseMinTask();
				task->src = sat_depth.data(); task->tmp_min = tmp_min.data(); task->res = res; task->rc = rc;
				task->v_begin = (int)(((int64)res * t)       / num_strips);
				task->v_end   = (int)(((int64)res * (t + 1)) / num_strips);
				tasks[t] = task;
				group->tasks.push_back(task);
			}
			task_manager->runTaskGroup(group);
		}
		{
			glare::TaskGroupRef group = new glare::TaskGroup();
			js::Vector<Reference<GsSatCloseFinishTask>, 16> tasks(num_strips);
			for(int t=0; t<num_strips; ++t)
			{
				Reference<GsSatCloseFinishTask> task = new GsSatCloseFinishTask();
				task->src = sat_depth.data(); task->tmp_min = tmp_min.data(); task->dst = closed.data(); task->res = res; task->rc = rc;
				task->v_begin = (int)(((int64)res * t)       / num_strips);
				task->v_end   = (int)(((int64)res * (t + 1)) / num_strips);
				tasks[t] = task;
				group->tasks.push_back(task);
			}
			task_manager->runTaskGroup(group);
			for(int t=0; t<num_strips; ++t)
				total.closed += tasks[t]->stats.closed;
		}
	}

	if(out_closed) *out_closed = total.closed;
	sat_depth = closed;
}


// SESSION080: the erosion split over the task manager by rows. Read-only source, disjoint destination rows - nothing to
// synchronise, same shape as the build's strips.
class GsSatErodeTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		gsSatErodeRegionRows(src, dst, res, region_radius, v_begin, v_end, &stats);
	}

	const float* src;
	float* dst;
	int res, v_begin, v_end;
	float region_radius;
	GsSatErodeStats stats;
};


// SESSION080: shared tail of both builds - allocate the destination, run the erosion (parallel where a pool was handed
// in), swap it into place. No-op when region_radius is 0, which keeps the R = 0 path bit-identical to before.
// SESSION081 PLAN ETAP 0: out_close_ms/out_erode_ms, when non-null, receive the wall time of each sub-pass separately -
// both used to be folded into the caller's one grid_ms timer with no way to tell how much either cost (the plan's
// "sum the parts against the whole" check). Left NULL (no Timer calls at all) when the caller does not ask.
static void gsSatApplyRegionErosion(js::Vector<float, 16>& sat_depth, int res,
	float region_radius, int closing_radius_tiles, glare::TaskManager* task_manager, size_t* out_erode_stats,
	double* out_close_ms, double* out_erode_ms)
{
	if(out_erode_stats)
		out_erode_stats[0] = out_erode_stats[1] = out_erode_stats[2] = out_erode_stats[3] = 0;
	if(out_close_ms) *out_close_ms = 0.0;
	if(out_erode_ms) *out_erode_ms = 0.0;
	if(!(region_radius > 0.f) || res == 0)
		return;

	// SESSION081: closing runs first, in place, so the erosion below sees a mask with pinhole noise already removed -
	// see gsSatApplyClosing()'s doc comment for why order matters here.
	Timer close_timer;
	size_t closed_count = 0;
	gsSatApplyClosing(sat_depth, res, closing_radius_tiles, task_manager, &closed_count);
	if(out_erode_stats) out_erode_stats[3] = closed_count;
	if(out_close_ms) *out_close_ms = close_timer.elapsed() * 1.0e3;

	Timer erode_timer;
	js::Vector<float, 16> eroded(sat_depth.size());

	const int concurrency = task_manager ? myMax(1, (int)task_manager->getConcurrency()) : 1;
	const int num_strips = myClamp(res / gs_sat_min_strip_rows, 1, concurrency);
	GsSatErodeStats total;
	if(num_strips <= 1)
	{
		gsSatErodeRegionRows(sat_depth.data(), eroded.data(), res, region_radius, 0, res, &total);
	}
	else
	{
		glare::TaskGroupRef group = new glare::TaskGroup();
		js::Vector<Reference<GsSatErodeTask>, 16> tasks(num_strips);
		for(int t=0; t<num_strips; ++t)
		{
			Reference<GsSatErodeTask> task = new GsSatErodeTask();
			task->src = sat_depth.data();
			task->dst = eroded.data();
			task->res = res;
			task->region_radius = region_radius;
			task->v_begin = (int)(((int64)res * t)       / num_strips);
			task->v_end   = (int)(((int64)res * (t + 1)) / num_strips);
			tasks[t] = task;
			group->tasks.push_back(task);
		}
		task_manager->runTaskGroup(group);

		for(int t=0; t<num_strips; ++t)
		{
			total.by_ceiling += tasks[t]->stats.by_ceiling;
			total.by_window  += tasks[t]->stats.by_window;
			total.max_radius = myMax(total.max_radius, tasks[t]->stats.max_radius);
		}
	}

	if(out_erode_stats)
	{
		out_erode_stats[0] = total.by_ceiling;
		out_erode_stats[1] = total.by_window;
		out_erode_stats[2] = total.max_radius;
	}
	if(out_erode_ms) *out_erode_ms = erode_timer.elapsed() * 1.0e3;

	sat_depth = eroded;
}


void gsBuildSaturationGrid(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius, int closing_radius_tiles,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers, size_t* out_tile_writes,
	js::Vector<float, 16>* out_accum_t, js::Vector<float, 16>* out_amp_sum, size_t* out_tile_stats,
	size_t* out_erode_stats, double* out_close_ms, double* out_erode_ms)
{
	const size_t num_tiles = (size_t)res * (size_t)res;
	sat_depth_out.resizeNoCopy(num_tiles);
	for(size_t i=0; i<num_tiles; ++i)
		sat_depth_out[i] = std::numeric_limits<float>::infinity();

	// SESSION077 DIAGNOSTIC: unbounded companion to accum_t below - see out_amp_sum's header comment.
	js::Vector<float, 16> amp_sum;
	if(out_amp_sum)
		amp_sum.resize(num_tiles, 0.f);

	// SESSION074: running per-tile transmittance, local scratch only (not stored on the frontier - sat_depth_out is
	// the only thing callers need). Reset to 1 (fully transparent) before the sequential pass below.
	js::Vector<float, 16> accum_t(num_tiles, 1.f);

	GsSatStripStats stats;
	gsSatBuildStrip(px, py, pz, radius, alpha, n, anchor_pos_ws, res,
		myClamp(1.f - saturation_threshold, 0.f, 1.f), region_radius,
		sat_depth_out.data(), accum_t.data(), out_amp_sum ? amp_sum.data() : NULL,
		(out_accum_t != NULL) || (out_amp_sum != NULL),
		0, res, (out_writers || out_tile_writes || out_tile_stats) ? &stats : NULL);

	if(out_writers)     *out_writers     = stats.writers;
	if(out_tile_writes) *out_tile_writes = stats.tile_writes;
	if(out_tile_stats) { out_tile_stats[0] = stats.tile_iters; out_tile_stats[1] = stats.tail_rej; out_tile_stats[2] = stats.sat_skip; }

	// SESSION080: the erosion runs LAST, on the finished mask - see gsSatApplyRegionErosion(). Serial entry point, so no
	// pool: this path is the overlay/no-TaskManager one, where a few ms more is not what anyone is measuring.
	gsSatApplyRegionErosion(sat_depth_out, res, region_radius, closing_radius_tiles, NULL, out_erode_stats, out_close_ms, out_erode_ms);

	// SESSION077 DIAGNOSTIC: hand the accumulator fields out for the debug overlay - see the header.
	if(out_accum_t)
		*out_accum_t = accum_t;
	if(out_amp_sum)
		*out_amp_sum = amp_sum;
}


// SESSION079 phase 1: turn a contiguous run of occluders into records. A pure map - no shared state, no order
// dependence - so it parallelises over node ranges with nothing to synchronise.
//
// Records are written into this task's own slice of the output array, sized to its input range, and `count` says how
// many of that slice were used. Nodes are visited in ascending order and appended, so the records stay in front-to-back
// order within the slice, and the slices themselves are in order - which is what lets phase 2 keep the accumulation
// exact.
class GsSatRecordTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		start_ms = clock->elapsed() * 1.0e3; // SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC - see the field.
		Timer task_timer; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC - see task_ms. Nodes are FRONT-TO-BACK and the cheap/expensive split correlates with distance, so an equal-COUNT chunking may not be an equal-WORK chunking; this measures whether that is actually true before anything is rebalanced.
		const float amp_reject_k = 3.14159265f * (float)res * (float)res / 18.f;
		size_t w = 0;
		size_t g1 = 0; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC - see gate1_survivors.
		for(size_t i=i_begin; i<i_end; ++i)
		{
			const float dx = px[i] - anchor_pos_ws.x[0];
			const float dy = py[i] - anchor_pos_ws.x[1];
			const float dz = pz[i] - anchor_pos_ws.x[2];
			const float dist_sq = dx*dx + dy*dy + dz*dz;
			if(dist_sq < 1.0e-12f)
				continue; // Degenerate: node sits at the anchor. No meaningful direction.

			const float r = radius[i];
			const float a = myClamp(alpha[i], 0.f, 1.f);
			if(a * r * r * amp_reject_k < gs_sat_min_occluder_amp * dist_sq)
				continue; // Upper bound on amp is already below the cutoff - see the same test in gsSatBuildStrip().
			++g1; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY - survived the cheap gate; about to pay for sqrt+oct+tile-angle whether or not it becomes a record.

			const float inv_dist = 1.f / std::sqrt(dist_sq);
			const float ang_radius = r * inv_dist;

			const float l1 = std::fabs(dx) + std::fabs(dy) + std::fabs(dz);
			const Vec2f oct = gsDirToOctWithL1(Vec4f(dx, dy, dz, 0.f), l1);
			const float radius_tiles = ang_radius * gsSatGridInvLocalTileAngle(l1, inv_dist, res);

			const float sigma_tiles = radius_tiles * (1.f / gs_sat_occluder_sigmas);
			const float sigma_sq = sigma_tiles * sigma_tiles;
			const float var = sigma_sq + (1.f / 12.f);
			const float two_pi = 6.28318531f;
			const float amp = a * (two_pi * sigma_sq) / myMax(1.f, two_pi * var);
			if(amp < gs_sat_min_occluder_amp)
				continue; // Integrated occlusion is nil - see gs_sat_min_occluder_amp.

			GsSatOccluderRec& rec = out[w++];
			rec.cu = (oct.x * 0.5f + 0.5f) * (float)res;
			rec.cv = (oct.y * 0.5f + 0.5f) * (float)res;
			rec.span = radius_tiles + gs_sat_tile_half_diag;
			rec.amp = amp;
			rec.far_edge = (dist_sq * inv_dist) + r + region_radius;
			rec.inv_2var = 0.5f / var;
		}
		count = w;
		gate1_survivors = g1; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC.
		task_ms = task_timer.elapsed() * 1.0e3; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC.
		end_ms = clock->elapsed() * 1.0e3;      // SESSION081 SCHEDULING PROBE - see start_ms.
	}

	const float *px, *py, *pz, *radius, *alpha;
	size_t i_begin, i_end, count;
	size_t gate1_survivors; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC - see run(). gate1_survivors - count is how many paid for sqrt+oct+tile-angle only to be rejected by the exact amp test - the ceiling on a tighter second-tier reject, measured BEFORE writing one.
	// SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC. The rec-balance probe above measures how long a chunk RUNS; it
	// cannot say when the chunk ran, and that turned out to be the question. Measured 2026-09-01 over 9 builds: the sum
	// of chunk run-times is near-constant (269-333ms, 1.23x spread) while rec_ms swings 31-116ms (3.74x), and in the
	// slowest build the group took four times its own slowest chunk. That rules out load imbalance, but it does NOT
	// distinguish the two remaining stories - chunks QUEUED behind other pool work and starting late, versus chunks
	// starting on time and being preempted mid-run (which inflates task_ms itself, since it is wall time).
	//
	// These two timestamps separate them. Both are read from ONE clock shared by every chunk in the group (see
	// GsSatRecordTask::clock) rather than each chunk timing itself - separate zero points would make the starts
	// incomparable, which is the whole measurement. If start_ms is spread across most of rec_ms, the chunks were
	// queued; if every chunk starts near zero and rec_ms is still long, they were preempted while running.
	double start_ms, end_ms; // Chunk start/end, ms after the group's clock was reset - see run().
	const Timer* clock;      // Shared group clock, owned by the caller. Timer::elapsed() is const, so concurrent reads are safe.
	double task_ms; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC - this chunk's own wall time. Chunks split the input range by equal COUNT; if the cheap/expensive split correlates with distance (front-to-back order), the near chunk can do far more real work than the far one while both wait on the same runTaskGroup() - this is what would explain rec_ms exceeding the DRAM-bandwidth-bound + arithmetic estimate.
	Vec4f anchor_pos_ws;
	int res;
	float region_radius;
	GsSatOccluderRec* out;
};


// SESSION081: one chunk's share of the phase-1 compaction.
//
// Phase 1's chunks write into slices sized to their INPUT range, so the surviving records sit in num_chunks dense runs
// separated by gaps. Closing those gaps used to be a serial memmove loop on the calling thread, and the scheduling
// probe caught what that cost: 12.8-25.1ms per build, mean 18.2 - THIRTY PERCENT of rec_ms, with the other sixteen
// threads idle through all of it. It had never been timed apart from rec_ms, so every rec_ms figure this session
// quoted had it folded in.
//
// SESSION081 BUGFIX - the first parallel cut of this compacted IN PLACE, and was wrong. Its argument ran: a chunk's
// records only ever move down (out[c] <= slice_begin[c]), and chunk c's destination [out[c], out[c+1]) lies below
// slice_begin[c'] for every LATER chunk c' - then asserted "in both directions" without checking the other one. The
// other one does not hold. A late chunk's destination lands inside an EARLY chunk's source, because the destinations
// contract towards zero by the survival rate while the sources do not: at the measured 16.7% survival (num_recs 2.25M
// out of 13.5M occluders) chunk 6 writes [1.00L, 1.17L) while chunk 1 is still reading [1.00L, 1.17L), L being a
// slice's length. Whichever ran first decided the contents.
//
// The damage was not a crash or a lost record - counts are unaffected, which is why the writers/num_recs checks that
// commit leaned on did not catch it. It was silent corruption of record CONTENT, which breaks the one thing the whole
// grid rests on: strict front-to-back order. A tile then latches its barrier off whichever occluder happened to be
// sitting in that slot, so the grid stops being a function of its input. Caught by the etap-2 prefix probe's
// determinism check (2026-09-01): three builds from an identical 13,473,650-occluder array produced sat_tiles of
// 82.8% / 94.4% / 82.8%, and 2,029 tiles that all three agreed were saturated carried different barrier distances.
//
// Fixed by giving the compaction a separate destination array. Sources are then in 'recs' and destinations in
// 'packed' - two distinct allocations - so no task can write a region another task reads, whatever the survival rate,
// and no argument about index arithmetic is needed to see it. Costs one extra num_recs-sized buffer (~54MB on the
// reference scene, against the 512MB record scratch already held) and turns the memmove into a memcpy.
//
// Expect bandwidth, not thread count, to set the ceiling here: this is ~57MB of pure copy on the reference scene, and
// a handful of threads already saturate a desktop's memcpy bandwidth. The win is real but it will not be 12x.
class GsSatCompactTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		std::memcpy(dst, src, count * sizeof(GsSatOccluderRec)); // Disjoint arrays - see the note above.
	}

	GsSatOccluderRec* dst;
	const GsSatOccluderRec* src;
	size_t count;
};


// SESSION081 PHASE 1c: which strips each record reaches, and the per-strip index lists phase 2 walks instead of
// re-scanning the whole record array once per strip.
//
// The measurement that motivated this, from the 2026-09-01 sub sweep (three grid resolutions, which is what finally
// broke the collinearity between scan count and tile count and let the two be separated):
//
//   sub    strips  scan_iters  ->  scan     tile_iters -> tile      sum     measured dep_busy
//   0.30     12      27.0M        173ms       9.70M      167ms      340.6      340
//   0.25     10      17.9M        115ms       7.22M      124ms      239.5      241
//   0.20      8      10.0M         64ms       4.68M       81ms      145.1      144
//
// The strip scan is HALF of phase 2's work, at 6.4ns per iteration. Note what that number means: the scan is bound by
// its iteration count, not by the bytes it touches. That is the post-mortem on this session's rejected 8-byte
// strip-key experiment, which cut the scan's traffic from 57MB to 19MB and bought nothing measurable - it was the
// right target reached by the wrong mechanism.
//
// Cutting the count instead: 12 * num_recs iterations become one binning pass over num_recs plus each strip walking
// only its own ~9% of the records.
//
// Why this does NOT repeat the plan's stage 5 (slabs), which died because reordering destroyed the front-to-back
// order the deposit's saturation early-out depends on: nothing is reordered here. A strip's list is built in ascending
// record order, so within a strip the occluders arrive in exactly the sequence they did before, and the barrier is
// still each tile's first crossing. Bit-identical.
//
// The strip set is EXACT, not conservative. gsSatDepositRec() restricts a record to rows [max(v0_grid, v_lo),
// min(v1_grid, v_hi-1)] and returns when that is empty, so a record deposits into strip t precisely when
// [v0_grid, v1_grid] meets t's rows - i.e. for t in [strip_of_row[v0_grid], strip_of_row[v1_grid]], nothing wider and
// nothing narrower. v0_grid/v1_grid are computed here with the same two expressions the deposit uses, so the two agree
// by construction rather than by a second derivation that could drift from it.
struct GsSatBinTask : public glare::Task
{
	// Pass A counts, pass B (after the caller's prefix sum) scatters. The strip range is computed in A and kept in
	// rec_range so B does not recompute it - two bytes per record, against recomputing a clamp and two ceils.
	virtual void run(size_t /*thread_index*/)
	{
		if(counting)
		{
			for(int t=0; t<num_strips; ++t) counts[t] = 0;
			for(size_t i=i_begin; i<i_end; ++i)
			{
				const GsSatOccluderRec& rec = recs[i];
				const int v0 = myClamp(gsSatCeilToInt(rec.cv - rec.span - 0.5f), 0, res - 1); // Same two expressions as gsSatDepositRec().
				const int v1 = myClamp((int)         (rec.cv + rec.span - 0.5f), 0, res - 1);
				const int s0 = strip_of_row[v0], s1 = strip_of_row[v1];
				rec_range[i] = (uint16)(s0 | (s1 << 8));
				for(int t=s0; t<=s1; ++t) ++counts[t];
			}
		}
		else
		{
			for(size_t i=i_begin; i<i_end; ++i)
			{
				const uint16 r = rec_range[i];
				const int s0 = r & 0xFF, s1 = r >> 8;
				for(int t=s0; t<=s1; ++t) lists[offsets[t]++] = (uint32)i;
			}
		}
	}

	const GsSatOccluderRec* recs;
	const int* strip_of_row;
	uint16* rec_range;
	uint32* counts;  // Pass A output: this chunk's per-strip count. num_strips entries.
	uint32* offsets; // Pass B input/cursor: where this chunk's entries for each strip start. num_strips entries.
	uint32* lists;
	size_t i_begin, i_end;
	int res, num_strips;
	bool counting;
};


// SESSION081 PHASE 2 PREFETCH: how far ahead GsSatStripTask reads the next record. Same mechanism as stage 4B's
// gs_sat_gather_prefetch_dist, and the same warning applies: this wants to be SWEPT on the target, not reasoned to.
//
// The starting value is derived rather than guessed, so the sweep has somewhere sensible to start. In the gather each
// iteration is ~5-7ns and 32 was measured best; here an iteration is far heavier (a whole footprint, ~3.4 tile
// iterations), so once the miss is covered an iteration should land near ~10ns, and a DRAM latency of ~80ns needs only
// ~8 iterations of lead. 16 doubles that for slack, because 4B's sweep showed the wide end wins for a reason that
// applies here too: this build shares a pool with the traversal, and the deciding factor there was not the median but
// the SPREAD - a distance that only just covers the latency falls apart when the core is pulled away mid-loop.
//
// Too far is a real failure mode, not a free margin: ~10-12 line-fill buffers per core means prefetches issued too
// early are evicted before use, which is what made 64 worse than 32 in the gather.
static const size_t gs_sat_deposit_prefetch_dist = 16;


// SESSION079 phase 2: deposit every record that reaches this strip's rows.
//
// Strips own disjoint tiles, so no two of these touch the same accumulator entry - no locks, no atomics. Each strip
// reads the whole record array but the reject is now a single float compare against a span it does not have to derive,
// which is the entire point: the first cut of this had each strip redo the geometry, and that repeated work was 91% of
// the parallel build's time.
class GsSatStripTask : public glare::Task
{
public:
	virtual void run(size_t /*thread_index*/)
	{
		start_ms = clock->elapsed() * 1.0e3; // SESSION081 SCHEDULING PROBE - see GsSatRecordTask's start_ms for what this answers.
		if(list)
		{
			// SESSION081 PHASE 1c: no scan and no reject - every entry here is a record that reaches these rows, in
			// front-to-back order. See GsSatBinTask.
			//
			// SESSION081 PHASE 2 PREFETCH: recs[list[j]] is a scattered read the hardware prefetcher cannot follow -
			// the stride is whatever the list says - but the list itself is sequential, so the address wanted a few
			// iterations from now is already readable. Exactly the shape stage 4B fixed in the gather, and the
			// arithmetic says it is where this phase's time actually goes: on the reference build, 2.76M record visits
			// at a DRAM latency each is ~221ms against 9.39M tile iterations at ~2ns = ~19ms, summing to ~240ms against
			// a measured dep busy of 234ms. That is ~94% of phase 2 spent waiting for records rather than depositing
			// them, and it is why the strip lists do not shrink the cost by holding smaller records: consecutive
			// entries are ~20 records apart, which is 7.6 cache lines at 24B and still 3.8 at 12B, so the LINE COUNT
			// is unchanged either way. Latency, not bytes - so a prefetch, not a repack.
			for(size_t j=0; j<list_len; ++j)
			{
				if(j + gs_sat_deposit_prefetch_dist < list_len)
					_mm_prefetch((const char*)&recs[list[j + gs_sat_deposit_prefetch_dist]], _MM_HINT_T0);

				gsSatDepositRec(recs[list[j]], res, remaining_threshold, sat_depth, accum, NULL, false, v_lo, v_hi, &stats);
			}
		}
		else
		{
			// Single strip: the list would hold every record, so the binning is pure overhead and this walks the array.
			const float lo = (float)v_lo, hi = (float)v_hi;
			for(size_t i=0; i<num_recs; ++i)
			{
				const GsSatOccluderRec& rec = recs[i];
				if(rec.cv + rec.span < lo || rec.cv - rec.span >= hi)
					continue; // Footprint cannot reach this strip. Exact, not a bound - span is the real one.
				gsSatDepositRec(rec, res, remaining_threshold, sat_depth, accum, NULL, false, v_lo, v_hi, &stats);
			}
		}
		end_ms = clock->elapsed() * 1.0e3;
	}

	// SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC. Phase 2 is measured alongside phase 1 because the suspected
	// cause - this build sharing a task pool with the traversal that spawned it - would hit both, and telling "the whole
	// stage is starved" apart from "phase 1 specifically is" costs nothing extra here but a whole measurement round if
	// asked later. Same shared clock, same reasoning: see GsSatRecordTask's copy of these fields.
	double start_ms, end_ms;
	const Timer* clock;

	const GsSatOccluderRec* recs;
	const uint32* list; // SESSION081 PHASE 1c: this strip's records, ascending. NULL for the single-strip case - see run().
	size_t list_len;
	size_t num_recs;
	int res;
	float remaining_threshold;
	float *sat_depth, *accum;
	int v_lo, v_hi;
	GsSatStripStats stats;
};


void gsBuildSaturationGridParallel(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius, int closing_radius_tiles,
	js::Vector<float, 16>& sat_depth_out, glare::TaskManager& task_manager,
	size_t* out_writers, size_t* out_tile_writes, size_t* out_tile_stats,
	GsSatBuildScratch* scratch, size_t* out_erode_stats,
	double* out_close_ms, double* out_erode_ms, double* out_rec_ms, double* out_dep_ms,
	size_t* out_num_recs, int* out_num_strips, size_t* out_num_blocks,
	size_t* out_gate1_survivors, // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC.
	double* out_rec_task_ms_min, double* out_rec_task_ms_max, double* out_rec_task_ms_mean, // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC.
	size_t* out_inf_hist, // SESSION082 PLAN §1.1, TEMPORARY DIAGNOSTIC - 5 counters, layout documented in the header.
	double* out_sched_stats) // SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC - 9 doubles, layout documented in the header.
{
	const size_t num_tiles = (size_t)res * (size_t)res;
	sat_depth_out.resizeNoCopy(num_tiles);
	for(size_t i=0; i<num_tiles; ++i)
		sat_depth_out[i] = std::numeric_limits<float>::infinity();

	const int concurrency = myMax(1, (int)task_manager.getConcurrency());
	const float remaining_threshold = myClamp(1.f - saturation_threshold, 0.f, 1.f);

	// SESSION081: was clamp(res/8, 1, concurrency) - one strip per thread, and a thickness the per-strip scan (now gone,
	// see GsSatBinTask) used to justify. Both halves are now derived from the thread count instead - see
	// gsSatDepositStripCount() for the sweep behind it and for why a number tuned on this desktop is the wrong shape of
	// answer for a client that also runs on phones.
	const int num_strips = gsSatDepositStripCount(res, concurrency);
	if(out_num_strips) *out_num_strips = num_strips;
	double rec_ms_total = 0.0, dep_ms_total = 0.0;

	// SESSION081 SCHEDULING PROBE, TEMPORARY DIAGNOSTIC - see GsSatRecordTask::start_ms for the question these answer.
	double rec_start_max_total = 0.0, rec_start_sum_total = 0.0, rec_busy_sum_total = 0.0, rec_group_wall_total = 0.0;
	double dep_start_max_total = 0.0, dep_busy_sum_total = 0.0, move_ms_total = 0.0, bin_ms_total = 0.0;
	size_t rec_chunk_n_total = 0, dep_strip_n_total = 0;
	size_t num_recs_total = 0;
	size_t num_blocks_total = 0; // SESSION081 PLAN, ETAP 0 follow-up DIAGNOSTIC: how many blocks (and therefore task-group launch pairs) this build actually ran - see gs_sat_rec_scratch_bytes's ceiling-measurement comment.
	size_t gate1_survivors_total = 0; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC - see GsSatRecordTask::gate1_survivors.
	double rec_task_ms_min_total = std::numeric_limits<double>::infinity(), rec_task_ms_max_total = 0.0, rec_task_ms_sum_total = 0.0;
	size_t rec_task_n_total = 0; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC - see GsSatRecordTask::task_ms.

	// The occluders are processed in BLOCKS rather than all at once, to bound the record scratch - see
	// gs_sat_rec_scratch_bytes for the trade-off and the numbers behind the budget.
	//
	// Blocking costs nothing in correctness: blocks are processed in order and records within a block are in order, so
	// each tile still sees its occluders front-to-back.
	const size_t block_n = myMin(n, myMax((size_t)65536, gs_sat_rec_scratch_bytes / sizeof(GsSatOccluderRec)));

	// SESSION081: the working buffers, from the caller when it supplied a scratch and freshly allocated when it did not
	// - see GsSatBuildScratch for why supplying one matters (the fresh path re-faults ~54MB of demand-zero pages on
	// every build, inside rec_ms, where it reads as work). Behaviour is identical either way; only the allocation is.
	GsSatBuildScratch local_scratch;
	GsSatBuildScratch& s = scratch ? *scratch : local_scratch;

	js::Vector<GsSatOccluderRec, 16>& recs = s.recs;
	recs.resizeNoCopy(block_n);

	// SESSION081 BUGFIX: the compaction's destination, deliberately a DIFFERENT array from the one phase 1 writes -
	// see GsSatCompactTask for the in-place version's write-before-read hazard and how it showed up. Everything
	// downstream of the compaction (binning, deposit) reads this one, never 'recs'.
	js::Vector<GsSatOccluderRec, 16>& packed = s.packed;

	// SESSION081 PHASE 1c: scratch for the per-strip record lists - see GsSatBinTask. strip_of_row is the strip that
	// owns each grid row, which is what makes a record's strip range two array lookups instead of a search.
	js::Vector<int, 16>&    strip_of_row = s.strip_of_row;
	js::Vector<uint16, 16>& rec_range    = s.rec_range;
	js::Vector<uint32, 16>& bin_counts   = s.bin_counts;
	js::Vector<uint32, 16>& strip_lists  = s.strip_lists;
	js::Vector<uint32, 16>& strip_list_begin = s.strip_list_begin;
	strip_of_row.resizeNoCopy(res); // Was a sized construction; now a resize, since a reused buffer may already hold a previous build's rows.
	for(int t=0; t<num_strips; ++t)
	{
		const int v_lo = (int)(((int64)res * t)       / num_strips);
		const int v_hi = (int)(((int64)res * (t + 1)) / num_strips);
		for(int v=v_lo; v<v_hi; ++v)
			strip_of_row[v] = t;
	}

	// SESSION081: from the scratch too, so it is not reallocated per build. A reused buffer holds the previous build's
	// values, so the fill is explicit rather than a side effect of construction - it was never optional, only implicit.
	js::Vector<float, 16>& accum = s.accum;
	accum.resizeNoCopy(num_tiles);
	for(size_t i=0; i<num_tiles; ++i)
		accum[i] = 1.f;

	GsSatStripStats total;

	for(size_t block_begin=0; block_begin<n; block_begin += block_n)
	{
		++num_blocks_total; // SESSION081 PLAN, ETAP 0 follow-up DIAGNOSTIC.
		const size_t block_end = myMin(block_begin + block_n, n);
		const size_t block_len = block_end - block_begin;

		// ---- Phase 1: geometry, once, in parallel over node ranges within this block. ----
		Timer rec_timer; // SESSION081 PLAN ETAP 0: covers the task group AND the compaction memmove below - both are phase 1's cost, not phase 2's.
		// SESSION081: 4 chunks per thread, not one. The input is sorted FRONT-TO-BACK and the expensive path (sqrt, oct
		// unwrap, local tile angle, amplitude) is only taken by occluders that clear the cheap bound - which the near
		// ones do and the far ones do not. So an equal-COUNT split is not an equal-WORK split: measured 2026-09-01, the
		// slowest of 17 chunks took 30.3ms against a 16.5ms mean, and runTaskGroup() waits for it while the other 16
		// idle. Oversubscribing lets the pool work-steal the tail instead. Same shape (and the same *4) the occluder
		// gather next door already uses, for the same reason.
		const int num_chunks = (int)myMin((size_t)concurrency * 4, block_len);
		js::Vector<Reference<GsSatRecordTask>, 16> rec_tasks(num_chunks);
		Timer rec_group_clock; // SESSION081 SCHEDULING PROBE: ONE clock every chunk reads - see GsSatRecordTask::start_ms. Reset immediately before the launch below so its zero point is the launch, not the task allocation loop.
		{
			glare::TaskGroupRef group = new glare::TaskGroup();
			for(int c=0; c<num_chunks; ++c)
			{
				Reference<GsSatRecordTask> t = new GsSatRecordTask();
				t->clock = &rec_group_clock; // SESSION081 SCHEDULING PROBE.
				t->start_ms = t->end_ms = 0.0;
				t->px = px; t->py = py; t->pz = pz; t->radius = radius; t->alpha = alpha;
				t->i_begin = block_begin + (block_len * (size_t)c)       / (size_t)num_chunks;
				t->i_end   = block_begin + (block_len * (size_t)(c + 1)) / (size_t)num_chunks;
				t->count = 0;
				t->anchor_pos_ws = anchor_pos_ws;
				t->res = res;
				t->region_radius = region_radius;
				t->out = recs.data() + (t->i_begin - block_begin); // Own slice, sized to its input range - see GsSatRecordTask.
				rec_tasks[c] = t;
				group->tasks.push_back(t);
			}
			rec_group_clock.reset(); // SESSION081 SCHEDULING PROBE: zero the shared clock at the launch itself.
			task_manager.runTaskGroup(group);
			rec_group_wall_total += rec_group_clock.elapsed() * 1.0e3;
		}

		// Close the gaps the chunks left, copying into 'packed'. Destinations are assigned in ascending chunk order, so
		// the records come out front-to-back exactly as the serial version left them; the copies are mutually
		// independent because source and destination are separate arrays - see GsSatCompactTask, including what the
		// in-place version of this got wrong.
		//
		// Counting has to finish before any copy starts (a chunk's destination is the sum of every earlier chunk's
		// count), so the task-building loop below runs to completion first and the group is launched after it.
		size_t num_recs = 0;
		for(int c=0; c<num_chunks; ++c)
			num_recs += rec_tasks[c]->count;
		packed.resizeNoCopy(num_recs);

		glare::TaskGroupRef move_group = new glare::TaskGroup();
		size_t packed_pos = 0;
		for(int c=0; c<num_chunks; ++c)
		{
			const GsSatRecordTask& t = *rec_tasks[c];
			const size_t slice_begin = t.i_begin - block_begin;
			if(t.count > 0)
			{
				Reference<GsSatCompactTask> mt = new GsSatCompactTask();
				mt->dst = packed.data() + packed_pos;
				mt->src = recs.data() + slice_begin;
				mt->count = t.count;
				move_group->tasks.push_back(mt);
			}
			packed_pos += t.count;
			rec_start_max_total = myMax(rec_start_max_total, t.start_ms); // SESSION081 SCHEDULING PROBE.
			rec_start_sum_total += t.start_ms;
			rec_busy_sum_total  += t.end_ms - t.start_ms;
			++rec_chunk_n_total;
			gate1_survivors_total += t.gate1_survivors; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC.
			rec_task_ms_min_total = myMin(rec_task_ms_min_total, t.task_ms); // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC.
			rec_task_ms_max_total = myMax(rec_task_ms_max_total, t.task_ms);
			rec_task_ms_sum_total += t.task_ms;
			++rec_task_n_total;
		}
		Timer move_timer; // SESSION081: times the compaction alone, as the probe's serial version did, so the before/after is like for like.
		if(move_group->tasks.size() > 0)
			task_manager.runTaskGroup(move_group);
		move_ms_total += move_timer.elapsed() * 1.0e3;
		rec_ms_total += rec_timer.elapsed() * 1.0e3;
		num_recs_total += num_recs;
		if(num_recs == 0)
			continue;

		// ---- Phase 1c: bin the records by strip, so phase 2 walks lists instead of re-scanning. See GsSatBinTask. ----
		Timer bin_timer;
		const bool want_bins = num_strips > 1;
		if(want_bins)
		{
			rec_range.resizeNoCopy(num_recs);
			const int num_bin_chunks = (int)myMin((size_t)concurrency * 4, num_recs);
			bin_counts.resizeNoCopy((size_t)num_bin_chunks * (size_t)num_strips);

			// A Task carries its own queue links and a pointer to the group it belongs to, so the two passes get
			// SEPARATE task objects rather than the same ones pushed into a second group. 2*num_bin_chunks small
			// allocations against a pass over millions of records is not worth reasoning about task reuse for.
			glare::TaskGroupRef count_group = new glare::TaskGroup();
			for(int c=0; c<num_bin_chunks; ++c)
			{
				Reference<GsSatBinTask> t = new GsSatBinTask();
				t->recs = packed.data(); // SESSION081 BUGFIX: the compaction OUTPUT - see GsSatCompactTask.
				t->strip_of_row = strip_of_row.data();
				t->rec_range = rec_range.data();
				t->counts = bin_counts.data() + (size_t)c * (size_t)num_strips;
				t->offsets = t->counts; // Rewritten into absolute cursors by the prefix sum below, then consumed by pass B.
				t->lists = NULL;
				t->i_begin = (num_recs * (size_t)c)       / (size_t)num_bin_chunks;
				t->i_end   = (num_recs * (size_t)(c + 1)) / (size_t)num_bin_chunks;
				t->res = res;
				t->num_strips = num_strips;
				t->counting = true;
				count_group->tasks.push_back(t);
			}
			task_manager.runTaskGroup(count_group);

			// Prefix sum, strip-major then chunk-major: strip t's entries are contiguous, and within them the chunks
			// appear in ascending order - which is what keeps each list front-to-back. Serial, but it is only
			// num_bin_chunks*num_strips entries (~816).
			strip_list_begin.resizeNoCopy((size_t)num_strips + 1);
			size_t running = 0;
			for(int t=0; t<num_strips; ++t)
			{
				strip_list_begin[t] = (uint32)running;
				for(int c=0; c<num_bin_chunks; ++c)
				{
					const size_t idx = (size_t)c * (size_t)num_strips + (size_t)t;
					const uint32 cnt = bin_counts[idx];
					bin_counts[idx] = (uint32)running; // Becomes this chunk's write cursor for strip t.
					running += cnt;
				}
			}
			strip_list_begin[num_strips] = (uint32)running;

			strip_lists.resizeNoCopy(running);
			glare::TaskGroupRef scatter_group = new glare::TaskGroup();
			for(int c=0; c<num_bin_chunks; ++c)
			{
				Reference<GsSatBinTask> t = new GsSatBinTask();
				t->recs = packed.data(); // SESSION081 BUGFIX: the compaction OUTPUT - see GsSatCompactTask.
				t->strip_of_row = strip_of_row.data();
				t->rec_range = rec_range.data();
				t->counts = NULL;
				t->offsets = bin_counts.data() + (size_t)c * (size_t)num_strips; // Now this chunk's write cursors.
				t->lists = strip_lists.data();
				t->i_begin = (num_recs * (size_t)c)       / (size_t)num_bin_chunks;
				t->i_end   = (num_recs * (size_t)(c + 1)) / (size_t)num_bin_chunks;
				t->res = res;
				t->num_strips = num_strips;
				t->counting = false;
				scatter_group->tasks.push_back(t);
			}
			task_manager.runTaskGroup(scatter_group);
		}
		bin_ms_total += bin_timer.elapsed() * 1.0e3;

		// ---- Phase 2: deposit, in parallel over strips. ----
		Timer dep_timer; // SESSION081 PLAN ETAP 0.
		Timer dep_group_clock; // SESSION081 SCHEDULING PROBE: phase 2's shared clock - see GsSatStripTask::clock.
		glare::TaskGroupRef group = new glare::TaskGroup();
		js::Vector<Reference<GsSatStripTask>, 16> strip_tasks(num_strips);
		for(int t=0; t<num_strips; ++t)
		{
			const int v_lo = (int)(((int64)res * t)       / num_strips);
			const int v_hi = (int)(((int64)res * (t + 1)) / num_strips);

			Reference<GsSatStripTask> task = new GsSatStripTask();
			task->clock = &dep_group_clock; // SESSION081 SCHEDULING PROBE.
			task->start_ms = task->end_ms = 0.0;
			task->recs = packed.data(); // SESSION081 BUGFIX: the compaction OUTPUT - see GsSatCompactTask.
			task->list     = want_bins ? (strip_lists.data() + strip_list_begin[t]) : NULL; // SESSION081 PHASE 1c.
			task->list_len = want_bins ? (size_t)(strip_list_begin[t + 1] - strip_list_begin[t]) : 0;
			task->num_recs = num_recs;
			task->res = res;
			task->remaining_threshold = remaining_threshold;
			task->sat_depth = sat_depth_out.data();
			task->accum = accum.data() + (size_t)v_lo * (size_t)res; // Strip-local rows start at v_lo - see gsSatDepositRec().
			task->v_lo = v_lo;
			task->v_hi = v_hi;
			strip_tasks[t] = task;
			group->tasks.push_back(task);
		}

		// runTaskGroup() processes work on the CALLING thread as well as the pool's, and steals back any of its own tasks
		// the pool has not picked up. That is what makes this safe to call from inside the traversal task, which is itself
		// running on this same TaskManager - there is no configuration in which it waits on a thread that never arrives.
		// It is also why a build degrades gracefully to serial where there are no worker threads, with no separate path.
		dep_group_clock.reset(); // SESSION081 SCHEDULING PROBE: zero at the launch itself, as phase 1 does.
		task_manager.runTaskGroup(group);
		dep_ms_total += dep_timer.elapsed() * 1.0e3;

		for(int t=0; t<num_strips; ++t)
		{
			dep_start_max_total = myMax(dep_start_max_total, strip_tasks[t]->start_ms); // SESSION081 SCHEDULING PROBE.
			dep_busy_sum_total += strip_tasks[t]->end_ms - strip_tasks[t]->start_ms;
			++dep_strip_n_total;
			total.writers     += strip_tasks[t]->stats.writers;
			total.tile_writes += strip_tasks[t]->stats.tile_writes;
			total.tile_iters  += strip_tasks[t]->stats.tile_iters;
			total.tail_rej    += strip_tasks[t]->stats.tail_rej;
			total.sat_skip    += strip_tasks[t]->stats.sat_skip;
		}

	}

	if(out_rec_ms) *out_rec_ms = rec_ms_total;
	if(out_dep_ms) *out_dep_ms = dep_ms_total;
	if(out_sched_stats) // SESSION081 SCHEDULING PROBE - layout documented in the header.
	{
		out_sched_stats[0] = rec_start_max_total;
		out_sched_stats[1] = rec_chunk_n_total > 0 ? (rec_start_sum_total / (double)rec_chunk_n_total) : 0.0;
		out_sched_stats[2] = rec_busy_sum_total;
		out_sched_stats[3] = rec_group_wall_total;
		out_sched_stats[4] = move_ms_total;
		out_sched_stats[5] = dep_start_max_total;
		out_sched_stats[6] = dep_busy_sum_total;
		out_sched_stats[7] = (double)concurrency;
		out_sched_stats[8] = bin_ms_total; // SESSION081 PHASE 1c - what the binning ADDS, against the scan it removes.
	}
	if(out_num_recs) *out_num_recs = num_recs_total;
	if(out_num_blocks) *out_num_blocks = num_blocks_total;
	if(out_gate1_survivors) *out_gate1_survivors = gate1_survivors_total; // SESSION081 PLAN, PRE-REJECT PROBE, TEMPORARY DIAGNOSTIC.
	if(out_rec_task_ms_min) *out_rec_task_ms_min = (rec_task_n_total > 0) ? rec_task_ms_min_total : 0.0; // SESSION081 PLAN, REC-BALANCE PROBE, TEMPORARY DIAGNOSTIC.
	if(out_rec_task_ms_max) *out_rec_task_ms_max = rec_task_ms_max_total;
	if(out_rec_task_ms_mean) *out_rec_task_ms_mean = (rec_task_n_total > 0) ? (rec_task_ms_sum_total / (double)rec_task_n_total) : 0.0;

	// NOTE: `writers` counts records that reached the tile loop, so one straddling a strip boundary is counted once per
	// strip it touches - the parallel total is the serial one plus the boundary crossings. The per-TILE counters
	// (tile_writes, tile_iters, tail_rej, sat_skip) are exact, and tile_writes is the one to check against the serial
	// build.
	if(out_writers)     *out_writers     = total.writers;
	if(out_tile_writes) *out_tile_writes = total.tile_writes;
	if(out_tile_stats) { out_tile_stats[0] = total.tile_iters; out_tile_stats[1] = total.tail_rej; out_tile_stats[2] = total.sat_skip; }

	// SESSION082 PLAN §1.1, TEMPORARY DIAGNOSTIC: what the unsaturated (+inf) tiles are MADE OF - see out_inf_hist's
	// header comment for the question. Placed here, after every block has deposited but before closing/erosion, because
	// both of those rewrite +inf tiles: this has to see the tiles as the deposit actually left them.
	//
	// accum is the right source and needs no extra bookkeeping: the deposit stops updating a tile once it saturates
	// (keep_accumulating_past_saturation is false on this path), but every tile counted here never saturated, so its
	// accum is the complete accumulation of everything that reached it. accum == 1 exactly means no occluder ever
	// deposited into the tile at all - float equality is the intended test, not a tolerance: the value is the untouched
	// initialiser, not the result of arithmetic that might land near it.
	if(out_inf_hist)
	{
		for(int i=0; i<5; ++i) out_inf_hist[i] = 0;
		const float inf = std::numeric_limits<float>::infinity();
		const float sat_thresh = 1.f - remaining_threshold; // The live saturation_threshold, after the same clamp the deposit used.
		for(size_t t=0; t<num_tiles; ++t)
		{
			if(sat_depth_out[t] != inf)
				continue;
			const float a = accum[t];
			if(a >= 1.f) { ++out_inf_hist[0]; continue; } // Nothing was ever deposited here.
			const float progress = (sat_thresh > 0.f) ? ((1.f - a) / sat_thresh) : 0.f;
			const int bin = myClamp(1 + (int)(progress * 4.f), 1, 4); // Four equal quarters of the way to the threshold; clamped because progress is in [0, 1) only up to float rounding.
			++out_inf_hist[bin];
		}
	}

	// SESSION080: after every block has deposited, never per block - the mask has to be complete before it is eroded,
	// or a silhouette would be measured against a half-built neighbourhood.
	gsSatApplyRegionErosion(sat_depth_out, res, region_radius, closing_radius_tiles, &task_manager, out_erode_stats, out_close_ms, out_erode_ms);
}


// SESSION085 ETAP 3 - see the header for why this is the right question for a bias and the wrong one for a prune.
float gsSatBuriedRatioSq(const Vec4f& offset, float dist_sq, const js::Vector<float, 16>& sat_depth, int res)
{
	if(res == 0)
		return 0.f; // No grid built - never bias. Same convention as gsSatOccluded().

	if(dist_sq < 1.0e-12f)
		return 0.f; // Degenerate: node at the anchor, no meaningful direction.

	const float sat_d = sat_depth[gsSatGridTileForDir(offset, res)];
	if(sat_d == std::numeric_limits<float>::infinity())
		return 0.f; // This direction never saturates.

	const float thresh_sq = sat_d * sat_d;
	if(dist_sq <= thresh_sq)
		return 0.f; // In front of the barrier, or on it.

	return dist_sq / thresh_sq;
}


GsSatGridStats::GsSatGridStats()
:	num_tiles(0), num_finite(0), d_p10(0.f), d_med(0.f), d_p90(0.f), d_max(0.f),
	num_pairs(0), num_edge_pairs(0), num_both_finite(0),
	adj_p50(1.f), adj_p90(1.f), adj_max(1.f), num_adj_gt2(0), num_adj_gt4(0), max_window_frac(0.f)
{}


// SESSION088 DIAGNOSTIC - see the header for what this is measuring and why.
void gsSatGridStats(const js::Vector<float, 16>& sat_depth, int res, GsSatGridStats& out)
{
	out = GsSatGridStats();
	if(res <= 2 || sat_depth.size() != (size_t)res * (size_t)res)
		return; // No grid, or a res too small to have any interior pair at all.

	out.num_tiles = sat_depth.size();

	// Depth percentiles over the finite tiles. Collected into a scratch copy and sorted outright rather than estimated -
	// this runs twice a second at most, and an exact median is worth more than the microseconds a histogram would save.
	std::vector<float> finite;
	finite.reserve(sat_depth.size());
	for(size_t t=0; t<sat_depth.size(); ++t)
		if(std::isfinite(sat_depth[t]))
			finite.push_back(sat_depth[t]);

	out.num_finite = finite.size();
	if(!finite.empty())
	{
		std::sort(finite.begin(), finite.end());
		const size_t last = finite.size() - 1;
		out.d_p10 = finite[(size_t)(0.10 * (double)last)];
		out.d_med = finite[(size_t)(0.50 * (double)last)];
		out.d_p90 = finite[(size_t)(0.90 * (double)last)];
		out.d_max = finite[last];
	}

	// Adjacent-tile contrast. Right and down neighbours only, so each pair is visited exactly once; the outermost ring is
	// skipped rather than wrapped - see the header for why that is the honest choice on an octahedral map.
	std::vector<float> ratios;
	ratios.reserve(sat_depth.size() * 2);
	size_t num_equal = 0;
	for(int v=1; v<res-1; ++v)
		for(int u=1; u<res-1; ++u)
		{
			const float here = sat_depth[(size_t)v * (size_t)res + (size_t)u];
			const float nbr[2] = {
				sat_depth[(size_t)v       * (size_t)res + (size_t)(u + 1)],
				sat_depth[(size_t)(v + 1) * (size_t)res + (size_t)u]
			};
			for(int k=0; k<2; ++k)
			{
				const float other = nbr[k];
				const bool a_fin = std::isfinite(here), b_fin = std::isfinite(other);
				++out.num_pairs;
				if(a_fin != b_fin)
				{
					++out.num_edge_pairs; // Mask silhouette: one side biased, the other not touched at all.
					continue;
				}
				if(!a_fin)
					continue; // Both +inf - no bias on either side, nothing to contrast.

				++out.num_both_finite;
				if(here == other)
					++num_equal; // Exactly equal: the max-window's signature - see max_window_frac.

				const float lo = myMin(here, other), hi = myMax(here, other);
				const float ratio = (lo > 0.f) ? (hi / lo) : 1.f;
				ratios.push_back(ratio);
				if(ratio > 2.f) ++out.num_adj_gt2;
				if(ratio > 4.f) ++out.num_adj_gt4;
			}
		}

	if(!ratios.empty())
	{
		std::sort(ratios.begin(), ratios.end());
		const size_t last = ratios.size() - 1;
		out.adj_p50 = ratios[(size_t)(0.50 * (double)last)];
		out.adj_p90 = ratios[(size_t)(0.90 * (double)last)];
		out.adj_max = ratios[last];
		out.max_window_frac = (float)((double)num_equal / (double)ratios.size());
	}
}


// SESSION085 ETAP 6: gsSatOccluded() - the prune's per-node verdict - is gone with the prune. The barrier is read
// by gsSatBuriedRatioSq() above now, which asks a different question (see its comment) and is the only reader left.
