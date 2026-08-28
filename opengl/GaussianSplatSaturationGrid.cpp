/*=====================================================================
GaussianSplatSaturationGrid.cpp
---------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatSaturationGrid.h"


#include "../maths/mathstypes.h"
#include "../utils/TaskManager.h" // SESSION079: gsBuildSaturationGridParallel().
#include "../utils/Reference.h"
#include <cstring> // SESSION079: memmove() in gsBuildSaturationGridParallel().
#include <cmath>
#include <limits>


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


int gsSatGridResForFocal(float focal_px, float coarse_pixel_scale, float tile_subdiv)
{
	// SESSION074: derive tile angular size from the existing coarse_pixel_scale/focal_px knobs rather than exposing a
	// new UI parameter - project rule, no manual per-scene tuning. A coarse node subtends about
	// coarse_pixel_scale/focal_px radians; the subdivision sizes the tile against that.
	// SESSION076 CALIBRATION: tile_subdiv replaces the fixed K - see the header.
	const float safe_focal = myMax(focal_px, 1.f);
	const float tile_ang = myMax(coarse_pixel_scale, 1.0e-3f) / (safe_focal * myMax(tile_subdiv, 0.01f));

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


// SESSION079: floor on how thin a strip gsBuildSaturationGridParallel() will cut. Every strip walks the WHOLE occluder
// array - that is the price of not reordering anything - so thinner strips buy less tile work each while paying the same
// walk. 8 rows keeps a strip's two accumulators (8*res*4 bytes each) comfortably inside L1 at any res this stage
// produces, which is the property the split exists for.
static const int gs_sat_min_strip_rows = 8;


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
static const size_t gs_sat_rec_scratch_bytes = 32 * 1024 * 1024;


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
static void gsSatApplyRegionErosion(js::Vector<float, 16>& sat_depth, int res,
	float region_radius, glare::TaskManager* task_manager, size_t* out_erode_stats)
{
	if(out_erode_stats)
		out_erode_stats[0] = out_erode_stats[1] = out_erode_stats[2] = 0;
	if(!(region_radius > 0.f) || res == 0)
		return;

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

	sat_depth = eroded;
}


void gsBuildSaturationGrid(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers, size_t* out_tile_writes,
	js::Vector<float, 16>* out_accum_t, js::Vector<float, 16>* out_amp_sum, size_t* out_tile_stats,
	size_t* out_erode_stats)
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
	gsSatApplyRegionErosion(sat_depth_out, res, region_radius, NULL, out_erode_stats);

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
		const float amp_reject_k = 3.14159265f * (float)res * (float)res / 18.f;
		size_t w = 0;
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
	}

	const float *px, *py, *pz, *radius, *alpha;
	size_t i_begin, i_end, count;
	Vec4f anchor_pos_ws;
	int res;
	float region_radius;
	GsSatOccluderRec* out;
};


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
		const float lo = (float)v_lo, hi = (float)v_hi;
		for(size_t i=0; i<num_recs; ++i)
		{
			const GsSatOccluderRec& rec = recs[i];
			if(rec.cv + rec.span < lo || rec.cv - rec.span >= hi)
				continue; // Footprint cannot reach this strip. Exact, not a bound - span is the real one.
			gsSatDepositRec(rec, res, remaining_threshold, sat_depth, accum, NULL, false, v_lo, v_hi, &stats);
		}
	}

	const GsSatOccluderRec* recs;
	size_t num_recs;
	int res;
	float remaining_threshold;
	float *sat_depth, *accum;
	int v_lo, v_hi;
	GsSatStripStats stats;
};


void gsBuildSaturationGridParallel(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius,
	js::Vector<float, 16>& sat_depth_out, glare::TaskManager& task_manager,
	size_t* out_writers, size_t* out_tile_writes, size_t* out_tile_stats,
	js::Vector<GsSatOccluderRec, 16>* scratch_recs, size_t* out_erode_stats)
{
	const size_t num_tiles = (size_t)res * (size_t)res;
	sat_depth_out.resizeNoCopy(num_tiles);
	for(size_t i=0; i<num_tiles; ++i)
		sat_depth_out[i] = std::numeric_limits<float>::infinity();

	const int concurrency = myMax(1, (int)task_manager.getConcurrency());
	const float remaining_threshold = myClamp(1.f - saturation_threshold, 0.f, 1.f);

	const int num_strips = myClamp(res / gs_sat_min_strip_rows, 1, concurrency);

	// The occluders are processed in BLOCKS rather than all at once, to bound the record scratch - see
	// gs_sat_rec_scratch_bytes for the trade-off and the numbers behind the budget.
	//
	// Blocking costs nothing in correctness: blocks are processed in order and records within a block are in order, so
	// each tile still sees its occluders front-to-back.
	const size_t block_n = myMin(n, myMax((size_t)65536, gs_sat_rec_scratch_bytes / sizeof(GsSatOccluderRec)));

	js::Vector<GsSatOccluderRec, 16> local_recs;
	js::Vector<GsSatOccluderRec, 16>& recs = scratch_recs ? *scratch_recs : local_recs;
	recs.resizeNoCopy(block_n);

	js::Vector<float, 16> accum(num_tiles, 1.f);

	GsSatStripStats total;

	for(size_t block_begin=0; block_begin<n; block_begin += block_n)
	{
		const size_t block_end = myMin(block_begin + block_n, n);
		const size_t block_len = block_end - block_begin;

		// ---- Phase 1: geometry, once, in parallel over node ranges within this block. ----
		const int num_chunks = (int)myMin((size_t)concurrency, block_len);
		js::Vector<Reference<GsSatRecordTask>, 16> rec_tasks(num_chunks);
		{
			glare::TaskGroupRef group = new glare::TaskGroup();
			for(int c=0; c<num_chunks; ++c)
			{
				Reference<GsSatRecordTask> t = new GsSatRecordTask();
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
			task_manager.runTaskGroup(group);
		}

		// Close the gaps the chunks left, in ascending chunk order so the records stay front-to-back. Each chunk's block
		// only ever moves to a LOWER offset, so the copies never overlap forwards.
		size_t num_recs = 0;
		for(int c=0; c<num_chunks; ++c)
		{
			const GsSatRecordTask& t = *rec_tasks[c];
			const size_t slice_begin = t.i_begin - block_begin;
			if(t.count > 0 && num_recs != slice_begin)
				std::memmove(recs.data() + num_recs, recs.data() + slice_begin, t.count * sizeof(GsSatOccluderRec));
			num_recs += t.count;
		}
		if(num_recs == 0)
			continue;

		// ---- Phase 2: deposit, in parallel over strips. ----
		glare::TaskGroupRef group = new glare::TaskGroup();
		js::Vector<Reference<GsSatStripTask>, 16> strip_tasks(num_strips);
		for(int t=0; t<num_strips; ++t)
		{
			const int v_lo = (int)(((int64)res * t)       / num_strips);
			const int v_hi = (int)(((int64)res * (t + 1)) / num_strips);

			Reference<GsSatStripTask> task = new GsSatStripTask();
			task->recs = recs.data();
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
		task_manager.runTaskGroup(group);

		for(int t=0; t<num_strips; ++t)
		{
			total.writers     += strip_tasks[t]->stats.writers;
			total.tile_writes += strip_tasks[t]->stats.tile_writes;
			total.tile_iters  += strip_tasks[t]->stats.tile_iters;
			total.tail_rej    += strip_tasks[t]->stats.tail_rej;
			total.sat_skip    += strip_tasks[t]->stats.sat_skip;
		}

	}

	// NOTE: `writers` counts records that reached the tile loop, so one straddling a strip boundary is counted once per
	// strip it touches - the parallel total is the serial one plus the boundary crossings. The per-TILE counters
	// (tile_writes, tile_iters, tail_rej, sat_skip) are exact, and tile_writes is the one to check against the serial
	// build.
	if(out_writers)     *out_writers     = total.writers;
	if(out_tile_writes) *out_tile_writes = total.tile_writes;
	if(out_tile_stats) { out_tile_stats[0] = total.tile_iters; out_tile_stats[1] = total.tail_rej; out_tile_stats[2] = total.sat_skip; }

	// SESSION080: after every block has deposited, never per block - the mask has to be complete before it is eroded,
	// or a silhouette would be measured against a half-built neighbourhood.
	gsSatApplyRegionErosion(sat_depth_out, res, region_radius, &task_manager, out_erode_stats);
}


bool gsSatOccluded(const Vec4f& offset, float dist_sq, float node_radius,
	const js::Vector<float, 16>& sat_depth, int res, float region_radius, bool* out_aggressive)
{
	if(out_aggressive)
		*out_aggressive = false;

	if(res == 0)
		return false; // No grid built - never drop. See the header.

	if(dist_sq < 1.0e-12f) // Degenerate: node at the anchor, no meaningful direction.
		return false;

	// SESSION074: the node's own angular radius, same ratio-instead-of-asin form the build pass uses (see its comment)
	// - but here it costs no sqrt at all, because the span only needs the ratio and the depth test below is squared.
	// node_radius^2 / dist_sq is the squared ratio; comparing spans in tile units needs the ratio itself, so one
	// reciprocal-sqrt's worth of work is the whole trigonometric cost of this function.
	const float inv_dist = 1.f / std::sqrt(dist_sq);
	// SESSION078: region pruning - the node may only be dropped if it is occluded from EVERY camera position in a ball
	// of region_radius around the anchor. Moving the viewpoint by R sweeps this node's apparent direction over an extra
	// R/d, so the set of tiles that must all agree is its footprint DILATED by that - the exact mirror of the build
	// side's erosion (see gsBuildSaturationGrid()). Both sides widen the conservatism; neither can drop more than the
	// R = 0 case. See the header.
	const float ang_radius = (node_radius + region_radius) * inv_dist;

	float cu, cv;
	const float radius_tiles = gsSatGridFootprint(offset, ang_radius, res, inv_dist, cu, cv);

	// SESSION074: the read side takes every tile the footprint TOUCHES - the opposite rounding from the build pass, see
	// gsSatGridFootprint()'s comment. floor() on both ends is exactly the touched set in 1D: it names every tile whose
	// [u, u+1) interval meets [cu - radius, cu + radius].
	const int u0 = myClamp((int)(cu - radius_tiles), 0, res - 1); // SESSION079 PERF: truncation is exact here - see gsSatCeilToInt()'s comment.
	const int u1 = myClamp((int)(cu + radius_tiles), 0, res - 1);
	const int v0 = myClamp((int)(cv - radius_tiles), 0, res - 1);
	const int v1 = myClamp((int)(cv + radius_tiles), 0, res - 1);
	if(u1 < u0 || v1 < v0)
		return false;

	// SESSION074 DIAGNOSTIC: the deliberately-wrong upper bound - centre tile only, node treated as a point (no
	// radius). Strictly more permissive than the real test below on both axes (one tile instead of the whole span, no
	// radius margin), so it bounds what this mechanism could ever cut if every conservatism were abandoned. Never
	// consulted by a drop decision; see [gsr-sat]'s would_drop_aggr and the plan snapshot's Gate A.
	if(out_aggressive)
	{
		const int cu_i = myClamp((int)cu, 0, res - 1);
		const int cv_i = myClamp((int)cv, 0, res - 1);
		const float sat_d_centre = sat_depth[(size_t)cv_i * (size_t)res + (size_t)cu_i];
		*out_aggressive = (sat_d_centre != std::numeric_limits<float>::infinity()) && (dist_sq > sat_d_centre * sat_d_centre);
	}

	// SESSION074: every tile the footprint touches must agree this node is occluded. This replaces the first cut's
	// blanket 3x3-max dilation of the whole grid - same conservative intent (a node straddling a boundary must be
	// safe in both tiles), but at the node's real scale instead of a fixed +-1-tile margin, and with no extra pass
	// over the grid. A single unsaturated (+inf) tile in the span makes the whole test fail, which is exactly the
	// silhouette-edge behaviour wanted: nothing is culled where saturated coverage borders open space.
	for(int v=v0; v<=v1; ++v)
	{
		const float* const depth_row = &sat_depth[(size_t)v * (size_t)res];
		for(int u=u0; u<=u1; ++u)
		{
			const float sat_d = depth_row[u];
			if(sat_d == std::numeric_limits<float>::infinity())
				return false; // This direction never saturates - node may be visible.

			// Squared form of (dist - node_radius - region_radius) > sat_d, i.e. dist > sat_d + node_radius +
			// region_radius. All terms are non-negative (sat_d is a distance, the other two radii), so squaring
			// preserves the comparison and the sqrt of dist_sq is never needed.
			// SESSION078: + region_radius because the nearest ball point to this node stands R closer to it than the
			// anchor does - the node has to clear the barrier from there too, not just from the anchor.
			const float threshold = sat_d + node_radius + region_radius;
			if(dist_sq <= threshold * threshold)
				return false; // Not behind the saturation depth in this tile.
		}
	}
	return true;
}
