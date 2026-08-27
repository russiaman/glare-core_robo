/*=====================================================================
GaussianSplatSaturationGrid.cpp
---------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatSaturationGrid.h"


#include "../maths/mathstypes.h"
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


void gsBuildSaturationGrid(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers, size_t* out_tile_writes,
	js::Vector<float, 16>* out_accum_t, js::Vector<float, 16>* out_amp_sum, size_t* out_tile_iters)
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

	const float remaining_threshold = myClamp(1.f - saturation_threshold, 0.f, 1.f); // Below this remaining transmittance, a tile counts as saturated.

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

		float cu, cv;
		const float radius_tiles = gsSatGridFootprint(Vec4f(dx, dy, dz, 0.f), ang_radius, res, inv_dist, cu, cv);

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
		const float span = radius_tiles + gs_sat_tile_half_diag;
		const int u0 = myClamp((int)std::ceil (cu - span - 0.5f), 0, res - 1);
		const int u1 = myClamp((int)std::floor(cu + span - 0.5f), 0, res - 1);
		const int v0 = myClamp((int)std::ceil (cv - span - 0.5f), 0, res - 1);
		const int v1 = myClamp((int)std::floor(cv + span - 0.5f), 0, res - 1);
		if(u1 < u0 || v1 < v0)
			continue;

		if(out_writers) ++(*out_writers); // SESSION076 DIAGNOSTIC: this node contributes something.

		const float inv_2var = 0.5f / var;

		for(int v=v0; v<=v1; ++v)
		{
			float* const accum_row = &accum_t[(size_t)v * (size_t)res];
			float* const depth_row = &sat_depth_out[(size_t)v * (size_t)res];
			float* const sum_row = out_amp_sum ? &amp_sum[(size_t)v * (size_t)res] : NULL;
			const float dv = ((float)v + 0.5f) - cv;
			const float dv_sq = dv * dv;
			for(int u=u0; u<=u1; ++u)
			{
				if(out_tile_iters) ++(*out_tile_iters); // SESSION079 DIAGNOSTIC - see the parameter.

				const float du = ((float)u + 0.5f) - cu;
				const float x = (du * du + dv_sq) * inv_2var;
				if(x >= gs_sat_exp_lut_max)
					continue; // Contribution below the lookup's tail - see gs_sat_exp_lut_max.

				const float w = gsSatExpNeg(x); // Gaussian falloff to this tile - shared by both accumulators below.
				const bool already_saturated = depth_row[u] != std::numeric_limits<float>::infinity();

				// SESSION077: the fast path (skip a tile once its barrier is set) is unchanged when nobody wants the
				// diagnostic accumulators - sat_depth_out's semantics (barrier = FIRST crossing's far edge) do not need
				// any writes past that point, so production runs (prune, or diag off) keep paying for exactly what they
				// use.
				//
				// With out_accum_t/out_amp_sum requested, keep accumulating past saturation too: sat_depth_out's reader
				// only ever asks "occluded or not", so nothing beyond the crossing point was ever computed anywhere
				// before - accum_t alone showed whatever value first tripped the threshold, near-identical for every
				// saturated tile regardless of how much more geometry piled up behind it (read by the owner, correctly,
				// as "clamped"); amp_sum below is the actual fix for that - an unbounded running total, not a [0, 1]
				// product, so a tile with ten occluders reads roughly 10x one with a single occluder. depth_row[u]
				// itself is still only ever set on the FIRST crossing (the `!already_saturated` guard below), so the
				// barrier this grid actually prunes by is bit-identical either way.
				if(already_saturated && !out_accum_t && !out_amp_sum)
					continue;

				if(out_tile_writes) ++(*out_tile_writes); // SESSION076 DIAGNOSTIC: write amplification.

				accum_row[u] *= (1.f - amp * w);
				if(!already_saturated && accum_row[u] <= remaining_threshold)
					depth_row[u] = far_edge;

				if(sum_row) sum_row[u] += amp * w;
			}
		}
	}

	// SESSION077 DIAGNOSTIC: hand the accumulator fields out for the debug overlay - see the header.
	if(out_accum_t)
		*out_accum_t = accum_t;
	if(out_amp_sum)
		*out_amp_sum = amp_sum;
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
	const int u0 = myClamp((int)std::floor(cu - radius_tiles), 0, res - 1);
	const int u1 = myClamp((int)std::floor(cu + radius_tiles), 0, res - 1);
	const int v0 = myClamp((int)std::floor(cv - radius_tiles), 0, res - 1);
	const int v1 = myClamp((int)std::floor(cv + radius_tiles), 0, res - 1);
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
