/*=====================================================================
GaussianSplatSaturationGrid.cpp
---------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatSaturationGrid.h"


#include "../maths/mathstypes.h"
#include <cmath>
#include <limits>


// SESSION074: headroom between a coarse node's angular size and a grid tile's.
//
// 1 means "a tile is about as big as the node meant to fill it", which is what the rasterising build pass below wants:
// a node then covers 1-4 tiles, so its footprint is captured almost exactly, with neither the ~16x write amplification
// a finer grid would cost nor the coverage loss a coarser one would force. The first cut used 4 (tile a quarter of a
// node per axis) while writing only the node's centre tile - the worst of both, and it also pinned res at
// gsSatGridMaxTiles in every measured view, so the derivation never actually chose anything. See the plan snapshot.
static const float gs_sat_grid_tile_headroom_K = 1.f;

// SESSION074: fixed margin for the octahedral mapping's area distortion (not equal-area; the literature bounds the
// stretch under 2x, worst near the square's corners). Applied to the tile's nominal angular size wherever a footprint
// is compared against it, so every such comparison errs towards "this node covers fewer tiles than the ideal geometry
// says" - the safe direction for this stage, since under-covering only weakens the cull, never breaks the picture.
static const float gs_sat_grid_oct_margin = 1.5f;


Vec2f gsDirToOct(const Vec4f& dir)
{
	// SESSION074: identical formula to float32x3_to_oct() in opengl/shaders/frag_utils.glsl (Cigolle et al.) - project
	// the sphere onto the octahedron, then onto the xy plane, then reflect the lower hemisphere's folds over the
	// diagonals so the whole sphere maps into one [-1, 1]^2 square.
	//
	// Note this divides by the vector's own L1 norm, so it is invariant under positive scaling - callers pass raw
	// offsets, never normalised directions. See the header.
	const float abs_sum = std::fabs(dir.x[0]) + std::fabs(dir.x[1]) + std::fabs(dir.x[2]);
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


float gsSatGridTileAngle(int res)
{
	// SESSION074: res*res tiles cover the whole sphere (4*pi steradians). Treating a tile as approximately square in
	// angle, tile_ang^2 ~= 4*pi / res^2, hence tile_ang = sqrt(4*pi) / res. Inflated by the octahedral distortion
	// margin so every caller's footprint comparison stays on the conservative side - see that constant's comment.
	return (std::sqrt(4.f * 3.14159265f) / myMax((float)res, 1.f)) * gs_sat_grid_oct_margin;
}


int gsSatGridResForFocal(float focal_px, float coarse_pixel_scale)
{
	// SESSION074: derive tile angular size from the existing coarse_pixel_scale/focal_px knobs rather than exposing a
	// new UI parameter - project rule, no manual per-scene tuning. A coarse node subtends about
	// coarse_pixel_scale/focal_px radians; K sizes the tile against that (see the constant's comment).
	const float safe_focal = myMax(focal_px, 1.f);
	const float tile_ang = myMax(coarse_pixel_scale, 1.0e-3f) / (safe_focal * gs_sat_grid_tile_headroom_K);

	// Inverse of gsSatGridTileAngle(), including its distortion margin, so that the res chosen here and the tile size
	// reported there are consistent with each other rather than two independent approximations.
	const float res_raw = (std::sqrt(4.f * 3.14159265f) * gs_sat_grid_oct_margin) / tile_ang;

	int res = (int)std::ceil(res_raw);
	res = myMax(res, gsSatGridMinRes);
	const int max_res = (int)std::sqrt((float)gsSatGridMaxTiles); // SESSION074: hard ceiling - see gsSatGridMaxTiles's comment.
	res = myMin(res, max_res);
	return res;
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
// handled), it is bounded in oct space by scaling the angular radius into tile units via gsSatGridTileAngle() - the
// same quantity the resolution was solved for, already carrying the distortion margin.
//
// SESSION076: both sides now use the TOUCH span. They used to differ - the write side demanded full coverage of a
// tile, on the reasoning that "this tile is behind opaque coverage" is a claim a partly-covering node has not
// established. That reasoning was sound only while the contribution was binary. Now a partly-covering node contributes
// a correspondingly small alpha instead of a full one, so the claim it makes is already proportionate to what it
// covers, and demanding full coverage would simply discard it. See the write loop for the model.
static inline float gsSatGridFootprint(const Vec4f& offset, float ang_radius, int res, float& cu, float& cv)
{
	const Vec2f oct = gsDirToOct(offset);
	cu = (oct.x * 0.5f + 0.5f) * (float)res;
	cv = (oct.y * 0.5f + 0.5f) * (float)res;
	return ang_radius / gsSatGridTileAngle(res);
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
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers, size_t* out_tile_writes)
{
	const size_t num_tiles = (size_t)res * (size_t)res;
	sat_depth_out.resizeNoCopy(num_tiles);
	for(size_t i=0; i<num_tiles; ++i)
		sat_depth_out[i] = std::numeric_limits<float>::infinity();

	// SESSION074: running per-tile transmittance, local scratch only (not stored on the frontier - sat_depth_out is
	// the only thing callers need). Reset to 1 (fully transparent) before the sequential pass below.
	js::Vector<float, 16> accum_t(num_tiles, 1.f);

	const float remaining_threshold = myClamp(1.f - saturation_threshold, 0.f, 1.f); // Below this remaining transmittance, a tile counts as saturated.

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

		// SESSION074: angular radius without trigonometry. The exact value is asin(r/dist); for the small angles this
		// pass deals with, r/dist is within a fraction of a percent of it, and it is only ever compared against tile
		// sizes that already carry a 1.5x safety margin. Kept as the ratio to avoid a per-node asin AND the sqrt that
		// feeding it would require - both measured as a large part of this pass's first-cut cost. Where a node is
		// close enough for the approximation to drift (r comparable to dist), it drifts towards a LARGER angle, i.e.
		// towards covering more tiles - so the comparison below is bounded by the exact one, not looser than it.
		const float inv_dist = 1.f / std::sqrt(dist_sq); // One sqrt per node here is unavoidable: sat_depth is stored as a real distance, and the span needs a real angle.
		const float ang_radius = r * inv_dist;

		float cu, cv;
		const float radius_tiles = gsSatGridFootprint(Vec4f(dx, dy, dz, 0.f), ang_radius, res, cu, cv);

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
		//   amp  = alpha * sigma^2 / var              (peak scaled so the integral is unchanged)
		//   a_eff(d) = amp * exp(-d^2 / (2*var))
		//
		// Large node (sigma >> 1 tile): amp -> alpha, a_eff -> the alpha at that tile. Small node (sigma << 1 tile):
		// amp -> alpha * 12 * sigma^2, and the total spread over the neighbourhood comes to alpha * 2*pi*sigma^2 - the
		// true integral of the node's occlusion. Neither regime is special-cased.
		const float sigma_tiles = radius_tiles * (1.f / gs_sat_occluder_sigmas);
		const float sigma_sq = sigma_tiles * sigma_tiles;
		const float var = sigma_sq + (1.f / 12.f);
		const float a = myClamp(alpha[i], 0.f, 1.f);
		const float amp = a * (sigma_sq / var);
		if(amp < gs_sat_min_occluder_amp)
			continue; // Integrated occlusion is nil - see gs_sat_min_occluder_amp. Skips the fine-scale bulk cheaply.

		// SESSION074: recorded at the node's FAR edge (dist + radius), not its centre or near edge - deliberately
		// conservative. The true saturation point lies somewhere within this node's footprint (we don't know exactly
		// where), and placing the cut past its entire extent guarantees this stage only ever drops fine geometry
		// unambiguously behind the coarse mass that caused saturation - never something that might still be in front
		// of, or interleaved with, the very geometry that saturated the tile.
		const float far_edge = (1.f / inv_dist) + r;

		// Touch span, not coverage span: with a falloff weight there is no longer any reason to demand full coverage,
		// and a tile the footprint merely clips now correctly receives a small contribution instead of none.
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
			const float dv = ((float)v + 0.5f) - cv;
			const float dv_sq = dv * dv;
			for(int u=u0; u<=u1; ++u)
			{
				const float du = ((float)u + 0.5f) - cu;
				const float x = (du * du + dv_sq) * inv_2var;
				if(x >= gs_sat_exp_lut_max)
					continue; // Contribution below the lookup's tail - see gs_sat_exp_lut_max.

				if(depth_row[u] != std::numeric_limits<float>::infinity())
					continue; // Already saturated by something nearer (front-to-back order) - nothing more to do for this tile.

				if(out_tile_writes) ++(*out_tile_writes); // SESSION076 DIAGNOSTIC: write amplification.

				accum_row[u] *= (1.f - amp * gsSatExpNeg(x));
				if(accum_row[u] <= remaining_threshold)
					depth_row[u] = far_edge;
			}
		}
	}
}


bool gsSatOccluded(const Vec4f& offset, float dist_sq, float node_radius,
	const js::Vector<float, 16>& sat_depth, int res, bool* out_aggressive)
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
	const float ang_radius = node_radius * inv_dist;

	float cu, cv;
	const float radius_tiles = gsSatGridFootprint(offset, ang_radius, res, cu, cv);

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

			// Squared form of (dist - node_radius) > sat_d, i.e. dist > sat_d + node_radius. Both sides are
			// non-negative (sat_d is a distance, node_radius a radius), so squaring preserves the comparison and
			// the sqrt of dist_sq is never needed.
			const float threshold = sat_d + node_radius;
			if(dist_sq <= threshold * threshold)
				return false; // Not behind the saturation depth in this tile.
		}
	}
	return true;
}
