/*=====================================================================
GaussianSplatCoplanarMerge.cpp
-------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatCoplanarMerge.h"


#include "GaussianSplatLodTree.h" // For makeGaussianSplatLodLeafNode()/mergeGaussianSplatLodNodes() - the replacement splat is fitted with exactly the maths the LoD tree's own merge uses.
#include "../maths/mathstypes.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <unordered_map>


namespace
{


// The direction a splat's flat face points: its smallest scale axis, rotated out of the quaternion.  A 3DGS optimiser
// fits near-flat discs to surfaces, so this is the surface normal wherever the fit is behaving, and it is what decides
// whether two splats in the same place are two views of one surface or two different surfaces meeting at an edge.
Vec4f splatNormal(const Vec3f& scale, const Vec4f& rotation)
{
	const float qx = rotation[0], qy = rotation[1], qz = rotation[2], qw = rotation[3];
	const Vec4f r_col0(1.f - 2.f*(qy*qy + qz*qz),        2.f*(qx*qy + qz*qw),        2.f*(qx*qz - qy*qw), 0.f);
	const Vec4f r_col1(       2.f*(qx*qy - qz*qw), 1.f - 2.f*(qx*qx + qz*qz),        2.f*(qy*qz + qx*qw), 0.f);
	const Vec4f r_col2(       2.f*(qx*qz + qy*qw),        2.f*(qy*qz - qx*qw), 1.f - 2.f*(qx*qx + qy*qy), 0.f);

	Vec4f n;
	if(scale.x <= scale.y && scale.x <= scale.z)      n = r_col0;
	else if(scale.y <= scale.x && scale.y <= scale.z) n = r_col1;
	else                                              n = r_col2;

	// A disc has no front and back, so n and -n describe the same orientation.  Pinning the sign by the largest component
	// keeps two splats facing opposite ways from being told apart by nothing but a sign.
	const float ax = std::fabs(n[0]), ay = std::fabs(n[1]), az = std::fabs(n[2]);
	const float dominant = (ax >= ay && ax >= az) ? n[0] : ((ay >= az) ? n[1] : n[2]);
	if(dominant < 0.f)
		n = Vec4f(-n[0], -n[1], -n[2], 0.f);
	return n;
}


// Area of the ellipse the renderer actually rasterises for one splat, seen face-on: pi times its two largest scale axes,
// times the square of the cutoff radius the vertex shader sizes the quad to.
//
// Two of the three things separating this from the on-screen footprint - distance and the foreshortening of the viewing
// angle - are properties of the camera, not of how the splats were grouped, so they cancel out of the "does this group
// pay?" comparison this exists for.  The third does not cancel and is included: the quad is cut off where alpha decays to
// alpha_cutoff, which is sqrt(2 * ln(opacity / alpha_cutoff)) sigmas out (capped at 3, as the shader caps it), so a more
// opaque splat is drawn over a wider ellipse.  A collapsed group is exactly the case where that matters - the replacement
// is more opaque than any of its members - so leaving it out would count merges as paying when they do not.
float geometricArea(const Vec3f& scale) // pi * the two largest scale axes: one sigma, before the cutoff radius is applied.
{
	const float s_max = myMax(scale.x, myMax(scale.y, scale.z));
	const float s_min = myMin(scale.x, myMin(scale.y, scale.z));
	const float s_mid = (scale.x + scale.y + scale.z) - s_max - s_min;
	return Maths::pi<float>() * s_max * s_mid;
}

float drawnArea(const Vec3f& scale, float opacity, float alpha_cutoff)
{
	if(opacity <= alpha_cutoff)
		return 0.f; // Invisible even at its centre; the shader makes a degenerate zero-area quad of it.

	const float sigma_cutoff = myMin(std::sqrt(2.f * std::log(opacity / alpha_cutoff)), 3.f);
	return geometricArea(scale) * sigma_cutoff * sigma_cutoff;
}


// Hashed uniform grid over the splat centres, as a per-cell singly linked list - compact, and no per-cell allocation.
// This is a neighbour index only: what a splat is grouped with is decided by honest distances against the seed, never by
// which cell it landed in.  (An earlier, measurement-only version of this grouping quantised the coordinates and matched
// equal keys instead.  That splits a real group whenever its members straddle any one cell edge, and worse, it selects
// for members that are far apart within the cell rather than near - it reported merging as a loss at every setting.)
struct NeighbourGrid
{
	float inv_cell;
	std::unordered_map<uint64, uint32> cell_head;
	js::Vector<uint32, 16> next_in_cell;

	static inline uint64 cellKey(int32 cx, int32 cy, int32 cz)
	{
		return ((uint64)(uint32)cx * 73856093ull) ^ ((uint64)(uint32)cy * 19349663ull) ^ ((uint64)(uint32)cz * 83492791ull);
	}

	void build(const Vec3f* positions, size_t num_splats, float cell_size)
	{
		inv_cell = 1.f / cell_size;
		cell_head.clear();
		cell_head.reserve(num_splats);
		next_in_cell.resize(num_splats);

		for(size_t i=0; i<num_splats; ++i)
		{
			const uint64 key = cellKey((int32)std::floor(positions[i].x * inv_cell), (int32)std::floor(positions[i].y * inv_cell), (int32)std::floor(positions[i].z * inv_cell));
			std::unordered_map<uint64, uint32>::iterator it = cell_head.find(key);
			next_in_cell[i] = (it == cell_head.end()) ? 0xFFFFFFFFu : it->second;
			cell_head[key] = (uint32)i;
		}
	}
};


} // end anonymous namespace


GaussianSplatCoplanarMergeStats coplanarMergeSplats(js::Vector<Vec3f, 16>& positions, js::Vector<Vec3f, 16>& scales, js::Vector<Vec4f, 16>& rotations,
	js::Vector<Vec4f, 16>& colours, const GaussianSplatCoplanarMergeParams& params)
{
	const size_t num_splats = positions.size();
	assert(scales.size() == num_splats && rotations.size() == num_splats && colours.size() == num_splats);

	GaussianSplatCoplanarMergeStats stats;
	stats.splats_in = num_splats;
	stats.splats_out = num_splats;
	stats.groups_merged = 0;
	stats.groups_refused = 0;
	stats.largest_group_merged = 0;
	stats.groups_total = 0;
	stats.largest_group_found = 0;
	stats.area_in = 0;
	stats.area_out = 0;

	js::Vector<Vec4f, 16> normals(num_splats);
	js::Vector<float, 16> areas(num_splats);
	for(size_t i=0; i<num_splats; ++i)
	{
		normals[i] = splatNormal(scales[i], rotations[i]);
		areas[i] = drawnArea(scales[i], colours[i][3], params.alpha_cutoff);
		stats.area_in += areas[i];
	}

	if(num_splats < 2)
		return stats;

	// Seed order: biggest splat first.  A group's replacement has to cover its largest member, so growing outwards from
	// that member is what keeps the replacement as small as the group allows.  Seeding in arbitrary order instead lets a
	// small splat claim a big neighbour and then have to grow to cover it - which, on the "must not paint more" test
	// below, mostly means the group is refused and the chance is wasted.
	js::Vector<uint32, 16> seed_order(num_splats);
	for(size_t i=0; i<num_splats; ++i)
		seed_order[i] = (uint32)i;
	{
		struct AreaDesc
		{
			const float* areas;
			inline bool operator () (uint32 a, uint32 b) const { return (areas[a] != areas[b]) ? (areas[a] > areas[b]) : (a < b); } // Index breaks ties, so the result doesn't depend on the sort's own ordering of equal keys.
		};
		AreaDesc pred; pred.areas = areas.data();
		std::sort(seed_order.data(), seed_order.data() + seed_order.size(), pred);
	}

	// The 27 cells around a seed cover everything within one cell side of it along each axis, so the cell has to be at
	// least as long as the reach can be: the reach is a disc of radius 'across' with 'through' of thickness either side,
	// whose furthest point is sqrt(across^2 + through^2) away.  Taking max(through, across) instead quietly loses the
	// neighbours out at the rim on thick settings.
	const float across = myMax(params.across, 1.0e-6f), through = myMax(params.through, 1.0e-6f);
	const float cos_tol = std::cos(params.angle_tol_deg * Maths::pi<float>() / 180.f);

	NeighbourGrid grid;
	grid.build(positions.data(), num_splats, std::sqrt(across*across + through*through));

	js::Vector<uint8, 16> assigned(num_splats);
	for(size_t i=0; i<num_splats; ++i)
		assigned[i] = 0;

	js::Vector<Vec3f, 16> out_positions;  out_positions.reserve(num_splats);
	js::Vector<Vec3f, 16> out_scales;     out_scales   .reserve(num_splats);
	js::Vector<Vec4f, 16> out_rotations;  out_rotations.reserve(num_splats);
	js::Vector<Vec4f, 16> out_colours;    out_colours  .reserve(num_splats);

	js::Vector<uint32, 16> group; // Members of the group being grown, seed first.  Reused across seeds.
	std::vector<GaussianSplatLodNode> group_nodes;

	for(size_t si=0; si<seed_order.size(); ++si)
	{
		const uint32 seed_i = seed_order[si];
		if(assigned[seed_i])
			continue;

		const Vec3f seed_pos = positions[seed_i];
		const Vec4f seed_normal = normals[seed_i];
		const Vec4f seed_col = colours[seed_i];

		group.clear();
		group.push_back(seed_i);

		// Marked before the walk rather than after it, so that the walk's own unlinking takes the seed out of the grid
		// too.  A seed is consumed either way - if the group below is refused it goes through on its own - so it never
		// needs to stay findable.
		assigned[seed_i] = 1;

		const int32 bx = (int32)std::floor(seed_pos.x * grid.inv_cell);
		const int32 by = (int32)std::floor(seed_pos.y * grid.inv_cell);
		const int32 bz = (int32)std::floor(seed_pos.z * grid.inv_cell);

		for(int dz=-1; dz<=1; ++dz)
		for(int dy=-1; dy<=1; ++dy)
		for(int dx=-1; dx<=1; ++dx)
		{
			std::unordered_map<uint64, uint32>::iterator it = grid.cell_head.find(NeighbourGrid::cellKey(bx+dx, by+dy, bz+dz));
			if(it == grid.cell_head.end())
				continue;

			// Walk the cell, unlinking members that have already been taken by an earlier group.  Without this the later
			// seeds re-walk the whole cloud and the pass goes quadratic on a dense capture.  Note that a member taken by
			// *this* group is not unlinked here: the group may still be refused below, in which case its members have to
			// stay findable by other seeds.  They are unlinked by whichever later walk next passes over them.
			uint32 prev = 0xFFFFFFFFu;
			uint32 j = it->second;
			while(j != 0xFFFFFFFFu)
			{
				const uint32 next = grid.next_in_cell[j];
				if(assigned[j])
				{
					if(prev == 0xFFFFFFFFu) it->second = next; else grid.next_in_cell[prev] = next;
					j = next;
					continue;
				}

				const Vec3f& b_pos = positions[j];
				const float dx_s = b_pos.x - seed_pos.x, dy_s = b_pos.y - seed_pos.y, dz_s = b_pos.z - seed_pos.z;
				const float along = dx_s*seed_normal[0] + dy_s*seed_normal[1] + dz_s*seed_normal[2];
				const float lat_sq = myMax(0.f, (dx_s*dx_s + dy_s*dy_s + dz_s*dz_s) - along*along);

				if((std::fabs(along) <= through) && (lat_sq <= across * across))
				{
					const Vec4f& b_normal = normals[j];
					const Vec4f& b_col = colours[j];
					if(std::fabs(seed_normal[0]*b_normal[0] + seed_normal[1]*b_normal[1] + seed_normal[2]*b_normal[2]) >= cos_tol &&
						std::fabs(seed_col[0] - b_col[0]) <= params.colour_tol &&
						std::fabs(seed_col[1] - b_col[1]) <= params.colour_tol &&
						std::fabs(seed_col[2] - b_col[2]) <= params.colour_tol)
					{
						group.push_back(j);
					}
				}

				prev = j;
				j = next;
			}
		}

		stats.groups_total++;
		stats.largest_group_found = myMax(stats.largest_group_found, group.size());

		if(group.size() == 1)
		{
			// Nothing to merge with, so it goes through exactly as it was.
			out_positions.push_back(positions[seed_i]);
			out_scales   .push_back(scales[seed_i]);
			out_rotations.push_back(rotations[seed_i]);
			out_colours  .push_back(colours[seed_i]);
			stats.area_out += areas[seed_i];
			continue;
		}

		// Fit the replacement with the tree's own merge: a weighted moment match (each member's covariance plus the
		// spread of its centre from the merged centre), which is the tightest single Gaussian that still covers the
		// group.  Sharing that function rather than writing a second version of the same maths also means a group that
		// happens to be a stack collapses to the same shape either code path would give it.
		group_nodes.resize(group.size());
		double sum_drawn_area = 0, sum_geo_area = 0, sum_alpha_geo_area = 0, sum_log_transmittance = 0;
		for(size_t k=0; k<group.size(); ++k)
		{
			const uint32 idx = group[k];
			group_nodes[k] = makeGaussianSplatLodLeafNode(positions[idx], scales[idx], rotations[idx], colours[idx]);

			const float opacity = myClamp(colours[idx][3], 0.f, 1.f - 1.0e-6f); // Held below 1 so the log below stays finite; a splat at opacity 1 is opaque either way.
			const double geo_area = geometricArea(scales[idx]);
			sum_drawn_area += areas[idx];
			sum_geo_area += geo_area;
			sum_alpha_geo_area += geo_area * opacity; // The ink the member lays down, which is a property of the Gaussian itself - unlike the drawn area above, it doesn't move with where the quad is cut off.
			sum_log_transmittance += std::log(1.0 - opacity);
		}

		GaussianSplatLodNode merged = mergeGaussianSplatLodNodes(group_nodes.data(), group_nodes.size());

		if(params.flatten_onto_surface)
		{
			// Collapse the group's depth: project every member onto the plane through the merged centre, then fit again.
			// Projecting along the normal leaves the weighted mean where it was (the mean of the projections is the
			// projection of the mean, and the mean is already on the plane), so the replacement does not move - only the
			// along-normal part of the spread term disappears from its covariance.
			//
			// Fitted twice rather than once with the weights worked out here: the weighting is mergeGaussianSplatLodNodes()'s
			// business, and a second copy of it here would be a second thing to keep in step with it.
			for(size_t k=0; k<group.size(); ++k)
			{
				const Vec3f d = group_nodes[k].centre_os - merged.centre_os;
				const float along = d.x*seed_normal[0] + d.y*seed_normal[1] + d.z*seed_normal[2];
				group_nodes[k].centre_os = group_nodes[k].centre_os - Vec3f(seed_normal[0], seed_normal[1], seed_normal[2]) * along;
			}

			merged = mergeGaussianSplatLodNodes(group_nodes.data(), group_nodes.size());
		}

		// Opacity, which mergeGaussianSplatLodNodes()'s own answer is not right for here.  That one derives an amplitude
		// from weight over *volume*, which is the correct reading for a stand-in that will be looked at instead of a
		// subtree; what this replacement has to preserve is what the group put on the screen.  Two bounds say what that
		// is, and the answer is the smaller:
		//
		//  - Conserving ink: sum(alpha_i * area_i) / area_replacement.  A replacement covering more area than one member
		//    did has to be correspondingly fainter to lay the same amount down.
		//  - What compositing can actually reach.  Ink adds linearly, alpha does not: five layers of 0.1 show 0.41, not
		//    0.5, so past a point the first bound asks for an opacity that no arrangement of these members ever showed,
		//    and the replacement comes out harder than what it replaced.
		//
		// The second bound is over the layers the areas imply, not over the whole group.  How many members overlap at a
		// typical point is sum(their areas) / the replacement's area - k for a group stacked exactly on itself, 1 for one
		// spread out until its members no longer touch - so the cap is 1 - t^n at the members' geometric-mean
		// transmittance t.  Using the whole group's product instead assumes total overlap, which is only true of a true
		// stack: measured on the reference capture it cost about a tenth of the cloud's ink once the replacements got
		// small enough for it to bind, visible directly as a drop in the report's "Sum of alpha".
		//
		// Where the group really is a stack this reduces to 1 - prod(1 - alpha_i) exactly, and that is the mechanism by
		// which merging cuts the number of layers a pixel blends before it saturates - a saving separate from the area one
		// below.
		const double merged_geo_area = geometricArea(merged.scale);
		const double alpha_over_area = (merged_geo_area > 1.0e-20) ? (sum_alpha_geo_area / merged_geo_area) : 1.0;
		const double effective_layers = (merged_geo_area > 1.0e-20) ? myMax(sum_geo_area / merged_geo_area, 0.0) : (double)group.size();
		const double alpha_composite = 1.0 - std::exp(sum_log_transmittance * effective_layers / (double)group.size());
		const float merged_alpha = (float)myClamp(myMin(alpha_over_area, alpha_composite), 0.0, 1.0);

		// The rule this whole operation turns on: a group is only collapsed if what replaces it paints less than what it
		// replaced.  The LoD tree's merge has no such test - it is trading splat count, not area - and that is exactly why
		// coarsening the tree removes six times the splats for a third of the fill.  Here, a group that would widen the
		// cloud is simply left alone; on a capture without real stacking that refuses most groups, which is the correct
		// answer rather than a failure to find any.
		//
		// Note this is decided after the opacity above, not before: the replacement's opacity is what decides how far out
		// its quad is drawn, so the two cannot be settled in the other order.
		const double merged_drawn_area = drawnArea(merged.scale, merged_alpha, params.alpha_cutoff);

		if(!(merged_drawn_area < sum_drawn_area))
		{
			stats.groups_refused++;

			out_positions.push_back(positions[seed_i]);
			out_scales   .push_back(scales[seed_i]);
			out_rotations.push_back(rotations[seed_i]);
			out_colours  .push_back(colours[seed_i]);
			stats.area_out += areas[seed_i];
			continue;
		}

		for(size_t k=1; k<group.size(); ++k) // Committed: the members are consumed.  0 is the seed, already marked.
			assigned[group[k]] = 1;

		out_positions.push_back(merged.centre_os);
		out_scales   .push_back(merged.scale);
		out_rotations.push_back(merged.rotation);
		out_colours  .push_back(Vec4f(merged.colour[0], merged.colour[1], merged.colour[2], merged_alpha));
		stats.area_out += merged_drawn_area;

		stats.groups_merged++;
		stats.largest_group_merged = myMax(stats.largest_group_merged, group.size());
	}

	positions.swapWith(out_positions);
	scales   .swapWith(out_scales);
	rotations.swapWith(out_rotations);
	colours  .swapWith(out_colours);

	stats.splats_out = positions.size();
	return stats;
}
