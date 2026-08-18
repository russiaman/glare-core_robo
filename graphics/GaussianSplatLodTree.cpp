/*=====================================================================
GaussianSplatLodTree.cpp
-------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "GaussianSplatLodTree.h"


#include "../maths/Matrix4f.h"
#include "../maths/Quat.h"
#include "../maths/mathstypes.h"
#include "../utils/TestUtils.h"
#include "../utils/ConPrint.h"
#include <assert.h>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <functional>


namespace
{


// Builds the 3x3 covariance matrix a Gaussian with the given scale/rotation represents: cov = R * diag(scale^2) * R^T, embedded in a Matrix4f (4th row/col = (0, 0, 0, 1), matching Quat::toMatrix()'s convention -
// see Matrix4f::scaleMatrix()/Quat::toMatrix()). Only the upper-left 3x3 is ever read back out of a matrix built this way.
Matrix4f covarianceFromScaleRotation(const Vec3f& scale, const Vec4f& rotation_xyzw)
{
	const Quat<float> q(rotation_xyzw.x[0], rotation_xyzw.x[1], rotation_xyzw.x[2], rotation_xyzw.x[3]);
	const Matrix4f R = q.toMatrix();

	Matrix4f S;
	S.setToScaleMatrix(scale.x, scale.y, scale.z);

	const Matrix4f M = R * S; // M * M^T = R * diag(scale) * diag(scale)^T * R^T = R * diag(scale^2) * R^T, since diag(scale) is its own transpose.
	return M * M.getTranspose();
}


// v * v^T, embedded the same way covarianceFromScaleRotation()'s result is, so the two can be added directly (used for the "spread of centres" term in the merge - see mergeGaussianSplatLodNodes()).
Matrix4f outerProduct3(const Vec3f& v)
{
	Matrix4f m;
	m.setColumn(0, Vec4f(v.x * v.x, v.y * v.x, v.z * v.x, 0.f));
	m.setColumn(1, Vec4f(v.x * v.y, v.y * v.y, v.z * v.y, 0.f));
	m.setColumn(2, Vec4f(v.x * v.z, v.y * v.z, v.z * v.z, 0.f));
	m.setColumn(3, Vec4f(0, 0, 0, 1));
	return m;
}


// One Givens rotation zeroing a.elem(p, q) (and a.elem(q, p)) in place, accumulating the same rotation into v's columns. Textbook symmetric-Jacobi rotation (Golub & Van Loan's tan/cos/sin-from-tan form - avoids
// calling atan2/cos/sin directly, the standard numerically-robust way to do this). 'a' must be symmetric; only p, q in {0, 1, 2} are used - see jacobiEigenSolveSymmetric3x3().
void jacobiRotate(Matrix4f& a, Matrix4f& v, int p, int q)
{
	const float apq = a.elem((unsigned)p, (unsigned)q);
	if(std::fabs(apq) < 1.0e-20f)
		return;

	const float app = a.elem((unsigned)p, (unsigned)p);
	const float aqq = a.elem((unsigned)q, (unsigned)q);

	const float theta = (aqq - app) / (2.f * apq);
	const float t = (theta >= 0.f ? 1.f : -1.f) / (std::fabs(theta) + std::sqrt(theta * theta + 1.f));
	const float c = 1.f / std::sqrt(t * t + 1.f);
	const float s = t * c;

	a.elem((unsigned)p, (unsigned)p) = app - t * apq;
	a.elem((unsigned)q, (unsigned)q) = aqq + t * apq;
	a.elem((unsigned)p, (unsigned)q) = 0.f;
	a.elem((unsigned)q, (unsigned)p) = 0.f;

	for(int k = 0; k < 3; ++k)
	{
		if(k != p && k != q)
		{
			const float akp = a.elem((unsigned)k, (unsigned)p);
			const float akq = a.elem((unsigned)k, (unsigned)q);
			a.elem((unsigned)k, (unsigned)p) = c * akp - s * akq;
			a.elem((unsigned)p, (unsigned)k) = a.elem((unsigned)k, (unsigned)p);
			a.elem((unsigned)k, (unsigned)q) = s * akp + c * akq;
			a.elem((unsigned)q, (unsigned)k) = a.elem((unsigned)k, (unsigned)q);
		}

		const float vkp = v.elem((unsigned)k, (unsigned)p);
		const float vkq = v.elem((unsigned)k, (unsigned)q);
		v.elem((unsigned)k, (unsigned)p) = c * vkp - s * vkq;
		v.elem((unsigned)k, (unsigned)q) = s * vkp + c * vkq;
	}
}


// Eigen-decomposition of a symmetric 3x3 matrix (embedded per this file's Matrix4f convention - see covarianceFromScaleRotation()), via cyclic Jacobi rotation. Converges to float precision within a handful of
// sweeps for a matrix this small. eigenvectors_out's columns are orthonormal but not guaranteed a right-handed (det == +1) triple - see decomposeCovarianceToScaleRotation() for the fix-up needed before treating
// them as a rotation matrix.
void jacobiEigenSolveSymmetric3x3(Matrix4f a, Vec3f& eigenvalues_out, Matrix4f& eigenvectors_out)
{
	eigenvectors_out = Matrix4f::identity();

	for(int sweep = 0; sweep < 24; ++sweep)
	{
		const float off = std::fabs(a.elem(0, 1)) + std::fabs(a.elem(0, 2)) + std::fabs(a.elem(1, 2));
		if(off < 1.0e-12f)
			break;

		// Cyclic (as opposed to "largest off-diagonal element first") sweep order - simplest correct choice, and fine for a matrix this small (3 pairs/sweep either way).
		jacobiRotate(a, eigenvectors_out, 0, 1);
		jacobiRotate(a, eigenvectors_out, 0, 2);
		jacobiRotate(a, eigenvectors_out, 1, 2);
	}

	eigenvalues_out = Vec3f(a.elem(0, 0), a.elem(1, 1), a.elem(2, 2));
}


// Inverse of covarianceFromScaleRotation(): recovers a (scale, rotation) pair whose covariance matches 'cov' as closely as floating point / Jacobi convergence allows.
void decomposeCovarianceToScaleRotation(const Matrix4f& cov, Vec3f& scale_out, Vec4f& rotation_out)
{
	Vec3f eigenvalues;
	Matrix4f eigenvectors;
	jacobiEigenSolveSymmetric3x3(cov, eigenvalues, eigenvectors);

	// Eigenvalues of a covariance matrix are variances, so should be >= 0 in exact arithmetic - clamp away any tiny negative floating-point noise before taking the square root.
	scale_out = Vec3f(std::sqrt(myMax(0.f, eigenvalues.x)), std::sqrt(myMax(0.f, eigenvalues.y)), std::sqrt(myMax(0.f, eigenvalues.z)));

	Matrix4f rot = eigenvectors;
	rot.setColumn(3, Vec4f(0, 0, 0, 1)); // jacobiEigenSolveSymmetric3x3() never touches column/row 3 (eigenvectors_out started as identity() there), but set this explicitly rather than relying on that.

	if(rot.upperLeftDeterminant() < 0.f) // Jacobi's eigenvectors are orthonormal but not guaranteed right-handed; Quat::fromMatrix() requires a genuine rotation matrix.
		rot.setColumn(2, rot.getColumn(2) * -1.f); // Flipping one axis turns a reflection back into a rotation without disturbing the other two (still-mutually-orthogonal) axes.

	rotation_out = Quat<float>::fromMatrix(rot).v;
}


} // end anonymous namespace


GaussianSplatLodNode makeGaussianSplatLodLeafNode(const Vec3f& centre_os, const Vec3f& scale, const Vec4f& rotation, const Vec4f& colour)
{
	GaussianSplatLodNode node;
	node.centre_os = centre_os;
	node.scale = scale;
	node.rotation = rotation;
	node.colour = colour;
	node.feature_size = 2.f * myMax(scale.x, myMax(scale.y, scale.z));
	node.bounding_radius_os = 1.5f * node.feature_size; // SESSION059: a leaf's own 3-sigma cutoff radius - see the field's comment.
	node.layer_density = 0.f; // Nothing to recurse into - see the field's own comment.
	node.child_start = 0;
	node.child_count = 0;
	return node;
}


GaussianSplatLodNode mergeGaussianSplatLodNodes(const GaussianSplatLodNode* children, size_t num_children)
{
	assert(num_children >= 1);

	// weight_i = opacity_i * volume_i, volume_i = scale.x * scale.y * scale.z - a cheap proxy for "how much this splat contributes" (Spark's docs say "opacity * surface_area of the ellipsoid" without pinning down
	// the exact area formula - this is the simplest defensible stand-in. None of the maths below depends on which proxy is used, only on it being >= 0 and increasing in size/opacity).
	float total_weight = 0.f;
	for(size_t i = 0; i < num_children; ++i)
	{
		const Vec3f& s = children[i].scale;
		total_weight += children[i].colour.x[3] * (s.x * s.y * s.z);
	}
	const bool degenerate = total_weight <= 1.0e-12f; // Every child ~fully transparent and/or ~zero volume - fall back to a plain unweighted average (below) rather than dividing by ~0.
	const float inv_w = 1.f / (degenerate ? (float)num_children : total_weight);

	// Weighted centre.
	Vec3f centre(0.f, 0.f, 0.f);
	for(size_t i = 0; i < num_children; ++i)
	{
		const Vec3f& s = children[i].scale;
		const float w = degenerate ? 1.f : (children[i].colour.x[3] * (s.x * s.y * s.z));
		centre += children[i].centre_os * w;
	}
	centre = centre * inv_w;

	// Weighted colour - rgb only. Opacity is NOT part of this average; it's derived separately below from how much total weight ends up packed into the parent's own (now-decomposed) footprint, not from averaging
	// the children's opacities directly - see the comment further down.
	Vec4f colour_rgb(0.f, 0.f, 0.f, 0.f);
	for(size_t i = 0; i < num_children; ++i)
	{
		const Vec3f& s = children[i].scale;
		const float w = degenerate ? 1.f : (children[i].colour.x[3] * (s.x * s.y * s.z));
		colour_rgb += children[i].colour * w;
	}
	colour_rgb = colour_rgb * inv_w;

	// Weighted covariance merge (moment matching / mixture-of-Gaussians collapse, as in Kerbl et al.): each child's own covariance PLUS the spread of its centre from the merged centre. The spread term is what
	// makes e.g. two small, separated splats merge into one bigger, elongated blob spanning both of them, rather than one implausibly small blob sitting between them.
	Matrix4f cov_sum;
	for(int i = 0; i < 16; ++i) cov_sum.e[i] = 0.f;
	for(size_t i = 0; i < num_children; ++i)
	{
		const Vec3f& s = children[i].scale;
		const float w = degenerate ? 1.f : (children[i].colour.x[3] * (s.x * s.y * s.z));

		const Matrix4f child_cov = covarianceFromScaleRotation(children[i].scale, children[i].rotation);
		const Vec3f d = children[i].centre_os - centre;
		const Matrix4f contribution = child_cov + outerProduct3(d);

		for(int k = 0; k < 16; ++k)
			cov_sum.e[k] += contribution.e[k] * w;
	}
	for(int k = 0; k < 16; ++k) cov_sum.e[k] *= inv_w;

	GaussianSplatLodNode parent;
	parent.centre_os = centre;
	decomposeCovarianceToScaleRotation(cov_sum, parent.scale, parent.rotation);

	// Opacity: A = total_weight / area_parent (Kerbl et al.'s amplitude), clamped to 1 rather than Spark's extended-D shifted-profile fix for A > 1 - see the header comment for why that's deferred.
	// Using total_weight here (not the degenerate-case uniform per-child weight) matters: a group of near-fully-transparent children should merge into something with ~0 opacity, not jump to full opacity just
	// because the degenerate fallback above used equal weights for the centre/colour/covariance averages.
	const float area_parent = myMax(parent.scale.x * parent.scale.y * parent.scale.z, 1.0e-12f);
	const float A = total_weight / area_parent;
	parent.colour = Vec4f(colour_rgb.x[0], colour_rgb.x[1], colour_rgb.x[2], myClamp(A, 0.f, 1.f));

	parent.feature_size = 2.f * myMax(parent.scale.x, myMax(parent.scale.y, parent.scale.z));

	// SESSION059: bottom-up enclosing-sphere bound - see the field's comment. Exact (not a statistical fit): every
	// child's own bound, offset by how far that child's centre sits from the parent's, is guaranteed to still contain
	// that child's whole subtree, so the max over children bounds all of them regardless of how they're distributed
	// spatially (unlike parent.feature_size/scale above, which can end up smaller than the true spread - see the field's
	// comment for why that specifically bites at real-world scale).
	float bounding_radius_os = 0.f;
	for(size_t i = 0; i < num_children; ++i)
	{
		const float d = (children[i].centre_os - centre).length();
		bounding_radius_os = myMax(bounding_radius_os, d + children[i].bounding_radius_os);
	}
	parent.bounding_radius_os = bounding_radius_os;

	parent.layer_density = 0.f; // Caller fills this in where it knows the merge voxel's step (buildGaussianSplatLodTree()) - this function has no notion of a voxel grid, so 0 (meaning "unknown/not applicable") is the only defensible default here (e.g. the tests in this file merge arbitrary node pairs with no voxel step at all).
	parent.child_start = 0;
	parent.child_count = 0;
	return parent;
}


namespace
{


// Integer voxel-grid coordinate (floor(centre / step) per axis) - the grouping key buildGaussianSplatLodTree() merges same-voxel nodes on at each level.
struct VoxelCoord
{
	int64 x, y, z;
	bool operator == (const VoxelCoord& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelCoordHasher
{
	size_t operator () (const VoxelCoord& c) const
	{
		// Simple hash-combine (boost::hash_combine's constant) - collisions just cost a bit of unordered_map bucket-chasing, not correctness (operator== is the real disambiguator).
		size_t h = std::hash<int64>()(c.x);
		h ^= std::hash<int64>()(c.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		h ^= std::hash<int64>()(c.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		return h;
	}
};


VoxelCoord voxelCoordForStep(const Vec3f& centre, float step)
{
	VoxelCoord c;
	c.x = (int64)std::floor(centre.x / step);
	c.y = (int64)std::floor(centre.y / step);
	c.z = (int64)std::floor(centre.z / step);
	return c;
}


// A splat's cross-section area, 3 sigma out - not feature_size's own 1 sigma (feature_size = 2 * max(scale), i.e. a
// 2-sigma diameter) - matches the actual on-screen cutoff radius the renderer uses (GaussianSplatRenderer.cpp's
// splat_cutoff_sigmas / the vertex shader's 3-sigma quad), so density estimates end up in the same units as what
// actually gets drawn, not a ~9x-smaller 1-sigma proxy that would silently make every real value look far below any
// sane threshold.
float splatCrossSectionArea(float feature_size)
{
	const float r = feature_size * 1.5f;
	return (float)NICKMATHS_PI * r * r;
}


// See GaussianSplatLodNode::layer_density's comment. 'cumulative_leaf_area' is the sum of splatCrossSectionArea()
// over every *original leaf* splat beneath this node (not just its immediate merge children, which may themselves
// already be coarse stand-ins for thousands of leaves further down - using only their own, much smaller, area would
// wildly understate how packed the node's footprint really is).
//
// 'footprint_sum' is deliberately the sum of the *direct children's own* splatCrossSectionArea(feature_size) - not
// splatCrossSectionArea(this node's own, just-computed feature_size). The two aren't the same thing: a widely-spread
// merge's re-fit covariance can end up more compact along its single largest axis (what feature_size measures) than
// the union of its children's footprints actually is, which silently produced a parent *denser* than every one of its
// own children - observed in practice as the whole tree collapsing to a single root-level blob at any threshold below
// the root's own inflated density, with no gradation. Building footprint_sum from the children's own areas instead
// makes layer_density(parent) = (sum of child_i's cumulative area) / (sum of child_i's own footprint) - a weighted
// average of the children's own density_i = cumulative_area_i / footprint_i - which is mathematically guaranteed to
// land between min(density_i) and max(density_i). A parent can now never exceed the densest thing beneath it.
float layerDensityFromSums(float cumulative_leaf_area, float footprint_sum)
{
	return cumulative_leaf_area / myMax(footprint_sum, 1.0e-12f);
}


} // end anonymous namespace


std::vector<GaussianSplatLodNode> buildGaussianSplatLodTree(const Vec3f* centres, const Vec3f* scales, const Vec4f* rotations, const Vec4f* colours, size_t num_splats, float lod_base)
{
	assert(num_splats >= 1);
	assert(lod_base > 1.f);

	// raw_nodes/raw_children hold the tree in build order (leaves 0..num_splats-1, then merged parents appended as they're created) - NOT yet the final, contiguous-children layout the caller gets; see the
	// linearisation pass below for that. Keeping these two concerns separate (build the parent/child relationships first, decide array positions after) is what makes the "children are contiguous, indices
	// increase root-to-leaf" layout possible at all - a single bottom-up pass can't satisfy that on its own, since a node's children are only known to be *its* children after it's created, not where in a
	// final array they'll end up living.
	std::vector<GaussianSplatLodNode> raw_nodes;
	std::vector<std::vector<uint32> > raw_children;
	std::vector<float> raw_leaf_area_sum; // Parallel to raw_nodes: sum of splatCrossSectionArea() over every original leaf beneath each node - build-time-only bookkeeping for layer_density, not part of the final struct (would cost 4 bytes/node forever for a value only needed once, at merge time).
	raw_nodes.reserve(num_splats * 2); // A generous guess (real trees seen in testing run ~1.3x-1.8x num_splats total nodes) - just avoids a few reallocations, doesn't need to be exact.
	raw_children.reserve(num_splats * 2);
	raw_leaf_area_sum.reserve(num_splats * 2);

	for(size_t i = 0; i < num_splats; ++i)
	{
		raw_nodes.push_back(makeGaussianSplatLodLeafNode(centres[i], scales[i], rotations[i], colours[i]));
		raw_children.push_back(std::vector<uint32>()); // Leaves have no children.
		raw_leaf_area_sum.push_back(splatCrossSectionArea(raw_nodes.back().feature_size));
	}

	// Admission order: smallest splats first, so the coarsening process starts from the finest detail and works outward. A node only becomes eligible to be grouped with others once the current level's voxel
	// step has grown at least as large as its own feature_size (admitted_up_to tracks how far into 'order' that's happened so far).
	std::vector<uint32> order(num_splats);
	for(size_t i = 0; i < num_splats; ++i) order[i] = (uint32)i;
	std::sort(order.begin(), order.end(), [&](uint32 a, uint32 b) { return raw_nodes[a].feature_size < raw_nodes[b].feature_size; });

	size_t admitted_up_to = 0;
	std::vector<uint32> active;

	const float min_feature_size = myMax(raw_nodes[order[0]].feature_size, 1.0e-6f);
	int level = (int)std::ceil(std::log(min_feature_size) / std::log(lod_base));

	const int max_iters = 10000; // Generous safety cap - real convergence takes on the order of log_base(spatial extent / smallest splat) iterations, nowhere near this; exists purely so a maths edge case fails loudly instead of hanging.
	int iter = 0;
	for(; iter < max_iters; ++iter)
	{
		const float step = std::pow(lod_base, (float)level);

		// Admit any not-yet-admitted splats (in ascending feature_size order) whose feature_size now fits within this level's voxel step.
		while(admitted_up_to < num_splats && raw_nodes[order[admitted_up_to]].feature_size <= step)
		{
			active.push_back(order[admitted_up_to]);
			++admitted_up_to;
		}

		if(active.size() <= 1 && admitted_up_to >= num_splats)
			break; // Everything admitted, at most one node left standing - that's the root (or, for num_splats == 1, the single leaf itself).

		if(active.empty())
		{
			++level;
			continue; // Nothing admitted yet at this step (only possible before the very first admission) - grow the step and try again.
		}

		// Group the current active set by voxel at this level's step, merging every group of more than one node into a new parent.
		std::unordered_map<VoxelCoord, std::vector<uint32>, VoxelCoordHasher> voxel_groups;
		for(uint32 idx : active)
			voxel_groups[voxelCoordForStep(raw_nodes[idx].centre_os, step)].push_back(idx);

		std::vector<uint32> new_active;
		new_active.reserve(voxel_groups.size());
		bool any_merge_happened = false;
		for(auto& kv : voxel_groups)
		{
			const std::vector<uint32>& group = kv.second;
			if(group.size() == 1)
				new_active.push_back(group[0]); // Lone node at this level - carries forward unmerged, may still find company at a coarser level.
			else
			{
				std::vector<GaussianSplatLodNode> group_nodes(group.size());
				for(size_t i = 0; i < group.size(); ++i)
					group_nodes[i] = raw_nodes[group[i]];

				float cumulative_leaf_area = 0.f;
				float footprint_sum = 0.f;
				for(uint32 idx : group)
				{
					cumulative_leaf_area += raw_leaf_area_sum[idx];
					footprint_sum += splatCrossSectionArea(raw_nodes[idx].feature_size);
				}

				GaussianSplatLodNode parent = mergeGaussianSplatLodNodes(group_nodes.data(), group_nodes.size());
				parent.layer_density = layerDensityFromSums(cumulative_leaf_area, footprint_sum);

				const uint32 new_raw_id = (uint32)raw_nodes.size();
				raw_nodes.push_back(parent);
				raw_children.push_back(group); // group.size() is realistically always far below 65535 (a voxel holding tens of thousands of splats isn't a scene this builder is tuned for), so the eventual uint16 child_count cast is safe.
				raw_leaf_area_sum.push_back(cumulative_leaf_area);
				new_active.push_back(new_raw_id);
				any_merge_happened = true;
			}
		}
		active = new_active;
		++level;

		// Safety valve: everything's been admitted, but a whole pass produced no merges at all (every voxel group was a singleton) - rather than waiting out however many further step-doublings it'd take for
		// the remaining scattered nodes to naturally share a voxel, just force them all into one root now. Still a mathematically valid tree (mergeGaussianSplatLodNodes() doesn't care how far apart its inputs
		// are - see the "spread" term in its covariance maths), just a root with an unusually large footprint.
		if(admitted_up_to >= num_splats && !any_merge_happened && active.size() > 1)
		{
			std::vector<GaussianSplatLodNode> group_nodes(active.size());
			for(size_t i = 0; i < active.size(); ++i)
				group_nodes[i] = raw_nodes[active[i]];

			float cumulative_leaf_area = 0.f;
			float footprint_sum = 0.f;
			for(uint32 idx : active)
			{
				cumulative_leaf_area += raw_leaf_area_sum[idx];
				footprint_sum += splatCrossSectionArea(raw_nodes[idx].feature_size);
			}

			GaussianSplatLodNode root = mergeGaussianSplatLodNodes(group_nodes.data(), group_nodes.size());
			root.layer_density = layerDensityFromSums(cumulative_leaf_area, footprint_sum);
			const uint32 new_raw_id = (uint32)raw_nodes.size();
			raw_nodes.push_back(root);
			raw_children.push_back(active);
			raw_leaf_area_sum.push_back(cumulative_leaf_area);
			active.assign(1, new_raw_id);
			break;
		}
	}
	assert(iter < max_iters); // See max_iters' comment - if this ever fires, something about the admission/grouping logic above has a real bug, not just an unlucky splat layout.
	assert(active.size() == 1);
	const uint32 raw_root = active[0];

	// Linearise: breadth-first flatten from raw_root into the final array, so that (a) the root ends up at index 0, (b) every node's children occupy a contiguous range starting at child_start (assigned the
	// moment the node itself is popped off the queue - at that point every child's future index is already reserved, in order, right after every one of the node's siblings that were queued earlier), and
	// (c) child indices always exceed their parent's, since a child is never appended before its parent has already been placed.
	std::vector<GaussianSplatLodNode> out;
	out.reserve(raw_nodes.size());
	std::unordered_map<uint32, uint32> final_index_of_raw;
	final_index_of_raw[raw_root] = 0;
	out.push_back(raw_nodes[raw_root]); // child_start/child_count still 0/0 here - filled in below once/if this node is popped from the queue and its children (if any) are placed.

	std::deque<uint32> queue;
	queue.push_back(raw_root);
	while(!queue.empty())
	{
		const uint32 raw_id = queue.front();
		queue.pop_front();

		const uint32 my_final_index = final_index_of_raw[raw_id];
		const std::vector<uint32>& children = raw_children[raw_id];
		if(!children.empty())
		{
			out[my_final_index].child_start = (uint32)out.size();
			out[my_final_index].child_count = (uint16)children.size();

			for(uint32 child_raw_id : children)
			{
				final_index_of_raw[child_raw_id] = (uint32)out.size();
				out.push_back(raw_nodes[child_raw_id]);
				queue.push_back(child_raw_id);
			}
		}
	}

	return out;
}


namespace GaussianSplatLodTreeTests
{


// Structural validity check shared by the buildGaussianSplatLodTree() tests below: every node is reachable from the root exactly once (it's actually a tree, not a DAG or something with orphans), every non-leaf
// node's children occupy a contiguous, in-bounds range starting after the node's own index, and every original leaf splat (identified by child_count == 0) is present exactly once. Returns the total node count.
static size_t checkTreeIsValid(const std::vector<GaussianSplatLodNode>& tree, size_t expected_num_leaves)
{
	testAssert(!tree.empty());

	std::vector<bool> visited(tree.size(), false);
	std::vector<uint32> stack(1, 0); // Start at the root.
	visited[0] = true;
	size_t leaf_count = 0;

	while(!stack.empty())
	{
		const uint32 i = stack.back();
		stack.pop_back();

		const GaussianSplatLodNode& node = tree[i];
		if(node.child_count == 0)
			++leaf_count;
		else
		{
			testAssert(node.child_start > i); // Children always come after their parent in the linearised array.
			testAssert((size_t)node.child_start + node.child_count <= tree.size());
			for(uint32 c = node.child_start; c < node.child_start + node.child_count; ++c)
			{
				testAssert(!visited[c]); // Would mean two parents claim the same child - not a tree.
				visited[c] = true;
				stack.push_back(c);
			}
		}
	}

	for(size_t i = 0; i < tree.size(); ++i)
		testAssert(visited[i]); // No orphaned nodes - everything must be reachable from the root.
	testAssert(leaf_count == expected_num_leaves);

	return tree.size();
}


void test()
{
	conPrint("GaussianSplatLodTreeTests::test()");

	// Test 1: a diagonal covariance (no off-diagonal terms) should be recovered exactly, with no rotation, and no Jacobi sweeps needed.
	{
		Matrix4f diag_cov;
		for(int i = 0; i < 16; ++i) diag_cov.e[i] = 0.f;
		diag_cov.elem(0, 0) = 4.f;
		diag_cov.elem(1, 1) = 9.f;
		diag_cov.elem(2, 2) = 16.f;
		diag_cov.elem(3, 3) = 1.f;

		Vec3f scale;
		Vec4f rotation;
		decomposeCovarianceToScaleRotation(diag_cov, scale, rotation);

		testAssert(epsEqual(scale.x, 2.f, 1.0e-4f));
		testAssert(epsEqual(scale.y, 3.f, 1.0e-4f));
		testAssert(epsEqual(scale.z, 4.f, 1.0e-4f));
	}

	// Test 2: round trip - build a covariance from a known (non-axis-aligned) scale/rotation, decompose it, and rebuild the covariance from the decomposed result. Sidesteps eigenvector-ordering ambiguity by
	// comparing the covariance matrices themselves rather than trying to match individual axes back to the original scale components.
	{
		const Vec3f scale(2.f, 1.f, 0.5f);
		const Quat<float> quat = Quat<float>::fromAxisAndAngle(Vec4f(0, 0, 1, 0), (float)(NICKMATHS_PI / 2)); // 90 degrees around Z.
		const Vec4f rotation(quat.v);

		const Matrix4f cov = covarianceFromScaleRotation(scale, rotation);

		Vec3f scale2;
		Vec4f rotation2;
		decomposeCovarianceToScaleRotation(cov, scale2, rotation2);
		const Matrix4f cov2 = covarianceFromScaleRotation(scale2, rotation2);

		for(int i = 0; i < 16; ++i)
			testAssert(epsEqual(cov.e[i], cov2.e[i], 1.0e-3f));
	}

	// Test 3: merging two identical splats should reproduce (approximately) the same splat.
	{
		const GaussianSplatLodNode leaf = makeGaussianSplatLodLeafNode(Vec3f(1, 2, 3), Vec3f(0.5f, 0.5f, 0.5f), Vec4f(0, 0, 0, 1), Vec4f(1, 0, 0, 1));
		const GaussianSplatLodNode children[2] = { leaf, leaf };
		const GaussianSplatLodNode parent = mergeGaussianSplatLodNodes(children, 2);

		testAssert(epsEqual(parent.centre_os.x, 1.f, 1.0e-3f) && epsEqual(parent.centre_os.y, 2.f, 1.0e-3f) && epsEqual(parent.centre_os.z, 3.f, 1.0e-3f));
		testAssert(epsEqual(parent.scale.x, 0.5f, 1.0e-3f) && epsEqual(parent.scale.y, 0.5f, 1.0e-3f) && epsEqual(parent.scale.z, 0.5f, 1.0e-3f));
		testAssert(epsEqual(parent.colour.x[3], 1.f, 1.0e-3f)); // Opacity should stay ~1, not double up.
	}

	// Test 4: two splats separated in space should merge into a bigger blob (feature_size strictly greater than either child's) centred at their midpoint - the "spread" term in the covariance merge is what's
	// responsible for this (without it, merging two far-apart splats would wrongly produce one implausibly small blob).
	{
		const GaussianSplatLodNode a = makeGaussianSplatLodLeafNode(Vec3f(0, 0, 0), Vec3f(0.3f, 0.3f, 0.3f), Vec4f(0, 0, 0, 1), Vec4f(1, 1, 1, 1));
		const GaussianSplatLodNode b = makeGaussianSplatLodLeafNode(Vec3f(2, 0, 0), Vec3f(0.3f, 0.3f, 0.3f), Vec4f(0, 0, 0, 1), Vec4f(1, 1, 1, 1));
		const GaussianSplatLodNode children[2] = { a, b };
		const GaussianSplatLodNode parent = mergeGaussianSplatLodNodes(children, 2);

		testAssert(parent.feature_size > a.feature_size);
		testAssert(epsEqual(parent.centre_os.x, 1.f, 1.0e-3f) && epsEqual(parent.centre_os.y, 0.f, 1.0e-3f) && epsEqual(parent.centre_os.z, 0.f, 1.0e-3f));
	}

	// Test 5: merging two low-opacity splats should stay low-opacity, not clamp up to 1 just because the merge internally falls back to equal weighting somewhere.
	{
		const GaussianSplatLodNode a = makeGaussianSplatLodLeafNode(Vec3f(0, 0, 0), Vec3f(0.2f, 0.2f, 0.2f), Vec4f(0, 0, 0, 1), Vec4f(0, 1, 0, 0.05f));
		const GaussianSplatLodNode b = makeGaussianSplatLodLeafNode(Vec3f(0.1f, 0, 0), Vec3f(0.2f, 0.2f, 0.2f), Vec4f(0, 0, 0, 1), Vec4f(0, 1, 0, 0.05f));
		const GaussianSplatLodNode children[2] = { a, b };
		const GaussianSplatLodNode parent = mergeGaussianSplatLodNodes(children, 2);

		testAssert(parent.colour.x[3] < 0.3f);
	}

	// Test 6: buildGaussianSplatLodTree() on a single splat should return a one-node tree (just the leaf itself, as the root).
	{
		const Vec3f centre(1, 2, 3), scale(0.2f, 0.2f, 0.2f);
		const Vec4f rotation(0, 0, 0, 1), colour(1, 1, 1, 1);
		const std::vector<GaussianSplatLodNode> tree = buildGaussianSplatLodTree(&centre, &scale, &rotation, &colour, 1);

		testAssert(tree.size() == 1);
		testAssert(tree[0].child_count == 0);
		checkTreeIsValid(tree, /*expected_num_leaves=*/1);
	}

	// Test 7: a regular 4x4x4 grid of small, closely-spaced splats should build into a single well-formed tree covering all of them - checks the admission/voxel-grouping/linearisation machinery end to end,
	// not just the merge maths (already covered by tests 3-5).
	{
		std::vector<Vec3f> centres, scales;
		std::vector<Vec4f> rotations, colours;
		for(int x = 0; x < 4; ++x)
			for(int y = 0; y < 4; ++y)
				for(int z = 0; z < 4; ++z)
				{
					centres.push_back(Vec3f((float)x, (float)y, (float)z));
					scales.push_back(Vec3f(0.1f, 0.1f, 0.1f));
					rotations.push_back(Vec4f(0, 0, 0, 1));
					colours.push_back(Vec4f(1, 1, 1, 1));
				}

		const std::vector<GaussianSplatLodNode> tree = buildGaussianSplatLodTree(centres.data(), scales.data(), rotations.data(), colours.data(), centres.size());

		const size_t total_nodes = checkTreeIsValid(tree, /*expected_num_leaves=*/64);
		testAssert(total_nodes > 64); // At least one merge must have happened - a "tree" that's just the 64 leaves with no root grouping them would fail checkTreeIsValid() anyway (no single root reaching all of them).
		testAssert(tree[0].child_count > 0); // Root isn't itself a leaf.
		// The root should be big enough to cover the whole ~3-unit-wide cluster, not just one corner splat.
		testAssert(tree[0].feature_size > 1.0f);
	}

	// Test 8: two splats far apart in space should still build into a single, valid two-leaf tree (exercises the "safety valve" forced final merge - see buildGaussianSplatLodTree()'s comment - since nothing
	// naturally groups them at any reasonably-sized voxel step).
	{
		const Vec3f centres[2] = { Vec3f(0, 0, 0), Vec3f(1000, 1000, 1000) };
		const Vec3f scales[2] = { Vec3f(0.1f, 0.1f, 0.1f), Vec3f(0.1f, 0.1f, 0.1f) };
		const Vec4f rotations[2] = { Vec4f(0, 0, 0, 1), Vec4f(0, 0, 0, 1) };
		const Vec4f colours[2] = { Vec4f(1, 1, 1, 1), Vec4f(1, 1, 1, 1) };

		const std::vector<GaussianSplatLodNode> tree = buildGaussianSplatLodTree(centres, scales, rotations, colours, 2);

		checkTreeIsValid(tree, /*expected_num_leaves=*/2);
		testAssert(tree.size() == 3); // Two leaves plus one root merging them.
	}

	// Test 9: many splats coincident at the exact same point (a degenerate case - zero spatial spread) shouldn't hang or crash the builder.
	{
		std::vector<Vec3f> centres(50, Vec3f(0, 0, 0));
		std::vector<Vec3f> scales(50, Vec3f(0.1f, 0.1f, 0.1f));
		std::vector<Vec4f> rotations(50, Vec4f(0, 0, 0, 1));
		std::vector<Vec4f> colours(50, Vec4f(1, 1, 1, 1));

		const std::vector<GaussianSplatLodNode> tree = buildGaussianSplatLodTree(centres.data(), scales.data(), rotations.data(), colours.data(), centres.size());
		checkTreeIsValid(tree, /*expected_num_leaves=*/50);
	}

	conPrint("GaussianSplatLodTreeTests::test() done");
}


} // end namespace GaussianSplatLodTreeTests
