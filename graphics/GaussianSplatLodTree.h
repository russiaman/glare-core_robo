/*=====================================================================
GaussianSplatLodTree.h
-----------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../maths/vec3.h"
#include "../maths/Vec4f.h"
#include <vector>


/*=====================================================================
GaussianSplatLodTree
---------------------
CPU-side on-the-fly LoD tree for a Gaussian Splat cloud.

Modelled on Spark's (sparkjs.dev) Tiny-LoD approach: a bottom-up voxel-grid
merge builds a hierarchy of "virtual" splats, each a statistically-faithful
stand-in for the splats below it, so that at render time a much smaller
"frontier" of nodes (large/coarse ones standing in for many small ones where
the camera is far away or the budget is tight) can represent the full splat
count while staying visually close to the un-thinned cloud.

A tree is built once per GaussianSplatData, in object space (see centre_os
below), from the same flat positions/scales/rotations/colours arrays the
decoder already produces - see GaussianSplatData::lod_tree. Building in
object space (rather than world space) is what lets a moved/scaled object's
tree be re-baked to world space the same way its leaf splats already are,
without rebuilding the tree itself - a merged node transforms identically to
a leaf one under a rigid + uniform-scale transform.

Documented simplification: opacity is plain [0, 1], clamped at merge time
(Spark's extended "D parameter" opacity model - a sharper-than-Gaussian
falloff for densely-merged nodes instead of a hard clamp - is deferred, not
implemented here).
=====================================================================*/


// One node of the LoD tree - either an original ("leaf") splat, or a merged stand-in for child_count children starting at child_start.
struct GaussianSplatLodNode
{
	Vec3f centre_os; // Object-space centre, in whatever space buildGaussianSplatLodTree()'s input was given in - matches GaussianSplatData::positions' convention.
	Vec3f scale; // Linear scale factors (already exponentiated), one per axis - matches GaussianSplatData::scales' convention.
	Vec4f rotation; // Unit quaternion (x, y, z, w) - matches GaussianSplatData::rotations' convention.
	Vec4f colour; // (r, g, b, opacity), all in [0, 1] - matches GaussianSplatData::colours' convention. See the file header comment re: the deferred D-parameter opacity model.
	float feature_size; // 2 * max(scale.x, scale.y, scale.z) - the "how big does this node look" metric traversal prioritises nodes by.

	// SESSION059: object-space radius of a sphere centred at centre_os that is GUARANTEED to enclose every original leaf
	// splat beneath this node (its full 3-sigma cutoff footprint, matching what the renderer actually draws) - used for
	// frustum-cull margin instead of feature_size. feature_size is a statistical fit (moment-matched covariance of the
	// merge) and is correct for "how big does the merged stand-in look on screen", but is NOT a bound on the true spread
	// of a node's descendants - see the mismatch this exact quantity is already known to have for layer_density, in the
	// comment on that field below ("re-fit covariance can end up more compact along its single largest axis than the
	// union of its children"). At small scenes that mismatch stayed under the traversal's frustum-cull margin and never
	// showed; at real-world (km-scale) scenes it doesn't, and a wrongly-culled high-level node drops its entire subtree,
	// visible as a whole rectangular chunk of the scene vanishing during straight-line camera translation (see session059
	// snapshot). Leaves: 1.5 * feature_size (the same 3-sigma cutoff radius the shader already draws to). Merged nodes:
	// max over children of (dist(child.centre_os, this.centre_os) + child.bounding_radius_os) - a proper bottom-up
	// enclosing-sphere bound, exact regardless of how the children are distributed.
	float bounding_radius_os;

	float layer_density; // Estimated overdraw (splat layers a ray would see if this node were expanded all the way down to full leaf resolution): (sum over direct children of that child's own cumulative leaf cross-section area) / (sum over direct children of that child's own cross-section area, pi * (1.5 * feature_size)^2 - the same 3-sigma cutoff radius the renderer actually draws out to) - see buildGaussianSplatLodTree()'s bookkeeping for where it's computed, and layerDensityFromSums()'s comment for why the denominator deliberately sums the *children's own* footprints rather than using this node's own (re-fit) feature_size: a widely-spread merge's covariance can end up more compact along its single largest axis than the true union of its children, which without this would let a parent come out denser than every one of its own children - this formulation is a weighted average of the children's own density values, so it's mathematically bounded between their min and max instead. 0 for leaves (nothing beneath them). Scale-invariant (areas scale as length^2 under a uniform scale, same on both sides of the ratio), so valid to read directly off the object-space tree without re-baking to world space - see GaussianSplatLodTraversalTask::run()'s use of it as a live-tunable traversal cutoff, independent of feature_size/pixel_scale_limit's screen-space one.

	uint32 child_start; // Index of the first child in the tree's linearised array. Always 0 on a node fresh out of mergeGaussianSplatLodNodes()/makeGaussianSplatLodLeafNode() - the tree builder is what actually places nodes into the array and fills this in.
	uint16 child_count; // 0 = leaf (an original, unmerged splat). Always 0 on a node fresh out of mergeGaussianSplatLodNodes()/makeGaussianSplatLodLeafNode(), for the same reason as child_start.
};


// SESSION071: which formulation derives a merged node's colour + opacity from its children. Selectable so the two can be
// A/B'd live on a loaded scene - see recolorLodTree().
enum GaussianSplatMergeColourMode
{
	// The session063 formulation, as built into the tree by buildGaussianSplatLodTree(): weight = opacity * VOLUME,
	// amplitude A = sum(weight) / parent volume, and where A > 1 the excess is multiplied into the colour before both are
	// clamped to [0, 1]. Two failure modes, both visible as the LoD coarsens (see session071):
	//   - A > 1 (children overlapping the parent's footprint more than once) drives the colour hard into the clamp, so a
	//     bright dense region merges to flat white and loses all its modulation.
	//   - The parent's volume includes the merge's spread-of-centres term, so a widely-spread group divides by a volume
	//     far larger than its children's, collapsing A towards 0 and letting the background show through.
	GaussianSplatMergeColourMode_Legacy = 0,

	// SESSION071: derived from conservation of on-screen premultiplied energy instead. A splat's contribution integrates
	// to alpha * colour * (silhouette area), so requiring the parent to match the sum of its children gives:
	//   colour = sum(alpha_i * colour_i * area_i) / sum(alpha_i * area_i)   (weight = opacity * AREA, not volume - a
	//                                                                        splat contributes in proportion to the area
	//                                                                        it covers, not to its volume)
	//   n      = sum(area_i) / area_parent                                  (how many layers deep the children cover the
	//                                                                        parent's own footprint)
	//   A      = 1 - pow(1 - mean_alpha, n)                                 (saturating composite of n layers)
	// A can no longer exceed 1 by construction, so the colour is never scaled up into the clamp; and a spread-out group
	// saturates towards its children's own opacity rather than collapsing towards transparent.
	GaussianSplatMergeColourMode_Energy = 1
};


// SESSION071: recomputes every MERGED node's colour + opacity under the given formulation, over colour/scale arrays held
// OUTSIDE the tree but indexed in lockstep with it (element i describes tree[i]) - which is how the renderer stores its
// world-baked copy, so an A/B toggle can re-derive colours on a loaded scene without rebuilding any tree or touching the
// shared, cross-thread-readable GaussianSplatData.
//
// Leaves are left exactly as they are: an original splat's colour is ground truth, never re-derived. Merged nodes are
// visited in one REVERSE pass, which is bottom-up here because buildGaussianSplatLodTree()'s breadth-first linearisation
// guarantees a node's children sit later in the array than the node itself - so every child has already been recomputed
// by the time its parent is reached, exactly as during the bottom-up build.
//
// Both formulations depend only on ratios of areas/volumes, so passing world-baked scales (leaf scale * uniform world
// scale) gives the same answer as object-space ones - the uniform factor cancels top and bottom.
//
// Re-running this is idempotent per mode and safe to switch back and forth: each merged node is fully rederived from its
// children, never accumulated onto its previous value.
//
// SESSION071 alpha_boost is a DIAGNOSTIC multiplier on a merged node's opacity (clamped to 1), for separating the two
// candidate causes of the residual darkening of bright dense surfaces that survived the Energy formulation: either the
// merged opacity is still too low there and the dark background shows through (in which case boosting it removes the
// darkening), or the colour itself is being averaged with interior splats that real front-to-back compositing never
// shows (in which case boosting only flattens the image and the grey stays). 1 = no change. Applies to merged nodes only,
// never to leaves, and compounds up the tree exactly as the merge itself does - so it is a probe, not a calibration knob.
void recolorLodTree(const std::vector<GaussianSplatLodNode>& tree, const Vec3f* scales, Vec4f* colours, GaussianSplatMergeColourMode mode, float alpha_boost = 1.f);


// Builds a leaf node directly from one splat's un-merged attributes (e.g. from GaussianSplatData's parallel arrays). child_count = 0, feature_size computed from scale.
GaussianSplatLodNode makeGaussianSplatLodLeafNode(const Vec3f& centre_os, const Vec3f& scale, const Vec4f& rotation, const Vec4f& colour);

// Merges 'num_children' nodes (leaves, already-merged nodes, or a mix) into one parent node's attributes - see the .cpp for the maths. Only centre_os/scale/rotation/colour/feature_size are set on the result;
// child_start/child_count are left at 0 - the tree builder, not this function, knows where the returned node will end up living in the linearised array. num_children must be >= 1.
GaussianSplatLodNode mergeGaussianSplatLodNodes(const GaussianSplatLodNode* children, size_t num_children);

// Builds a complete LoD tree from a flat splat cloud (Tiny-LoD style: bottom-up voxel-grid merge - see the .cpp for the algorithm). Transform-agnostic (see the centre_os field comment above) - operates in
// whichever space 'centres' is given in, doesn't know or care about world transforms.
//
// Returns the tree linearised into a single array: the result's [0] is always the root; every node's children occupy the contiguous range [child_start, child_start + child_count) later in the array.
//
// num_splats must be >= 1. lod_base is the grid-step growth factor between levels (> 1) - Spark's default for this method is 1.5 (a non-integer base gives smoother LoD transitions than doubling).
std::vector<GaussianSplatLodNode> buildGaussianSplatLodTree(const Vec3f* centres, const Vec3f* scales, const Vec4f* rotations, const Vec4f* colours, size_t num_splats, float lod_base = 1.5f);


namespace GaussianSplatLodTreeTests
{
	void test();
}
