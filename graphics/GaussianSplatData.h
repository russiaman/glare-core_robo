/*=====================================================================
GaussianSplatData.h
-------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "GaussianSplatLodTree.h"
#include "../maths/vec3.h"
#include "../maths/Vec4f.h"
#include "../physics/jscol_aabbox.h"
#include "../utils/Vector.h"
#include "../utils/Reference.h"
#include "../utils/ThreadSafeRefCounted.h"
#include <vector>


/*=====================================================================
GaussianSplatData
-----------------
A decoded Gaussian splat cloud in object space: one entry per splat in each
of the parallel arrays below.

This is the CPU-side, GPU-agnostic representation - it says nothing about how
the splats are packed into textures for rendering (see
opengl/GaussianSplatRenderer.h for that), and nothing about the file format
they were decoded from (see SOGDecoder.h).

Note that only the DC term (sh0) of the spherical harmonics is kept, so
colours are view-independent.
=====================================================================*/
class GaussianSplatData : public ThreadSafeRefCounted
{
public:
	size_t numSplats() const { return positions.size(); }

	// How many entries a GPU upload of this cloud actually needs: every tree node (leaves + merged) if lod_tree has been built, else just the leaves. GaussianSplatRenderer uploads and accounts capacity in
	// terms of this, not numSplats(), once a tree exists - see bakeMember() in GaussianSplatRenderer.cpp.
	size_t numNodes() const { return lod_tree.empty() ? numSplats() : lod_tree.size(); }

	js::Vector<Vec3f, 16> positions;
	js::Vector<Vec3f, 16> scales; // Per-axis linear scale factors (already exponentiated out of the log domain the file stores them in).
	js::Vector<Vec4f, 16> rotations; // Unit quaternions, stored as (x, y, z, w).
	js::Vector<Vec4f, 16> colours; // (r, g, b, opacity), all in [0, 1].

	js::AABBox aabb_os; // Bounds the splat centres only - does not account for the extent of the splats themselves.

	// On-the-fly LoD tree, built asynchronously from positions/scales/rotations/colours above right after this GaussianSplatData is decoded (see LoadModelTask.cpp's .sog branch), in this same object space, so
	// that a moved/scaled object's tree can be re-baked to world space the same way its leaf splats already are, without rebuilding the tree itself (see GaussianSplatLodTree.h). std::vector rather than
	// js::Vector like the members above, matching buildGaussianSplatLodTree()'s return type directly. Empty if the build hasn't happened yet, or failed - not fatal to rendering the splat cloud, callers must
	// treat an empty tree the same as "no LoD, render every splat".
	std::vector<GaussianSplatLodNode> lod_tree;
};


typedef Reference<GaussianSplatData> GaussianSplatDataRef;
