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
#include "../maths/mathstypes.h"
#include <vector>
#include <cmath>


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


/*
The live opacity adjustment: alpha' = gain * alpha^gamma, clamped to the [0, 1] the stored alpha lives in.

Kept here, next to the data it applies to, because it has to be applied in more than one place and they have to agree:
gaussian_splat_vert_shader.glsl applies it to what is drawn, and the CPU-side cost predictions (see splatFootprint() and
drawnArea()) have to describe the same splats.  The shader carries its own copy of these two lines - GLSL cannot include
this - so a change here is a change there.

Applied as early as anything reads the alpha, i.e. before the quad radius is derived from it, so it moves the area a
splat is rasterised over and not just the colour of the pixels under it.  gain = gamma = 1 is the identity, and is what
the renderer holds unless the panel says otherwise.

The gamma goes first: it reshapes the distribution (it is the one that can lift the low-opacity bulk the fill sits in
relative to the rest), and the gain then scales the result.  The other order is the same family of curves with a
different meaning per knob, but this way gain stays a plain "how much denser", independent of gamma.
*/
inline float adjustSplatAlpha(float alpha, float gain, float gamma)
{
	const float adjusted = (gamma == 1.f) ? alpha : std::pow(alpha, gamma); // Skip the pow() in the default case: this runs per splat over millions of them in the report paths.
	return myMin(gain * adjusted, 1.f);
}
