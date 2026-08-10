/*=====================================================================
GaussianSplatCoplanarMerge.h
-----------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../maths/vec3.h"
#include "../maths/Vec4f.h"
#include "../utils/Vector.h"


/*=====================================================================
GaussianSplatCoplanarMerge
---------------------------
Collapses groups of near-duplicate splats - splats sitting in the same place,
facing the same way and carrying the same colour - into one splat each.

This is a different operation from the LoD tree's merge, and the difference is
the whole point.  The tree merges whatever falls in a grid cell, so its stand-in
is as big as the cell: it cuts the splat count without cutting the area painted,
which is why coarsening the tree buys six times the splats for a third of the
fill.  The rule here is the opposite one - a group is only collapsed if the
replacement paints *less* than the members did, and the group is refused
outright otherwise.  So this reduces fill, which is what the splat pass is bound
by, rather than reducing splat count.

The cost model is a sum of ellipse areas, not their union: every splat in a
stack is rasterised in full whether or not the one in front of it covered the
same pixels, so a stack of k splats really does cost k times one splat's area.
That is what makes collapsing a stack a genuine saving, and it is also why the
window is narrow - see the note on the tolerances below.

Operates on the flat parallel arrays GaussianSplatData holds (and in whatever
space they are given in - object space at load time), in place, so it can run
either before buildGaussianSplatLodTree() at load time or against a live cloud
whose tree is then rebuilt.  It knows nothing about world transforms, the LoD
tree, or the renderer.

Choosing the tolerances
-----------------------
'across' is the one that decides whether this pays at all, and it wants to be
small.  Merging k splats spread over a lateral radius d replaces k*pi*r^2 of
fill with pi*(r+d)^2, so it only pays while d stays under about (sqrt(k)-1)*r -
and k itself only grows as d^2.  Measured on a reference capture (median
spacing 2.7 cm against a splat radius of 5 cm), an 'across' of 1 cm is around
break-even and 5 cm loses badly.  What does pay is depth: 'through' collapses a
stack without widening anything, so it can be set well past 'across'.
=====================================================================*/


struct GaussianSplatCoplanarMergeParams
{
	// Reach around a seed splat, in the units the arrays are given in.  'through' is measured along the seed's own facing
	// (the direction its flattest axis points), 'across' at right angles to it - so the reach is a thin disc lying in the
	// surface, not a ball.  They are separated because they buy different things: see the header note.
	float across;
	float through;

	float colour_tol; // Allowed difference per RGB channel, on the [0, 1] scale the colours are stored in.
	float angle_tol_deg; // Allowed difference in facing.  Compared as an absolute dot product: a disc has no front and back, so opposite normals are the same facing.

	// The alpha the renderer cuts a splat's quad off at, not a merge tolerance: it is set from the renderer's own
	// getAlphaCutoff() rather than by whoever is choosing the tolerances above.
	//
	// It is here because a splat is drawn out to sqrt(2 * ln(opacity / alpha_cutoff)) sigmas, capped at 3 - so a more
	// opaque splat is rasterised over a wider ellipse than a faint one of the same shape.  A collapsed group is more
	// opaque than its members were, which means the replacement is drawn wider than its scale alone would say, and
	// leaving that out of the "does this group pay?" test below would flatter every merge by up to a factor of 1.5.
	float alpha_cutoff;
};


struct GaussianSplatCoplanarMergeStats
{
	size_t splats_in, splats_out;
	size_t groups_merged;  // Groups of two or more that were collapsed.
	size_t groups_refused; // Groups of two or more whose replacement would have painted at least as much as its members did, so they were left alone.  Expected to be the majority on a capture with little real stacking - that is the safety condition working, not a failure.
	size_t largest_group_merged; // Largest group actually collapsed.

	// What the grouping found, before any of it was judged.  These are here to keep "nothing was merged" readable, because
	// on its own it has two completely different meanings: the grouping found nothing to merge, or it found plenty and the
	// area test refused it.  Those want opposite next steps, and the last time this distinction was left to inference the
	// answer came out backwards for a whole session.  groups_total counts every group including the singletons, so
	// splats_in / groups_total is the mean group size; largest_group_found says whether a crowded neighbourhood was
	// reachable at all at these tolerances.
	size_t groups_total;
	size_t largest_group_found;
	double area_in, area_out; // Summed drawn area (the ellipse the renderer actually rasterises, seen face-on) before and after, in the input's own units squared.  The ratio is the fill saving this merge would give a camera that sees every splat face-on, which is the closest a view-independent measure gets to the renderer's own fill.
};


// Collapses near-duplicate splats in the four parallel arrays, in place: the arrays are rewritten with the merged cloud
// and resized to it.  All four must be the same length; a cloud of fewer than two splats is returned untouched.
//
// Splat order is not preserved.
GaussianSplatCoplanarMergeStats coplanarMergeSplats(js::Vector<Vec3f, 16>& positions, js::Vector<Vec3f, 16>& scales, js::Vector<Vec4f, 16>& rotations,
	js::Vector<Vec4f, 16>& colours, const GaussianSplatCoplanarMergeParams& params);
