/*=====================================================================
GaussianSplatSaturationGrid.h
------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../maths/vec2.h"
#include "../maths/Vec4f.h"
#include "../utils/Vector.h"


/*=====================================================================
GaussianSplatSaturationGrid
-----------------------------
SESSION074: CPU-side pre-GPU occlusion cull. See snapshots/2026-08-24-session074-cpu-saturation-prefilter-plan.md
for the full design and rationale; this is the terse version.

The problem: session045 measured that a typical pixel in the stress scene is composited ~145 times, but the first
~20 layers already give 95% of the final image quality - so ~90% of the blend cost (the single largest cost
component in the splat pass) is spent on splats that contribute almost nothing, because something nearer on the
same ray already made the pixel opaque. The existing saturation gate (vertex shader) and layer/coverage cap
(fragment shader) both act AFTER a splat has already cost its vertex-stage issue (K); this stage instead decides,
on the CPU, before a node is even considered for the draw list, whether it is behind already-saturated geometry -
cutting both K and the blend cost, not just the blend cost.

The approach, in one sentence: walk the LoD tree's existing coarse floor (K4, session063 - a cheap, already-sorted,
already-complete low-detail cut of the whole scene, ~800k nodes vs ~11-16M in the fine frontier) front-to-back once,
accumulating a per-direction "depth beyond which this direction is saturated" map (sat_depth), then let the
per-frame fine-node filter reject any fine node whose near edge lies beyond its direction's sat_depth - a single
read + compare per node, no cross-node dependency, so the filter keeps its existing SIMD-friendly shape.

Why a DIRECTION map, not a screen-space tile grid: a screen-space grid depends on orientation and would have to be
rebuilt every frame (the cost the plan's §2.3 estimate worried about). A grid anchored on unit directions from the
camera position is orientation-INVARIANT, so it can be built once inside the same async traversal task that builds
U(P) (session062-063 split architecture) and reused across every filter kick of a pure rotation, exactly like U(P)
itself - meaning the expensive sequential accumulation pass costs nothing per frame; only the read in the filter
does.

The direction <-> tile mapping is the octahedral encoding already used elsewhere in this engine for irradiance
probes (float32x3_to_oct()/oct_to_float32x3() in opengl/shaders/frag_utils.glsl, itself from Cigolle et al., "A
Survey of Efficient Representations for Independent Unit Vectors") - reimplemented here in plain C++ rather than
shared with the GLSL, since this runs on the CPU inside the traversal/filter tasks, not in a shader.

sat_depth semantics: for each tile, the WORLD-SPACE DISTANCE from the anchor position beyond which that direction is
considered occluded by nearer coarse geometry, or +inf if no coarse node saturated that direction. Recorded at a
saturating node's FAR edge (dist + radius), not its centre or near edge - deliberately conservative (see
gsBuildSaturationGrid()'s comment): places the cut only past the entire mass of coarse geometry that caused the
saturation, never inside or in front of it, so this stage can only drop fine geometry that is unambiguously behind
already-opaque coverage, never something that might still be visible in front of or alongside it.
=====================================================================*/


// SESSION074: fixed safety ceiling on grid cell count, independent of scene/settings. num_tiles grows quadratically
// with focal_px (zoom/narrow FOV, e.g. photoMode) and inversely with coarse_pixel_scale, so an unclamped formula can
// run away under an owner-chosen combination of the two. Exceeding this ceiling simply coarsens the grid (bigger
// cells -> more conservative, cheaper, weaker cull) rather than growing the grid unboundedly - see
// gsSatGridResForFocal(). Not a per-scene tuning knob (project rule: no manual per-scene tuning) - a fixed code
// constant, same in every scene.
//
// SESSION074 REVISION: was 300000 while the grid ran at K=4 (tile a quarter of a coarse node). Measured on the stress
// scene, that combination pinned res at the ceiling in EVERY view - the derivation below never got to express itself,
// the ceiling was silently the only thing setting resolution. With K=1 (see gsSatGridResForFocal()) the natural res
// is ~4x smaller per axis, so this ceiling is back to being what it was meant to be: an unreachable backstop for
// extreme zoom, not the operating point.
static const int gsSatGridMaxTiles = 300000;

// SESSION074: floor on grid resolution, guarding degenerate inputs (focal_px near zero, etc.) from producing a
// grid too coarse to be useful at all. 8x8 = 64 tiles is a deliberately generous floor - never expected to bind in
// practice, just a sanity backstop.
static const int gsSatGridMinRes = 8;


// SESSION076: how far out, in sigmas of the node's own Gaussian, this stage treats a node as an OCCLUDER.
//
// This is NOT splat_cutoff_sigmas (3), which is where the shader stops drawing the splat because it has become
// invisible. The build pass stamps "this tile is behind opaque coverage" uniformly across every tile the node's disc
// covers, so the radius it is given must be the radius out to which the node is actually near-opaque - and a Gaussian
// at 3 sigma is exp(-4.5) ~= 0.011 of its peak, i.e. transparent. Using the cutoff radius made every occluder cast a
// shadow 3x too wide in angle and ~9x too large in area.
//
// Measured consequence, and how this was found (session076): the owner reported glasses and cups standing on a table -
// 100x200 px objects, plainly visible - vanishing when the stage was switched on, with which object vanished changing
// under a 10-20cm camera step, "as if the mask were attached to the camera rather than to the objects". That is the
// signature of shadows far wider than the geometry casting them. The same run showed the saturation threshold to be
// nearly inert between 0.96 and 0.9999 (only exactly 1.0, which disables the mechanism, changed anything), meaning
// per-tile transmittance was collapsing to ~0 regardless - the second half of the same problem, since a merged node's
// alpha is a saturating composite (1 - (1-mean_alpha)^n, see GaussianSplatMergeColourMode_Energy) that goes to 1 for
// any node standing in for more than a handful of splats. An over-wide disc of near-opaque alpha is exactly what
// drives every tile to zero.
//
// 1 sigma (exp(-0.5) ~= 0.61 of peak) is the value under test. It also makes the grid self-consistent for the first
// time: gsSatGridResForFocal() sizes a tile to coarse_pixel_scale/focal_px, which is one node DIAMETER at 1 sigma, so
// a node covers about one tile - the "1-4 tiles" that constant's own comment claims and never actually got.
static const float gs_sat_occluder_sigmas = 1.f;


// Octahedral direction -> unit-square coordinate, in [-1, 1]^2.
//
// SESSION074: 'dir' does NOT have to be unit length. The mapping divides by the vector's own L1 norm, so it is
// invariant under any positive scaling - feeding a raw (node_pos - anchor) offset gives exactly the same answer as
// feeding its normalised form, which is why no caller normalises (a per-node sqrt + 3 divides that measured as a
// large part of this stage's first-cut cost, removed once this property was noticed).
//
// Matches float32x3_to_oct() in frag_utils.glsl exactly (same formula, same reference - Cigolle et al.) so the two
// are provably the same mapping if ever compared, not just conceptually similar.
Vec2f gsDirToOct(const Vec4f& dir);

// Grid resolution (grid is res x res, covering the WHOLE sphere in one octahedral square - see gsDirToOct()) derived
// from the existing coarse_pixel_scale/focal_px knobs, no new UI parameter - see the .cpp for the derivation.
int gsSatGridResForFocal(float focal_px, float coarse_pixel_scale);

// Tile index (row-major, [0, res*res)) for a direction (any positive length - see gsDirToOct()).
int gsSatGridTileForDir(const Vec4f& dir, int res);

// SESSION076 DIAGNOSTIC: the smallest pixel_scale a node can have and still write to the grid, as a MULTIPLE of
// coarse_pixel_scale.
//
// The build pass only accepts a node that fully covers at least one tile (see gsBuildSaturationGrid()'s cover_radius
// test), which makes "can this node contribute at all?" a pure function of pixel_scale. Collecting the algebra in one
// place so the traversal's diagnostic can predict the build pass's own verdict without duplicating its constants:
//
//   a tile subtends           coarse_pixel_scale / focal_px  radians   (by construction, gsSatGridResForFocal())
//   a node's angular radius   0.5*cutoff_sigmas*feature_size / dist    (the radius the renderer feeds this grid)
//                           = 0.5*cutoff_sigmas * pixel_scale / focal_px
//   so radius_tiles         = 0.5*cutoff_sigmas * pixel_scale / coarse_pixel_scale
//   and cover_radius > 0    <=>  pixel_scale > (tile half-diagonal / (0.5*cutoff_sigmas)) * coarse_pixel_scale
//
// Pass the OCCLUDER sigmas (gs_sat_occluder_sigmas), the same value the radius fed to the build pass is derived from -
// not the shader's cutoff sigmas. At 3 the factor was ~0.47, below the capture rule's own threshold; at 1 it is ~1.41,
// above it, which means no node captured merely for being at or under coarse_pixel_scale can write at all and only the
// larger terminal captures reach the grid. That is a real narrowing of what feeds the grid, and it is the point: those
// were the nodes whose over-wide shadows were removing visible geometry.
float gsSatGridMinWritingPixelScaleFactor(float splat_cutoff_sigmas);

// Angular size of one grid tile at the given resolution - the SAME quantity gsSatGridResForFocal() solves for (a tile
// of solid angle ~= tile_ang^2 covering, together with res*res of its neighbours, the whole sphere's 4*pi steradians),
// returned here so every caller shares one consistent notion of "how big is a tile" rather than re-deriving its own
// approximation (an earlier draft of this file did exactly that, inconsistently - see the session074 plan snapshot's
// implementation log). Includes a fixed distortion-margin factor (see the .cpp) since the octahedral mapping is not
// equal-area; stays on the generous (bigger-tile) side everywhere rather than being exact anywhere - correctness-safe
// direction for this stage, see this file's header comment.
float gsSatGridTileAngle(int res);


// SESSION074: pass 1 - the coarse-floor accumulation pass. Walks the coarse-only SoA (already front-to-back sorted,
// since it is a filtered subsequence of the traversal's globally-sorted output - see GaussianSplatUnculledFrontier)
// ONCE, sequentially (this is the only sequential, order-dependent part of the whole mechanism - see the file
// header), building a per-tile saturation depth.
//
// SESSION074 REVISION - each node is rasterised into EVERY tile its angular footprint fully covers, not just the one
// its centre direction lands in. The first cut wrote a single tile per node while simultaneously *requiring* the node
// to be wider than a tile - i.e. it established full coverage of a neighbourhood and then discarded all but 1/16th of
// it (measured: a coarse node at pixel scale 30 spans ~4 tiles per axis). That both suppressed the saturated fraction
// directly and, worse, made the saturated set speckled rather than contiguous, which the conservative neighbourhood
// test on the read side then penalised a second time. Covering the real footprint fixes both at once, and is why the
// tile is now sized to the node (K=1) instead of a quarter of it - see gsSatGridResForFocal().
//
// alpha here is expected to be each coarse node's own (possibly merge-derived) opacity - see
// GaussianSplatMergeColourMode_Energy in GaussianSplatLodTree.h for why that value is expected to already represent
// a saturating composite of what the node stands in for, not a single splat's raw opacity. This function does not
// second-guess that value; it is the caller's job (and this stage's Gate A, see the plan snapshot) to confirm it
// actually saturates before trusting drop decisions built on it.
//
// KNOWN GAP, deliberately left: alpha is the STORED opacity, not what the renderer actually draws - the live
// gain/gamma adjustment (adjustSplatAlpha() in GaussianSplatData.h) is applied at draw time and is not accounted for
// here. Inert while the panel's "ignore" alpha-adjust switch is on, which is its default; if that is ever turned off
// with a gain far from 1, this grid's opacities are the wrong ones. Fix belongs with whoever wires the adjustment in,
// not here.
//
// n arrays (px/py/pz/radius/alpha) must all be the same length and given in the SAME front-to-back order as the
// source frontier. saturation_threshold is the existing splat_saturation_threshold knob (no new threshold is
// introduced - see the plan's "no manual per-scene tuning" constraint). sat_depth_out is resized to res*res and
// filled; a tile with no saturating coverage is left at +inf.
//
// SESSION076 DIAGNOSTIC: out_writers / out_tile_writes, when non-null, are INCREMENTED (not reset - the caller owns
// their zeroing) with the number of nodes that cleared the full-coverage test and the total number of per-tile writes
// those nodes made. Both are gated by the "sat diag" checkbox at the call site and left null otherwise, so the hot
// loop pays only a null test per node in normal operation.
void gsBuildSaturationGrid(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers = NULL, size_t* out_tile_writes = NULL);


// SESSION074: pass 2's per-node read - true if this fine node is unambiguously behind saturated coarse geometry, so
// it can be dropped before it ever costs a GPU instance. offset is (node_pos - anchor_pos_ws), UNNORMALISED (see
// gsDirToOct()); dist_sq is its squared length, which every caller already has.
//
// SESSION074 REVISION - tests the tiles the node's own angular footprint actually touches, rather than reading a
// blanket 3x3-dilated map. The first cut dilated the whole grid by a 3x3 max so a node straddling a tile boundary
// would be compared against a safe value; that is a +-1-whole-tile margin applied to a node whose angular radius is
// a fraction of a tile, and it interacted badly with the sparse writes the build pass used to produce. Now the node's
// footprint is turned into a small tile span (usually 1 tile, 2 or 4 when it straddles) and every touched tile must
// agree the node is occluded - the same conservative intent, at the node's true scale rather than the grid's.
//
// The depth test is done squared, against (sat_depth + radius)^2, so no sqrt of dist is ever taken.  Returns false
// when res == 0 ("no grid built" - stage disabled, or the coarse floor was empty), so a missing grid can never cause
// a drop.
//
// out_aggressive, if non-null, receives a deliberately-wrong upper-bound verdict computed from the same tile data:
// centre tile only, node treated as a point. Purely Stage A's "how much could this ever cut" ceiling - see
// [gsr-sat]'s would_drop_aggr. Nothing in the real drop path reads it.
bool gsSatOccluded(const Vec4f& offset, float dist_sq, float node_radius,
	const js::Vector<float, 16>& sat_depth, int res, bool* out_aggressive = NULL);
