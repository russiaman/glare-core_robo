/*=====================================================================
GaussianSplatSaturationGrid.h
------------------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../maths/vec2.h"
#include "../maths/vec3.h" // SESSION077: per-axis splat scales - see gsSatProjectedRadius().
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

The approach, in one sentence: walk the frontier front-to-back once, accumulating a per-direction "depth beyond which
this direction is saturated" map (sat_depth), then reject any node whose near edge lies beyond its direction's
sat_depth - a single read + compare per node, no cross-node dependency, so the filter keeps its SIMD-friendly shape.

SESSION076: the accumulation walks the FINE frontier, not the coarse floor it was originally written against. The
coarse version was measured to a dead end: at the only setting that preserved visible geometry it dropped 1.0% with a
ceiling of 10.8%, and no tuning moved that. The reason is structural - a coarse node is a round blob, while occlusion
boundaries follow silhouettes, which are not round. Any disc-shaped claim either under-covers (useless) or reaches
past the geometry it stands for and removes things that are plainly visible; there is no working point between. The
fine frontier is the actual geometry at the actual selected scale, so opacity accumulates where geometry IS, and the
boundary between a table and a chair back behind it falls out of the data instead of being arbitrated by a rule. This
is, in effect, a very low resolution software rasterisation of the same alpha the GPU accumulates every frame - see
the session076 snapshot for why we compute it rather than read it back.

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
// SESSION076 CALIBRATION: raised from 300000. The subdivision factor is now a live knob (see gsSatGridResForFocal())
// while the working point is being found, and at the values under test the old ceiling bound in every view - which is
// precisely the failure mode this constant's own comment warns about, the ceiling silently becoming the only thing
// setting resolution. Revisit downwards once the subdivision is frozen back into a constant.
static const int gsSatGridMaxTiles = 2500000;

// SESSION074: floor on grid resolution, guarding degenerate inputs (focal_px near zero, etc.) from producing a
// grid too coarse to be useful at all. 8x8 = 64 tiles is a deliberately generous floor - never expected to bind in
// practice, just a sanity backstop.
static const int gsSatGridMinRes = 8;


// SESSION076: how far out, in sigmas of the node's own Gaussian, this stage lets a node occlude - and the sigma the
// falloff within that extent is measured against. Equal to the shader's own draw cutoff, deliberately: the build pass
// now WEIGHTS each tile's contribution by the Gaussian instead of stamping it uniformly, so a tile at 3 sigma receives
// ~1% of the node's alpha, which is what it should receive. There is no reason to truncate the extent early once the
// weighting is right.
//
// Two earlier cuts got this wrong in opposite directions, and both failures are worth keeping because they are what
// established that the error was never in the radius:
//
//  - At 3 sigma with a UNIFORM stamp, every occluder asserted near-opaque coverage out to where it is ~1% of peak, so
//    each shadow was 3x too wide in angle and ~9x too large in area. Owner-visible as 100x200 px objects standing on a
//    table vanishing outright, and as WHICH object vanished changing under a 10-20cm camera step - "as if the mask were
//    attached to the camera rather than to the objects", the exact signature of shadows far wider than the geometry
//    casting them. The same run found the saturation threshold nearly inert between 0.96 and 0.9999 (only exactly 1.0,
//    which disables the stage, changed anything): per-tile transmittance was collapsing to ~0 regardless. A merged
//    node's alpha is a saturating composite, 1 - (1-mean_alpha)^n (GaussianSplatMergeColourMode_Energy), which goes to
//    1 for any node standing in for more than a handful of splats - so an over-wide disc of near-opaque alpha drove
//    every tile straight to zero.
//
//  - Shrunk to 1 sigma to fix that, the disc became exactly one tile ACROSS at the coarse scale (gsSatGridResForFocal()
//    sizes a tile to coarse_pixel_scale/focal_px), so the full-coverage write rule could not be satisfied by a
//    coarse-scale node at all. Measured: cA went to exactly 0 - every node captured for being at or below the coarse
//    pixel scale became unable to write - leaving only near-field terminal leaves as occluders. A chair three metres
//    away then had no representative in the grid and stopped occluding anything behind it, while thin near geometry
//    (a floor underfoot) still did.
//
// The threshold did become responsive at 1 sigma (dropped 1.2% at 0.96 against 25.6% at 0.06, where before it had been
// flat), which confirmed the transmittance-collapse half of the diagnosis even as the coverage half broke. Hence the
// current model: the node is a Gaussian, not a disc of any radius, and is treated as one. See the write loop.
static const float gs_sat_occluder_sigmas = 3.f;


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
//
// SESSION076 CALIBRATION: tile_subdiv divides the tile's angular size, i.e. res scales with it. It exists because the
// grid's angular resolution turned out to be the binding constraint on silhouette accuracy: a tile spans
// coarse_pixel_scale (30 px at the owner's settings), so a 30-50 px feature - a chair back protruding above a table -
// lands inside a single tile together with the near geometry beside it, and one scalar barrier per tile cannot keep one
// and drop the other. Live knob while the working point is found; to be frozen as a constant afterwards.
int gsSatGridResForFocal(float focal_px, float coarse_pixel_scale, float tile_subdiv);

// Tile index (row-major, [0, res*res)) for a direction (any positive length - see gsDirToOct()).
int gsSatGridTileForDir(const Vec4f& dir, int res);


// SESSION077: the world-space radius of the disc with the SAME AREA as this splat's cross-section as seen from
// 'unit_dir' - i.e. the footprint it can actually occlude with, in that one direction.
//
// Replaces max(scale.xyz), which the grid used before. That was the splat's bounding radius: correct for culling (it
// cannot reach further in ANY direction) and badly wrong for occlusion, because a 3DGS splat is typically a thin disc.
// Measured on the owner's interior: mean max/min scale ratio 101, 67% of occluders past 10:1, worst 63000:1 - so most
// splats were claiming a footprint orders of magnitude larger than they occlude with. Worse, the error is not a
// constant factor: it depends on the angle between the splat's thin axis and the view, so an opaque wall's mask broke
// up by how far each part sat from the surface normal instead of being uniform (owner-visible as a clear patch that
// stayed in the middle of a wall and painted everywhere else - session077).
//
// Derivation: for an ellipsoid with covariance C = R diag(s^2) R^T, the silhouette along unit d has area
// pi * sqrt(det C) * sqrt(d^T C^-1 d). Equal-area radius is the square root of (area/pi), giving
//
//   r_eff = sqrt( s0*s1*s2 * sqrt( sum_k (d . axis_k)^2 / s_k^2 ) )
//
// which needs only three dot products against the rotation's own axes - no matrix build, no 2D projection. Sanity
// checks: isotropic (s,s,s) gives exactly s from every direction; a disc (a,a,b) gives a seen down its thin axis and
// sqrt(a*b) seen edge-on, both the correct equal-area radii.
//
// Returned in the same units as 'scales', i.e. one sigma - the caller applies whatever sigma multiple it draws/occludes
// out to (see gs_sat_occluder_sigmas). rot is (x, y, z, w).
float gsSatProjectedRadius(const Vec3f& scales, const Vec4f& rot, const Vec4f& unit_dir);

// SESSION076: the pixel_scale write threshold that used to live here is gone. It existed because the write rule was
// binary (a node either fully covered a tile or contributed nothing), which made "can this node write at all?" a pure
// function of pixel_scale. With the Gaussian falloff there is no such cliff - every node contributes in proportion to
// its integrated occlusion - so no pixel_scale threshold can exclude a node without discarding real signal. The
// negligible-amplitude cutoff in the .cpp replaces it, applied to the contribution itself rather than to a proxy.

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
// SESSION078: alpha here is expected to already be the DRAWN opacity (gain/gamma applied - see adjustSplatAlpha() in
// GaussianSplatData.h), not the stored one - the caller (GaussianSplatLodTraversalTask::run()) applies the same
// transform the vertex shader does before building the occluder arrays this feeds. Closes what used to be a KNOWN GAP
// here: with the "ignore" alpha-adjust switch off, raising alpha in translucent regions (gain > 1, or gamma < 1) now
// genuinely raises what this stage can prune there, exactly as it raises what the renderer draws - the two agree by
// construction rather than one silently lagging the other.
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
//
// SESSION076: the coverage-entitlement parameter is gone with the coarse source it was invented for. Entitlement was an
// attempt to stop a round blob from claiming directions its silhouette never covered; the fix was to stop feeding the
// grid round blobs. With the FINE frontier as input, a node's footprint IS the geometry's footprint, and every node
// contributes in proportion to how much of a tile it actually covers - which is the whole model, with nothing left to
// arbitrate. See the .cpp's write loop.
// SESSION077 DIAGNOSTIC: out_accum_t, when non-null, receives the pass's final per-tile TRANSMITTANCE (res*res, 1 =
// nothing occludes this direction, 0 = fully blocked) - the running quantity the barrier decision is made from, which
// otherwise dies with this function's local scratch. Only the debug overlay reads it, and only while the "diag"
// checkbox is on: the binary "is this tile saturated?" that sat_depth_out records is a single threshold test away from
// it (t <= 1 - saturation_threshold), so the overlay can show either view from this one array.
//
// Requesting this ALSO changes what gets computed, not just what gets kept: with out_accum_t null (the production
// path - prune, or diag off), a tile stops being written to the instant it first crosses the threshold, so its
// transmittance is whatever value first tripped it, not the true product of every occluder that touched it - cheap,
// and irrelevant to sat_depth_out, which only ever asks "occluded or not". With out_accum_t non-null, every touching
// occluder keeps multiplying in regardless, so the overlay reads the UNCLAMPED value - see the .cpp write loop's
// already_saturated guard. sat_depth_out's own barrier is bit-identical either way: it is still latched on the FIRST
// crossing only, so nothing about the prune's behaviour changes by asking for this.
// SESSION077 DIAGNOSTIC: out_amp_sum, when non-null, receives the pass's per-tile SUM of every touching occluder's
// weighted contribution (amp * gaussian falloff to the tile) - unlike out_accum_t's transmittance, which is a PRODUCT
// of (1 - contribution) terms and is therefore bounded to [0, 1] and visually saturates towards 1 however much
// occluding mass actually piled up, this sum has no ceiling: a tile ten occluders deep reads roughly 10x a tile with
// one. Owner-requested (session077) specifically to see the RANGE of how much is contributing, not just whether the
// threshold was crossed - out_accum_t alone cannot distinguish "barely occluded" from "extremely occluded" once both
// are near 1. Computed under the same condition as out_accum_t (both are requested together by the one diag call
// site) - see the .cpp write loop.
// SESSION078 - REGION PRUNING: region_radius (R) makes the whole grid an assertion about a BALL of camera positions of
// that radius around anchor_pos_ws, not about the single point it is built from. R = 0 is exactly the old behaviour.
//
// Why: sat_depth was a statement about one camera position, so it went stale the moment the camera moved, and the
// ~300ms between "camera started moving" and "a fresh unpruned frontier arrived" showed as shadow-shaped holes (see the
// session076 snapshot's 5.1). Making the barrier valid for a whole neighbourhood up front means it does not go stale
// inside that neighbourhood at all - there is nothing to switch between and nothing to re-latch.
//
// A ball, not a box, deliberately: the radius enters as one scalar in every direction. A cube's support function is
// R*(|ux|+|uy|+|uz|), which swings from R along the world axes to R*sqrt(3) along the diagonals, so it would either cost
// a flat sqrt(3) of extra dilation (bounding it by the circumscribed sphere) or need per-direction handling whose
// angular half is a hexagonal cross-section rather than a disc. A ball is also rotation-invariant, which matches the
// orientation-invariance sat_depth already has by construction. If a voxel CACHE is layered on later, the cube is the
// right shape for the addressing (floor(pos/S) as a key) and the ball is still the right shape for the math - a cell of
// side S is covered by R = S*sqrt(3)/2.
//
// Both sides pay for it, and both move in the safe direction (neither can ever drop more than R = 0 would):
//  - build: the occluder's barrier is pushed out to dist + r + R, the furthest its far edge sits from any ball point.
//    Its angular footprint is deliberately NOT eroded - see the write loop, which records why the first cut did erode
//    it and why that was wrong (per-atom erosion shrinks a dense surface's interior coverage, not just its silhouette,
//    and at R = 0.02 deleted most of a near wall's occluders outright).
//  - read: a node is dropped only if occluded from EVERY ball point, so its footprint is DILATED by R/d (every tile in
//    the widened span must agree) and it must clear the barrier by an extra R in depth. This is where the angular half
//    of the conservatism lives, acting on the finished aggregate mask rather than on individual occluders.
//
// The cost is concentrated near the camera and negligible far away, which is the useful shape: R/d is the angular
// penalty, so at the owner's settings (tile ~= 4.8 deg) an R of 0.6m costs 0.35 tiles at 20m but 2.4 tiles at 3m - and
// what this stage drops is distant geometry, while what does the occluding is near.
void gsBuildSaturationGrid(const float* px, const float* py, const float* pz, const float* radius, const float* alpha, size_t n,
	const Vec4f& anchor_pos_ws, int res, float saturation_threshold, float region_radius,
	js::Vector<float, 16>& sat_depth_out, size_t* out_writers = NULL, size_t* out_tile_writes = NULL,
	js::Vector<float, 16>* out_accum_t = NULL, js::Vector<float, 16>* out_amp_sum = NULL,
	size_t* out_tile_iters = NULL); // SESSION079 DIAGNOSTIC: per-tile loop iterations, against out_tile_writes which counts only those reaching the accumulator - the two together say how much of the tile loop is wasted work.


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
	const js::Vector<float, 16>& sat_depth, int res, float region_radius, bool* out_aggressive = NULL);
