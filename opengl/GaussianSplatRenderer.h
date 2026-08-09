/*=====================================================================
GaussianSplatRenderer.h
-----------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../graphics/GaussianSplatData.h"
#include "../maths/Quat.h"
#include "../maths/Vec4f.h"
#include "../physics/jscol_aabbox.h"
#include "../utils/Platform.h"
#include "../utils/Reference.h"
#include "../utils/ThreadMessage.h"
#include "../utils/ThreadSafeQueue.h"
#include "../utils/Vector.h"
#include <map>
#include <string>
#include <vector>


class OpenGLEngine;
class OpenGLProgram;
namespace glare { class TaskManager; }
class SplatCloud; // Defined in GaussianSplatRenderer.cpp - one drawable cloud, holding one or more splat objects.
struct CloudMember; // Defined in GaussianSplatRenderer.cpp - one registered splat object within a cloud.
class GaussianSplatSortScratch; // Defined in GaussianSplatRenderer.cpp - the reusable working buffers a background depth-sort uses.
class GaussianSplatLodTraversalScratch; // Defined in GaussianSplatRenderer.cpp - the reusable working buffers a background LoD traversal uses.


/*=====================================================================
GaussianSplatRenderer
---------------------
Renders Gaussian splat clouds.  Owned by OpenGLEngine - get at it with
OpenGLEngine::getSplatRenderer().  Callers register a cloud with addObject()
and keep the returned Handle to later move or remove it.

Splats are semi-transparent and have to be composited in depth order, so every
cloud carries a depth sort.  That order is front-to-back (nearest first), blended
with the "under" operator - algebraically the same composite as the more usual
back-to-front "over", but the only direction in which a pixel can be known to
have saturated while there is still work left to skip.  Clouds are drawn by
OpenGLEngine::drawSplatClouds(), which is a pass of its own rather than part of
the alpha-blended pass; see the comment there for why.

That pass blends into an accumulation buffer of its own and resolves it onto the
main colour buffer afterwards, because splats have to be blended in the
display-referred sRGB space 3DGS fits them in rather than in the engine's linear
space - see gaussian_splat_frag_shader.glsl.

Partitioning
------------
Each registered object usually gets a drawable cloud of its own, which is what
makes frustum culling, cheap add/remove and a per-cloud sort budget possible.
That only works while the clouds can be correctly ordered against each other.
Two clouds with disjoint AABBs always can be: a disjoint pair is separated by an
axis-aligned plane, and every ray from the camera crosses that plane at most
once, so the cloud on the camera's side is nearer along every ray that hits
both - true even where the two overlap in screen space.

Objects whose bounds *do* intersect have no such plane, so they are merged into
one cloud, where the per-splat sort orders them against each other and there is
no cross-cloud ordering left to get wrong.  Merging is by union AABB and runs to
a fixpoint, since a merged cloud's bounds can in turn intersect a third cloud.
That over-merges in some arrangements - the union of two diagonally placed boxes
contains corners neither of them occupies - which costs culling granularity but
never correctness.  Under-merging is the direction that would break, which is
why member bounds are padded by the splat extent rather than bounding the splat
centres alone.

Within a cloud, splat data is baked into world space, so the GLObject's
ob_to_world_matrix stays identity, and each member owns a [offset, count) range
of the shared arrays.

The depth sort runs on a worker thread in two stages: a fast approximate
counting sort is posted first so the view updates promptly, followed by a
precise radix sort.  Splats are sorted by distance from the camera rather than
by depth along the view axis, which makes the resulting order invariant to
camera rotation, so only camera *movement* triggers a re-sort - and the distance
a cloud's camera must move to earn one scales with how far away the cloud is.

Concurrency: a sort worker never reads the live splat arrays, since the main
thread can reallocate them.  think() copies the positions into a snapshot buffer
on the main thread when it kicks a sort off, and the worker only reads that.  A
plain append while a sort is in flight is safe without extra bookkeeping,
because the in-flight result only covers a prefix of the splats and the appended
tail is already in identity order.  Anything that renumbers a cloud bumps its
structure_generation, which every result carries and which think() checks before
applying a result; results for a cloud that has since been merged away are
dropped by cloud id.

Not handled:
 - order-independent transparency.
 - non-uniform scaling of a cloud.
 - more splats in a single cloud than one texture can address (see
   maxSplatsPerCloud()).  A merge that would exceed it is refused, leaving the
   clouds separate and their relative order approximate.
=====================================================================*/
class GaussianSplatRenderer
{
public:
	// Doesn't touch OpenGL: the shader program is built on the first addObject() call, so an engine that never renders
	// a splat cloud doesn't pay for it.
	GaussianSplatRenderer(OpenGLEngine& opengl_engine);
	~GaussianSplatRenderer();

	// Identifies a splat cloud that has been registered.  Callers keep this to later move or remove it.
	typedef uint64 Handle;
	static const Handle invalid_handle = 0;

	// Bakes splat_data's positions/scales/rotations into world space with the given pose, and registers the result.
	// Note that non-uniform scaling isn't supported.
	//
	// Throws glare::Exception if splat_data has more splats than maxSplatsPerCloud()
	Handle addObject(const GaussianSplatDataRef& splat_data, const Vec4f& translation_ws, const Quat<float>& rotation_ws, float uniform_scale_ws);

	// Re-bakes a cloud with a new pose.  Returns false if the handle isn't valid.
	bool updateObjectTransform(Handle handle, const Vec4f& translation_ws, const Quat<float>& rotation_ws, float uniform_scale_ws);

	// Returns false if the handle isn't valid.
	bool removeObject(Handle handle);

	// Removes every object, invalidating all outstanding handles.
	void removeAllObjects();

	bool isValidHandle(Handle handle) const;

	// The most splats one drawable cloud can hold, given the real GL_MAX_TEXTURE_SIZE.  Note that this is a per-cloud
	// limit, not a world-wide one: several clouds of this size can coexist, so long as they don't have to be merged.
	size_t maxSplatsPerCloud() const;

	size_t numSplatsInWorld() const;
	size_t numObjectsInWorld() const;
	size_t numDrawableClouds() const { return clouds.size(); } // How many clouds the partition has settled on.

	// The program OpenGLEngine::drawSplatClouds() resolves the splat accumulation buffer with.  Null until the first
	// addObject(), like the splat program itself.
	const Reference<OpenGLProgram>& getResolveProgram() const { return resolve_prog; }
	const Reference<OpenGLProgram>& getShaderProgram()  const { return shader_prog; }

	// Multi-line summary of the partition, the sort state and GPU/CPU memory use, for the diagnostics display.
	// Returns an empty string if no splat object is registered, so it costs nothing in a world without any.
	std::string getDiagnostics() const;

	// One-off diagnostic (GaussianSplatSettingsWidget's "Count in frustum" button, Qt only): counts splats, across every
	// cloud, that are both in the camera's current frustum and pass the size clamp above (same test as the shader's,
	// including the invert flag - if the clamp is disabled, every in-frustum splat counts). O(total splats in the
	// world); meant to be triggered once by a button click, not called per-frame. Answers "how many of what the size
	// filter is currently isolating are actually in view right now" without needing a GPU capture.
	size_t countSplatsInFrustum() const;

	// One-off diagnostic (GaussianSplatSettingsWidget's "Frustum report" button, Qt only): a multi-line breakdown of what
	// the LoD hierarchy offers at the current camera position and what the traversal actually took from it, for every
	// cloud in the world rather than any particular object.
	//
	// Answers one question the per-frame diagnostics can't: for each drawn node, *why* the traversal stopped unfolding
	// there. That splits the drawn splats into the ones a LoD parameter could still remove (converged, or stopped by the
	// density/depth/budget caps - a coarser stand-in exists above them) and the ones no parameter can (leaves, where the
	// tree has nothing coarser left and only merging or pruning the source cloud can reduce the fill cost). Alongside it,
	// how much of the tree is unfolded in view, the depth and branching of the trees themselves, and the layer_density
	// distribution of what is being drawn - which is what says whether the density cap has anything to bite on.
	//
	// Runs one full traversal per cloud synchronously, on the main thread, with the current camera and the current live
	// parameters (the per-frame frontier isn't kept on the CPU to read back), so it costs a visible hitch on a large cloud
	// - meant for a button click, never per frame. Changes no renderer state: the traversal's result is read and dropped,
	// not applied.
	std::string getFrustumStructureReport();

	// Forces every cloud's LoD frontier to be recomputed on the next think()/kickOffTraversals(), bypassing the normal
	// camera-movement threshold - for GaussianSplatSettingsWidget's live traversal parameters (pixel_scale_limit,
	// max_splats_budget, max_layer_density, max_tree_depth), so a value change is visible immediately rather than
	// waiting for the camera to move far enough to naturally trigger a re-traversal.
	void forceTraversalRefresh();

	// Per-frame update: refreshes the uniforms the shader needs for the covariance projection, applies any completed
	// background sorts, and kicks off new ones for clouds whose camera has moved far enough.  Called by
	// OpenGLEngine::draw(), after the frame's camera transform has been set.
	void think();

	// Live-tunable LoD traversal parameters (GaussianSplatSettingsWidget, Qt only) - take effect on a cloud's next
	// traversal kick-off, no reload needed.  See kickOffTraversals()/GaussianSplatLodTraversalTask for how each is used.
	float getPixelScaleLimit() const { return lod_pixel_scale_limit; }
	void setPixelScaleLimit(float v) { lod_pixel_scale_limit = v; }
	size_t getMaxSplatsBudget() const { return lod_max_splats_budget; }
	void setMaxSplatsBudget(size_t v) { lod_max_splats_budget = v; }
	float getResortMoveThresholdWS() const { return lod_resort_move_threshold_ws; }
	void setResortMoveThresholdWS(float v) { lod_resort_move_threshold_ws = v; }

	// Live-tunable traversal cutoff on GaussianSplatLodNode::layer_density (see its own comment) - a node estimated to
	// be this dense with overdraw is kept as its own merged approximation rather than expanded further, regardless of
	// pixel_scale_limit. 0 (default) disables the check, matching the "0 = unlimited" convention used elsewhere in this
	// panel (size_clamp, "Splats list" search radius).
	float getMaxLayerDensity() const { return lod_max_layer_density; }
	void setMaxLayerDensity(float v) { lod_max_layer_density = v; }

	// Live-tunable hard ceiling on traversal depth from the tree root - "never unfold finer than this, anywhere in the
	// world", independent of pixel_scale_limit/max_layer_density/max_splats_budget. 0 (default) disables the check.
	int getMaxTreeDepth() const { return lod_max_tree_depth; }
	void setMaxTreeDepth(int v) { lod_max_tree_depth = v; }

	// Diagnostic tool, not a LoD parameter: culls any splat whose feature_size (2 * max scale axis, matching
	// GaussianSplatLodNode::feature_size) falls outside [min, max], directly in the vertex shader, regardless of
	// whether the cloud has a LoD tree. Used to locate abnormally large/degenerate splats (e.g. under-reconstructed
	// scene edges) by narrowing the range until the offending splat drops out. 0 disables the respective bound
	// (matches the "0 = unlimited" convention the Splats list panel's search radius uses) - default (0, 0) is off.
	float getSizeClampMin() const { return splat_size_clamp_min; }
	void setSizeClampMin(float v) { splat_size_clamp_min = v; }
	float getSizeClampMax() const { return splat_size_clamp_max; }
	void setSizeClampMax(float v) { splat_size_clamp_max = v; }
	// false (default) = cull outside [min, max] (isolate a size range); true = cull inside [min, max] instead
	// (exclude a size range, leaving the rest of the cloud untouched). No effect while the clamp itself is disabled.
	bool getSizeClampInvert() const { return splat_size_clamp_invert; }
	void setSizeClampInvert(bool v) { splat_size_clamp_invert = v; }

	// Per-splat quad radius is cut to exactly where alpha decays to this value (opacity * exp(-0.5*k^2) = alpha_cutoff),
	// instead of a fixed 3-sigma bound - see the derivation in gaussian_splat_vert_shader.glsl. Default 1/255 matches
	// the fragment shader's own discard threshold, so it changes zero pixels vs the old fixed-3-sigma behaviour; raising
	// it trims low-opacity splats' quads further (less overdraw, at the cost of their faintest edge).
	float getAlphaCutoff() const { return splat_alpha_cutoff; }
	void setAlphaCutoff(float v) { splat_alpha_cutoff = v; }

	// How many consecutive sub-ranges each cloud's depth-sorted splats are drawn in, nearest range first. 1 (default) is
	// one draw per cloud, exactly as before this existed. Higher values change nothing on their own - the ranges are
	// drawn back to back in the same order, so the blend is identical - and exist to give the front-to-back saturation
	// test somewhere to run: it can only look at what has accumulated so far between draws, never during one. See
	// OpenGLEngine::drawSplatClouds(). The right value is a scene-dependent trade (a slice has to be fine enough to end
	// inside a dense region, and each boundary costs a full pass over the region), hence a live parameter rather than a
	// constant.
	int getNumDrawSlices() const { return splat_num_draw_slices; }
	void setNumDrawSlices(int v) { splat_num_draw_slices = v; }

	// Ratio between the sizes of consecutive draw slices: each slice holds this many times as many splats as the one
	// before it. 1 (default) makes them all equal, which is what slicing did before this existed. Values below 1 make
	// each slice smaller than the last, i.e. a coarse start and a finely sliced tail.
	//
	// This knob exists because *where* the boundaries fall decides how much the saturation test can skip, and equal
	// spacing is only one guess at that. Which way to lean follows from the scene: a pixel is finished once its
	// accumulated coverage passes the threshold, so with a mean per-splat alpha of a it takes ln(1 - threshold)/ln(1 - a)
	// layers to get there - about 63 at a = 0.07, against the 350 or so layers a dense region of the measured scene has.
	// So nothing is saturated over roughly the first fifth of a depth-sorted cloud, and a boundary placed inside that
	// region costs a full-screen pass and skips nothing. Past it, pixels keep finishing steadily, so each further check
	// catches a new batch: what pays is asking *often after the knee*, not asking early.
	//
	// Measured on the reference scene, both directions lose to equal spacing, and monotonically. With 4 slices, growth 2
	// (boundaries at 7/20/47%) was worth under a millisecond and growth 3 (2/10/33%) cost two. With 6 slices, moving the
	// boundaries the other way - growth 0.85, 0.7, 0.5, i.e. first boundary at 24%, 34%, 51% - cost 1.4, 2.3 and 2.6 ms.
	// The reason both ends lose is that a check's value is the number of pixels already finished times the number of
	// splats still to come: crowding the checks before the knee makes the first factor zero, crowding them into the tail
	// makes the second one small, and spreading them evenly integrates the product best.
	//
	// So placement is not the lever; the number of checks is, and that is limited by what one check costs. The knob is
	// kept because the knee is a property of the scene, not a constant: anything that raises the mean per-splat alpha -
	// pruning, merging near-coplanar splats - moves it earlier and makes growth above 1 worth re-testing.
	float getSliceGrowth() const { return splat_slice_growth; }
	void setSliceGrowth(float v) { splat_slice_growth = v; }

	// Whether to reject splats at pixels the composite has already finished with. false (default) draws every slice in
	// full, so the slice count alone stays a no-op and the two can be compared directly. Needs more than one slice to do
	// anything, and is ignored in the overdraw debug views, where the blend is additive and the accumulated alpha counts
	// layers rather than coverage. See OpenGLEngine::markSaturatedSplatPixels().
	bool getSaturationGateEnabled() const { return splat_saturation_gate_enabled; }
	void setSaturationGateEnabled(bool v) { splat_saturation_gate_enabled = v; }

	// How many screen pixels across one texel of the saturation mask covers. 1 = a mask at full resolution, 4 (default)
	// = a mask with a sixteenth of the pixels.
	//
	// The mask is what the gate marks and what the splat shader tests, so its resolution sets both what a mark costs and
	// how precisely finished regions can be described. Coarsening it is close to free on the second count: a texel is
	// marked only if *every* pixel under it is finished (see the mask shader), so the picture stays exactly correct and
	// the only loss is the partly finished texels along the boundary of a saturated region, which are a shrinking
	// fraction of it as the region grows. On the first count it is most of the win: the mark pass writes 1/16 as many
	// pixels at 4, and 1/16 of a pass is what makes checking often affordable, which is the lever the measurements
	// pointed at.
	int getSaturationMaskDownscale() const { return splat_saturation_mask_downscale; }
	void setSaturationMaskDownscale(int v) { splat_saturation_mask_downscale = v; }

	// Accumulated coverage at or above which the gate treats a pixel as finished. Default 1 - 1/255: the light still
	// getting through is then under one 8-bit level, which is where the reference 3DGS rasteriser ends its own per-pixel
	// loop. Lowering it skips more work and starts to be visible, so it is the knob for trading the two off. Note the
	// accumulation buffer is RGBA16F, whose steps near 1.0 are about 0.0005 wide - settings closer to 1 than that are
	// not distinguishable.
	float getSaturationThreshold() const { return splat_saturation_threshold; }
	void setSaturationThreshold(float v) { splat_saturation_threshold = v; }

	// Whether the accumulation buffer is RGBA8 rather than the default RGBA16F, halving the bytes every splat fragment
	// blends. Measured on an RTX 3070 at 5.0 Mpixel with 2.9M splats drawn: about a third off the splat pass, which is
	// what says the pass is bound by blend bandwidth more than by anything else. The cost is precision: the composite
	// is a chain of hundreds of blends per pixel, and the resolve then divides the accumulated colour by the accumulated
	// coverage, so quantisation is amplified exactly where coverage is small - cloud silhouettes and thin, sparse
	// regions, not the dense areas where the fill cost is. Whether that is acceptable is a judgement about a given
	// scene, which is why this is a switch and not a new default. Like the saturation gate, toggling it rebuilds the
	// accumulation framebuffer, so it is not free to change every frame.
	bool getAccumBuffer8Bit() const { return splat_accum_buffer_8bit; }
	void setAccumBuffer8Bit(bool v) { splat_accum_buffer_8bit = v; }

	// The program OpenGLEngine::markSaturatedSplatPixels() marks finished pixels with. Null until the first addObject(),
	// like the splat program itself.
	const Reference<OpenGLProgram>& getSaturationMaskProgram() const { return saturation_mask_prog; }

	// The program that halves the mask, taking the minimum of each 2x2 block, to build the pyramid the vertex shader
	// picks a level from. Null until the first addObject(), like the programs above.
	const Reference<OpenGLProgram>& getMaskReduceProgram() const { return mask_reduce_prog; }

	// Location of the splat program's saturation-mask sampler, or -1 if the program has no such uniform. Resolved on
	// first use rather than in buildShadersIfNeeded(), since the build may still be in flight there; the draw path only
	// asks once the program reports isBuilt(). The engine binds the texture itself - the material path has no support
	// for sampler uniforms - hence a location rather than a setter here.
	int getSplatMaskTexUniformLoc();

	// How many screen pixels across one mask texel covers, or 0 to switch the test in the splat shader off entirely.
	// A block size rather than its reciprocal because the shader divides by it in integer arithmetic: the mask's size is
	// a rounded-up division, so a float scale would round the wrong way for some pixels and send them to a neighbouring
	// texel - which is not conservative, and so shows. Set by the draw path each frame rather than by think(), because
	// whether the gate actually runs is the draw path's decision - and a splat shader that tests a mask nobody wrote
	// would be reading stale texels.
	void setSplatMaskBlockSize(int block_size, int max_level);

	// Sets that program's uniforms from the current getSaturationThreshold(), plus how many accumulation-buffer pixels
	// across one mask texel covers, which is the caller's since it depends on the size the mask was actually allocated
	// at. Called once the mask program is bound, for the same reason setResolveOverdrawUniforms() exists: the pass is
	// one manual full-viewport quad with no material, so it bypasses the generic per-object uniform path.
	void setSaturationMaskUniforms(int block_size) const;

	// Overdraw debug view: 0 (default) = normal rendering. Non-zero = every splat writes a flat additive increment
	// instead of its real colour, into the same accumulation buffer as normal, which the resolve pass then colour-ramps
	// - see gaussian_splat_frag_shader.glsl and OpenGLEngine::drawSplatClouds(). Mode 1 sums 1.0 per fragment, giving
	// depth complexity (layers per pixel), which is what drives blend cost rather than raw splat count. Mode 2 sums the
	// fragments' alpha instead; read against mode 1's layer count it estimates how much fragment work a front-to-back
	// transmittance cutoff could skip (~sum_alpha / 5.54) - see the derivation in the fragment shader's uniform comment.
	int getShowOverdrawMode() const { return splat_show_overdraw_mode; }
	void setShowOverdrawMode(int v) { splat_show_overdraw_mode = v; }

	// Whether any overdraw debug mode is active - what the draw path needs, since every mode shares the same additive
	// blend func and only the accumulated value differs.
	bool getShowOverdraw() const { return splat_show_overdraw_mode != 0; }

	// Range the overdraw ramp maps to blue..green..red, in whichever units getShowOverdrawMode() is accumulating -
	// see gaussian_splat_resolve_frag_shader.glsl. No effect while getShowOverdraw() is false.
	float getOverdrawRangeMin() const { return splat_overdraw_range_min; }
	void setOverdrawRangeMin(float v) { splat_overdraw_range_min = v; }
	float getOverdrawRangeMax() const { return splat_overdraw_range_max; }
	void setOverdrawRangeMax(float v) { splat_overdraw_range_max = v; }

	// Sets the resolve program's overdraw-view uniforms from the current getShowOverdrawMode()/getOverdrawRangeMin()/
	// getOverdrawRangeMax() values. Called by OpenGLEngine::resolveSplatAccumBuffer() right after the resolve program
	// is bound: that draw is one manual full-viewport quad with no material, so it bypasses the generic per-object
	// user-uniform path the main splat program's addObject()/think() machinery uses for its own uniforms.
	void setResolveOverdrawUniforms() const;

private:
	GLARE_DISABLE_COPY(GaussianSplatRenderer);

	void buildShadersIfNeeded();

	Reference<SplatCloud> allocCloud(); // Builds an empty cloud with its GLObject.  Not added to the engine until it holds a member - see addCloudToEngineIfNeeded().
	void addCloudToEngineIfNeeded(SplatCloud& cloud); // Adds the cloud's GLObject to the engine, once it has real bounds and an instance count for the engine to cache.
	void destroyCloud(const Reference<SplatCloud>& cloud); // Removes the cloud's GLObject from the engine and drops the cloud.

	void ensureGpuCapacity(SplatCloud& cloud, size_t needed_splats); // Grows the data texture and instance index VBO if needed.
	void rebuildVAO(SplatCloud& cloud); // Rebuilds vert_vao against the current instance index VBO - needed whenever that VBO is replaced.
	void uploadTexelRowsForSplatRange(SplatCloud& cloud, size_t first_splat, size_t num_splats_to_upload); // Repacks and re-uploads just the texture rows spanning the given splat range.
	void writeIdentityIndices(SplatCloud& cloud, size_t first_splat, size_t num_splats); // Writes an identity draw order over the given range of the instance index VBO.
	void rebuildCloudAABB(SplatCloud& cloud); // Recomputes the cloud AABB as the union of its members' bounds.  O(num members), not O(num splats).

	void appendMemberToCloud(SplatCloud& cloud, const CloudMember& member); // Fast path: bakes one member onto the tail and uploads only the affected rows.  member.offset is assigned here.
	void rebuildCloud(SplatCloud& cloud); // Re-bakes every member from its stored pose.  Used after a merge or a removal, where offsets change.
	void mergeIntersectingClouds(SplatCloud& seed_cloud); // Merges any cloud whose AABB intersects seed_cloud into it, to a fixpoint.

	void drainSortResults();
	void kickOffSorts();

	void writePlaceholderSelection(SplatCloud& cloud); // Synchronous stand-in frontier (root-only per member with a tree, everything for a member without one), written after any structural change, until the next background traversal's result supersedes it.
	void drainTraversalResults();
	void kickOffTraversals();
	void fillTraversalScratch(const SplatCloud& cloud, GaussianSplatLodTraversalScratch& scratch) const; // Freezes a cloud's world-space node data and member layout into a scratch, ready for a traversal to read without touching the live arrays.

	Reference<OpenGLProgram> shader_prog; // Shared by every cloud.  Null until the first addObject().
	Reference<OpenGLProgram> resolve_prog; // Resolves the splat accumulation buffer onto the main colour buffer.  Built alongside shader_prog.
	Reference<OpenGLProgram> saturation_mask_prog; // Marks finished pixels between draw slices.  Built alongside shader_prog.
	Reference<OpenGLProgram> mask_reduce_prog; // Halves the mask, by minimum, to build its pyramid.  Built alongside shader_prog.

	OpenGLEngine* opengl_engine;

	std::vector<Reference<SplatCloud> > clouds;
	std::map<Handle, SplatCloud*> handle_to_cloud; // Kept in step with the member lists, since a merge moves members between clouds.
	Handle next_handle;
	uint64 next_cloud_id;

	// Sort scratch is pooled rather than per-cloud: at large splat counts these buffers run to hundreds of MB, so N
	// clouds must not mean N copies of them.  Borrowed for the duration of a sort, returned when its precise result lands.
	std::vector<Reference<GaussianSplatSortScratch> > free_scratch;
	int num_sorts_in_flight;

	ThreadSafeQueue<Reference<ThreadMessage> > sort_result_queue; // Shared by every cloud; results carry the cloud id they belong to.
	js::Vector<Reference<ThreadMessage>, 16> completed_msgs;

	// LoD traversal: same pooling/in-flight-tracking shape as the sort state above, kept separate since the two are
	// independent pipelines (see kickOffTraversals()/drainTraversalResults()).
	std::vector<Reference<GaussianSplatLodTraversalScratch> > free_traversal_scratch;
	int num_traversals_in_flight;
	ThreadSafeQueue<Reference<ThreadMessage> > traversal_result_queue;
	js::Vector<Reference<ThreadMessage>, 16> completed_traversal_msgs;

	// Live-tunable via GaussianSplatSettingsWidget (Qt only); hardcoded defaults if that panel's saved settings are never
	// applied (e.g. no UI). pixel_scale_limit is roughly "stop refining once a node projects to about this many pixels";
	// max_splats_budget is a per-cloud cap on how many nodes one traversal may select, matched to what the old
	// (pre-partitioning) renderer settled on after real-world testing at multi-million-splat scale (see
	// Claude_LOD_plan.md's session notes) - a starting point, not a measured value for this architecture specifically.
	// resort_move_threshold_ws is kickOffTraversals()'s own move-threshold floor, separate from the plain sort's
	// min_resort_move_threshold_ws constant (non-LoD clouds aren't affected by this setting).
	float lod_pixel_scale_limit;
	size_t lod_max_splats_budget;
	float lod_resort_move_threshold_ws;

	// See getMaxLayerDensity() above. 0 disables the check.
	float lod_max_layer_density;

	// See getMaxTreeDepth() above. 0 disables the check.
	int lod_max_tree_depth;

	// See getSizeClampMin()/getSizeClampMax() above. Defaults (0, 0) disable both bounds, so never exclude a real splat.
	float splat_size_clamp_min;
	float splat_size_clamp_max;
	bool splat_size_clamp_invert;

	// See getAlphaCutoff() above. Default 1/255 is lossless (matches the fragment shader's fixed discard threshold).
	float splat_alpha_cutoff;

	// See getNumDrawSlices() above. Default 1 = one draw per cloud, i.e. no slicing.
	int splat_num_draw_slices;

	// See getSliceGrowth() above. Default 1 = every slice the same size.
	float splat_slice_growth;

	// See getSaturationGateEnabled()/getSaturationThreshold()/getSaturationMaskDownscale() above. Off by default.
	bool splat_saturation_gate_enabled;
	float splat_saturation_threshold;
	int splat_saturation_mask_downscale;

	// See getSplatMaskTexUniformLoc(). -2 means "not looked up yet", which -1 cannot mean, that being GL's answer for
	// a uniform the linker dropped.
	int splat_mask_tex_uniform_loc;

	// See getAccumBuffer8Bit() above. Off by default, i.e. the RGBA16F buffer this pass has always used.
	bool splat_accum_buffer_8bit;

	// See getShowOverdrawMode() above. 0 = off.
	int splat_show_overdraw_mode;

	// See getOverdrawRangeMin()/getOverdrawRangeMax() above.
	float splat_overdraw_range_min;
	float splat_overdraw_range_max;
};
