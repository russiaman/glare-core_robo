/*=====================================================================
GaussianSplatRenderer.h
-----------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../graphics/GaussianSplatCoplanarMerge.h" // For the parameters applyCoplanarMerge() takes.
#include "../graphics/GaussianSplatData.h"
#include "../maths/Matrix4f.h" // For the frozen hide-overdraw mask's view matrix.
#include "../maths/Quat.h"
#include "../maths/Vec4f.h"
#include "../physics/jscol_aabbox.h"
#include "../utils/Platform.h"
#include "../utils/Reference.h"
#include "../utils/Timer.h" // For the camera-motion EMA prev-think time reference used by kickOffTraversals() dilation.
#include "../utils/ThreadMessage.h"
#include "../utils/ThreadSafeQueue.h"
#include "../utils/Vector.h"
#include <map>
#include <string>
#include <vector>


class OpenGLEngine;
class OpenGLProgram;
struct GLObject; // Only ever passed through by pointer here - see visibleFractionToDrawIndex().  Declared the same way MeshPrimitiveBuilding.h and TransformGizmo.h do it.
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

	// SESSION059: debug tool - leave one splat object out of the draw entirely (frustum-cull path, sort path and the
	// synchronous placeholder frontier all respect it), without touching its baked data. Deliberately in-memory only,
	// never persisted - every session starts with every object visible. Returns false if the handle isn't valid.
	bool setObjectHidden(Handle handle, bool hidden);
	bool getObjectHidden(Handle handle) const; // Returns false (not hidden) if the handle isn't valid.

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
	// cloud, that are in the camera's current frustum AND pass both the size clamp and the distance slice above (same tests
	// as the shader's, including the invert flags - if a slice is at its keep-everything default, it excludes nothing).
	// SESSION066: the distance slice is now mirrored here too, so cutting with dist_clamp moves the count. O(total splats in
	// the world); meant to be triggered once by a button click, not called per-frame. Returns in_frustum and total (leaves +
	// merged internal nodes summed across all clouds) so callers can report both the absolute number and the fraction -
	// used to size the ceiling for traversal frustum-cull work (session054 §2A).
	// SESSION066: 'drawn' is the current LoD draw-list size S(P,R) (sum of num_instances_to_draw) - the pre-slice selection,
	// including the dilation band that sits just outside the tight frustum. 'visible' iterates that same draw list and
	// counts only the splats that pass the frustum + size/distance slices this frame - i.e. what really reaches the screen;
	// it is the number that drops when the pixel_scale limit or the distance slice tighten. So the button reports the
	// geometric-in-frustum ceiling (in_frustum/total), the raw draw count (drawn), and the real on-screen count (visible).
	struct FrustumCounts { size_t in_frustum; size_t total; size_t drawn; size_t visible; };
	FrustumCounts countSplatsInFrustum() const;

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
	// - meant for a button click, never per frame. The traversal's result is read and dropped, not applied.
	//
	// One deliberate exception to "changes no renderer state": each run folds its per-splat importance into a running
	// record on the cloud, so that the pruning ceiling - a single-camera figure by construction, since a splat hidden from
	// here may be the front layer from there - can also be reported over every viewpoint visited so far. Accumulating on
	// the button press rather than continuously is what keeps the set of viewpoints a deliberate choice. Persists for the
	// life of the cloud (no reset control - see session log for removal rationale), and thrown away by itself if the
	// cloud's node numbering changes underneath it.
	// merge_colour_tol and merge_angle_tol_deg govern only the merge section: how alike two splats' colours and facings
	// have to be before the report is willing to call them the same surface seen twice. They are arguments rather than
	// renderer state because nothing outside the report reads them, and the point of them is to be swept - the honest
	// tolerance and a deliberately reckless one, so the gap between the two says how much of the merge prize the safety
	// conditions are costing. Defaults are a tenth per colour channel and 26 degrees.
	std::string getFrustumStructureReport(float merge_colour_tol = 0.1f, float merge_angle_tol_deg = 26.f);

	// Collapses near-duplicate splats in every registered cloud - see GaussianSplatCoplanarMerge.h for what that means and
	// what it is for - and rebuilds each affected member's LoD tree from the merged splats, since the tree describes the
	// cloud it was built from and nothing else. Returns a one-line-per-figure summary for the log.
	//
	// params_ws.alpha_cutoff is filled in here from getAlphaCutoff() and need not be set by the caller; the rest of the
	// struct is the tolerances, which are the caller's.
	//
	// The tolerances are in world metres and are converted to each member's own object space here, using the member's
	// uniform scale - so a capture placed in the world at half size is merged to the same world-space tolerance as one
	// placed at full size, which is what makes the setting mean something across a world of differently-scaled objects.
	//
	// Synchronous, on the main thread, and it re-runs the LoD tree build - which is a load-time-scale cost, seconds on a
	// multi-million-splat capture. Meant for a button press while standing still, exactly like getFrustumStructureReport().
	//
	// The pre-merge data is kept (see CloudMember::unmerged_splat_data) rather than being reloaded from the source file,
	// which is what makes restoreUnmergedSplats() possible and what keeps a second press from merging an already-merged
	// cloud again: every press starts from the original splats, so the tolerances mean the same thing every time.
	std::string applyCoplanarMerge(const GaussianSplatCoplanarMergeParams& params_ws, float lod_base);

	// Puts every merged cloud back to the splats it was loaded with, rebuilding the trees again. The point of it is
	// eyes-on A/B at a fixed camera: a merge that changes the picture is only visible against the picture it changed.
	std::string restoreUnmergedSplats();

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

	// Note what this budget is counting: selected nodes, wherever they are, including the ones behind the camera - the
	// traversal has no frustum test, on purpose, for reasons set out at GaussianSplatLodTraversalTask.  At the default
	// (10M against a tree of under 4M nodes) it never binds, so that costs nothing today.  Where it does bind, though, it
	// truncates detail in view in order to pay for nodes out of it, which is not merely suboptimal - it is the wrong thing
	// to spend the last of a budget on.  getDiagnostics() prints "budget cap is limiting detail on this cloud" whenever
	// this has stopped a traversal, which is the signal that this paragraph has become relevant.  The planned cluster-based
	// LoD (see the same comment) removes the interaction rather than reweighting it, since an off-screen cluster is
	// rejected before it can ask for budget at all.
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

	// SESSION055: turns on frustum-culling inside the traversal's expand loop - each popped node is tested against the
	// scene's frustum planes (dilated by 1.5 * feature_size to keep large-radius nodes whose centre is just outside),
	// and out-of-frustum nodes are neither recorded nor expanded, pruning entire subtrees behind the camera. Paired with
	// an angular re-kick threshold in kickOffTraversals(), so that turning on the spot re-runs the traversal.
	//
	// Trades the property that the old comment on GaussianSplatLodTraversalTask leans on ("rotation stays free") for a
	// large drop in nodes visited on interior scenes - see session054 §2A. Kept as a live switch to allow A/B comparison
	// and to fall back if a rotation-heavy workflow shows the extra re-kicks costing more than the cull saves.
	bool getFrustumCullEnabled() const { return lod_frustum_cull_enabled; }
	void setFrustumCullEnabled(bool v) { lod_frustum_cull_enabled = v; }

	// SESSION063: split filter architecture (session062 §9.3). When on, the traversal runs cull-off and builds the
	// orientation-independent unculled frontier U(P); a cheap per-orientation SSE frustum filter derives the actual draw
	// list S(P,R) from it, so a pure rotation no longer needs a fresh ~450ms traversal. Mutually exclusive in effect with
	// the in-traversal frustum cull above (kickOffTraversals() forces cull_active false when this is on). Live switch for
	// A/B against the cull path. See GaussianSplatUnculledFrontier / filterUnculledFrontier() / drainTraversalResults().
	bool getSplitFilterEnabled() const { return split_filter_enabled; }
	void setSplitFilterEnabled(bool v) { split_filter_enabled = v; }

	// SESSION063 K3: live-tunable dilation of the per-orientation filter (see kickOffFilters()). The filter keeps nodes
	// slightly outside the frustum so the edge doesn't trail during the filter's latency window; how far is these three.
	// latency (s): the motion-prediction window - margin = motion_rate * latency, so bigger = wider edge coverage but more
	// over-inclusion. rot/trans floors: the baseline rate assumed even from a standstill, so the static->moving transition
	// is covered. Exposed so the owner can find the smallest values that hide the edge without paying for it.
	float getFilterDilationLatency() const { return filter_dilation_latency; }
	void setFilterDilationLatency(float v) { filter_dilation_latency = v; }
	float getFilterMinRotRateDegPerS() const { return filter_min_rot_rate_deg_per_s; }
	void setFilterMinRotRateDegPerS(float v) { filter_min_rot_rate_deg_per_s = v; }
	float getFilterMinTransRateMPerS() const { return filter_min_trans_rate_m_per_s; }
	void setFilterMinTransRateMPerS(float v) { filter_min_trans_rate_m_per_s = v; }

	// SESSION063 K4: coarse floor. The traversal also captures a low-detail cut (first node per branch at
	// coarse_pixel_scale), drawn after the fine set; the filter dilates this cut with its own (wider) latency so cheap big
	// splats plug motion-revealed edges/holes without the fine set paying for a wide dilation. See GaussianSplatUnculledFrontier.
	bool getCoarseFloorEnabled() const { return split_coarse_floor_enabled; }
	void setCoarseFloorEnabled(bool v) { split_coarse_floor_enabled = v; }
	float getCoarsePixelScale() const { return split_coarse_pixel_scale; }
	void setCoarsePixelScale(float v) { split_coarse_pixel_scale = v; }
	float getFilterCoarseDilationLatency() const { return filter_coarse_dilation_latency; }
	void setFilterCoarseDilationLatency(float v) { filter_coarse_dilation_latency = v; }

	// SESSION063 K4 debug: draw ONLY the coarse floor (skip the fine set), so its screen coverage can be inspected in
	// isolation - see drainFilterResults(). Off = normal (fine + coarse).
	bool getCoarseLayerDebug() const { return coarse_layer_debug; }
	void setCoarseLayerDebug(bool v) { coarse_layer_debug = v; }


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

	// Diagnostic tool, not a LoD parameter: keeps only splats whose distance from the camera lies inside [min, max]
	// metres, so a capture can be sliced open and inspected a shell at a time - the near wall taken away to see what is
	// behind it, or one shell isolated to see what is in it. Applied in the vertex shader, so it costs nothing when the
	// range is wide open and needs no re-traversal to change.
	//
	// Distance from the camera position, not depth along the view axis: a spherical shell keeps the same splats when the
	// camera turns on the spot, which is what makes it usable for looking around inside a slice. Default (0, 1000) keeps
	// everything at any sane scene size, so unlike the size clamp there is no "0 = unlimited" case to encode.
	float getDistClampMin() const { return splat_dist_clamp_min; }
	void setDistClampMin(float v) { splat_dist_clamp_min = v; }
	float getDistClampMax() const { return splat_dist_clamp_max; }
	void setDistClampMax(float v) { splat_dist_clamp_max = v; }
	// false (default) = keep what is inside the range; true = keep what is outside it, i.e. cut that shell away.
	bool getDistClampInvert() const { return splat_dist_clamp_invert; }
	void setDistClampInvert(bool v) { splat_dist_clamp_invert = v; }

	// Per-splat quad radius is cut to exactly where alpha decays to this value (opacity * exp(-0.5*k^2) = alpha_cutoff),
	// instead of a fixed 3-sigma bound - see the derivation in gaussian_splat_vert_shader.glsl. Default 1/255 matches
	// the fragment shader's own discard threshold, so it changes zero pixels vs the old fixed-3-sigma behaviour; raising
	// it trims low-opacity splats' quads further (less overdraw, at the cost of their faintest edge).
	float getAlphaCutoff() const { return splat_alpha_cutoff; }
	void setAlphaCutoff(float v) { splat_alpha_cutoff = v; }

	// Live rescaling of every splat's stored opacity: alpha' = gain * alpha^gamma, clamped to 1 - see adjustSplatAlpha()
	// in graphics/GaussianSplatData.h for the transform itself and for why it is defined there.
	//
	// Applied in the vertex shader before the quad is sized, which is the earliest point that can still be live, so it
	// moves how wide each splat is rasterised as well as how much it contributes per pixel. Everything downstream is
	// therefore of the adjusted cloud: the saturation gate's coverage, the overdraw views, and the cost predictions in
	// getFrustumStructureReport(), which apply the same transform on the CPU.
	//
	// What it does not reach is the LoD tree, whose merged splats were built from the stored alpha when the cloud was
	// loaded - a tree rebuild is what folds a setting in permanently. 1 and 1 is the cloud as captured.
	float getAlphaGain() const { return splat_alpha_gain; }
	void setAlphaGain(float v) { splat_alpha_gain = v; }
	float getAlphaGamma() const { return splat_alpha_gamma; }
	void setAlphaGamma(float v) { splat_alpha_gamma = v; }

	// How many consecutive sub-ranges each cloud's depth-sorted splats are drawn in, nearest range first. 1 (default) is
	// one draw per cloud, exactly as before this existed. Higher values change nothing on their own - the ranges are
	// drawn back to back in the same order, so the blend is identical - and exist to give the front-to-back saturation
	// test somewhere to run: it can only look at what has accumulated so far between draws, never during one. See
	// OpenGLEngine::drawSplatClouds(). The right value is a scene-dependent trade (a slice has to be fine enough to end
	// inside a dense region, and each boundary costs a full pass over the region), hence a live parameter rather than a
	// constant.
	int getNumDrawSlices() const { return splat_num_draw_slices; }
	void setNumDrawSlices(int v) { splat_num_draw_slices = v; }

	// DIAGNOSTIC ONLY - stop drawing into a pixel once this many splats have blended into it.  0 (default) = no cap.
	//
	// Shares the hide-overdraw machinery entirely: a counting pass measures layers per pixel for the current camera, the
	// pixels at or above the cap are marked in the mask, and the splat vertex shader drops splats standing there.  So it
	// costs one counting pass when the camera or the settings move, and nothing at all on the frames after that - which
	// is what makes it usable as a measurement.  The frames after the rebuild contain exactly the work a cloud without
	// those splats would do.
	//
	// Not a per-pixel cut: the grain is a whole splat.  With the conservative test (see getHideTestConservative()) a
	// splat is only dropped when every pixel its quad can touch has already had 'cap' layers, so nothing can open a hole
	// - each pixel keeps at least 'cap' layers of paint, and the pixels near the edge of a dense region keep more.  That
	// is the reading of "stop compositing this pixel past depth N" that this machinery can actually express.
	//
	// A stencil buffer would express the exact per-pixel version, and was tried: the test is defeated by the fragment
	// shader's discard (which switches off the driver's early per-fragment tests, so the shading is paid anyway), and a
	// stencil can only be had here inside a packed depth buffer of our own, which costs 3.5-8.7 ms in lost Hi-Z plus
	// about 3.4 ms of tax on every depth test against it.  Measured; the net was negative.  Do not go back to it.
	int getLayerCap() const { return splat_layer_cap; }
	void setLayerCap(int v) { splat_layer_cap = v; }

	// What a pixel that reached the cap is composited as.  true: fully covered, keeping the colour its kept layers
	// averaged to, so the cut shows as a colour error.  false: as covered as those layers made it, so the background
	// shows through wherever the cap bit - which is the quickest way to see where it bit at all.
	// Shared with the coverage cap below, which cuts a pixel short in exactly the same way and wants the same choice about
	// how to show it.
	bool getLayerCapOpaque() const { return splat_layer_cap_opaque; }
	void setLayerCapOpaque(bool v) { splat_layer_cap_opaque = v; }

	// DIAGNOSTIC ONLY - the same per-fragment cut as getLayerCap(), but decided by accumulated coverage rather than by a
	// layer count.  0 (default) = off; otherwise the coverage at or above which a pixel stops being composited.
	//
	// The missing third combination of the two axes the mask machinery has.  The saturation gate marks by coverage and
	// kills whole splats in the vertex shader, conservatively - which is why it finds so little: a quad is dropped only
	// when every pixel it can touch is finished, and quads are larger than the connected finished patches.  The layer cap
	// marks by a count and kills fragments.  This marks by coverage and kills fragments, which is the aggressive reading
	// of the gate's own criterion.
	//
	// Why coverage rather than the layer count the owner measured "20 layers" with: a count says how many times a pixel
	// was painted, coverage says how much of it is already opaque, and only the second is a statement about what is still
	// visible.  Front-to-back compositing is what makes it sound - once accumulated coverage is at the threshold, nothing
	// drawn behind can change the pixel by more than what still gets through.  Twenty layers was a proxy for reaching that
	// point in the measured scene, not the criterion itself.
	//
	// The practical difference from the layer cap is that no per-pixel counter buffer is needed - see
	// wantsLayerCountBuffer().  That counter is a second full-screen attachment every splat fragment writes to, and the
	// pass is bound by blend bandwidth, so the measurement tool was paying a share of the very cost it was measuring.
	//
	// Its own threshold rather than getSaturationThreshold(): the gate and the cap want that number in different places
	// (the gate stops when nothing can be seen at all, this stops when little enough can), and sharing one field would
	// mean every switch between them retunes the other.  Only one of the three can run - there is one mask - so the draw
	// path stands the gate down while this is on and yields to the layer cap when both are set.
	float getCoverageCap() const { return splat_coverage_cap; }
	void setCoverageCap(float v) { splat_coverage_cap = v; }

	// DIAGNOSTIC ONLY - the ablation ladder.  0 (default) = off, i.e. the pass exactly as it is.  Higher values switch
	// progressively more of it back on, lightest first, so that reading the splat pass timer at each step and taking the
	// difference attributes cost to the one thing that step added.
	//
	//   1  CPU only: traversal, sort and instance buffer are all done, and the draw call is not issued.
	//   2  Instances emitted.  One texel fetched per vertex (the position), one-pixel quad, flat white, no blending.
	//   3  All four texels fetched and unpacked.  Still a one-pixel quad.
	//   4  The splat's own colour on those points.
	//   5  All of the projection maths - covariance, Jacobian, eigen-decomposition, radii - and still a one-pixel quad.
	//   6  The real quad size.  Flat opaque fill, still no blending.
	//   7  Blending on, with a flat alpha across the quad.
	//   8  The Gaussian falloff, but no discard.
	//
	// Stage 9 is this switched off - the full shader - and stage 10 is that with the alpha saturation cap on, so neither
	// needs a value here: the existing knobs already say it.  9 minus 8 is the cost of 'discard' itself, which is the
	// number this ladder exists for, and it should agree with what the draw slice limit says independently.
	//
	// 5 exists because 4 -> 6 moves two things at once, the vertex maths and the fill; splitting them is what makes each
	// readable on its own.  Bisecting the fill further is getQuadRadiusScale()'s job, not another stage's.
	//
	// Two things the readings cannot be asked: GPU cost is not a sum of its stages but a maximum over the units that
	// saturate, so a step reads as "the cost of this given everything below it" and the order is part of the answer; and
	// stage 1 leaves the splat pass timer at nothing, so the number to read there is the CPU frame time instead.
	//
	// The quad stays a quad throughout, shrunk to a pixel rather than swapped for GL_POINTS: a different primitive type
	// would change the rasterisation path as well as the area, and the step would then answer about both at once.
	int getAblationStage() const { return splat_ablation_stage; }
	void setAblationStage(int v) { splat_ablation_stage = v; }

	// DIAGNOSTIC ONLY - multiplies every splat's screen-space quad radius.  1 (default) is the real size.
	//
	// Bisects the fill by area rather than by stage: the splat count, all of the vertex work and the length of the blend
	// chain are untouched, and only rasterised area changes - as the square of this. So a cost that halves when this is
	// set to 0.7 is area-proportional (rasterisation, shading, blend bandwidth), and one that does not is per-splat
	// (instance emission, primitive setup) whatever the stage ladder says about it.
	//
	// Not the same as raising getAlphaCutoff(), which also shrinks quads: that trims each splat by its own opacity, so it
	// changes which splats shrink and by how much. This is a uniform scale, which is what an area law needs. The conic is
	// built from the scaled radii, so the Gaussian shrinks with the quad instead of being cut off by it - smaller splats,
	// not hard-edged ones.
	float getQuadRadiusScale() const { return splat_quad_radius_scale; }
	void setQuadRadiusScale(float v) { splat_quad_radius_scale = v; }

	// DIAGNOSTIC ONLY - makes getQuadRadiusScale()'s reduction area-dependent rather than flat. Both inert (reduction
	// stays flat, exactly as before these existed) while getQuadRadiusScale() is 1.
	//
	// Each splat gets a weight in [0, 1] from its own screen-space area (proportional to radius1*radius2, before either
	// scale is applied) against getAreaScaleRefPx()^2: 0 at zero area, 1 at or above the reference area. gamma is the
	// exponent on that ratio - 1 (default) makes the weight linear in the splat's own area; raising it concentrates
	// getQuadRadiusScale()'s reduction on splats bigger than the reference (smaller ones increasingly spared); lowering
	// it below 1 spreads a partial reduction onto smaller splats too. The splat's effective scale is then
	// mix(1, getQuadRadiusScale(), weight) - the reference-and-above splats get the full reduction, everything below it
	// a fraction of it, nothing gets more than the flat tool would.
	//
	// Exists to test session046 open question (2)/(5)'s premise directly: the ablation ladder found the pass's cost to
	// be rasterised area, and a handful of large splats (background, near-flat, low-parallax) can hold a disproportionate
	// share of it - so a reduction weighted toward them should buy more of the same saving at a smaller cost to visible
	// detail than the flat tool, which cuts foreground and background alike. Not yet measured; this is the toggle to do
	// that with.
	float getAreaScaleGamma() const { return splat_area_scale_gamma; }
	void setAreaScaleGamma(float v) { splat_area_scale_gamma = v; }
	// Radius (px, before either scale) at which a splat's own area reaches the reference area (ref^2) and so gets the
	// full getQuadRadiusScale() reduction - see getAreaScaleGamma() above for the rest of the formula.
	float getAreaScaleRefPx() const { return splat_area_scale_ref_px; }
	void setAreaScaleRefPx(float v) { splat_area_scale_ref_px = v; }

	// How wide the fade-out of splats the camera is getting inside of is, as a fraction of the ratio the test uses -
	// see the block that uses splat_near_fade_width in gaussian_splat_vert_shader.glsl for the whole story.
	//
	// A splat is an ellipsoid, not a point, and a big flat one (a metre-wide floor splat, which every capture has, since
	// a surface shot with little parallax does not constrain its splats' size) reaches far enough past its centre that
	// standing in the room puts part of it behind the camera. That part has negative depth, so the perspective divide
	// throws it above the horizon rather than below, and the splat is drawn as a streak climbing through the frame. The
	// projection of a splat straddling the camera plane genuinely does not exist, so this is not fixable by bounding the
	// radius: the splat has to go. The engine's own near-plane test does not catch it because it looks at the centre.
	//
	// 0 makes it a hard cull at the point the projection stops existing; the default spreads it over the last 30% of the
	// approach, since these splats are often the only thing covering their patch of floor and a hard test pops them in
	// and out as the camera moves. Fading costs nothing extra and saves area, as the quad's radius follows the opacity
	// through the sigma cutoff.
	// Switches all three corrections off together - the frustum cull, the honest-size bound on the radius, and the near
	// fade - leaving the projection as it was before any of them. Here so the three can be judged against the original as
	// one change, which is how they will be shipped: they share a cause, and turning them on one at a time only shows a
	// picture in which the cause is partly corrected.
	bool getEWAProjectionFixEnabled() const { return splat_ewa_fix_enabled; }
	void setEWAProjectionFixEnabled(bool v) { splat_ewa_fix_enabled = v; }

	float getNearFadeWidth() const { return splat_near_fade_width; }
	void setNearFadeWidth(float v) { splat_near_fade_width = v; }

	// Depth (metres) below which the splat vertex shader culls a splat by its centre - an unconditional
	// guard against the Jacobian's 1/d / 1/d^2 terms exploding at very small d.  Default 0.1 (the historical
	// hardcode); can be lowered when the app wants to accept the projection artefacts in exchange for a
	// closer near clip (e.g. Photo Mode's "Near clip" override).  See splat_near_epsilon in
	// gaussian_splat_vert_shader.glsl.
	float getNearEpsilon() const { return splat_near_epsilon; }
	void setNearEpsilon(float v) { splat_near_epsilon = v; }

	// How Gaussian splats contribute depth for post-process effects (Depth of Field, fog) that read the depth
	// buffer.  Splats normally never write depth (see drawSplatClouds()), so a splat pixel's depth is whatever
	// was behind it - which is why DoF blurs splats regardless of focus distance in the upstream code path.
	//
	// This whole feature only engages when a post-process consumer is asking for it (currently: dof_blur_strength
	// > 0).  Zero cost otherwise, regardless of the setting.
	enum SplatDoFDepthMode
	{
		// Upstream behaviour: splats do not write depth at all.  DoF/fog use the depth of whatever was behind the
		// splats, so splats blur wrong.  Kept as an option for A/B against the two modes below.
		SplatDoFDepthMode_Off = 0,

		// A: a second, colour-masked depth-only pass over every visible splat after the composite, using a
		// strict alpha threshold (see getDoFDepthPrepassAlphaMin()) so only meaningfully opaque fragments write
		// depth.  Cheap - one extra geometry pass per DoF frame - but the answer is "the frontmost surviving
		// fragment" per pixel, which is a discrete choice.  Neighbouring pixels can pick different splats, so
		// silhouettes step visibly where the frontmost splat changes.
		SplatDoFDepthMode_Prepass = 1,

		// B: accumulate depth * alpha * transmittance into a second colour attachment during the splat pass,
		// resolve pass divides through by coverage and writes gl_FragDepth.  Gives the expected depth of the
		// splat stack per pixel - smooth by construction, no silhouette steps - at the cost of a second
		// attachment (RGBA16F) blended alongside the accumulation buffer.  Not physically depth of a surface:
		// where a lacy foreground shows a distant background through it, the mean depth lands between them,
		// where nothing actually is.
		SplatDoFDepthMode_Weighted = 2,
	};
	int getDoFDepthMode() const { return splat_dof_depth_mode; }
	void setDoFDepthMode(int mode) { splat_dof_depth_mode = mode; }
	// Whether SplatDoFDepthMode_Weighted's second accumulation attachment should be allocated - decided by the
	// mode alone, not by whether DoF is live this frame, so dragging the DoF blur strength slider through zero
	// doesn't repeatedly allocate/free it.  See OpenGLScene::splat_dof_depth_renderbuffer.
	bool wantsDoFDepthBuffer() const { return splat_dof_depth_mode == SplatDoFDepthMode_Weighted; }

	// Alpha threshold that gates depth writes in SplatDoFDepthMode_Prepass.  Fragments below it are discarded
	// before writing depth, and vertex-side quads shrink to the same threshold (see splat_alpha_cutoff in
	// gaussian_splat_vert_shader.glsl).  Default 0.3: 1/255 (matching the main draw's own discard) lets the
	// almost-transparent wing of a foreground splat win depth for pixels the wing barely touches, which shows
	// as ghost silhouettes in the resulting DoF.  0.3 is the empirical setting that pushes those out without
	// obviously starving depth over regions where the frontmost splat is legitimately semi-transparent.
	float getDoFDepthPrepassAlphaMin() const { return splat_dof_depth_prepass_alpha_min; }
	void setDoFDepthPrepassAlphaMin(float v) { splat_dof_depth_prepass_alpha_min = v; }

	// DIAGNOSTIC ONLY - session046 open question (3): shrinks a splat's quad continuously by how covered the composite
	// already is under it, instead of the saturation gate's binary keep/drop (which needs every texel the quad touches
	// marked finished before it drops anything, and so rarely fires - see getSaturationGateEnabled()). 0 (default) is
	// off, and the only value that changes nothing: OpenGLEngine::markSaturatedSplatPixels() then still builds the
	// coverage pyramid's base level alongside the gate's mask (that part is unconditional, since it costs nothing beyond
	// a second framebuffer attachment on a pass already running) but never reduces or samples it. Above 0, a splat's
	// radii are multiplied by (1 - coverage_estimate * this), clamped to [0, 1], where coverage_estimate is the mean of
	// the accumulated coverage the composite already holds under the splat's quad, read from a *mean* pyramid built
	// alongside the gate's own *min* one - see gaussian_splat_saturation_mask_frag_shader.glsl and
	// gaussian_splat_mask_reduce_mean_frag_shader.glsl. 1 shrinks a splat over an already-fully-covered pixel to
	// nothing; 0.5 shrinks it by at most half.
	//
	// Requires the saturation gate to be on (more than one draw slice, gate enabled): it reads the coverage the gate's
	// own mark pass already computed rather than paying for a pass of its own, so it can only see what the gate sees.
	// Not yet measured - this is the toggle to do that with.
	float getCoverageShrinkStrength() const { return splat_coverage_shrink_strength; }
	void setCoverageShrinkStrength(float v) { splat_coverage_shrink_strength = v; }

	// Which formula the value above feeds. Both read the same coverage estimate, so switching between them changes only
	// what is done with it - which is what makes them comparable on one frame.
	//
	// 0: the original. Radii are scaled by (1 - coverage * value), and the conic is built from the scaled radii, so the
	// Gaussian narrows along with the quad. A splat over a half-covered pixel at full strength keeps a quarter of its
	// area but only half the light through that pixel was still available to lose, so this discards more than the
	// coverage says it can. That overshoot is what shows as a band wherever a slice boundary crosses a surface: the band
	// is not the step in the estimate at the boundary, which was implemented, measured and found to have no effect on it,
	// but the light this formula takes and the pixel had not finished with.
	//
	// 1: the value is a budget on light lost instead. What stopping the quad early throws away is the splat's own alpha
	// at that radius attenuated by the (1 - coverage) still reaching the eye, so the honest place to stop is where alpha
	// falls to a threshold the coverage sets - the question splat_alpha_cutoff already answers, asked with a threshold
	// raised by how finished the pixel is. The budget is added to splat_alpha_cutoff rather than replacing it, so an
	// empty pixel is left at exactly the alpha cutoff however high this is set, and the extra light given up works out
	// to budget * coverage. sigma_cutoff is scaled with the radii, so the same Gaussian is drawn over a shorter quad:
	// the faint edge is truncated rather than the splat being made smaller.
	//
	// 2: mode 1 without the 1/(1 - coverage) amplification, so the threshold is splat_alpha_cutoff + budget * coverage
	// and cannot exceed splat_alpha_cutoff + budget however finished the pixel is. Mode 1's derivation is sound for a
	// splat considered alone and wrong for a stack: the budget it licenses is spent again, in full, by every splat over
	// the same pixel, and the amplification makes that ruinous - at coverage 0.99 the factor is 99, so a 0.02 budget
	// asks for a threshold of 2.0 and removes every remaining splat outright instead of trimming its edge. The pixel
	// freezes at whatever coverage it had reached (0.92 on the measured cushion, against 0.999 without the shrink) and
	// the missing light shows as a hole onto what is behind - one that moves with the camera, since where a pixel
	// freezes depends on where the slice boundaries fell. This mode gives up mode 1's closed form to bound that.
	//
	// Measured against mode 0 and against raising splat_alpha_cutoff on its own, at matched cost on one frame: at the
	// same draw time as alpha cutoff 0.04, this mode's mean error was a quarter of it, and in the sparsely covered
	// regions - where the visible holes are - a seventh. Both take their saving from the same places on a solid wall;
	// only this one leaves the thin regions alone, because only this one asks whether there was anything there to lose.
	int getCoverageShrinkMode() const { return splat_coverage_shrink_mode; }
	void setCoverageShrinkMode(int v) { splat_coverage_shrink_mode = v; }

	// How the coverage pyramid the shrink above reads is reduced from one level to the next.
	//
	// 0 (mean, the original): a coarse texel is the average coverage under it. Cheap to reason about in the middle of a
	// solid surface, and wrong at its edge: a splat straddling a boundary between finished and unfinished screen gets an
	// answer describing neither half, and over the unfinished half that answer is too high. It then truncates itself as
	// if the pixels under it were done. On a low-opacity splat the truncation radius reaches zero at a modest threshold,
	// so the splat is not trimmed but removed outright - which is what an unfinished region surrounded by finished ones
	// (a cushion seen against a wall, say) shows as holes full of flickering splat-sized rectangles.
	//
	// 1 (min): a coarse texel is the *least* covered pixel under it, so the estimate becomes a lower bound - "under you
	// at least this much is covered" - and the straddling splat above is answered by its unfinished half and draws in
	// full. The error changes direction: the shrink can then only fail to save work, never remove light that was needed.
	// This is the same rule the binary gate's own pyramid has always used, and for the same reason.
	//
	// Kept as a switch rather than a replacement so the two can be A/B'd on one frame: the difference is entirely in
	// what the estimate says, and both formulas above read it identically.
	int getCoverageReduceMode() const { return splat_coverage_reduce_mode; }
	void setCoverageReduceMode(int v) { splat_coverage_reduce_mode = v; }

	// DIAGNOSTIC ONLY - draws the coverage pyramid itself over the frame, as grey levels, instead of the resolved splats:
	// this is what the shrink reads, at the level asked for, in the screen position it stands for. -1 (default) = off.
	//
	// The level is the point of it. Level 0 is one texel per getSaturationMaskDownscale() pixels and looks much like the
	// alpha snapshots; the coarse levels are what a big quad actually samples, and are where a mean pyramid smears a
	// small unfinished region away entirely. Comparing a level between the two reduce modes above is the whole use.
	//
	// Requires the saturation gate to be running, since the pyramid is built by its mark pass. Costs nothing while off:
	// the pyramid is not reduced and the resolve shader's branch is one integer compare on a pass that already runs.
	int getShowCoverageMapLevel() const { return splat_show_coverage_map_level; }
	void setShowCoverageMapLevel(int v) { splat_show_coverage_map_level = v; }

	// The program that halves the coverage pyramid by mean, once per level - see getMaskReduceProgram(), whose min
	// pyramid this sits beside. Null until the first addObject(), like the rest.
	const Reference<OpenGLProgram>& getMeanMaskReduceProgram() const { return mean_reduce_prog; }

	// Location of the splat program's coverage-mask sampler, or -1 if the program has no such uniform - see
	// getSplatMaskTexUniformLoc(), which this mirrors. Resolved on first use for the same reason that is.
	int getCoverageMaskTexUniformLoc();

	// Whether the splat pass needs its per-pixel layer counter this frame - see OpenGLScene::splat_layer_count_renderbuffer.
	// Deliberately not getCoverageCap(): that cap reads the coverage the accumulation buffer already holds, and being able
	// to leave this attachment off is most of the reason it exists.
	bool wantsLayerCountBuffer() const { return (splat_layer_cap > 0) || splat_layer_estimate_requested; }

	// SESSION066 - whether the "Clip" diagnostic wants its own overdraw-count buffer this frame (i.e. Clip is on). Kept
	// separate from the gate's mask so Clip no longer stands the gate down - see OpenGLScene::splat_hide_count_copy_texture.
	bool wantsHideCountBuffer() const { return getHideMode() != 0; }

	// Sampler location for that Clip overdraw-count texture in the main splat program, resolved on first use - mirrors
	// getSplatMaskTexUniformLoc(). See gaussian_splat_vert_shader.glsl.
	int getHideCountTexUniformLoc();

	// SESSION066 - the Clip per-splat cull uniforms, set by the draw path each frame (like setSplatMaskBlockSize): active
	// turns the vertex test on and off (0 = off, no sample), threshold is the red-zone value (getOverdrawRangeMax()).
	void setHideCountCull(bool active, float threshold);

	// Asks the next frame to measure how much fill a per-pixel cap would remove, and report it - see
	// OpenGLEngine::estimateSplatLayerCapSaving().  The frame that does it draws uncapped, so the counts it reads are
	// what the scene has rather than what the cap left.
	void requestLayerCapEstimate() { splat_layer_estimate_requested = true; }
	bool layerEstimateRequested() const { return splat_layer_estimate_requested; }
	void clearLayerEstimateRequest() { splat_layer_estimate_requested = false; }
	void setLayerEstimateResult(const std::string& s) { splat_layer_estimate_result = s; }
	const std::string& getLayerEstimateResult() const { return splat_layer_estimate_result; }

	// Which test the splat vertex shader uses against the mask, for the layer cap and for the hide-overdraw diagnostic
	// alike.  true (default): conservative - drop the splat only when its whole quad is inside the marked region, so
	// nothing that could still have shown is lost and no hole can open.  false: the centre test - drop it when the texel
	// under its centre is marked, which cuts through a marked region and leaves a hole, and is what shows how much of
	// the picture that region was carrying.  Quads are bigger than the connected marked patches, so the conservative
	// test removes far fewer splats; the two answer different questions and both are worth having.
	bool getHideTestConservative() const { return splat_hide_test_conservative; }
	void setHideTestConservative(bool v) { splat_hide_test_conservative = v; }

	// DIAGNOSTIC ONLY - draws only the first this-many slices of every cloud and stops, leaving the frame deliberately
	// incomplete. 0 (default) draws them all, which is the only value the picture is correct at.
	//
	// It exists to locate the fill in depth rather than on the screen. Slicing already splits each cloud into
	// consecutive depth ranges; raising the limit one step at a time and reading the splat pass time attributes the cost
	// to a range, which no screen-space diagnostic can do - the overdraw views say where the layers pile up, not which
	// of them is paying for it. The slice sizes are the ones getSliceGrowth() describes, so the steps are equal in splat
	// count, not in depth.
	int getDrawSliceLimit() const { return splat_draw_slice_limit; }
	void setDrawSliceLimit(int v) { splat_draw_slice_limit = v; }

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

	// Whether draw slice boundaries are placed by how many *visible* splats each slice holds, rather than by raw
	// position in the draw order. false (default) is the plain index split, i.e. slicing exactly as it was.
	//
	// The problem it addresses: the draw order is sorted by distance from the camera and covers the whole cloud,
	// including everything behind the viewer (deliberately - see GaussianSplatLodTraversalTask's comment on why the
	// traversal has no frustum test). So the nearest splats by distance are a sphere around the camera, of which only
	// the part inside the frustum is drawn to any pixels. Standing above a forest looking into the distance, the first
	// slices are almost entirely the ground below and behind, and the saturation census taken after each of them
	// photographs a mask that is still completely black - a full-screen pass that could not have found anything. The
	// same happens in reverse in a room: what is behind the viewer is a large share of the near end of the order.
	//
	// The fix is not to cull the order, which would have to be redone on every rotation and, being a frame or two late,
	// would show as holes along the screen edge. It is to keep drawing the whole order and move only where it is cut:
	// a fixed-size sample of the order is tested against the frustum each frame, and the boundaries are placed so each
	// slice holds an equal share of what is actually visible. Slicing provably cannot change the picture - the slices
	// tile the same order either way - so a sample that is coarse, stale or simply wrong can misplace a census and can
	// do nothing else. That asymmetry is the whole reason this is the affordable version of the idea.
	//
	// It also handles the second half of the problem for free: a near frustum that is empty of geometry (air above the
	// forest) contributes no samples either, so the boundaries move past it for the same reason.
	//
	// Not yet measured - this is the switch to A/B it against a fixed camera with.
	bool getVisibleSlicingEnabled() const { return splat_visible_slicing; }
	void setVisibleSlicingEnabled(bool v) { splat_visible_slicing = v; }

	// Draw index at which the given fraction of everything visible in 'ob's cloud has been passed, for placing one draw
	// slice boundary - see getVisibleSlicingEnabled(). draw_count is how many instances the caller is slicing, which is
	// also what a fraction of 1 maps to.
	//
	// Returns -1 when the question cannot be answered - the feature is off, the cloud is not one of ours, the sample
	// does not describe the order being drawn, or nothing of the cloud is in view - in which case the caller falls back
	// to the plain index split. Deliberately a per-boundary query rather than a precomputed list, so that the geometric
	// slice spacing (see getSliceGrowth()) stays owned by the draw path that already implements it.
	int visibleFractionToDrawIndex(const GLObject* ob, double fraction, int draw_count) const;

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
	// DIAGNOSTIC ONLY - see OpenGLEngine::fillCappedSplatPixels().  Null until the first addObject(), like the rest.
	const Reference<OpenGLProgram>& getCapFillProgram() const { return cap_fill_prog; }
	int getCapFillMaskTexUniformLoc(); // Resolved on first use, like getSplatMaskTexUniformLoc(): a sampler uniform is not something the material path can carry.
	void setCapFillUniforms(int block_size) const;

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
	//
	// centre_test picks which of two tests the shader applies. false is the conservative one the saturation gate needs: a
	// splat goes only if every mask texel its quad can touch is marked, so it is never dropped where it could still have
	// changed a pixel. true is the opposite, and is for the hide-overdraw diagnostic: the splat goes if the texel under
	// its centre is marked, which cuts straight through a marked region and leaves a visible hole in it. That hole is the
	// point there - it is what shows how much of the picture the marked region was carrying - and it is also why the
	// conservative test cannot serve: quads are larger than the connected marked patches, so almost every splat overlaps
	// the edge of one and survives.
	// DIAGNOSTIC ONLY - block size for the per-fragment mask test the layer cap uses, 0 to switch it off.  Separate from
	// the vertex test's: the cap wants the pixels rejected and the splats left alone, which is the opposite of the gate.
	void setSplatFragMaskBlockSize(int block_size);

	void setSplatMaskBlockSize(int block_size, int max_level, bool centre_test = false);

	// Sets that program's uniforms, plus how many accumulation-buffer pixels across one mask texel covers, which is the
	// caller's since it depends on the size the mask was actually allocated at. Called once the mask program is bound,
	// for the same reason setResolveOverdrawUniforms() exists: the pass is one manual full-viewport quad with no
	// material, so it bypasses the generic per-object uniform path.
	//
	// from_layer_count picks which of the two things the mask is marking: false is the saturation gate, marking where the
	// composite has finished, thresholded at getSaturationThreshold(); true is the "hide overdraw" diagnostic, marking
	// where the layer count has reached getOverdrawRangeMax(). Both come out as the same one-bit mask, so everything
	// downstream - the pyramid, the vertex shader - is shared.
	// layer_count_threshold: the count being thresholded is the per-pixel layer cap's rather than the overdraw ramp's red
	// end - the two share this pass and differ only in what number they compare against.
	// coverage_cap_threshold: same idea on the coverage side - the coverage being thresholded is getCoverageCap()'s rather
	// than the gate's, the two being the same measure read at different places.  Ignored when from_layer_count is true.
	void setSaturationMaskUniforms(int block_size, bool from_layer_count, bool layer_count_threshold = false, bool coverage_cap_threshold = false) const;

	// Diagnostic, not an optimisation: draws a counting pass first, then throws away every splat that lands where the
	// layer count reached getOverdrawRangeMax() - that is, exactly the region the overdraw view paints solid red - and
	// draws what is left. Answers "are the red regions really where the time goes" directly, by removing them and looking
	// at the clock, rather than by inference from the histograms.
	//
	// The splats are dropped in the vertex shader, before rasterisation, which is the earliest point available. Costs an
	// extra full pass over the splats to build the mask, so the frame total is meaningless while this is on - read the
	// "draw splats" GPU timer, which is started after the counting pass and so measures the same thing in both states.
	//
	// Mutually exclusive with the saturation gate, which owns the same mask texture; the gate is switched off while this
	// is on.
	bool getHideOverdrawEnabled() const { return splat_hide_overdraw_enabled; }
	void setHideOverdrawEnabled(bool v) { splat_hide_overdraw_enabled = v; }

	// The same diagnostic against the other overdraw measure: the counting pass sums each fragment's alpha instead of
	// counting fragments, so what gets thrown away is the splats standing where a lot of alpha has piled up rather than
	// where a lot of layers have. The two answer different questions - many faint layers show up in the first and not the
	// second - and the threshold is the same range maximum, read in whichever units the counting pass is accumulating.
	bool getHideAlphaEnabled() const { return splat_hide_alpha_enabled; }
	void setHideAlphaEnabled(bool v) { splat_hide_alpha_enabled = v; }

	// 0 = neither, 1 = hide by layer count, 2 = hide by summed alpha. Only one mask exists, so the two cannot both run;
	// layers win, being the coarser and more usual question.
	int getHideMode() const { return splat_hide_overdraw_enabled ? 1 : (splat_hide_alpha_enabled ? 2 : 0); }

	// The counting pass is only redone when something it depends on has changed, so a still camera pays for it once and
	// then not at all. That is what makes the frame total comparable: with the mask frozen, the frame contains exactly
	// the work a cloud with those splats removed would do, and nothing else.
	//
	// The cost is that the mask is a screen-space thing pinned to the camera that built it, so while the camera moves it
	// describes where the crowded regions used to be. That is acceptable here and nowhere else: this is a measurement
	// taken standing still.
	//
	// Everything the mask depends on is compared, not just the camera: the viewport it was built at, the threshold, the
	// mask resolution, and how many splats were drawn - the last of which is what catches a LoD parameter changing the
	// drawn set without the camera moving at all.
	bool hideOverdrawMaskNeedsRebuild(const Matrix4f& view_matrix, int viewport_w, int viewport_h, int mask_block, uint64 num_splats_drawn) const;
	void noteHideOverdrawMaskBuilt(const Matrix4f& view_matrix, int viewport_w, int viewport_h, int mask_block, uint64 num_splats_drawn);
	void invalidateHideOverdrawMask() { splat_hide_overdraw_mask_valid = false; }
	uint64 getHideOverdrawMaskRebuilds() const { return splat_hide_overdraw_mask_rebuilds; }

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
	//
	// coverage_map_block is how many screen pixels across one level-0 texel of the coverage pyramid covers, needed by
	// getShowCoverageMapLevel()'s view to map a fragment back to the texel standing for it. The caller's, since it
	// depends on the size the mask was actually allocated at - the same reason setSaturationMaskUniforms() takes it.
	void setResolveOverdrawUniforms(int coverage_map_block) const;

	// Which texture unit the resolve pass reads the coverage pyramid on, for getShowCoverageMapLevel()'s view. Unit 0 is
	// the accumulation buffer that pass already reads; this is a pass of its own, so the numbering is local to it and has
	// nothing to do with the splat program's units. Named here because the bind (OpenGLEngine) and the sampler uniform
	// (setResolveOverdrawUniforms() above) have to agree, and they live in different files.
	static const int RESOLVE_COVERAGE_MAP_TEXTURE_UNIT_INDEX = 1;

	// Location of that sampler, or -1 while the program is still building. Kept here rather than indexed into the
	// user-uniform list at the call site, so that the list's layout stays this class's business.
	int getResolveCoverageMapTexUniformLoc() const;

	// SplatDoFDepthMode_Weighted: tells the resolve pass whether to divide the weighted-depth accumulation
	// (OpenGLScene::splat_dof_depth_renderbuffer) through by coverage and write gl_FragDepth from it this frame,
	// and gives it the near-clip distance the view-depth -> device-depth conversion needs (this program has no
	// use for the MaterialCommonUniforms block the rest of the engine's shaders get near_clip_dist from - see
	// setResolveOverdrawUniforms()'s own comment on why this pass's uniforms are all set manually). Called by
	// OpenGLEngine::resolveSplatAccumBuffer() alongside setResolveOverdrawUniforms().
	void setResolveDoFDepthUniforms(bool write_weighted_depth, float near_clip_dist) const;

	// See RESOLVE_COVERAGE_MAP_TEXTURE_UNIT_INDEX's comment - same reasoning, a different unit.
	static const int RESOLVE_DOF_DEPTH_TEXTURE_UNIT_INDEX = 2;
	int getResolveDoFDepthTexUniformLoc() const;

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
	void kickOffFilters();      // SESSION063: split architecture - see the .cpp.
	void drainFilterResults();  // SESSION063

	void noteDrawOrderForSlicing(SplatCloud& cloud, const uint32* draw_indices, size_t count); // Refreshes the sample of a cloud's draw order the frustum-aware slicing works from - see getVisibleSlicingEnabled().  Called from every place that writes the instance index VBO.
	void buildVisibleSliceCDFs(); // Per-frame, from think(): re-tests each cloud's sample against the current frustum.  The one part of the draw order that depends on where the camera is looking rather than where it is.
	void fillTraversalScratch(SplatCloud& cloud, GaussianSplatLodTraversalScratch& scratch) const; // Freezes a cloud's world-space node data and member layout into a scratch, ready for a traversal to read without touching the live arrays. SESSION058: non-const - may cache the snapshot on the cloud (see SplatCloud::cached_traversal_geom) so repeated kicks against an unchanged cloud reuse it instead of re-copying.

	Reference<OpenGLProgram> shader_prog; // Shared by every cloud.  Null until the first addObject().
	Reference<OpenGLProgram> resolve_prog; // Resolves the splat accumulation buffer onto the main colour buffer.  Built alongside shader_prog.
	Reference<OpenGLProgram> saturation_mask_prog; // Marks finished pixels between draw slices.  Built alongside shader_prog.
	Reference<OpenGLProgram> cap_fill_prog; // DIAGNOSTIC ONLY - fills capped pixels out to full coverage, see getLayerCapOpaque().  Built alongside shader_prog.
	int cap_fill_mask_tex_uniform_loc;

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

	// SESSION063: per-orientation filter pipeline of the split architecture - independent of the traversal pipeline above.
	// A filter is stateless apart from its input U(P) (held by the task via Reference), so no scratch pool is needed.
	int num_filters_in_flight;
	static const int max_concurrent_filters = 2; // A filter is ~13ms; a couple in flight covers multi-cloud scenes without oversubscribing the worker pool.
	ThreadSafeQueue<Reference<ThreadMessage> > filter_result_queue;
	js::Vector<Reference<ThreadMessage>, 16> completed_filter_msgs;

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

	// See getFrustumCullEnabled() above. On by default from session055.
	bool lod_frustum_cull_enabled;

	// SESSION063: see getSplitFilterEnabled() above. Off by default - the split path is opt-in for A/B while it's built out.
	bool split_filter_enabled;

	// SESSION063 K3: filter dilation knobs - see the getters above.
	float filter_dilation_latency;      // s
	float filter_min_rot_rate_deg_per_s;
	float filter_min_trans_rate_m_per_s;

	// SESSION063 K4: coarse floor knobs - see the getters above.
	bool split_coarse_floor_enabled;
	float split_coarse_pixel_scale;         // pixel_scale threshold for the coarse cut (>> pixel_scale_limit).
	float filter_coarse_dilation_latency;   // s - the coarse tail's (wider) dilation window.
	bool coarse_layer_debug;                // Draw only the coarse floor - see getCoarseLayerDebug().

	// SESSION055: camera-motion tracker for anisotropic frustum-cull dilation. think() diffs the current cam pose against
	// the previous one to compute an instantaneous velocity and angular speed, feeds them through an EMA with a
	// max(instant, ema) override (rise-fast, fall-slow, so movement start doesn't lag). kickOffTraversals() turns them
	// into per-plane translation dilation and a per-node-distance rotation dilation rate, so the traversal keeps enough
	// margin for nodes that will enter the frustum before the async result lands - see the cull block in
	// GaussianSplatLodTraversalTask::run(). Zero-init: first think() only records the pose, no dilation yet.
	bool have_prev_think_cam_state;
	Vec4f prev_think_cam_pos_ws;
	Vec4f prev_think_cam_forward_ws;
	Timer prev_think_timer;                  // Reset each think(); elapsed() between resets is dt for the velocity diff.
	Vec4f cam_velocity_ema_ws;               // World-space linear velocity (m/s), EMA-smoothed.
	float cam_angular_speed_ema;             // Scalar angular speed (rad/s), max(inst, blended).
	float cam_angular_speed_peak;            // SESSION055: slow-decay peak of angular speed, so a mouse flick keeps rotation dilation elevated for the next ~1s of kicks - covers subsequent bursts that neither EMA nor empirical predict in time.
	float cam_inst_angular_speed;            // SESSION064: this frame's raw instantaneous angular speed (rad/s), NOT smoothed. Drives the filter's per-frame re-filter trigger (kickOffFilters()): it is 0 the moment the camera stops, whereas the EMA/peak above coast down over ~2s and would keep re-filtering (and boiling) a static camera - see session064 snapshot.

	// See getSizeClampMin()/getSizeClampMax() above. Defaults (0, 0) disable both bounds, so never exclude a real splat.
	float splat_size_clamp_min;
	float splat_size_clamp_max;
	bool splat_size_clamp_invert;

	// See getDistClampMin()/getDistClampMax()/getDistClampInvert() above. Default (0, 1000) keeps every splat.
	float splat_dist_clamp_min;
	float splat_dist_clamp_max;
	bool splat_dist_clamp_invert;

	// See getAlphaCutoff() above. Default 1/255 is lossless (matches the fragment shader's fixed discard threshold).
	float splat_alpha_cutoff;

	// See getAlphaGain()/getAlphaGamma() above. Both 1 = the stored alpha untouched.
	float splat_alpha_gain;
	float splat_alpha_gamma;

	// See getNumDrawSlices() above. Default 1 = one draw per cloud, i.e. no slicing.
	int splat_num_draw_slices;

	// DIAGNOSTIC ONLY - see getDrawSliceLimit() above. Default 0 = draw every slice, i.e. a complete frame.
	int splat_draw_slice_limit;

	// See getLayerCap()/getHideTestConservative() above. Default 0 = uncapped, conservative test.
	int splat_layer_cap;
	bool splat_layer_cap_opaque;

	// See getCoverageCap() above. Default 0 = off.
	float splat_coverage_cap;

	// See getAblationStage() above. Default 0 = off, i.e. the full pass.
	int splat_ablation_stage;

	// See getQuadRadiusScale() above. Default 1 = the real size.
	float splat_quad_radius_scale;
	bool splat_hide_test_conservative;

	// See getAreaScaleGamma()/getAreaScaleRefPx() above. Default gamma 1 = linear in area; ref_px is inert while
	// splat_quad_radius_scale is 1, so its default value only matters once that is lowered.
	float splat_area_scale_gamma;
	float splat_area_scale_ref_px;

	// See getCoverageShrinkStrength() above. Default 0 = off.
	float splat_coverage_shrink_strength;

	// See getCoverageShrinkMode() above. Defaults to the original formula, so that a session that turns the strength up
	// sees exactly what it saw before this existed.
	int splat_coverage_shrink_mode;

	// See getCoverageReduceMode() above. Defaults to the mean pyramid, i.e. to what the shrink read before the switch
	// existed, so that turning the strength up reproduces the earlier sessions' numbers unchanged.
	int splat_coverage_reduce_mode;

	// See getShowCoverageMapLevel() above. -1 = off.
	int splat_show_coverage_map_level;

	// See getEWAProjectionFixEnabled()/getNearFadeWidth() above. Defaults: on, and a fade over the last 30% of the approach.
	bool splat_ewa_fix_enabled;
	float splat_near_fade_width;
	float splat_near_epsilon; // See getNearEpsilon(). Default 0.1 (historical hardcode).
	int splat_dof_depth_mode; // See getDoFDepthMode(). Default SplatDoFDepthMode_Off.
	float splat_dof_depth_prepass_alpha_min; // See getDoFDepthPrepassAlphaMin(). Default 0.3.
	Reference<OpenGLProgram> mean_reduce_prog; // See getMeanMaskReduceProgram() above. Built alongside mask_reduce_prog.
	int coverage_mask_tex_uniform_loc; // See getCoverageMaskTexUniformLoc() above. -2 means "not looked up yet", as with splat_mask_tex_uniform_loc.

	// See requestLayerCapEstimate() above.
	bool splat_layer_estimate_requested;
	std::string splat_layer_estimate_result;

	// See getSliceGrowth() above. Default 1 = every slice the same size.
	float splat_slice_growth;

	// See getVisibleSlicingEnabled() above. Off by default, i.e. the plain index split slicing has always used.
	bool splat_visible_slicing;

	// See getSaturationGateEnabled()/getSaturationThreshold()/getSaturationMaskDownscale() above. Off by default.
	bool splat_saturation_gate_enabled;
	float splat_saturation_threshold;
	int splat_saturation_mask_downscale;

	// See getSplatMaskTexUniformLoc(). -2 means "not looked up yet", which -1 cannot mean, that being GL's answer for
	// a uniform the linker dropped.
	int splat_mask_tex_uniform_loc;
	int splat_hide_count_tex_uniform_loc; // SESSION066 - sampler loc for splat_hide_count_texture; -2 = not looked up yet, as with splat_mask_tex_uniform_loc.

	// See getAccumBuffer8Bit() above. Off by default, i.e. the RGBA16F buffer this pass has always used.
	bool splat_accum_buffer_8bit;

	// See getShowOverdrawMode() above. 0 = off.
	int splat_show_overdraw_mode;

	// See getHideOverdrawEnabled()/getHideAlphaEnabled() above. Both off by default.
	bool splat_hide_overdraw_enabled;
	bool splat_hide_alpha_enabled;

	// State of the frozen hide-overdraw mask - see hideOverdrawMaskNeedsRebuild().
	bool splat_hide_overdraw_mask_valid;
	Matrix4f splat_hide_overdraw_mask_view;
	int splat_hide_overdraw_mask_viewport_w, splat_hide_overdraw_mask_viewport_h;
	int splat_hide_overdraw_mask_block;
	float splat_hide_overdraw_mask_threshold;
	int splat_hide_overdraw_mask_mode; // Which of the two measures built it - switching between them has to recount.
	uint64 splat_hide_overdraw_mask_splats_drawn;
	uint64 splat_hide_overdraw_mask_rebuilds; // Surfaced in the diagnostics: while this is still, the frame carries no counting pass and the total means something.

	// See getOverdrawRangeMin()/getOverdrawRangeMax() above.
	float splat_overdraw_range_min;
	float splat_overdraw_range_max;

	// One line describing the coplanar merge the clouds are currently carrying, or empty if they are as they were loaded.
	// Printed by getFrustumStructureReport(), so that a report taken after a merge says so in its own text - a log full of
	// reports whose state has to be reconstructed from what was pressed before them is a log that will eventually be read
	// wrong.  Set by applyCoplanarMerge(), cleared by restoreUnmergedSplats().
	std::string last_merge_description;

	// What the last getFrustumStructureReport() press measured about the vertex shader's own culling: how many splats
	// reached the rasteriser, out of how many were in the frustum before it.  Kept so getDiagnostics() can show it in the
	// panel, where the per-frame counters next to it cannot answer that question at all - the culls happen on the GPU,
	// and the only exact answer available is the CPU pass the report runs.  Zero until a report has been taken.
	uint64 last_report_reached_rasteriser, last_report_in_frustum;
};
