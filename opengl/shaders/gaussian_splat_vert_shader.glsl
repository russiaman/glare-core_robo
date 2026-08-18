
// gaussian_splat_vert_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Projects each Gaussian splat to a screen-space ellipse (EWA splatting), drawn as one instanced quad.
// See opengl/GaussianSplatRenderer.h for the overall design.

precision highp float; // Override the engine's default "precision mediump float;" for Emscripten - the covariance maths is precision-sensitive.

in vec3 position_in; // Local quad corner, in [-1, 1] x [-1, 1], z = 0.  Scaled and oriented per splat below.
in uint splat_index_in; // Per-instance index into albedo_texture, reordered each frame by the depth sort.

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 proj_matrix;

uniform sampler2D albedo_texture; // Packed splat data, 4 RGBA32F texels per splat - see packSplatTexels() in GaussianSplatRenderer.cpp.
uniform vec2 viewport_dims_px;
uniform vec2 focal_len_px;
uniform int splat_tex_width;
uniform vec2 splat_size_clamp_min_max; // Diagnostic tool (GaussianSplatSettingsWidget, Qt only): culls any splat whose feature_size (2 * max scale axis, matching GaussianSplatLodNode::feature_size) falls outside [x, y]. 0 disables the respective bound (matches the "0 = unlimited" convention used by the Splats list search radius) - (0, 0), the default, disables the filter entirely.
uniform int splat_size_clamp_invert; // 0 (default) = cull outside [min, max] (isolate a size range); non-zero = cull inside [min, max] instead (exclude a size range, leaving the rest of the cloud untouched). No effect while the clamp itself is disabled.

// Diagnostic tool (GaussianSplatSettingsWidget, Qt only): keeps only splats whose distance from the camera falls inside
// [x, y] metres, so the cloud can be sliced open and looked into a shell at a time.  Tested against distance from the
// camera position rather than depth along the view axis, so the slice is a spherical shell centred on the viewer and
// turning on the spot does not change which splats survive.  Default (0, 1000) keeps everything at any sane scene size.
uniform vec2 splat_dist_clamp_min_max;
uniform int splat_dist_clamp_invert; // 0 (default) = keep what is inside the range; non-zero = keep what is outside it, i.e. cut the shell away and see through it.
// The saturation gate: a mask of the pixels the composite has already finished with, rebuilt between draw slices - see
// gaussian_splat_saturation_mask_frag_shader.glsl.  splat_saturation_mask_block is how many screen pixels across one
// mask texel covers, or 0 when the gate is not running, which is also the only state in which the texture may hold
// anything meaningless.
//
// Tested here, per splat, rather than per fragment: a fragment-side test was measured and saved nothing at all, because
// one texture fetch across the hundreds of millions of fragments this pass shades costs about what the blending it
// skips is worth.  Here the fetch is paid once per splat - a hundred times less often - and a splat that fails takes
// its rasterisation and its whole quad's worth of shading with it, not just the blending.
uniform sampler2D splat_saturation_mask_texture;
uniform int splat_saturation_mask_block;
uniform int splat_saturation_mask_max_level; // Coarsest level of the mask's min-pyramid, so that a big quad can be answered with one fetch rather than many.

// 0: the conservative test the gate needs - drop the splat only if every texel its quad can touch is marked, so it is
// never dropped where it could still have changed a pixel.
// 1: the hide-overdraw diagnostic - drop it if the texel under its centre is marked. Not conservative, and deliberately
// so: it cuts through a marked region and leaves a hole, which is what shows how much of the picture that region was
// carrying. The conservative test cannot do that, because quads are bigger than the connected marked patches and so
// nearly all of them overlap an unmarked edge and survive.
uniform int splat_mask_centre_test;

uniform float splat_alpha_cutoff; // GaussianSplatSettingsWidget, Qt only. Sets the per-splat quad radius to exactly where alpha decays to this value, instead of the fixed 3-sigma bound below - see the derivation where it's used. Default 1/255 matches the fragment shader's own discard threshold exactly (lossless); raising it trims low-opacity splats' quads further, trading a sliver of their faint edge for less overdraw.

// (gain, gamma) of the live opacity adjustment alpha' = gain * alpha^gamma - see adjustSplatAlpha() in
// graphics/GaussianSplatData.h, which is where this transform is defined and which the CPU-side cost predictions use.
// Duplicated here rather than shared because GLSL has no way to include it: keep the two in step.
// (1, 1) is the identity, and is what the renderer sends unless the panel says otherwise.
uniform vec2 splat_alpha_gain_gamma;

// DIAGNOSTIC ONLY - the ablation ladder, see GaussianSplatRenderer::getAblationStage().  0 (default) = the full shader
// below, i.e. every frame that is not measuring.  Higher stages switch progressively more of the pass back on, so that
// the frame time can be read at each step and the jump between two steps attributes cost to the one thing that changed.
// Stage 1 (CPU only) never reaches a shader - the draw path skips the draw call - so the lowest value seen here is 2.
// The quad is kept a quad at every stage, shrunk to a pixel rather than swapped for GL_POINTS: changing the primitive
// type would change the rasterisation path as well as the area, and then the step would answer about both.
uniform int splat_ablation_stage;

// DIAGNOSTIC ONLY - multiplies every splat's screen-space quad radius, see GaussianSplatRenderer::getQuadRadiusScale().
// 1 (default) is the real size.  Rasterised area goes as the square of it, with the splat count, the vertex work and the
// blend chain all untouched - which is what makes it a way to ask whether a cost is area-proportional at all.
uniform float splat_quad_radius_scale;

// DIAGNOSTIC ONLY - makes splat_quad_radius_scale's reduction area-dependent instead of flat, see
// GaussianSplatRenderer::getAreaScaleGamma()/getAreaScaleRefPx(). Both are inert (reduction stays flat, as before these
// existed) while splat_quad_radius_scale is 1, since the formula below then reduces to 1 regardless of area_weight.
uniform float splat_area_scale_gamma; // 1 (default) = weight is linear in the splat's own screen-space area.  Raising it
                                       // concentrates the reduction on splats with more area than splat_area_scale_ref_px;
                                       // lowering it spreads a partial reduction onto smaller splats too.
uniform float splat_area_scale_ref_px; // Radius (px, pre-scale) at which area_weight reaches 1, i.e. the splat gets the
                                        // full splat_quad_radius_scale reduction.  Below it, the reduction fades toward
                                        // none as the splat's own area falls, at the rate splat_area_scale_gamma sets.

// DIAGNOSTIC ONLY - the coverage-based quad shrink, see GaussianSplatRenderer::getCoverageShrinkStrength() and the
// block below that uses these.  0 (default) is off - splat_coverage_mask_texture is then whatever the gate last built
// it as, or uninitialised if the gate has never run, and is never sampled while this is 0.
uniform sampler2D splat_coverage_mask_texture;
uniform float splat_coverage_shrink_strength;

// Which of the two shrink formulas splat_coverage_shrink_strength above feeds - see
// GaussianSplatRenderer::getCoverageShrinkMode().  0 = scale the radii by (1 - coverage * strength), which narrows the
// Gaussian with the quad; 1 = spend the value as a light-loss budget, which truncates the same Gaussian instead.  The
// two read the identical coverage estimate and differ only in what they do with it, so the A/B is on the formula alone.
uniform int splat_coverage_shrink_mode;

// Turns off all three of the corrections to the affine projection below - the frustum cull, the honest-size bound on the
// radius, and the near fade - leaving the projection exactly as it was before them.  For A/B comparison; see
// GaussianSplatRenderer::getEWAProjectionFixEnabled().
uniform int splat_ewa_fix_enabled;

// How wide, as a fraction of the ratio the test uses, the fade-out of splats the camera is getting inside of is spread -
// see where it is used, and GaussianSplatRenderer::getNearFadeWidth().  0 makes it a hard cull at the point the splat's
// projection stops existing; the default spreads it over the last part of the approach so nothing pops.
uniform float splat_near_fade_width;

// Depth below which a splat centre is culled unconditionally (independent of the EWA-fix radius clamp),
// because the Jacobian's 1/d and 1/d^2 terms produce numerically extreme covariances there.  Default 0.1 m
// (the historical hardcode); can be lowered by the Photo Mode "Near clip" override for very close-up shots,
// at the cost of the artefacts near_epsilon exists to hide.  See GaussianSplatRenderer::getNearEpsilon().
uniform float splat_near_epsilon;

out vec2 frag_screen_offset_px; // Pixel-space offset of this vertex from the splat's projected centre.
out vec3 frag_conic; // Inverse 2D covariance (A, B, C) of [[A, B], [B, C]], for the per-pixel Gaussian evaluation.
out vec4 frag_colour; // (r, g, b, opacity)


ivec2 splatTexelCoord(int texel_index)
{
	return ivec2(texel_index % splat_tex_width, texel_index / splat_tex_width);
}


// DIAGNOSTIC ONLY - a one-pixel quad at this splat's position, used by ablation stages 2-4 to show where the splats are
// while rasterising as close to nothing as a quad can.  Everything the full path does between the projection and the
// quad offset - covariance, eigen-decomposition, radii - is skipped, which is the point: these stages measure what it
// costs to emit and place the instances, with the fill taken out.
void emitSplatPoint(vec3 pos_os)
{
	vec4 clip_pos = proj_matrix * (view_matrix * (model_matrix * vec4(pos_os, 1.0)));
	clip_pos.xy += (position_in.xy / viewport_dims_px) * clip_pos.w; // One pixel across, in the same clip-space form the full path below uses.
	gl_Position = clip_pos;
	frag_conic = vec3(0.0);
	frag_screen_offset_px = vec2(0.0);
}


void main()
{
	int base_texel = int(splat_index_in) * 4;

	vec4 t0 = texelFetch(albedo_texture, splatTexelCoord(base_texel + 0), 0);

	// DIAGNOSTIC ONLY - stage 2: the one texel a position needs, and nothing else.  The three fetches below are what
	// stage 3 adds, so the step between them is the cost of the extra vertex texture reads - open question 8.2.
	if(splat_ablation_stage == 2)
	{
		emitSplatPoint(t0.xyz);
		frag_colour = vec4(1.0);
		return;
	}

	vec4 t1 = texelFetch(albedo_texture, splatTexelCoord(base_texel + 1), 0);
	vec4 t2 = texelFetch(albedo_texture, splatTexelCoord(base_texel + 2), 0);
	vec4 t3 = texelFetch(albedo_texture, splatTexelCoord(base_texel + 3), 0);

	vec3 pos_os = t0.xyz;
	vec3 scale  = vec3(t0.w, t1.x, t1.y);
	vec4 rot    = vec4(t1.z, t1.w, t2.x, t2.y); // (x, y, z, w)
	frag_colour = vec4(t2.z, t2.w, t3.x, t3.y); // (r, g, b, opacity)

	// The live opacity adjustment, applied here - the first thing done with the splat's own data, and before the quad is
	// sized from it below - so that everything downstream is of the adjusted splat: the sigma cutoff that sets how wide
	// this splat is rasterised, the alpha the fragment shader blends, and through those the coverage the saturation gate
	// thresholds and the layer counts the overdraw views show.  See splat_alpha_gain_gamma above.
	frag_colour.a = min(splat_alpha_gain_gamma.x * pow(frag_colour.a, splat_alpha_gain_gamma.y), 1.0);

	// DIAGNOSTIC ONLY - stages 3 and 4: all four texels read and unpacked, still a one-pixel quad.  The degenerate-scale
	// test is here so that t1's fetch is genuinely consumed: a fetch the optimiser can prove unused is a fetch that did
	// not happen, and measuring that it happened is the whole of stage 3.  Stage 4 differs only in the fragment shader,
	// which outputs the splat's colour instead of flat white - that step should read near zero, and is a check on the
	// instrument as much as a measurement.
	if((splat_ablation_stage == 3) || (splat_ablation_stage == 4))
	{
		if(2.0 * max(scale.x, max(scale.y, scale.z)) <= 0.0)
		{
			gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
			frag_conic = vec3(0.0);
			frag_screen_offset_px = vec2(0.0);
			return;
		}
		emitSplatPoint(pos_os);
		return;
	}

	// Diagnostic size filter - see splat_size_clamp_min_max above.  Checked before any of the projection maths below,
	// since it needs only the raw world-space scale, not the view-dependent covariance.
	float feature_size = 2.0 * max(scale.x, max(scale.y, scale.z));
	bool clamp_active = (splat_size_clamp_min_max.x > 0.0) || (splat_size_clamp_min_max.y > 0.0);
	bool below_min = (splat_size_clamp_min_max.x > 0.0) && (feature_size < splat_size_clamp_min_max.x);
	bool above_max = (splat_size_clamp_min_max.y > 0.0) && (feature_size > splat_size_clamp_min_max.y);
	bool outside_range = below_min || above_max;
	bool cull_for_size = (splat_size_clamp_invert != 0) ? (clamp_active && !outside_range) : outside_range;
	if(cull_for_size)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	vec4 pos_vs = view_matrix * (model_matrix * vec4(pos_os, 1.0));

	// Note that the view_matrix the engine sends is in the standard OpenGL camera-space convention (x = right, y = up,
	// -z = forwards), not the engine's own raw (y = forwards, z = up) convention: OpenGLEngine builds it as
	// indigo_to_opengl_cam_matrix * world_to_camera_space_matrix before it reaches any shader.  So depth is -pos_vs.z,
	// and the Jacobian below is the textbook z-forward perspective derivative, with no engine-specific adaptation.
	// The distance slice - see splat_dist_clamp_min_max above.  Here rather than beside the size clamp because it needs
	// the view-space position, and before the covariance maths because that is the expensive part.
	float dist_to_cam = length(pos_vs.xyz);
	bool inside_dist_range = (dist_to_cam >= splat_dist_clamp_min_max.x) && (dist_to_cam <= splat_dist_clamp_min_max.y);
	bool cull_for_dist = (splat_dist_clamp_invert != 0) ? inside_dist_range : !inside_dist_range;
	if(cull_for_dist)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	float depth = -pos_vs.z;
	// Depth below which the splat's own centre is culled, to keep the Jacobian's 1/d / 1/d^2 terms bounded.
	// Was a hardcoded 0.1; now a uniform so the Photo Mode "Near clip" override can drop it for close-ups.
	if(depth < splat_near_epsilon)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	// Fade out splats the camera is getting inside of.
	//
	// The near-plane test above looks at the splat's centre only.  A splat is an ellipsoid with a real extent, though, and
	// a big flat one - a metre-wide, paper-thin splat lying along the floor, which every capture has plenty of, since a
	// surface photographed with little parallax does not constrain its splats' size - reaches well past its own centre.
	// Stand in the middle of a room and the floor splat under your feet has its centre a metre or two in front of you and
	// its far edge behind you.  The part behind the camera has negative depth, the perspective divide flips it, and it
	// lands above the horizon instead of below: that is the streak that climbs from the floor to the ceiling through the
	// frame.  It is not an artifact of the approximation - the projection of a splat straddling the camera plane genuinely
	// does not exist - so no bound on the radius can fix it.  Only not drawing the splat can.
	//
	// The condition is exactly "the camera is inside the splat's own extent": distance to its centre below its 3-sigma
	// radius.  That is the same idea as the near_epsilon test above, with the splat's own size in place of a fixed 10 cm.
	//
	// Faded rather than switched, because a splat this big is often the only thing covering its patch of floor, and a
	// hard test would pop it in and out as the camera moves.  splat_near_fade_width sets how much of the ratio the fade is
	// spread over: 0 gives exactly the hard test, and the fade costs nothing beyond this multiply - it in fact saves
	// rasterised area, since the sigma cutoff below derives the quad's radius from the opacity and so shrinks the quad as
	// the splat fades.
	if(splat_ewa_fix_enabled != 0)
	{
	float near_fade = clamp((1.0 - 3.0 * max(scale.x, max(scale.y, scale.z)) * length(model_matrix[0].xyz) / dist_to_cam) / max(splat_near_fade_width, 1.0e-6), 0.0, 1.0);
	if(near_fade <= 0.0)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}
	frag_colour.a *= near_fade;
	}

	// Cull splats that lie entirely outside the frustum's side planes.
	//
	// The projection below is an *affine* approximation of the perspective divide, taken about this splat's own position,
	// and it is only accurate near the view axis: the Jacobian's focal * v / depth^2 term grows with the off-axis angle
	// and eventually dominates, so the 2D covariance - and with it the quad - inflates without bound.  Off screen that
	// would not matter, except the quad is bounded only by max_radius_px below, which is twice the viewport: a splat whose
	// centre projects well outside the frame can still reach back into it and paint a screen-covering slab of colour.
	// Measured on an interior capture: ~20k splats per frame with their centre off screen were painting pixels this way,
	// their quads oversized by 6-10x against the radius their own scale and distance allow.  The near-plane test above and
	// the radius clamp cannot catch this between them, since neither knows where on screen the splat landed.
	//
	// Culling on the exact 3-sigma bounding sphere is what fixes it: outside the frustum the approximation is worthless,
	// and a splat whose whole sphere is out there could not legitimately have coloured any pixel anyway.
	//
	// The four side planes come out of focal_len_px and viewport_dims_px, which are already here, so this needs no new
	// uniform.  The right-hand edge of the frame is screen_x = +w with w = viewport_dims_px.x / 2, i.e.
	// focal_x * vx / depth = w; with depth = -vz that is the plane focal_x * vx + w * vz = 0, whose normal has length
	// sqrt(focal_x^2 + w^2), and the splat's signed distance outside it is the expression over that length.  The left
	// plane is the same with vx negated, so taking abs(focal_x * vx) covers both at once, and likewise in y.
	//
	// 3 sigma rather than the opacity-aware sigma_cutoff computed below: this bound has to hold for the widest quad the
	// splat could ever be drawn as, independently of the alpha cutoff in force.
	float model_scale = length(model_matrix[0].xyz); // Uniform scale assumed throughout this shader - see the covariance transform below.
	float bound_radius_vs = 3.0 * max(scale.x, max(scale.y, scale.z)) * model_scale;
	vec2 half_viewport_px = viewport_dims_px * 0.5;
	vec2 plane_normal_len = sqrt(focal_len_px * focal_len_px + half_viewport_px * half_viewport_px);
	vec2 dist_outside_vs = (abs(focal_len_px * pos_vs.xy) + half_viewport_px * pos_vs.z) / plane_normal_len; // pos_vs.z is negative in front of the camera, so this is negative for a splat inside the frame.
	if((splat_ewa_fix_enabled != 0) && (max(dist_outside_vs.x, dist_outside_vs.y) > bound_radius_vs))
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	// Build the rotation matrix from the quaternion.  Columns are built explicitly, rather than with a 9-scalar
	// mat3(...) literal, to avoid GLSL's column-major constructor order silently transposing this.
	float qx = rot.x, qy = rot.y, qz = rot.z, qw = rot.w;
	vec3 r_col0 = vec3(1.0 - 2.0*(qy*qy + qz*qz),       2.0*(qx*qy + qz*qw),       2.0*(qx*qz - qy*qw));
	vec3 r_col1 = vec3(      2.0*(qx*qy - qz*qw), 1.0 - 2.0*(qx*qx + qz*qz),       2.0*(qy*qz + qx*qw));
	vec3 r_col2 = vec3(      2.0*(qx*qz + qy*qw),       2.0*(qy*qz - qx*qw), 1.0 - 2.0*(qx*qx + qy*qy));
	mat3 R = mat3(r_col0, r_col1, r_col2);

	// 3D covariance in object space: Sigma = R * diag(scale^2) * R^T.
	mat3 RS = mat3(r_col0 * (scale.x*scale.x), r_col1 * (scale.y*scale.y), r_col2 * (scale.z*scale.z));
	mat3 cov_os = RS * transpose(R);

	// Transform to camera space.  This assumes model_matrix has no non-uniform scale or shear; non-uniform scaling would
	// need the inverse transpose here instead.
	mat3 W = mat3(view_matrix * model_matrix);
	mat3 cov_vs = W * cov_os * transpose(W);

	// Project the 3D covariance to a 2D screen-space covariance via the projection's Jacobian, evaluated at this splat's
	// view-space position.  Standard z-forward perspective projection: screen = focal * (x, y) / depth.
	float vx = pos_vs.x, vy = pos_vs.y;
	vec3 j_row0 = vec3(focal_len_px.x / depth, 0.0, focal_len_px.x * vx / (depth*depth));
	vec3 j_row1 = vec3(0.0, focal_len_px.y / depth, focal_len_px.y * vy / (depth*depth));

	vec3 cov_vs_j0 = cov_vs * j_row0;
	vec3 cov_vs_j1 = cov_vs * j_row1;

	float cov2d_a = dot(j_row0, cov_vs_j0) + 0.3; // The +0.3 is a low-pass filter on the diagonal, which avoids degenerate sub-pixel splats aliasing.  Standard 3DGS technique.
	float cov2d_b = dot(j_row0, cov_vs_j1);
	float cov2d_c = dot(j_row1, cov_vs_j1) + 0.3;

	float det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
	if(det <= 0.0)
	{
		gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Degenerate covariance, which shouldn't normally happen after the dilation above.
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	// Eigen-decomposition of the symmetric 2x2 [[a, b], [b, c]], giving the ellipse's screen-space axes and radii.
	float mid = 0.5 * (cov2d_a + cov2d_c);
	float half_span = sqrt(max(mid*mid - det, 0.0));
	float lambda1 = mid + half_span;
	float lambda2 = max(mid - half_span, 0.0);

	vec2 axis1 = (cov2d_b != 0.0) ? normalize(vec2(cov2d_b, lambda1 - cov2d_a)) : ((cov2d_a >= cov2d_c) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
	vec2 axis2 = vec2(-axis1.y, axis1.x);

	// Opacity-aware sigma cutoff: alpha at k sigma out is opacity * exp(-0.5 * k^2), so solving for the k where that
	// hits splat_alpha_cutoff gives k = sqrt(2 * ln(opacity / splat_alpha_cutoff)) - the exact radius beyond which this
	// splat is invisible at the chosen threshold, rather than always drawing out to a fixed 3 sigma regardless of how
	// transparent the splat is.  Clamped to the historical 3-sigma cap (99.7%) as a ceiling, and to 0 once opacity itself
	// is at or below the cutoff (splat invisible even at its centre - degenerate zero-area quad, cheaper than branching).
	float opacity = frag_colour.a;
	float sigma_cutoff = (opacity > splat_alpha_cutoff) ? min(sqrt(2.0 * log(opacity / splat_alpha_cutoff)), 3.0) : 0.0;

	// Clamp the screen-space radius.  Splats near the camera plane can have a legitimately huge but numerically extreme
	// projected size, which without a cap turns a single nearby splat into a screen-covering quad.  Twice the viewport's
	// larger dimension is generous enough never to visibly clip a real splat while still bounding the worst case.
	float max_radius_px = 2.0 * max(viewport_dims_px.x, viewport_dims_px.y);

	// Bound the radius by what the splat's own size and distance actually allow, which the clamp above does not.
	//
	// The 2D covariance comes from an *affine* approximation of the perspective divide, taken about this splat's position.
	// It is exact on the view axis and degrades away from it: the Jacobian's focal * v / depth^2 term grows with the
	// off-axis angle, and by the edge of a wide frame it dominates, inflating the ellipse several times over.  The clamp
	// above is twice the viewport, so it does not notice; the splat is then drawn as a slab of colour across the frame.
	// Measured on an interior capture: splats of about 1 m at 2.5 m, some 60 degrees off axis, projecting to 4000+ px
	// where their own size allows 1200.
	//
	// The honest bound needs no approximation at all.  The drawn extent is a ball of radius R = sigma_cutoff * sigma_max
	// about the splat's centre; a ball at distance d subtends a half-angle asin(R / d), so it can never cover more than
	// focal * tan(asin(R / d)) pixels however it is oriented.  Near the axis that equals focal * R / d, which is what the
	// covariance gives there anyway, so this is inert on splats the approximation projects correctly - checked against a
	// real frame, where it agreed with the computed radius to within a pixel on every well-behaved splat, and cut only
	// the diverged ones.
	//
	// R >= d means the camera is inside the splat's own extent.  Its projection is then genuinely unbounded - a splat you
	// are standing in does legitimately cover the frame - so leave those to the viewport clamp above, as before.
	float bound_radius_ws = sigma_cutoff * max(scale.x, max(scale.y, scale.z)) * length(model_matrix[0].xyz); // Uniform model scale assumed, as in the covariance transform above.
	float sin_half_angle = bound_radius_ws / max(dist_to_cam, 1.0e-6);
	if((splat_ewa_fix_enabled != 0) && (sin_half_angle < 0.999))
		max_radius_px = min(max_radius_px, max(focal_len_px.x, focal_len_px.y) * sin_half_angle / sqrt(1.0 - sin_half_angle * sin_half_angle));

	// DIAGNOSTIC ONLY - splat_quad_radius_scale, see GaussianSplatRenderer::getQuadRadiusScale().  1 (default) is the real
	// size.  Applied here, before the conic is built from these radii below, so the Gaussian shrinks with the quad instead
	// of being cut off by it: the picture then has smaller splats rather than hard-edged ones, and the only thing that
	// changed is rasterised area.  Area goes as the square of this, which is what makes it a bisection of the fill.
	float radius1_raw = min(sigma_cutoff * sqrt(lambda1), max_radius_px);
	float radius2_raw = min(sigma_cutoff * sqrt(lambda2), max_radius_px);

	// area_weight is how much of splat_quad_radius_scale's reduction this particular splat gets, from 0 (untouched) to 1
	// (the full reduction) - see the uniforms' own comments above. area_ratio compares this splat's own screen-space area
	// (radius1_raw * radius2_raw, proportional to it regardless of the ellipse's eccentricity) against the reference
	// area splat_area_scale_ref_px^2, clamped to 1 so nothing beyond the reference gets more than the full reduction.
	float area_ratio = clamp((radius1_raw * radius2_raw) / (splat_area_scale_ref_px * splat_area_scale_ref_px), 0.0, 1.0);
	float area_weight = pow(area_ratio, splat_area_scale_gamma);
	float effective_quad_scale = mix(1.0, splat_quad_radius_scale, area_weight);

	float radius1 = radius1_raw * effective_quad_scale;
	float radius2 = radius2_raw * effective_quad_scale;

	vec4 clip_pos = proj_matrix * pos_vs;

	// DIAGNOSTIC ONLY - stage 5: every bit of the projection maths above has run - covariance, Jacobian,
	// eigen-decomposition, radii - and the quad is still one pixel.  It exists because the step from points to full quads
	// moves two things at once, the vertex maths and the fill; this splits them, so 5 minus 4 is the vertex ALU alone and
	// 6 minus 5 is the fill alone.  The radii are tested rather than ignored so that none of that maths is dead code.
	if(splat_ablation_stage == 5)
	{
		if((radius1 <= 0.0) && (radius2 <= 0.0))
		{
			gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
			frag_conic = vec3(0.0);
			frag_screen_offset_px = vec2(0.0);
			return;
		}
		clip_pos.xy += (position_in.xy / viewport_dims_px) * clip_pos.w; // One pixel across, as in emitSplatPoint().
		gl_Position = clip_pos;
		frag_conic = vec3(0.0);
		frag_screen_offset_px = vec2(0.0);
		return;
	}

	// The saturation gate.  Drop the whole splat if every mask texel its quad can touch is already marked finished:
	// nothing it could contribute would survive the composite.
	//
	// Conservative throughout, deliberately.  The quad's half-extent is bounded by radius1 + radius2, which is at least
	// its true extent along either screen axis, so the texels tested always cover the quad; and every level of the mask
	// holds the *minimum* of its children, so a coarse texel reads as marked only when every pixel under it is finished.
	// A splat is therefore only ever dropped when it could not have changed a single pixel.
	//
	// The level is chosen so that the quad spans at most 2x2 texels there, which is what keeps this to four fetches
	// whatever the splat's size - a fixed level would either miss the big splats or answer too coarsely for the small
	// ones.
	if(splat_saturation_mask_block > 0)
	{
		vec2 centre_px = ((clip_pos.xy / clip_pos.w) * 0.5 + 0.5) * viewport_dims_px;
		float extent_px = radius1 + radius2;

		float block = float(splat_saturation_mask_block);

		// Clamped to the mask before anything else, both because a right shift of a negative value is
		// implementation-defined in GLSL and because it loses nothing: whatever falls outside the mask is off screen,
		// and produces no fragments whether this splat is culled or not.
		ivec2 mask_max = textureSize(splat_saturation_mask_texture, /*mip level=*/0) - ivec2(1);

		if(splat_mask_centre_test != 0)
		{
			// One fetch at the finest level, under the splat's centre.  Deliberately not conservative - see the uniform.
			ivec2 c = clamp(ivec2(floor(centre_px / block)), ivec2(0), mask_max);
			if(texelFetch(splat_saturation_mask_texture, c, /*mip level=*/0).r > 0.5)
			{
				gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
				frag_conic = vec3(0.0);
				frag_screen_offset_px = vec2(0.0);
				return;
			}
		}
		else
		{

		ivec2 lo = clamp(ivec2(floor((centre_px - extent_px) / block)), ivec2(0), mask_max);
		ivec2 hi = clamp(ivec2(floor((centre_px + extent_px) / block)), ivec2(0), mask_max);

		int level = 0;
		for(int i=0; i<16; ++i) // Bounded loop rather than a log2: the count is tiny, and this needs no float round-trip.
		{
			if((level >= splat_saturation_mask_max_level) || all(lessThanEqual((hi >> level) - (lo >> level), ivec2(1))))
				break;
			level++;
		}

		ivec2 lo_l = lo >> level;
		ivec2 hi_l = hi >> level;

		if(all(lessThanEqual(hi_l - lo_l, ivec2(1)))) // False only for a quad too big for even the coarsest level, which then just draws.
		{
			float marked = min(
				min(texelFetch(splat_saturation_mask_texture, ivec2(lo_l.x, lo_l.y), level).r, texelFetch(splat_saturation_mask_texture, ivec2(hi_l.x, lo_l.y), level).r),
				min(texelFetch(splat_saturation_mask_texture, ivec2(lo_l.x, hi_l.y), level).r, texelFetch(splat_saturation_mask_texture, ivec2(hi_l.x, hi_l.y), level).r));

			if(marked > 0.5)
			{
				gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Push outside the clip volume.
				frag_conic = vec3(0.0);
				frag_screen_offset_px = vec2(0.0);
				return;
			}
		}

		} // end of the conservative branch
	}

	// DIAGNOSTIC ONLY - shrinks the quad continuously by how covered the composite already is under it, rather than the
	// gate's binary keep/drop above - see GaussianSplatRenderer::getCoverageShrinkStrength(). A splat the gate would keep
	// (nothing above returned) can still be standing somewhere partly finished, and this is what acts on that: less
	// area for a splat contributing into an already-mostly-covered pixel, none of it lost outright the way the gate's
	// all-texels-marked test would need. Reuses the gate's own level selection (same block size, same conservative
	// bound on the quad's extent) but samples a *mean* pyramid built alongside the gate's min one - see
	// gaussian_splat_saturation_mask_frag_shader.glsl - so the answer is a continuous coverage estimate rather than a
	// pass/fail. Only meaningful, and only non-zero, while the gate itself is running: splat_saturation_mask_block is 0
	// otherwise, which is what the second half of the condition below is testing.
	if((splat_coverage_shrink_strength > 0.0) && (splat_saturation_mask_block > 0))
	{
		vec2 centre_px = ((clip_pos.xy / clip_pos.w) * 0.5 + 0.5) * viewport_dims_px;
		float extent_px = radius1 + radius2;
		float block = float(splat_saturation_mask_block);
		ivec2 mask_max = textureSize(splat_coverage_mask_texture, /*mip level=*/0) - ivec2(1);

		ivec2 lo = clamp(ivec2(floor((centre_px - extent_px) / block)), ivec2(0), mask_max);
		ivec2 hi = clamp(ivec2(floor((centre_px + extent_px) / block)), ivec2(0), mask_max);

		int level = 0;
		for(int i=0; i<16; ++i)
		{
			if((level >= splat_saturation_mask_max_level) || all(lessThanEqual((hi >> level) - (lo >> level), ivec2(1))))
				break;
			level++;
		}

		ivec2 lo_l = lo >> level;
		ivec2 hi_l = hi >> level;

		// The mean of the (up to 4) coarse texels the quad spans - not conservative like the gate's minimum, on purpose:
		// this is an estimate to weight a continuous shrink by, not a test that must never be wrong in one direction.
		// Falls back to 0 (no shrink) for a quad too big for even the coarsest level, same as the gate's own test does.
		//
		// The census this reads was taken before the slice started, so every splat in a slice reads one number and it
		// steps to a new one at the boundary.  Carrying the estimate across the slice to remove that step was built and
		// measured, and changed the bands not at all: the band is the light mode 0 takes and the pixel had not finished
		// with, not the step in the number.  The extrapolation is gone again rather than left as a knob that measures
		// nothing - session050.
		float coverage_estimate = all(lessThanEqual(hi_l - lo_l, ivec2(1))) ? (0.25 * (
			texelFetch(splat_coverage_mask_texture, ivec2(lo_l.x, lo_l.y), level).r + texelFetch(splat_coverage_mask_texture, ivec2(hi_l.x, lo_l.y), level).r +
			texelFetch(splat_coverage_mask_texture, ivec2(lo_l.x, hi_l.y), level).r + texelFetch(splat_coverage_mask_texture, ivec2(hi_l.x, hi_l.y), level).r)) : 0.0;

		if(splat_coverage_shrink_mode == 0)
		{
			// Scale both radii by the coverage.  The conic below is built from the radii, so the Gaussian narrows with
			// the quad: the splat gets smaller rather than clipped, and its total contribution falls as the square of
			// the scale even though the light still getting through the pixel only falls as (1 - coverage).  That
			// mismatch is the mode's known cost, and mode 1 is what it is being compared against.
			float coverage_shrink = clamp(1.0 - coverage_estimate * splat_coverage_shrink_strength, 0.0, 1.0);
			radius1 *= coverage_shrink;
			radius2 *= coverage_shrink;
		}
		else
		{
			// Spend the value as a budget on light lost instead.  What is thrown away by stopping the quad early is the
			// splat's own alpha at that radius, times the (1 - coverage) of the light still reaching the eye there - so
			// the radius at which the loss equals the budget is the radius at which alpha reaches budget/(1 - coverage).
			// That is the same question splat_alpha_cutoff already answers, asked with a threshold raised by how covered
			// the pixel is: same closed form, no new machinery, and no circularity, since coverage is measured, not
			// derived from the opacity this then changes.
			//
			// The budget is spent *on top of* splat_alpha_cutoff rather than instead of it, so that the threshold is
			// exactly the alpha cutoff wherever the pixel is empty, whatever this is set to.  Taking the larger of the
			// two instead - the first form of this - meant any budget above the alpha cutoff raised the threshold
			// everywhere, including over pixels with no coverage at all, which is what the plain alpha cutoff already
			// does; the knob was then half itself and half that, and the extra was measured taking seven times the error
			// out of sparse regions that the coverage-aware part alone takes.  In this form the extra light given up is
			// budget * coverage: none on an empty pixel, the full budget only where the composite is finished.
			//
			// sigma_cutoff itself is scaled by the same factor as the radii, so the conic below reconstructs the
			// identical Gaussian over a shorter quad: this truncates the faint edge rather than narrowing the splat.
			// Mode 2 drops the 1/transmittance amplification the line below carries, and is what the two are being
			// compared on. That factor is honest for one splat in isolation - the light it still had to lose through the
			// pixel is (1 - coverage), so spending a fixed budget of light needs a threshold that large - but the budget
			// is then spent again, in full, by every one of the dozens of splats standing over the same pixel. What the
			// pixel actually gives up is that budget times the number of splats, not the budget. The amplification is
			// what makes the total ruinous: at coverage 0.9 the factor is 9 and at 0.99 it is 99, so a 0.02 budget asks
			// for a threshold of 0.2 and then 2.0, and every splat whose own opacity is under that vanishes outright
			// rather than being trimmed. The pixel then freezes wherever it had got to - 0.92 on the measured cushion
			// against the 0.999 it reaches without the shrink - and the missing light is a hole showing what is behind.
			//
			// Without the factor the threshold cannot exceed alpha_cutoff + budget however finished the pixel is, so a
			// splat with any real opacity keeps a real radius and the wipe cannot happen. It gives up the closed-form
			// claim about light lost per splat, which was the mode's original argument for itself; that claim was about
			// a splat on its own and the picture is made of stacks.
			float transmittance = max(1.0 - coverage_estimate, 1.0e-4);
			float budget_cutoff = (splat_coverage_shrink_mode == 2) ?
				(splat_alpha_cutoff + splat_coverage_shrink_strength * coverage_estimate) :
				(splat_alpha_cutoff + splat_coverage_shrink_strength * coverage_estimate / transmittance);
			float budget_sigma = (opacity > budget_cutoff) ? min(sqrt(2.0 * log(opacity / budget_cutoff)), 3.0) : 0.0;

			float cutoff_ratio = budget_sigma / max(sigma_cutoff, 1.0e-6);
			radius1 *= cutoff_ratio;
			radius2 *= cutoff_ratio;
			sigma_cutoff = budget_sigma;
		}
	}

	vec2 screen_offset_px = position_in.x * radius1 * axis1 + position_in.y * radius2 * axis2;

	// Note that the mask test above used clip_pos as the splat's *centre*, before this offset moves it to this vertex's
	// corner: the test is about the whole quad, and all four of its vertices have to reach the same verdict.
	clip_pos.xy += (screen_offset_px / viewport_dims_px) * 2.0 * clip_pos.w; // Offset in clip space, premultiplied by w so it survives the perspective divide unchanged.
	gl_Position = clip_pos;

	// Build the conic from the possibly-clamped radii above, rather than by inverting the raw 2D covariance.  Without
	// this, an oversized splat (e.g. a huge near-flat background splat, common where a region has no parallax to
	// constrain its scale) has its quad clamped above but its alpha falloff still computed from the true, enormous
	// variance, which barely decays by the clamped quad edge and so produces a hard straight-edged cutoff instead of a
	// soft fade.  Using the clamped radius as the effective sigma makes alpha fade to ~0 at the quad boundary, and is a
	// no-op whenever the radius wasn't clamped.
	//
	// The radius is sigma_cutoff sigmas out, so lambda = (radius / sigma_cutoff)^2 - not (radius/3)^2, which held only
	// while the radius was always a fixed 3 sigma.  Dividing by 9 regardless narrows the Gaussian itself by
	// sigma_cutoff/3 whenever the opacity-aware cutoff above bites, so splat_alpha_cutoff quietly shrank low-opacity
	// splats rather than only trimming the tail the fragment shader discards anyway.
	float inv_cutoff_sq = 1.0 / max(sigma_cutoff * sigma_cutoff, 1.0e-6); // Guarded for sigma_cutoff = 0, i.e. a splat at or below the cutoff, whose quad is degenerate and rasterises nothing.
	float eff_lambda1 = (radius1 * radius1) * inv_cutoff_sq;
	float eff_lambda2 = (radius2 * radius2) * inv_cutoff_sq;
	float inv_l1 = 1.0 / eff_lambda1;
	float inv_l2 = 1.0 / eff_lambda2;
	frag_conic = vec3(
		axis1.x*axis1.x*inv_l1 + axis2.x*axis2.x*inv_l2,
		axis1.x*axis1.y*inv_l1 + axis2.x*axis2.y*inv_l2,
		axis1.y*axis1.y*inv_l1 + axis2.y*axis2.y*inv_l2); // (A, B, C) of the conic Ax^2 + 2Bxy + Cy^2.
	frag_screen_offset_px = screen_offset_px;
}
