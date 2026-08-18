
// gaussian_splat_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Evaluates the 2D Gaussian for one splat and blends it into the splat accumulation buffer, which
// gaussian_splat_resolve_frag_shader.glsl then composites onto the main colour buffer.
// See opengl/GaussianSplatRenderer.h for the overall design.

precision highp float; // Override the engine's default "precision mediump float;" for Emscripten - the exponent below is precision-sensitive for large splats.

in vec2 frag_screen_offset_px;
in vec3 frag_conic;
in vec4 frag_colour;
in float frag_view_depth;

// Debug view (GaussianSplatSettingsWidget, Qt only). 0 (default) = normal splat colour. Non-zero = every surviving
// fragment writes a flat value into the red channel instead, additively blended (drawSplatClouds() switches to
// GL_ONE, GL_ONE for these modes) into the same accumulation buffer as normal, so accum.r ends up holding a per-pixel
// sum that gaussian_splat_resolve_frag_shader.glsl maps to a colour ramp:
//   1 = write 1.0 per fragment, so the sum is the raw layer count (depth complexity).
//   2 = write the fragment's own alpha, so the sum is the total accumulated alpha over those layers.
// Mode 2 exists to size up a front-to-back transmittance cutoff - the "early-Z for splats" a hardware blend pipeline
// can't do per-fragment, since it never knows the accumulated alpha at shading time. Read together with mode 1:
// mean_alpha = sum_alpha / layers, and front-to-back accumulation crosses T < 1/255 at about
// ln(1/255) / ln(1 - mean_alpha) layers, so the fraction of fragment work such a cutoff could skip works out to
// roughly sum_alpha / 5.54, independent of the layer count itself. A region with hundreds of layers but a small
// sum_alpha is made of near-threshold splats that a cutoff would never reach - that case wants pruning, not early-out.
uniform int splat_show_overdraw;

// Note that the saturation gate is not tested here.  It was, and it saved nothing: a texture fetch per fragment, over
// the hundreds of millions of fragments this pass shades, costs about what the blending it skips is worth.  The test
// lives in the vertex shader instead, where it is paid once per splat - see gaussian_splat_vert_shader.glsl.

// Note that there's deliberately no order-independent-transparency variant here.  Splat clouds are drawn by
// drawSplatClouds(), which always renders to a single colour buffer with ordinary front-to-back alpha blending, never
// through the OIT path - see the material setup in GaussianSplatRenderer::addObject() for why.
layout(location = 0) out vec4 colour_out;

// DIAGNOSTIC ONLY - the per-pixel layer cap, see GaussianSplatRenderer::getLayerCap().  Zero disables both of the
// things below, which is every frame that is not measuring.
//
// This is the per-fragment mask test the note above says was removed for costing what it saved.  It is back only as a
// diagnostic: a cap on how deep the composite goes is a statement about pixels, and the conservative per-splat test
// cannot express it - a splat is either drawn everywhere or nowhere.  The cost is accepted here because what is being
// looked at is the picture, not the frame time.
uniform int splat_frag_mask_block; // Accumulation-buffer pixels per mask texel, or 0 to skip the test entirely.

// DIAGNOSTIC ONLY - see the ladder in main(), and GaussianSplatRenderer::getAblationStage().  0 = off, i.e. the full
// shader.  Shared with the vertex shader, which owns stages 2-4.
uniform int splat_ablation_stage;
uniform sampler2D splat_saturation_mask_texture; // The same mask the vertex shader tests; a sampler uniform is program-wide, so this is the same one, bound once by the draw path.
layout(location = 1) out vec4 layer_count_out; // A flat 1 per surviving fragment, blended additively into the layer counter - see OpenGLScene::splat_layer_count_renderbuffer.

// GaussianSplatRenderer::SplatDoFDepthMode_Weighted: (frag_view_depth * alpha, 0, 0, alpha), blended with the same
// front-to-back "under" op as colour_out, into OpenGLScene::splat_dof_depth_renderbuffer. The resolve pass divides
// the accumulated depth*alpha*T by the accumulated coverage (colour_out's own alpha channel - see the resolve
// shader) to get the splat stack's mean depth, and writes gl_FragDepth from it, for Depth of Field (and fog) to
// read afterwards. Written unconditionally, like layer_count_out - a declared output with no attachment bound to
// it is simply not captured, so there is no need to gate this on the mode being active.
layout(location = 2) out vec4 dof_depth_out;


void main()
{
	// DIAGNOSTIC ONLY - the ablation ladder, see GaussianSplatRenderer::getAblationStage().  Each stage adds one thing to
	// the one below it, so the step between two readings attributes cost to that one thing.  Handled by returning early
	// rather than by branching inside the code below, which leaves stage 0 - the only value that is not a measurement -
	// compiling to exactly what it did before this existed.
	//
	// Blending is switched on by the draw path from stage 7 up, so stages 2-6 write opaque and the step into 7 is the
	// blend bandwidth on its own.  Stages above 8 are not values of this uniform: stage 9 is the ladder switched off, and
	// stage 10 is the ladder off with the alpha saturation cap on, both of which the existing knobs already express.
	if(splat_ablation_stage != 0)
	{
		layer_count_out = vec4(1.0, 0.0, 0.0, 0.0);
		dof_depth_out = vec4(0.0); // Zero weight - the ablation ladder isn't meant to interact with DoF depth.

		vec3 flat_col = (splat_ablation_stage <= 3) ? vec3(1.0) : clamp(frag_colour.rgb, 0.0, 1.0); // 2 and 3 are white; the splat's own colour is what 4 adds.

		if(splat_ablation_stage <= 6)
			colour_out = vec4(flat_col, 1.0); // 2-6: opaque, no alpha, and the draw path has blending off - so this measures pure rasterised area.  Stages 4 and 5 differ only in the vertex shader.
		else if(splat_ablation_stage == 7)
			colour_out = vec4(flat_col * frag_colour.a, frag_colour.a); // Alpha and blending, but a flat alpha across the quad - the Gaussian is what 8 adds.
		else
		{
			// 8: the Gaussian, but no discard.  min() replaces the discard that would otherwise have thrown out the
			// fragments with a positive exponent, which are outside the ellipse and would blow alpha up past 1.
			float power7 = -0.5 * (frag_conic.x * frag_screen_offset_px.x * frag_screen_offset_px.x
			                      + 2.0 * frag_conic.y * frag_screen_offset_px.x * frag_screen_offset_px.y
			                      + frag_conic.z * frag_screen_offset_px.y * frag_screen_offset_px.y);
			float alpha7 = frag_colour.a * exp(min(power7, 0.0));
			colour_out = vec4(clamp(frag_colour.rgb, 0.0, 1.0) * alpha7, alpha7);
		}
		return;
	}

	// Evaluate the 2D Gaussian at this pixel: exponent = -0.5 * offset^T * conic * offset, where conic is the inverse 2D
	// covariance (see gaussian_splat_vert_shader.glsl).
	float power = -0.5 * (frag_conic.x * frag_screen_offset_px.x * frag_screen_offset_px.x
	                     + 2.0 * frag_conic.y * frag_screen_offset_px.x * frag_screen_offset_px.y
	                     + frag_conic.z * frag_screen_offset_px.y * frag_screen_offset_px.y);
	if(power > 0.0)
		discard;

	float alpha = frag_colour.a * exp(power);
	if(alpha < (1.0 / 255.0))
		discard;

	// The layer cap: this pixel has already had its layers, so nothing further is composited into it.  Tested after the
	// two discards above so that the count the mask was built from and the fragments this rejects are the same
	// population - a fragment that contributes nothing was never counted, and must not be capped either.
	if(splat_frag_mask_block > 0)
	{
		if(texelFetch(splat_saturation_mask_texture, ivec2(gl_FragCoord.xy) / splat_frag_mask_block, 0).r > 0.0)
			discard;
	}

	layer_count_out = vec4(1.0, 0.0, 0.0, 0.0); // Alpha zero: the pass blends with (GL_ONE_MINUS_DST_ALPHA, GL_ONE), so a destination alpha that stays at zero makes this plain addition.

	// Clamp the colour to a displayable range.  Evaluating the spherical harmonics can land outside [0, 1] - the DC term
	// alone reaches slightly negative values in real files - and out-of-range values here are not just wrong but
	// dangerous, since the blend below would carry them into the accumulation buffer, where the resolve pass's inverse
	// tone map is only conditioned for [0, 1].  This is also the right place for the clamp if view-dependent (shN) terms
	// are ever added, since those are summed per-fragment.
	vec3 base_col = clamp(frag_colour.rgb, 0.0, 1.0);

	// Write the splat's colour in the display-referred (non-linear sRGB) space the file stores it in, and let the blend
	// happen in that space.  That is not a rendering choice we're free to make: standard 3DGS training loads the
	// photographs as 8-bit sRGB, divides by 255 and optimises against them directly, with no sRGB->linear conversion
	// anywhere, so the fitted colours and opacities are exactly the values whose alpha composite *in gamma-encoded
	// sRGB* reproduces the captured images.  The compositing space is part of the model.  Blending the same splats in
	// linear space gives a different image wherever a pixel mixes splats of differing brightness - depth edges and
	// semi-transparent regions, which is precisely where a capture is most fragile.  It is physically wrong, and it is
	// a known wart of 3DGS, but reproducing the reference means reproducing the wart.
	//
	// So the engine's display transform can't be inverted here, per splat: the blend has to see the authored sRGB
	// values, and inverting first would blend in a space that is neither sRGB nor anything the model saw.  Since that
	// mapping is convex, the resulting average is biased even where it doesn't blow up.  drawSplatClouds() therefore
	// blends this pass into a buffer of its own and inverts the transform once, after the blend - see
	// gaussian_splat_resolve_frag_shader.glsl.
	//
	// Premultiplied by alpha: drawSplatClouds() blends with (GL_ONE_MINUS_DST_ALPHA, GL_ONE), the "under" operator, so
	// the accumulation buffer ends up holding (sum of c_i * a_i * T_i, coverage), which is what the resolve pass wants.
	if(splat_show_overdraw != 0)
	{
		colour_out = vec4((splat_show_overdraw == 2) ? alpha : 1.0, 0.0, 0.0, 1.0); // Alpha sum, or one layer - see the uniform comment above. 'alpha' is post-discard, so mode 2 sums exactly what the normal path would have blended.
		dof_depth_out = vec4(0.0); // Debug view - leave DoF depth alone, as with the ablation ladder above.
	}
	else
	{
		colour_out = vec4(base_col * alpha, alpha);
		dof_depth_out = vec4(frag_view_depth * alpha, 0.0, 0.0, alpha);
	}
}
