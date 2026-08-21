
// gaussian_splat_resolve_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Resolves the splat accumulation buffer that gaussian_splat_frag_shader.glsl blends into, and composites the result
// onto the main colour buffer.  See OpenGLEngine::drawSplatClouds() for how the pass is set up.
//
// The splats were blended in display-referred (non-linear sRGB) space, because that is the space 3DGS fits them in -
// see the comment in gaussian_splat_frag_shader.glsl.  That blend is finished by the time this runs, so this is the one
// place where the engine's display transform can be inverted without disturbing it.

precision highp float; // Override the engine's default "precision mediump float;" for Emscripten - dividing out a small alpha and inverting the tone map both want the range.
precision highp sampler2D; // Likewise: sampler2D defaults to lowp in GLSL ES, which would quantise the accumulated colour and coverage read below.

uniform sampler2D albedo_texture; // The splat accumulation buffer: (sum of c_i * a_i * T_i, coverage), non-linear sRGB, premultiplied.

uniform int splat_show_overdraw; // Overdraw debug view (GaussianSplatSettingsWidget, Qt only) - see gaussian_splat_frag_shader.glsl. Set via GaussianSplatRenderer::setResolveOverdrawUniforms(), which OpenGLEngine::resolveSplatAccumBuffer() calls, since this program is drawn manually (one full-viewport quad, no material) rather than through the generic per-object user-uniform path the main splat program uses.
uniform float splat_overdraw_range_min; // Accumulated value (layers, or summed alpha in mode 2) that maps to solid blue.
uniform float splat_overdraw_range_max; // Likewise for solid red. Between the two, the ramp runs blue -> green -> red.

// DIAGNOSTIC ONLY - the coverage map view (GaussianSplatSettingsWidget, Qt only), see
// GaussianSplatRenderer::getShowCoverageMapLevel(). Shows the pyramid the coverage shrink reads, at the level asked for,
// instead of the resolved splats. -1 = off, and then nothing below is sampled.
uniform int splat_show_coverage_map_level;
uniform int splat_coverage_map_block;   // Screen pixels across one level-0 texel of it.
uniform sampler2D splat_coverage_map_texture;

// GaussianSplatRenderer::SplatDoFDepthMode_Weighted - see GaussianSplatRenderer::setResolveDoFDepthUniforms().
// splat_dof_depth_texture holds (sum of view_depth_i * alpha_i * T_i, coverage) accumulated the same way as the
// main albedo_texture above; dividing its .r by the coverage this pass already recovers from albedo_texture.a
// gives the splat stack's mean view-space depth at this pixel (see main() below for why the two accumulations'
// alpha channels are guaranteed identical, rather than needing dof_depth_texture's own .a).
uniform int splat_write_weighted_depth;
uniform float splat_dof_near_clip_dist;
uniform sampler2D splat_dof_depth_texture;

// SESSION067 - the accumulation buffer can be smaller than the frame, see GaussianSplatRenderer::getAccumBufferScale().
// splat_accum_upsample: 0 = the buffer is the same size as the frame and is indexed one-to-one, exactly as this pass
// always did; 1 = nearest; 2 = bilinear.  Both sizes are given because at fractional scales the buffer's size is a
// *rounded* scaling of the viewport, so neither can be recovered from the other and the scale factor alone would put
// the last row and column slightly wrong.
uniform int splat_accum_upsample;
uniform vec2 splat_accum_dims_px;    // Size the accumulation buffer was actually allocated at.
uniform vec2 splat_resolve_dims_px;  // Size of the frame this pass is drawing into, i.e. the viewport.

// SESSION068 - post-processing enhancers on the upscaled splat buffer. See GaussianSplatRenderer::setResolveEnhanceUniforms().
// Both are independent on/off switches so the owner can A/B either alone or both together.  All four uniforms are
// forced inert (enabled=0) by the setter whenever the accumulation buffer is 1:1 with the frame - the whole point of
// this pass is to undo losses that only exist while the buffer is smaller.
uniform int splat_deconv_enabled;    // 0 = off, 1 = on.  Matched deconvolution of the +0.3 low-pass + bilinear tent, see applyMatchedDeconv().
uniform float splat_deconv_gain;     // A/B multiplier on the analytically-derived strength.  1.0 = as derived; higher/lower is a knob to verify the derivation by eye.
uniform int splat_rcas_enabled;      // 0 = off, 1 = on.  AMD FidelityFX RCAS (MIT), see applyRCAS().
uniform float splat_rcas_sharpness;  // 0..1, RCAS sharpness (fraction of its safe maximum lobe).

out vec4 colour_out;


// Reads one of the accumulation buffers at this fragment, honouring the mode above.
//
// Bilinear rather than nearest is what makes a downscaled buffer usable: a splat is a Gaussian, so its screen footprint
// is band-limited by construction (that is what the +0.3 low-pass in the vertex shader guarantees), and a smooth
// reconstruction of a band-limited signal is close to lossless - far more forgiving than the same downscale would be on
// ordinary geometry with real edges in it.  Both channels of interest ride along: the colour is premultiplied by
// coverage and the coverage is in alpha, so interpolating the vec4 interpolates a premultiplied pair, which is the form
// that composites correctly.  Interpolating a *straight* colour by the same weights would not.
//
// Nearest is not a fallback but a diagnostic: seeing the blocks is how one confirms the buffer really is smaller, and
// it is the honest baseline the bilinear cost is compared against.  It is selected by the sampler's own filter state,
// set by OpenGLEngine::resolveSplatAccumBuffer(), which is why both modes take the same fetch here.
vec4 sampleSplatAccum(sampler2D tex)
{
	if(splat_accum_upsample == 0)
		return texelFetch(tex, ivec2(gl_FragCoord.xy), /*mip level=*/0); // Same size as the frame: bit-for-bit the pre-SESSION067 path.

	// The buffer covers the same screen area at a lower resolution, so the normalised position within the frame is
	// also the normalised position within the buffer, whatever its size.  gl_FragCoord is at pixel centres, which is
	// what makes this land on texel centres rather than half a texel off.
	return texture(tex, gl_FragCoord.xy / splat_resolve_dims_px);
}


// This fragment's position in accumulation-buffer pixels.  The saturation and coverage masks are sized and blocked in
// those, not in frame pixels, so the diagnostic views that index them have to convert - at scale 1 this is the identity.
vec2 accumCoordPx()
{
	return (splat_accum_upsample == 0) ? gl_FragCoord.xy : (gl_FragCoord.xy * splat_accum_dims_px / splat_resolve_dims_px);
}


// SESSION068 - straight (un-premultiplied) colour of a neighbour texel of the accumulation buffer, offset by
// offset_texels texels of THAT buffer.  Falls back to centre_col where the neighbour is empty, so that the edge of a
// cloud is not pulled toward black: an empty texel is "no data here", not "black here".
//
// Stepping in ACCUM-BUFFER texels (not frame pixels) is essential: real information exists only at that spacing, and
// any finer step would sharpen the bilinear interpolant itself rather than the underlying signal.
vec3 sampleSplatColourOffset(vec2 offset_texels, vec3 centre_col)
{
	vec2 uv = gl_FragCoord.xy / splat_resolve_dims_px + offset_texels / splat_accum_dims_px;
	vec4 a = texture(albedo_texture, uv);
	return (a.a > 0.0) ? clamp(a.rgb / a.a, 0.0, 1.0) : centre_col;
}


// SESSION068 - matched deconvolution: approximate inverse of the KNOWN blur applied to this buffer, not a hand-tuned
// sharpen.  Two sources of blur, both ours: the +0.3 low-pass on the 2D covariance diagonal
// (gaussian_splat_vert_shader.glsl, variance 0.3 in accum-buffer texels squared) and the triangular bilinear-tent kernel
// used to upsample (variance 1/6).  Gaussian variances add, and a blur of variance v to first order is
// col + (v/2)*laplacian - so its inverse subtracts the same term.
//
// Only the EXCESS over scale 1 is undone: the +0.3 at scale 1 is legitimate anti-aliasing that prevents sub-pixel splats
// from flickering as the camera moves; taking it out would bring that flicker back.  After the laplacian's step
// (1 accum-texel = 1/s frame pixels) is folded in, the scale factor almost cancels and the strength collapses to
// a = (0.3 + 1/6 - 0.3*s*s) / 2.  Full derivation: session068 snapshot §4.1.
//
// The neighbourhood clamp at the end is not cosmetic: an unclamped laplacian rings on sharp transitions, and this
// clamp makes the filter incapable of producing a new local extremum by construction.
vec3 applyMatchedDeconv(vec3 e, vec3 b, vec3 d, vec3 f, vec3 h, float scale, float gain)
{
	float a = max((0.3 + (1.0/6.0) - 0.3 * scale * scale) * 0.5, 0.0) * gain;
	vec3 laplacian = b + d + f + h - 4.0 * e;
	vec3 sharp = e - a * laplacian;
	return clamp(sharp, min(min(b, d), min(f, h)), max(max(b, d), max(f, h)));
}


// SESSION068 - AMD FidelityFX RCAS (MIT).  Locally-contrast-adaptive sharpen: the result is bounded by the neighbourhood
// min/max by construction, so this filter cannot produce ringing or halos - unlike an ordinary unsharp mask.  Applied
// in display-referred [0, 1] (see the call site in main(), which is where main() puts col at the moment RCAS wants).
//
// sharpness: 0 = mild, 1 = the maximum the algorithm considers safe.
vec3 applyRCAS(vec3 e, vec3 b, vec3 d, vec3 f, vec3 h, float sharpness)
{
	vec3 mn = min(min(b, d), min(f, h));
	vec3 mx = max(max(b, d), max(f, h));

	// How far the centre lobe can be pushed without leaving the neighbourhood - taken per channel, then the tightest.
	vec3 hit_min = mn / (4.0 * mx + 1.0e-6);
	vec3 hit_max = (1.0 - mx) / (4.0 * mn - 4.0 - 1.0e-6);
	vec3 lobe_rgb = max(-hit_min, hit_max);
	float lobe = max(lobe_rgb.r, max(lobe_rgb.g, lobe_rgb.b));

	// -0.1875 is the point beyond which RCAS begins to ring; sharpness scales the safe cap down from there.
	lobe = clamp(lobe, -0.1875, 0.0) * sharpness;

	return (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
}


#if DO_POST_PROCESSING


// Inverse of ACESFilm() in frag_utils.glsl, which is f(x) = (x*(a*x + b)) / (x*(c*x + d) + e).
// Solving f(x) = y for x rearranges to the quadratic x^2*(y*c - a) + x*(y*d - b) + y*e = 0.
vec3 inverseACESFilm(vec3 y)
{
	const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;

	vec3 quad_a = y * c - a; // For y in [0, 1] this lies in [-2.51, -0.08], so it's never zero and the quadratic stays well conditioned.
	vec3 quad_b = y * d - b;
	vec3 quad_c = y * e;

	// Of the two roots, this is the one that is >= 0 over the range of interest.
	return (-quad_b - sqrt(max(quad_b*quad_b - 4.0*quad_a*quad_c, vec3(0.0)))) / (2.0 * quad_a);
}

#endif // DO_POST_PROCESSING


void main()
{
	// This quad covers the whole viewport, so the fragment's own coordinates locate it in the accumulation buffer -
	// directly when that buffer is the same size (the default), through sampleSplatAccum()'s filter when it is smaller.
	// Any MSAA samples were already resolved into it by the blit, which averages the premultiplied colour and the
	// coverage together - the right order, since dividing per-sample and then averaging would weight sparsely covered
	// samples equally with fully covered ones.
	vec4 accum = sampleSplatAccum(albedo_texture);

	// DIAGNOSTIC ONLY - see splat_show_coverage_map_level above.  Written before the overdraw view and the resolve
	// proper, since it replaces both: this is a view of what the shrink reads, not of what the splats produced.
	if(splat_show_coverage_map_level >= 0)
	{
		// One texel of the level covers block << level screen pixels, because the pyramid halves each step and its base
		// texel already stands for block of them.  Clamped for the same reason the vertex shader clamps: the viewport is
		// not necessarily a whole number of texels across, and the last row and column are partly outside.
		int level = splat_show_coverage_map_level;
		int texels_per_px = splat_coverage_map_block << level;
		ivec2 level_max = max(textureSize(splat_coverage_map_texture, level) - ivec2(1), ivec2(0));
		ivec2 texel = min(ivec2(accumCoordPx()) / ivec2(texels_per_px), level_max); // The block size is in accumulation-buffer pixels, so the fragment has to be expressed in them too - see accumCoordPx().

		// Grey, straight through: 0 = nothing covered, 1 = finished.  No ramp, because the question this view answers is
		// "how far off 1 is this number", and a colour ramp makes near-equal values look further apart than they are.
		float coverage = texelFetch(splat_coverage_map_texture, texel, level).r;
		colour_out = vec4(vec3(coverage), 1.0); // Opaque, so it replaces the background rather than blending over it - as the overdraw view below does.
		return;
	}

	if(splat_show_overdraw != 0)
	{
		// accum.r holds whichever per-pixel sum the mode asked for - layer count (mode 1) or summed alpha (mode 2),
		// blended additively per surviving fragment by drawSplatClouds() - see gaussian_splat_frag_shader.glsl. The
		// ramp is the same either way, only the units the range is read in differ. Map it through [range_min,
		// range_max] to a blue -> green -> red ramp and write it out fully opaque, so it replaces the background pixel
		// outright rather than blending over it (this pass composites with (GL_ONE, GL_ONE_MINUS_SRC_ALPHA), so
		// alpha = 1 does exactly that).
		float accum_sum = accum.r;
		if(accum_sum <= 0.0)
			discard; // No splat reached this pixel - leave the background alone, same as the coverage check below.

		float range = max(splat_overdraw_range_max - splat_overdraw_range_min, 1.0e-4);
		float t = clamp((accum_sum - splat_overdraw_range_min) / range, 0.0, 1.0);
		vec3 ramp_col = (t < 0.5)
			? mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t * 2.0)
			: mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
		colour_out = vec4(ramp_col, 1.0);
		return;
	}

	float coverage = accum.a; // = 1 - product of (1 - a_i), i.e. how much of this pixel the splat stack covers.
	if(coverage <= 0.0)
		discard; // No splat reached this pixel, so leave the background alone.  Also avoids the 0/0 below.

	// Divide out the alpha to recover the straight (non-premultiplied) colour of the splat stack: the colour the
	// capture says this pixel should be where the splats cover it.  The clamp only guards against half-float rounding
	// pushing a convex combination of values in [0, 1] just outside it, which would take inverseACESFilm() outside the
	// range it is conditioned for.
	vec3 col = clamp(accum.rgb / coverage, 0.0, 1.0);

	// SESSION068 - post-processing enhancers.  Placed HERE, between the un-premultiplying divide above and the display
	// transform inversion below, for three reasons that all matter:
	//   1. col is display-referred sRGB in [0, 1] at this point - the space both filters below are derived for.
	//      After inverseACESFilm() values leave [0, 1] into linear HDR, where the same coefficients ring differently.
	//   2. col is straight (un-premultiplied).  Sharpening the premultiplied accum.rgb would chase the alpha gradient,
	//      not the colour gradient.
	//   3. The diagnostic branches (coverage map, overdraw) returned earlier - they show raw data and are left alone.
	// Both enable flags are forced to 0 by setResolveEnhanceUniforms() whenever the accum buffer is 1:1 with the frame,
	// so the scale-1 path is bit-for-bit unchanged.  See session068 snapshot §2.
	if(splat_deconv_enabled != 0 || splat_rcas_enabled != 0)
	{
		// One shared set of four neighbour samples for both filters - they read the same cross of texels.
		vec3 b = sampleSplatColourOffset(vec2( 0.0, -1.0), col);
		vec3 d = sampleSplatColourOffset(vec2(-1.0,  0.0), col);
		vec3 f = sampleSplatColourOffset(vec2( 1.0,  0.0), col);
		vec3 h = sampleSplatColourOffset(vec2( 0.0,  1.0), col);

		// Deconvolution first, then RCAS: the deconvolution is the physical inverse (restores contrast the blur took
		// away), RCAS is perceptual on top (adds apparent sharpness).  Reversing them applies RCAS to a still-blurred
		// input and buys less at the same cost.  See session068 snapshot §4.2.
		if(splat_deconv_enabled != 0)
			col = applyMatchedDeconv(col, b, d, f, h, splat_accum_dims_px.x / splat_resolve_dims_px.x, splat_deconv_gain);
		if(splat_rcas_enabled != 0)
			col = clamp(applyRCAS(col, b, d, f, h, splat_rcas_sharpness), 0.0, 1.0);
	}

#if DO_POST_PROCESSING
	// col is what the splats should look like on screen, but the engine will apply toneMapToNonLinear() - that is,
	// toNonLinear(ACESFilm(col * 2.0)) - to this buffer later on.  So hand it the linear value that that transform maps
	// *to* col, by inverting the whole chain.  Note that a plain gamma conversion can't do this job: the display
	// transform isn't a gamma curve, and raising each channel to a power widens the ratios between channels, which
	// shows up as oversaturation.
	col = inverseACESFilm(fastApproxNonLinearSRGBToLinearSRGB(col)) * (1.0 / PRE_TONE_MAP_SCALE_FACTOR);
#endif

	// Premultiplied again for the composite: drawSplatClouds() blends this with (GL_ONE, GL_ONE_MINUS_SRC_ALPHA), so
	// the splat stack goes over the background with its accumulated coverage.
	colour_out = vec4(col * coverage, coverage);

	// SplatDoFDepthMode_Weighted: write this pixel's splat-stack mean depth into the real depth buffer, for Depth
	// of Field (and fog) to read afterwards. Safe without a depth test: any splat that contributed to `coverage`
	// here was already depth-tested against opaque geometry while it was drawn (see drawSplatClouds()), so the
	// mean depth recovered below can only be at or nearer than whatever was already there - never further.
	//
	// The two accumulation buffers' alpha channels are bit-identical: both blend with the exact same front-to-back
	// "under" operator (GL_ONE_MINUS_DST_ALPHA, GL_ONE), which applies the same blend factors to every channel
	// including alpha, over the exact same sequence of draw calls - so dof_depth_texture's own alpha recurrence is
	// identical to albedo_texture's, and `coverage` (already known > 0, from the discard above) serves both.
	if(splat_write_weighted_depth != 0)
	{
		float depth_accum = sampleSplatAccum(splat_dof_depth_texture).r; // Same buffer shape and the same filter as the colour above - see the identical-alpha argument in the comment.
		float mean_view_depth = depth_accum / coverage;
		gl_FragDepth = getDeviceDepthFromLinearDepth(splat_dof_near_clip_dist, mean_view_depth);
	}
}
