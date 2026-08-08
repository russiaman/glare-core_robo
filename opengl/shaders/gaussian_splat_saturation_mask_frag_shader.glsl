
// gaussian_splat_saturation_mask_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Marks the pixels that the front-to-back splat composite has already finished with, so that the splats still to be
// drawn behind them can be rejected before they are shaded.  Run between draw slices - see
// OpenGLEngine::markSaturatedSplatPixels() for the pass setup and OpenGLEngine::drawSplatClouds() for why slices exist.
//
// The mark is a depth write, not a colour one: colour writes are masked off, and a finished pixel is stamped with the
// near-plane depth value.  Everything drawn behind it then fails the depth comparison the splats already perform, so the
// rejection costs no extra per-fragment work and needs no buffer beyond the depth copy the pass already owns.
//
// Stencil would express this more directly, but a stencil buffer has to be attached alongside the depth one, and the
// combination of a depth-only format with a separate stencil buffer is refused outright by many drivers.  Making the
// depth buffer packed instead works, but measurably slows every depth test done against it - which, at hundreds of
// millions of splat fragments a frame, cost more than the gate saved.

precision highp float;
precision highp sampler2D;

// A copy of the splat accumulation buffer, since a buffer cannot be sampled while it is attached to the framebuffer
// being drawn to.  Only the alpha channel is read here.
uniform sampler2D albedo_texture;

// Accumulated coverage at or above which a pixel counts as finished.  Coverage is 1 - transmittance, so the meaningful
// setting is just below 1: at 1 - 1/255 the light still getting through is under one 8-bit level, which is where the
// reference 3DGS rasteriser stops its own per-pixel loop.
uniform float splat_saturation_threshold;

// The depth value that marks a pixel as finished - the near plane, whose value depends on the engine's depth direction,
// so it is passed in rather than assumed here.
uniform float splat_saturated_depth;

out vec4 colour_out;


void main()
{
	// Same indexing as the resolve pass: the accumulation buffer matches the viewport and this quad covers all of it, so
	// the fragment's own coordinates address it directly.
	float coverage = texelFetch(albedo_texture, ivec2(gl_FragCoord.xy), /*mip level=*/0).a;

	if(coverage < splat_saturation_threshold)
		discard; // Not finished - leave this pixel's depth alone, so later slices still draw into it.

	gl_FragDepth = splat_saturated_depth; // The mark.
	colour_out = vec4(0.0); // Masked off by glColorMask(); written only because a fragment shader must have an output.
}
