// gaussian_splat_cap_fill_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// DIAGNOSTIC ONLY - fills the pixels the per-pixel layer cap cut short out to full coverage, keeping the colour their
// kept layers produced.  See OpenGLEngine::fillCappedSplatPixels() and GaussianSplatRenderer::getLayerCapOpaque().
//
// A capped pixel stops accumulating before its stack is finished, so it is left partly covered and the resolve pass
// would composite the background through the rest of it: a wall goes translucent exactly where the cap saved the most
// work.  Rewriting it as fully covered turns that into a colour error instead - the pixel shows the average of the
// layers that were kept, which are the nearest ones.
//
// Which pixels those are is the saturation mask, holding the frame's last census: marked where the layer count reached
// the cap.  Everything else is left exactly as the splat pass wrote it.

precision highp float;
precision highp sampler2D;

uniform sampler2D albedo_texture; // A copy of the accumulation buffer: (premultiplied colour, coverage).
uniform sampler2D splat_saturation_mask_texture;
uniform int splat_mask_block_size; // Accumulation-buffer pixels per mask texel.

out vec4 colour_out;


void main()
{
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	if(texelFetch(splat_saturation_mask_texture, pixel / splat_mask_block_size, 0).r <= 0.0)
		discard; // Not capped: leave the pixel as the splat pass left it.

	vec4 accum = texelFetch(albedo_texture, pixel, 0);

	// Un-premultiply by the coverage reached and re-premultiply by 1, i.e. keep the colour and raise the coverage.  The
	// guard costs nothing, and 0/0 would put a NaN into the buffer for the resolve to spread.
	colour_out = (accum.a > 0.0) ? vec4(accum.rgb / accum.a, 1.0) : vec4(0.0);
}
