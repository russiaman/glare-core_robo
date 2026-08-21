
// gaussian_splat_depth_downsample_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// SESSION067 - reduces the scene's depth buffer onto the splat accumulation buffer's depth attachment, which is smaller
// than the frame whenever GaussianSplatRenderer::getAccumBufferScale() is below 1.  Run by OpenGLEngine::drawSplatClouds()
// once per frame, before any splat is drawn.
//
// Why a pass and not a blit.  The full-resolution path fills that attachment with glBlitFramebuffer, but a depth blit
// cannot scale: GL_DEPTH_BUFFER_BIT requires the source and destination rectangles to have the same size, and the
// stricter WebGL 2 validation rejects it outright.  Since the web client is the target that matters most here, this is
// done with a quad that reads a depth texture and writes gl_FragDepth, which behaves identically everywhere.
//
// Farthest, not nearest or an average.  This depth is only ever used as an occluder for the splats behind it, so the
// error has a safe direction: taking the farthest sample of the footprint can only let a splat survive that a
// full-resolution test would have hidden, never cut one that should have been drawn.  Nearest would do the opposite and
// eat real geometry along every silhouette; an average would produce a depth that exists nowhere in the scene.
//
// Note the sense of "farthest" depends on the projection - see USE_REVERSE_Z below.

precision highp float;
precision highp sampler2D;

uniform sampler2D albedo_texture; // The scene's depth, at full resolution, as a sampleable copy - see OpenGLScene::splat_depth_src_texture.

uniform vec2 splat_depth_src_dims_px; // Size of that copy.
uniform vec2 splat_depth_dst_dims_px; // Size of this pass's target, i.e. of the accumulation buffer.


void main()
{
	// The source footprint of this destination texel.  Derived from the two sizes rather than from a scale factor
	// because the destination is a *rounded* scaling of the source, so at fractional scales the ratio is not exactly
	// the number the user typed, and the last row and column would otherwise be sampled short.
	vec2 ratio = splat_depth_src_dims_px / splat_depth_dst_dims_px;

	vec2 footprint_begin = floor(gl_FragCoord.xy - 0.5) * ratio;
	vec2 footprint_end   = footprint_begin + ratio;

	ivec2 src_max   = ivec2(splat_depth_src_dims_px) - ivec2(1);
	ivec2 src_begin = clamp(ivec2(floor(footprint_begin)), ivec2(0), src_max);

	// Ceil, so that a footprint straddling a source texel boundary still covers it: leaving one out is exactly the case
	// where a thin occluder is lost.  At least one texel is always read even when the scale is 1.
	ivec2 src_end = clamp(ivec2(ceil(footprint_end)) - ivec2(1), src_begin, src_max);

	// Bounded so that a scale below the allowed minimum cannot turn this into an unbounded loop on the GPU.  Sized to
	// cover the clamped floor of 0.1, whose footprint is 10x10, with room to spare - clamping *below* the true footprint
	// would be unsafe, not merely approximate: the farthest sample of a subset is nearer than the farthest of the whole,
	// and a nearer occluder hides splats that should have been drawn, which is the one direction this pass must never
	// err in.  Total fetches are scale-independent by construction (fewer destination pixels, proportionally more taps
	// each), so raising this costs nothing at the scales that are actually usable.
	const int max_taps = 12;
	src_end = min(src_end, src_begin + ivec2(max_taps - 1));

#if USE_REVERSE_Z
	// Reversed z: the near plane is 1 and the far plane is 0, so the farthest sample is the smallest value.
	float acc = 1.0;
	for(int y = src_begin.y; y <= src_end.y; ++y)
		for(int x = src_begin.x; x <= src_end.x; ++x)
			acc = min(acc, texelFetch(albedo_texture, ivec2(x, y), /*mip level=*/0).r);
#else
	float acc = 0.0;
	for(int y = src_begin.y; y <= src_end.y; ++y)
		for(int x = src_begin.x; x <= src_end.x; ++x)
			acc = max(acc, texelFetch(albedo_texture, ivec2(x, y), /*mip level=*/0).r);
#endif

	gl_FragDepth = acc;
}
