// gaussian_splat_taa_depth_writeback_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// SESSION070 - GaussianSplatRenderer::SplatDoFDepthMode_Weighted, routed through TAA.  The resolve's own gl_FragDepth
// write (gaussian_splat_resolve_frag_shader.glsl) is a no-op while TAA is active: the resolve draws into a colour-only
// TAA current texture there, which has no depth attachment for the write to land in - see isTAAActiveForResolve()'s
// own comment.  This is a second, depth-ONLY draw of the same full-viewport quad, run straight after the TAA composite
// colour draw and into the same real framebuffer: OpenGLEngine::resolveSplatAccumBuffer() masks colour off for this
// draw (glColorMask(false,...)) so it cannot disturb what the composite draw just wrote, and this shader discards
// (not `return`) wherever this frame's splat stack had no coverage, so the real depth buffer is left exactly as
// drawNonTransparentMaterialBatches() found it there - never stomped to the quad's own rasterised depth. Same "safe
// without a depth test" argument as the resolve's own write: any splat that contributed to depth_accum_sample here was
// already depth-tested against opaque geometry while it was drawn, so the mean depth recovered below can only be at or
// nearer than whatever is already in the buffer - never further.

precision highp float;
precision highp sampler2D;

// (sum of view_depth_i * alpha_i * T_i, coverage) - the same low-res accumulation buffer
// gaussian_splat_resolve_frag_shader.glsl reads via splat_dof_depth_texture, same accumulation as the colour buffer
// (see that shader's own comment on why the two buffers' alpha channels are bit-identical).  Read directly here rather
// than through the TAA colour history: depth-of-field's blur radius does not need temporal supersampling the way
// colour does, so there is no history buffer for it - this frame's own resolved value is enough.
uniform sampler2D splat_dof_depth_texture;
uniform vec2 splat_resolve_dims_px; // Frame dims - same UV form as sampleSplatAccum() in the resolve shader.
uniform float splat_dof_near_clip_dist;

out vec4 colour_out; // Never actually reaches the framebuffer - OpenGLEngine masks colour off for this draw.

// Inverse of getDepthFromDepthTextureValue() in frag_utils.glsl.  Duplicated rather than pulling frag_utils.glsl into
// this small program's build: this is the only piece of it this shader needs.
float getDeviceDepthFromLinearDepth(float near_clip_dist_, float linear_depth)
{
#if USE_REVERSE_Z
	return near_clip_dist_ / linear_depth;
#else
	return 1.0 - near_clip_dist_ / linear_depth;
#endif
}

void main()
{
	vec4 depth_accum_sample = texture(splat_dof_depth_texture, gl_FragCoord.xy / splat_resolve_dims_px);
	if(depth_accum_sample.a <= 0.0)
		discard; // No splat coverage here this frame - leave the real depth buffer untouched.

	float mean_view_depth = depth_accum_sample.r / depth_accum_sample.a;
	gl_FragDepth = getDeviceDepthFromLinearDepth(splat_dof_near_clip_dist, mean_view_depth);
	colour_out = vec4(0.0);
}
