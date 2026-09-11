// gaussian_splat_taa_accum_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// SESSION069 - TAA accumulate pass.  Blends this frame's freshly resolved splat layer into the running history at
// weight 1 / (frame_count + 1), a running mean over all accumulated frames.  See OpenGLEngine::resolveSplatAccumBuffer()
// for how the pass is set up and GaussianSplatRenderer::updateTAAState() for the reset logic that decides when
// frame_count goes back to 0.
//
// Running mean rather than an exponential:  each Halton offset is a distinct sample of the underlying continuous scene,
// and all N of them should count equally in the reconstruction; an exponential would weight the newest frames higher
// and never converge to the correct integral.
//
// Both textures hold PREMULTIPLIED (colour * coverage, coverage) - exactly what the resolve shader outputs before
// composite.  Averaging premultiplied colour componentwise is linear, i.e. correct: composite the mean = mean of
// composites.  This is the reason the resolve writes vec4(0) instead of `discard` where coverage is zero (see
// splat_taa_active in gaussian_splat_resolve_frag_shader.glsl): an empty pixel this frame must contribute an explicit
// zero to the mean, not skip the update.

precision highp float;
precision highp sampler2D;

uniform sampler2D current_texture;   // Splat layer, resolved but NOT composited, from THIS frame.  Premultiplied.
uniform sampler2D history_texture;   // The accumulation so far - what the composite pass ends up sampling from.
uniform float taa_weight;            // 1 / (frame_count + 1).  Set by GaussianSplatRenderer::setTAAAccumulateUniform().

// SESSION094 - per-pixel history rejection.  updateTAAState() resets only on camera, settings and splat content, so a
// moving mesh (an animated avatar under a still camera) left stale coverage in the history: old splats drawn over where
// it arrived, a see-through trail where it left.  These hold the accum-resolution depth the splats were tested against
// (the depth downsample program's colour output), this frame and last; wherever it changed, the history was built under
// different occlusion and is dropped.
uniform sampler2D taa_depth_texture;
uniform sampler2D taa_prev_depth_texture;
uniform int taa_depth_reject_enabled; // 0 without a scene depth copy: splats are then not occluded by meshes at all.

out vec4 colour_out;

void main()
{
	// texelFetch (rather than sampled `texture`) because both textures are the same size as the frame, and this pass is
	// a full-viewport quad - so gl_FragCoord.xy already indexes them one-to-one.  No filter or wrap state is consulted.
	vec4 curr = texelFetch(current_texture, ivec2(gl_FragCoord.xy), 0);
	vec4 hist = texelFetch(history_texture, ivec2(gl_FragCoord.xy), 0);

	// SESSION094 - history rejection, see the uniforms.  3x3 because the resolve reads the accum buffer bilinearly and
	// jittered by up to half a texel, so this pixel's coverage draws on the accum texels within one of its own.  Exact
	// compare: meshes are not jittered, so under a still camera unchanged geometry reproduces its depth bit for bit.
	if(taa_depth_reject_enabled != 0)
	{
		ivec2 depth_dims = textureSize(taa_depth_texture, 0);
		ivec2 centre = ivec2(gl_FragCoord.xy * vec2(depth_dims) / vec2(textureSize(current_texture, 0)));
		for(int dy = -1; dy <= 1; ++dy)
			for(int dx = -1; dx <= 1; ++dx)
			{
				ivec2 t = clamp(centre + ivec2(dx, dy), ivec2(0), depth_dims - ivec2(1));
				if(texelFetch(taa_depth_texture, t, 0).r != texelFetch(taa_prev_depth_texture, t, 0).r)
				{
					colour_out = curr;
					return;
				}
			}
	}

	// mix(a, b, t) = a*(1-t) + b*t.  With t = 1/(n+1), this is the running mean:
	//   frame 0 (n=0, t=1):  hist doesn't exist yet (last-frame's contents are irrelevant on the first frame after a
	//                        reset).  Weight 1 makes the current replace it outright, so the initial garbage is gone.
	//   frame N (n=N, t=1/(N+1)):  new mean = old mean * N/(N+1) + curr * 1/(N+1) - the standard recurrence.
	colour_out = mix(hist, curr, taa_weight);
}
