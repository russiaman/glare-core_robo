
// gaussian_splat_mask_reduce_mean_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Builds one level of the coverage pyramid from the level above it, taking the mean of each 2x2 block - see
// OpenGLEngine::markSaturatedSplatPixels() and gaussian_splat_mask_reduce_frag_shader.glsl, whose min-pyramid this sits
// beside. Mean, not minimum: the coverage pyramid is not answering "is this whole area finished" (the min-pyramid's
// job, unchanged), it is answering "how covered is this area on average", for the quad-shrink weighting in
// gaussian_splat_vert_shader.glsl (see GaussianSplatRenderer::getCoverageShrinkStrength()) - a continuous estimate a
// strict AND cannot give.
//
// Base level (level 0) is written by gaussian_splat_saturation_mask_frag_shader.glsl's second output, as the mean of
// the accumulated alpha over the same per-texel block the min-pyramid's level 0 already loops over - see that shader.
// Same base-level restriction rule as the min reduce: the caller selects the source level with GL_TEXTURE_BASE_LEVEL.

precision highp float;
precision highp sampler2D;

uniform sampler2D albedo_texture; // The level above this one, as the only level the sampler can see.

out vec4 colour_out;


void main()
{
	ivec2 src = ivec2(gl_FragCoord.xy) * 2;
	ivec2 src_max = textureSize(albedo_texture, /*mip level=*/0) - ivec2(1); // An odd-sized level leaves the last row and column with only half a block; clamping re-reads a valid texel, which is the correct thing to average in twice for an edge block that is genuinely half the size.

	colour_out = vec4(0.25 * (
		texelFetch(albedo_texture, min(src,                  src_max), 0).r + texelFetch(albedo_texture, min(src + ivec2(1, 0), src_max), 0).r +
		texelFetch(albedo_texture, min(src + ivec2(0, 1), src_max), 0).r + texelFetch(albedo_texture, min(src + ivec2(1, 1), src_max), 0).r));
}
