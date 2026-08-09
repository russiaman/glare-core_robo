
// gaussian_splat_mask_reduce_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Builds one level of the saturation mask's pyramid from the level above it, taking the minimum of each 2x2 block -
// see OpenGLEngine::markSaturatedSplatPixels().
//
// Minimum, not average: a texel means "every pixel under this is finished", and that only stays true up the pyramid if
// a coarse texel is marked solely when all four of its children are.  That is what lets gaussian_splat_vert_shader.glsl
// answer "is this whole splat's quad finished?" with a single fetch at whichever level is coarse enough to cover the
// quad, rather than reading every texel the quad touches.
//
// The source level is selected with GL_TEXTURE_BASE_LEVEL by the caller rather than passed as a lod here: reading and
// writing different levels of one texture is only defined when the levels being read are excluded from the attachment,
// which is what restricting the base level does.

precision highp float;
precision highp sampler2D;

uniform sampler2D albedo_texture; // The level above this one, as the only level the sampler can see.

out vec4 colour_out;


void main()
{
	ivec2 src = ivec2(gl_FragCoord.xy) * 2;
	ivec2 src_max = textureSize(albedo_texture, /*mip level=*/0) - ivec2(1); // An odd-sized level leaves the last row and column with only half a block; clamping re-reads a valid texel, which can only make the minimum smaller, i.e. keep drawing.

	colour_out = vec4(min(
		min(texelFetch(albedo_texture, min(src,                  src_max), 0).r, texelFetch(albedo_texture, min(src + ivec2(1, 0), src_max), 0).r),
		min(texelFetch(albedo_texture, min(src + ivec2(0, 1), src_max), 0).r, texelFetch(albedo_texture, min(src + ivec2(1, 1), src_max), 0).r)));
}
