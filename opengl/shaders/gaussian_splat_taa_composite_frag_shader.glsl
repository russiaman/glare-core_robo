// gaussian_splat_taa_composite_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// SESSION069 - TAA composite pass.  Draws the freshly accumulated history onto the frame's colour buffer using the
// same front-to-back "under" blend the resolve shader used to do directly (see the pre-TAA resolveSplatAccumBuffer()
// path).  Splits out because the accumulate pass has to write into a texture, and the target-and-blend can't be one
// pass with a normal write.
//
// The history holds premultiplied (colour * coverage, coverage), same as the resolve output that fed it.  The
// composite therefore blends with (GL_ONE, GL_ONE_MINUS_SRC_ALPHA) exactly as the resolve did before TAA existed.

precision highp float;
precision highp sampler2D;

uniform sampler2D history_texture;

out vec4 colour_out;

void main()
{
	// Same texelFetch reasoning as the accumulate pass: full-viewport quad, history is frame-sized, one-to-one indexing.
	colour_out = texelFetch(history_texture, ivec2(gl_FragCoord.xy), 0);
}
