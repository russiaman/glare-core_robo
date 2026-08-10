
// gaussian_splat_saturation_mask_frag_shader.glsl
// Copyright Glare Technologies Limited 2026 -
//
// Builds the mask of pixels that the front-to-back splat composite has already finished with, so that the splats still
// to be drawn behind them can leave immediately.  Run between draw slices - see OpenGLEngine::markSaturatedSplatPixels()
// for the pass setup and OpenGLEngine::drawSplatClouds() for why slices exist.
//
// The mask is an ordinary colour texture, usually at a fraction of the screen resolution, which
// gaussian_splat_vert_shader.glsl samples to drop whole splats that can no longer contribute.  It used to be a depth
// buffer instead, marked with the near
// plane so that the rasteriser's own depth test threw the fragments out before shading - which rejects earlier and so
// saves more per fragment.  It was abandoned because of what it costs to have: the accumulation framebuffer cannot be
// allowed to write into the scene's depth buffer, so the gate needed a private copy of it, and a depth buffer filled by
// blitting has neither the coarse per-tile summaries nor the compression a rasterised one does.  Measured, that cost
// 3.5 ms with nothing occluding the cloud and 8.7 ms with it mostly occluded, against about 2 ms saved.  A texture of
// our own costs nothing of the sort, and being small is what makes checking often affordable.
//
// One texel is marked only when *every* pixel under it is finished, which is what keeps the picture exactly right at a
// coarse mask: a pixel is skipped only if it was itself saturated.  The cost of the conservatism is the partly finished
// texels around the edge of a saturated region, which keep being drawn - a shrinking share of the region as it grows.

precision highp float;
precision highp sampler2D;

// A copy of the splat accumulation buffer, since a buffer cannot be sampled while it is attached to the framebuffer
// being drawn to.  Only the alpha channel, the accumulated coverage, is read here.
uniform sampler2D albedo_texture;

// Accumulated coverage at or above which a pixel counts as finished.  Coverage is 1 - transmittance, so the meaningful
// setting is just below 1: at 1 - 1/255 the light still getting through is under one 8-bit level, which is where the
// reference 3DGS rasteriser stops its own per-pixel loop.
uniform float splat_saturation_threshold;

// How many accumulation-buffer pixels across one mask texel covers.  1 makes this a full-resolution mask and the loop
// below a single fetch.
uniform int splat_mask_block_size;

// 0: the normal gate above - mark where the composite has finished, reading accumulated coverage from alpha.
// 1: the "hide overdraw" diagnostic - mark where the *layer count* is at or above the threshold, reading it from red,
// which is where the overdraw counting pass puts it.  Same conservative rule and same output either way, so the pyramid
// built on top of this and the vertex shader that reads it need to know nothing about which one produced the mask.
uniform int splat_mask_from_layer_count;

out vec4 colour_out;


void main()
{
	// This pass draws into the mask, so the fragment's own coordinates are mask texels; the block they stand for starts
	// at block_size times that.
	ivec2 block_begin = ivec2(gl_FragCoord.xy) * splat_mask_block_size;
	ivec2 accum_size  = textureSize(albedo_texture, /*mip level=*/0);

	// Starts above anything either channel can hold, so the first fetch always wins: coverage tops out at 1, and a layer
	// count in a half-float accumulation buffer cannot reach this.
	float min_val = (splat_mask_from_layer_count != 0) ? 1.0e20 : 1.0;
	for(int y=0; y<splat_mask_block_size; ++y)
		for(int x=0; x<splat_mask_block_size; ++x)
		{
			// Clamped rather than skipped: the viewport is not necessarily a multiple of the block size, and re-reading
			// an in-range pixel is harmless here (it only ever makes the minimum smaller, i.e. keeps drawing), whereas
			// an out-of-range texelFetch is undefined.
			ivec2 coord = min(block_begin + ivec2(x, y), accum_size - ivec2(1));
			vec4 accum = texelFetch(albedo_texture, coord, /*mip level=*/0);
			min_val = min(min_val, (splat_mask_from_layer_count != 0) ? accum.r : accum.a);
		}

	// The minimum, in both modes, so a texel is marked only when *every* pixel under it qualifies.  That is what the
	// min-pyramid built on this assumes, and what lets the vertex shader answer a big quad with one coarse fetch.
	//
	// Written for every texel, not discarded: the mask is rebuilt in place each time, so an unmarked texel has to be
	// actively cleared rather than left holding what the previous pass put there.
	colour_out = vec4((min_val >= splat_saturation_threshold) ? 1.0 : 0.0);
}
