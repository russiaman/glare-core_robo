
// See sat_grid_debug_vert_shader.glsl.
//
// Drawn as a colour-only overlay (single draw buffer, depth test off) after drawSplatClouds(), so it sits on top of
// the scene without disturbing the normal buffer - see OpenGLEngine::drawSatGridDebugSphere().

in vec3 dir_os;

// Both R32F, res x res, one texel per saturation-grid tile - see GaussianSplatUnculledFrontier's sat_accum_t /
// sat_amp_sum for what each holds and why the ramp view needs a different quantity from the binary one.
uniform sampler2D sat_grid_tex;      // TRANSMITTANCE: 1 = nothing occludes this direction, 0 = fully blocked. Binary view only - exactly what the prune's threshold test acts on.
uniform sampler2D sat_grid_ramp_tex; // Unbounded SUM of every touching occluder's contribution. Ramp view only - has no ceiling, so it does not visually clamp towards one colour the way a threshold-normalised quantity would.

uniform float sat_grid_remaining_threshold; // 1 - saturation threshold. A tile counts as saturated at or below this - the same test gsBuildSaturationGrid() makes.
uniform int sat_grid_debug_mode;            // 0 = binary (saturated tiles only), 1 = continuous ramp over saturation degree.

// SESSION077: the ramp's blue/red endpoints, reusing the existing overdraw view's own range controls
// (splat_overdraw_range_min/max, "Show overdraw" row) rather than a new pair of spinboxes - same idea (map a
// per-fragment scalar to a fixed ramp), different scalar and units (0-1 saturation here vs. layer count/summed alpha
// there), so the owner has to re-enter a 0-1-ish range when switching this on, same as changing overdraw mode does.
uniform float sat_grid_ramp_range_min;
uniform float sat_grid_ramp_range_max;

layout(location = 0) out vec4 colour_out;


// Bondi Blue, for the binary view.
const vec3 SAT_GRID_DEBUG_COLOUR = vec3(0.0, 0.584, 0.714);
const float SAT_GRID_DEBUG_ALPHA = 0.35;


void main()
{
	vec3 dir = normalize(dir_os);

	// Matches gsDirToOct()/gsSatGridTileForDir() in GaussianSplatSaturationGrid.cpp exactly (same formula).
	vec2 oct_uv = float32x3_to_oct(dir) * 0.5 + vec2(0.5);

	if(sat_grid_debug_mode == 0)
	{
		// Binary: exactly what the prune acts on.
		float transmittance = texture(sat_grid_tex, oct_uv).r;
		if(transmittance <= sat_grid_remaining_threshold)
			colour_out = vec4(SAT_GRID_DEBUG_COLOUR, SAT_GRID_DEBUG_ALPHA);
		else
			colour_out = vec4(0.0); // Not saturated: fully transparent, real scene shows through unchanged.
		return;
	}

	// Ramp: how much occluding mass actually accumulated in each tile - see sat_grid_ramp_tex's declaration for why
	// this is a different, UNBOUNDED quantity from the binary view's transmittance rather than 1-transmittance of it.
	// A tile ten occluders deep reads roughly 10x one with a single occluder, instead of both reading close to the
	// same near-1 value - the "everything clamps to one colour" the owner saw when this first used transmittance.
	// Same blue -> green -> red ramp the overdraw view uses (gaussian_splat_resolve_frag_shader.glsl), for
	// familiarity, and the same range uniforms - see their declaration above.
	float amp_sum = texture(sat_grid_ramp_tex, oct_uv).r;

	if(amp_sum <= 0.0)
	{
		colour_out = vec4(0.0); // Nothing accumulated here at all - leave the scene alone rather than tinting it deep blue.
		return;
	}

	// SESSION077: opaque, exactly like the overdraw view (gaussian_splat_resolve_frag_shader.glsl) - it replaces the
	// scene pixel outright rather than blending a tint over it, so the ramp reads the same way in both places.
	float range = max(sat_grid_ramp_range_max - sat_grid_ramp_range_min, 1.0e-4);
	float t = clamp((amp_sum - sat_grid_ramp_range_min) / range, 0.0, 1.0);

	vec3 ramp_col = (t < 0.5)
		? mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t * 2.0)
		: mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);

	colour_out = vec4(ramp_col, 1.0);
}
