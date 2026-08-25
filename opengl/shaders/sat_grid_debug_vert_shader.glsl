
// SESSION076 §9: debug visualisation of the saturation grid (GaussianSplatSaturationGrid) - a giant sphere centred on
// the frontier's anchor_pos_ws, drawn from the inside, so its surface can be shaded per-direction from sat_depth.
// Modelled on probe_debug_vert_shader.glsl (same "unit sphere position is also the object-space direction" trick),
// but the sphere is sized to engulf the camera rather than sit small at a point - see
// OpenGLEngine::drawSatGridDebugSphere().

in vec3 position_in;

out vec3 dir_os;

uniform mat4 proj_matrix;
uniform mat4 view_matrix;
uniform vec4 sat_grid_sphere_pos_radius; // xyz = anchor_pos_ws, w = sphere radius.


void main()
{
	// MeshPrimitiveBuilding::makeSphereMesh() gives a unit sphere centred on the origin, so the vertex position
	// is also the direction from the anchor, in object space.
	dir_os = position_in;

	vec3 pos_ws = sat_grid_sphere_pos_radius.xyz + position_in * sat_grid_sphere_pos_radius.w;

	gl_Position = proj_matrix * (view_matrix * vec4(pos_ws, 1.0));
}
