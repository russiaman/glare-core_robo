/*=====================================================================
TransformGizmo.cpp
------------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#include "TransformGizmo.h"


#include "OpenGLEngine.h"
#include "MeshPrimitiveBuilding.h"
#include "../maths/vec2.h"
#include "../maths/plane.h"
#include "../maths/mathstypes.h"
#include "../maths/LineSegment4f.h"
#include "../graphics/SRGBUtils.h"
#include <chrono>


static const float GIZMO_ANIM_DURATION  = 0.2f; // seconds for all hover transitions
static const int   GRABBED_CENTER_SCALE      = 6; // grabbed_axis: uniform scale (0..2=translate, 3..5=rotation)
static const int   GRABBED_SCALE_PLANE_BASE  = 7; // grabbed_axis 7/8/9 = two-axis scale plane 0/1/2

static const Colour3f axis_arrows_default_cols[]   = { Colour3f(0.6f,0.2f,0.2f), Colour3f(0.2f,0.6f,0.2f), Colour3f(0.2f,0.2f,0.6f) };
static const Colour3f axis_arrows_mouseover_cols[] = { Colour3f(1,0.45f,0.3f),   Colour3f(0.3f,1,0.3f),    Colour3f(0.3f,0.45f,1) };

// For each direction x, y, z, the two other basis vectors. 
static const Vec4f basis_vectors[6] = { Vec4f(0,1,0,0), Vec4f(0,0,1,0), Vec4f(0,0,1,0), Vec4f(1,0,0,0), Vec4f(1,0,0,0), Vec4f(0,1,0,0) };

static const float arc_handle_half_angle = 1.5f;


TransformGizmo::TransformGizmo(OpenGLEngine* engine_, const Vec4f& gizmo_centre)
:	engine(engine_),
	grabbed_axis(-1),
	hovered_axis(-1),
	hovered_cube(-1),
	hovered_scale_plane(-1),
	hovered_translate_plane(-1),
	center_scale_engaged(false),
	center_cube_src_matrix(Matrix4f::identity()),
	center_cube_tgt_matrix(Matrix4f::identity()),
	center_cube_anim_t(1.0f),
	center_cube_prev_state(-1),
	axis_cube_src_side  {0.f, 0.f, 0.f},
	axis_cube_tgt_side  {0.f, 0.f, 0.f},
	axis_cube_src_cfrac {0.f, 0.f, 0.f},
	axis_cube_tgt_cfrac {0.f, 0.f, 0.f},
	axis_cube_anim_t    {1.f, 1.f, 1.f},
	axis_cube_prev_state{-1, -1, -1},
	shaft_src_scale  {1.f, 1.f, 1.f},
	shaft_tgt_scale  {1.f, 1.f, 1.f},
	shaft_anim_t     {1.f, 1.f, 1.f},
	shaft_prev_state {-1, -1, -1},
	tp_src_size  {0.f, 0.f, 0.f},
	tp_tgt_size  {0.f, 0.f, 0.f},
	tp_src_alpha {0.5f, 0.5f, 0.5f},
	tp_tgt_alpha {0.5f, 0.5f, 0.5f},
	tp_anim_t    {1.f, 1.f, 1.f},
	tp_prev_state{-1, -1, -1},
	last_frame_time(-1.0),
	grabbed_angle(0),
	original_grabbed_angle(0),
	grabbed_arc_angle_offset(0),
	grabbed_scale_mouse_px(Vec2f(0.f))
{
	{
		static const Vec4f axis_tip_pos[3] = { Vec4f(1,0,0,1), Vec4f(0,1,0,1), Vec4f(0,0,1,1) };
		auto cube_meshdata  = MeshPrimitiveBuilding::makeCubeMesh(*engine->vert_buf_allocator);
		for(int i = 0; i < NUM_AXIS_ARROWS; ++i)
		{
			// Shaft (translate handle) — standard cylinder mesh (local Z = shaft axis, radius 1).
			axis_arrow_objects[i] = engine->allocateObject();
			axis_arrow_objects[i]->ob_to_world_matrix = Matrix4f::identity();
			axis_arrow_objects[i]->mesh_data = engine->getCylinderMesh();
			axis_arrow_objects[i]->materials.resize(1);
			axis_arrow_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
			axis_arrow_objects[i]->always_visible = true;
			engine->addObject(axis_arrow_objects[i]);

			// Cube tip (scale handle) — separate object for independent hover/scale
			axis_scale_cube_objects[i] = engine->allocateObject();
			axis_scale_cube_objects[i]->ob_to_world_matrix = Matrix4f::translationMatrix(gizmo_centre);
			axis_scale_cube_objects[i]->mesh_data = cube_meshdata;
			axis_scale_cube_objects[i]->materials.resize(1);
			axis_scale_cube_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
			axis_scale_cube_objects[i]->always_visible = true;
			engine->addObject(axis_scale_cube_objects[i]);
		}
	}

	for(int i=0; i<3; ++i)
	{
		GLObjectRef ob = engine->allocateObject();
		ob->ob_to_world_matrix = Matrix4f::translationMatrix((float)i * 3, 0, 2);
		ob->mesh_data = MeshPrimitiveBuilding::makeRotationArcHandleMeshData(*engine->vert_buf_allocator, arc_handle_half_angle * 2);
		ob->materials.resize(1);
		ob->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
		ob->always_visible = true;
		rot_handle_arc_objects[i] = ob;
		engine->addObject(rot_handle_arc_objects[i]);
	}


	// Center scale cube: single cube that morphs into the hovered scale-plane shape.
	{
		auto cube_meshdata2 = MeshPrimitiveBuilding::makeCubeMesh(*engine->vert_buf_allocator);
		center_scale_cube_object = engine->allocateObject();
		center_scale_cube_object->mesh_data = cube_meshdata2;
		center_scale_cube_object->materials.resize(1);
		center_scale_cube_object->materials[0].albedo_linear_rgb = toLinearSRGB(Colour3f(0.55f, 0.55f, 0.55f));
		center_scale_cube_object->always_visible = true;
		center_scale_cube_object->ob_to_world_matrix = Matrix4f::identity();
		engine->addObject(center_scale_cube_object);
	}

	// Outer plane handles: same colours, offset along both axes
	auto quad_meshdata = MeshPrimitiveBuilding::makeUnitQuadMesh(*engine->vert_buf_allocator);
	for(int i = 0; i < NUM_PLANES; ++i)
	{
		translate_plane_objects[i] = engine->allocateObject();
		translate_plane_objects[i]->mesh_data = quad_meshdata;
		translate_plane_objects[i]->materials.resize(1);
		translate_plane_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
		translate_plane_objects[i]->materials[0].alpha = 0.5f;
		translate_plane_objects[i]->materials[0].alpha_blend = true;
		translate_plane_objects[i]->always_visible = true;
		translate_plane_objects[i]->ob_to_world_matrix = Matrix4f::identity();
		engine->addObject(translate_plane_objects[i]);
	}

	updateGizmoDrawTransform(gizmo_centre);
}


TransformGizmo::~TransformGizmo()
{
	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
		checkRemoveObAndSetRefToNull(*engine, axis_arrow_objects[i]);

	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
		checkRemoveObAndSetRefToNull(*engine, axis_scale_cube_objects[i]);

	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
		checkRemoveObAndSetRefToNull(*engine, rot_handle_arc_objects[i]);


	for(int i=0; i<NUM_PLANES; ++i)
		checkRemoveObAndSetRefToNull(*engine, translate_plane_objects[i]);

	checkRemoveObAndSetRefToNull(*engine, center_scale_cube_object);
}


// Avoids NaNs
static float safeATan2(float y, float x)
{
	const float a = std::atan2(y, x);
	if(!isFinite(a))
		return 0.f;
	else
		return a;
}


void TransformGizmo::update(const Vec4f& ob_pos_ws)
{
	// If grabbed something, don't update from external changes
	if(grabbed_axis == -1)
	{
		updateGizmoDrawTransform(ob_pos_ws);
	}
}


// Project a world-space point to pixel coordinates.
// Returns false if the point is behind the camera.
bool TransformGizmo::worldToPixel(const Vec4f& ws_pos, OpenGLEngine* engine, Vec2f& px_out)
{
	return engine->worldSpacePosToPixel(ws_pos, px_out);
}


/*
Given a world-space line (p_a_ws, p_b_ws) and the pixel coordinates of the mouse,
returns the world-space point on the line whose screen-space projection is closest to px.

Let line coords in ws be p_ws(t) = a + b * t

pixel coords for a point p_ws are

cam_to_p = p_ws - cam_origin

r_x =  dot(cam_to_p, cam_right) / dot(cam_to_p, cam_forw)
r_y = -dot(cam_to_p, cam_up)    / dot(cam_to_p, cam_forw)

and

pixel_x = gl_w * (lens_sensor_dist / sensor_width  * r_x + 1/2)
pixel_y = gl_h * (lens_sensor_dist / sensor_height * r_y + 1/2)

let R = lens_sensor_dist / sensor_width

so 

pixel_x = gl_w * (R *  dot(p_ws - cam_origin, cam_right) / dot(p_ws - cam_origin, cam_forw) + 1/2)
pixel_y = gl_h * (R * -dot(p_ws - cam_origin, cam_up)    / dot(p_ws - cam_origin, cam_forw) + 1/2)

pixel_x = gl_w * (R *  dot(a + b * t - cam_origin, cam_right) / dot(a + b * t - cam_origin, cam_forw) + 1/2)
pixel_y = gl_h * (R * -dot(a + b * t - cam_origin, cam_up)    / dot(a + b * t - cam_origin, cam_forw) + 1/2)

We know pixel_x and pixel_y, want to solve for t.

pixel_x = gl_w * (R * dot(a + b * t - cam_origin, cam_right) / dot(a + b * t - cam_origin, cam_forw) + 1/2)
pixel_x/gl_w = R * dot(a + b * t - cam_origin, cam_right) / dot(a + b * t - cam_origin, cam_forw) + 1/2
pixel_x/gl_w = R * [dot(a - cam_origin, cam_right) + dot(b * t, cam_right)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)] + 1/2
pixel_x/gl_w - 1/2 = R  * [dot(a - cam_origin, cam_right) + dot(b * t, cam_right)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)]
(pixel_x/gl_w - 1/2) / R = [dot(a - cam_origin, cam_right) + dot(b * t, cam_right)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)]

let A = dot(a - cam_origin, cam_forw)
let B = dot(b, cam_forw)
let C = (pixel_x/gl_w - 1/2) / R
let D = dot(a - cam_origin, cam_right)
let E = dot(b, cam_right)

so we get

C = [D + dot(b * t, cam_right)] / [A + dot(b * t, cam_forw)]
C = [D + dot(b, cam_right) * t] / [A + dot(b, cam_forw) * t]
C = [D + E * t] / [A + B * t]
[A + B * t] C = D + E * t
AC + BCt = D + Et
BCt - Et = D - AC
t(BC - E) = D - AC
t = (D - AC) / (BC - E)


For y (used when all x coordinates are ~ the same)
pixel_y = gl_h * (R * -dot(a + b * t - cam_origin, cam_up) / dot(a + b * t - cam_origin, cam_forw) + 1/2)
pixel_y/gl_h = R * -dot(a + b * t - cam_origin, cam_up) / dot(a + b * t - cam_origin, cam_forw) + 1/2
pixel_y/gl_h = R * -[dot(a - cam_origin, cam_up) + dot(b * t, cam_up)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)] + 1/2
pixel_x/gl_w - 1/2 = R  * -[dot(a - cam_origin, cam_up) + dot(b * t, cam_up)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)]
(pixel_x/gl_w - 1/2) / R = -[dot(a - cam_origin, cam_right) + dot(b * t, cam_right)] / [dot(a - cam_origin, cam_forw) + dot(b * t, cam_forw)]

let A = dot(a - cam_origin, cam_forw)
let B = dot(b, cam_forw)
let C = (pixel_y/gl_h - 1/2) / R
let D = dot(a - cam_origin, cam_up)
let E = dot(b, cam_up)

C = -[D + dot(b * t, cam_up)] / [A + dot(b * t, cam_forw)]
C = -[D + dot(b, cam_right) * t] / [A + dot(b, cam_forw) * t]
C = -[D + E * t] / [A + B * t]
[A + B * t] C = -[D + E * t]
AC + BCt = -D - Et
BCt + Et = -D - AC
t(BC + E) = -D - AC
t = (-D - AC) / (BC + E)
*/
Vec4f TransformGizmo::pointOnLineWorldSpace(const Vec4f& p_a_ws, const Vec4f& p_b_ws, const Vec2f& px) const
{
	const Vec4f cam_pos      = engine->getCurrentScene()->cam_to_world.getColumn(3); // = cam_to_world * Vec4f(0,0,0,1);
	const Vec4f cam_forwards = engine->getCurrentScene()->cam_to_world.getColumn(1); // = cam_to_world * Vec4f(0,1,0,0);
	const Vec4f cam_right    = engine->getCurrentScene()->cam_to_world.getColumn(0); // = cam_to_world * Vec4f(1,0,0,0);
	const Vec4f cam_up       = engine->getCurrentScene()->cam_to_world.getColumn(2); // = cam_to_world * Vec4f(0,0,1,0);

	const int viewport_w = engine->getViewPortWidth(); // = gl_w
	const int viewport_h = engine->getViewPortHeight();
	const float sensor_w = engine->getCurrentScene()->use_sensor_width;
	const float sensor_h = engine->getCurrentScene()->use_sensor_height;
	const float lens_sensor_dist = engine->getCurrentScene()->lens_sensor_dist;

	//const float sensor_height = sensor_w * (float)viewport_h / (float)viewport_w;

	const Vec4f a = p_a_ws;
	const Vec4f b = normalise(p_b_ws - p_a_ws);

	float A = dot(a - cam_pos, cam_forwards);
	float B = dot(b, cam_forwards);
	float C = (px.x / (float)viewport_w - 0.5f) * sensor_w / lens_sensor_dist;
	float D = dot(a - cam_pos, cam_right);
	float E = dot(b, cam_right);

	const float denom = B*C - E;
	float t;
	if(std::fabs(denom) > 1.0e-4f)
	{
		t = (D - A*C) / denom;
	}
	else
	{
		// Work with y instead

		A = dot(a - cam_pos, cam_forwards);
		B = dot(b, cam_forwards);
		C = (px.y / (float)viewport_h - 0.5f) * sensor_h / lens_sensor_dist;
		D = dot(a - cam_pos, cam_up);
		E = dot(b, cam_up);

		t = (-D - A*C) / (B*C + E);
	}

	return a + b * t;
}


static LineSegment4f clipLineSegmentToCameraFrontHalfSpace(const LineSegment4f& segment, const Planef& cam_front_plane)
{
	const float d_a = cam_front_plane.signedDistToPoint(segment.a);
	const float d_b = cam_front_plane.signedDistToPoint(segment.b);

	// If both endpoints are in front half-space, no clipping is required.  If both points are in back half-space, line segment is completely clipped.
	// In this case just return the unclipped line segment.
	if((d_a < 0 && d_b < 0) || (d_a > 0 && d_b > 0))
		return segment;

	/*
	
	a                  /         b
	------------------/----------
	d_a              /   d_b

	*/
	if(d_a > 0)
	{
		assert(d_b < 0);
		const float frac = d_a / (d_a - d_b); // = d_a / (d_a + |d_b|)
		return LineSegment4f(segment.a, Maths::lerp(segment.a, segment.b, frac));
	}
	else
	{
		assert(d_a < 0);
		assert(d_b >= 0);
		const float frac = -d_a / (-d_a + d_b); // = |d_a| / (|d_a| + d_b)
		return LineSegment4f(segment.b, Maths::lerp(segment.a, segment.b, frac));
	}
}

// See https://math.stackexchange.com/questions/1036959/midpoint-of-the-shortest-distance-between-2-rays-in-3d
// In particular this answer: https://math.stackexchange.com/a/2371053
static Vec4f closestPointOnLineToRay(const LineSegment4f& line, const Vec4f& origin, const Vec4f& unitdir)
{
	const Vec4f a = line.a;
	const Vec4f b = normalise(line.b - line.a);
	const Vec4f c = origin;
	const Vec4f d = unitdir;

	const float t = (dot(c - a, b) + dot(a - c, d) * dot(b, d)) / (1.f - Maths::square(dot(b, d)));
	return a + b * t;
}


// NOTE: pretty much the same as clipLineSegmentToCameraFrontHalfSpace, remove
inline static bool clipLineToPlaneBackHalfSpace(const Planef& plane, Vec4f& a, Vec4f& b)
{
	const float ad = plane.signedDistToPoint(a);
	const float bd = plane.signedDistToPoint(b);
	if(ad > 0 && bd > 0) // If both endpoints not in back half space:
		return false;

	if(ad <= 0 && bd <= 0) // If both endpoints in back half space:
		return true;

	// Else line straddles plane
	// ad + (bd - ad) * t = 0
	// t = -ad / (bd - ad)
	// t = ad / -(bd - ad)
	// t = ad / (-bd + ad)
	// t = ad / (ad - bd)

	const float t = ad / (ad - bd);
	const Vec4f on_plane_p = a + (b - a) * t;
	//assert(epsEqual(plane.signedDistToPoint(on_plane_p), 0.f));

	if(ad <= 0) // If point a lies in back half space:
		b = on_plane_p; // update point b
	else
		a = on_plane_p; // else point b lies in back half space, so update point a
	return true;
}


static float cubicEaseOut(float t) { const float u = 1.f - t; return 1.f - u * u * u; }

static Matrix4f lerpMatrix(const Matrix4f& a, const Matrix4f& b, float t)
{
	Matrix4f result;
	for(int c = 0; c < 4; ++c)
		result.setColumn(c, a.getColumn(c) + (b.getColumn(c) - a.getColumn(c)) * t);
	return result;
}


void TransformGizmo::updateGizmoDrawTransform(const Vec4f& new_gizmo_centre)
{
	const Vec4f cam_pos = engine->getCurrentScene()->cam_to_world.getColumn(3); // = cam_to_world * Vec4f(0,0,0,1);

	// Compute dt once per frame; used by all animation blocks below.
	const double now_s = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
	const float  dt    = (last_frame_time < 0.0) ? 0.f : myMin(0.1f, (float)(now_s - last_frame_time));
	last_frame_time    = now_s;

	const Vec4f gizmo_centre = new_gizmo_centre;
	const Vec4f cam_to_gizmo = gizmo_centre - cam_pos;
	const float control_scale = cam_to_gizmo.length() * 0.2f;

	const float arrow_len = control_scale * 0.9f;

	// Segments end exactly at the shaft visual end (80% of arrow_len).
	// The cube is positioned beyond segment.b — shaft picking covers only the shaft.
	const float shaft_end = arrow_len * 0.80f;
	axis_arrow_segments[0] = LineSegment4f(gizmo_centre, gizmo_centre + Vec4f(cam_to_gizmo[0] > 0 ? -shaft_end : shaft_end, 0, 0, 0));
	axis_arrow_segments[1] = LineSegment4f(gizmo_centre, gizmo_centre + Vec4f(0, cam_to_gizmo[1] > 0 ? -shaft_end : shaft_end, 0, 0));
	axis_arrow_segments[2] = LineSegment4f(gizmo_centre, gizmo_centre + Vec4f(0, 0, cam_to_gizmo[2] > 0 ? -shaft_end : shaft_end, 0));

	// Update shaft objects (translate handles).
	// Cylinder mesh: local XY = radius 1, local Z = [0..1] length.
	// Map: col0,col1 = perp axes * shaft_r; col2 = arrow direction (full length); col3 = start.
	const float shaft_r = arrow_len * 0.01f;
	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
	{
		const float tgt_scale  = (hovered_axis == i) ? 1.07f : 1.0f;
		const int   cur_state  = (hovered_axis == i) ? 1 : 0;

		if(shaft_prev_state[i] < 0)
		{
			shaft_src_scale[i]  = tgt_scale;
			shaft_tgt_scale[i]  = tgt_scale;
			shaft_anim_t[i]     = 1.0f;
			shaft_prev_state[i] = cur_state;
		}
		else if(cur_state != shaft_prev_state[i])
		{
			const float e = cubicEaseOut(shaft_anim_t[i]);
			shaft_src_scale[i]  = shaft_src_scale[i] + (shaft_tgt_scale[i] - shaft_src_scale[i]) * e;
			shaft_tgt_scale[i]  = tgt_scale;
			shaft_anim_t[i]     = 0.f;
			shaft_prev_state[i] = cur_state;
		}
		else
		{
			shaft_tgt_scale[i] = tgt_scale;
		}

		shaft_anim_t[i] = myMin(1.f, shaft_anim_t[i] + dt / GIZMO_ANIM_DURATION);
		const float sc = shaft_src_scale[i] + (shaft_tgt_scale[i] - shaft_src_scale[i]) * cubicEaseOut(shaft_anim_t[i]);

		const Vec4f dir = axis_arrow_segments[i].b - axis_arrow_segments[i].a;
		const Vec4f dn  = dir * (1.f / dir.length());
		const Vec4f arb = (std::fabs(dn[0]) < 0.9f) ? Vec4f(1,0,0,0) : Vec4f(0,1,0,0);
		const Vec4f u   = normalise(crossProduct(dn, arb));
		const Vec4f v   = crossProduct(dn, u);
		Matrix4f m;
		m.setColumn(0, u * (shaft_r * sc));
		m.setColumn(1, v * (shaft_r * sc));
		m.setColumn(2, dir * sc);
		m.setColumn(3, axis_arrow_segments[i].a);
		axis_arrow_objects[i]->ob_to_world_matrix = m;
		engine->updateObjectTransformData(*axis_arrow_objects[i]);
	}

	// Update axis cube tip objects (scale handles) — world-space axis-aligned boxes.
	{
		// segment b = shaft end (0.80 * arrow_len). Cube sits just beyond shaft end.
		// Express cube center and size as fractions of segment length:
		//   cube center = 0.83 * arrow_len = (0.83/0.80) * shaft_end
		//   cube half   = 0.03 * arrow_len = (0.03/0.80) * shaft_end
		static const float cx_mid = 0.83f / 0.80f;  // fraction of segment to cube center
		static const float cs     = 0.03f / 0.80f;  // cube half-size as fraction of segment
		const float seg_len = shaft_end; // segment length = shaft_end = 0.80 * arrow_len
		const float cube_ws_side = cs * 2.0f * seg_len; // = 0.06 * arrow_len (same as before)

		for(int i=0; i<NUM_AXIS_ARROWS; ++i)
		{
			const Vec4f dir_ws = axis_arrow_segments[i].b - axis_arrow_segments[i].a;

			// Target state: 0=default, 1=axis-hover, 2=cube-hover.
			float tgt_side, tgt_cfrac;
			int   cur_state;
			if(hovered_axis == i)
			{
				tgt_side  = cube_ws_side * 1.07f;
				tgt_cfrac = cx_mid * 1.07f;
				cur_state = 1;
			}
			else if(hovered_cube == i)
			{
				tgt_side  = cube_ws_side * 1.25f;
				tgt_cfrac = cx_mid;
				cur_state = 2;
			}
			else
			{
				tgt_side  = cube_ws_side;
				tgt_cfrac = cx_mid;
				cur_state = 0;
			}

			if(axis_cube_prev_state[i] < 0)
			{
				// First frame: snap to target, no animation.
				axis_cube_src_side[i]   = tgt_side;
				axis_cube_tgt_side[i]   = tgt_side;
				axis_cube_src_cfrac[i]  = tgt_cfrac;
				axis_cube_tgt_cfrac[i]  = tgt_cfrac;
				axis_cube_anim_t[i]     = 1.0f;
				axis_cube_prev_state[i] = cur_state;
			}
			else if(cur_state != axis_cube_prev_state[i])
			{
				// State changed: snapshot current interpolated pos using stored OLD target.
				const float e = cubicEaseOut(axis_cube_anim_t[i]);
				axis_cube_src_side[i]   = axis_cube_src_side[i]  + (axis_cube_tgt_side[i]  - axis_cube_src_side[i])  * e;
				axis_cube_src_cfrac[i]  = axis_cube_src_cfrac[i] + (axis_cube_tgt_cfrac[i] - axis_cube_src_cfrac[i]) * e;
				axis_cube_tgt_side[i]   = tgt_side;
				axis_cube_tgt_cfrac[i]  = tgt_cfrac;
				axis_cube_anim_t[i]     = 0.f;
				axis_cube_prev_state[i] = cur_state;
			}
			else
			{
				// Same state: keep target fresh (camera zoom changes cube_ws_side).
				axis_cube_tgt_side[i]  = tgt_side;
				axis_cube_tgt_cfrac[i] = tgt_cfrac;
			}

			axis_cube_anim_t[i] = myMin(1.f, axis_cube_anim_t[i] + dt / GIZMO_ANIM_DURATION);
			const float e = cubicEaseOut(axis_cube_anim_t[i]);
			const float side  = axis_cube_src_side[i]  + (axis_cube_tgt_side[i]  - axis_cube_src_side[i])  * e;
			const float cfrac = axis_cube_src_cfrac[i] + (axis_cube_tgt_cfrac[i] - axis_cube_src_cfrac[i]) * e;

			const Vec4f cube_center_ws = axis_arrow_segments[i].a + dir_ws * cfrac;
			const float half = side * 0.5f;

			Matrix4f m;
			m.setColumn(0, Vec4f(side, 0, 0, 0));
			m.setColumn(1, Vec4f(0, side, 0, 0));
			m.setColumn(2, Vec4f(0, 0, side, 0));
			m.setColumn(3, cube_center_ws - Vec4f(half, half, half, 0.f));
			axis_scale_cube_objects[i]->ob_to_world_matrix = m;
			engine->updateObjectTransformData(*axis_scale_cube_objects[i]);
		}
	}


	//----------------------- Update plane handles (2-axis translate), animated -----------------------
	{
		static const int plane_axes[3][2] = {{1,2},{0,2},{0,1}};
		const float plane_size   = arrow_len * 0.156f;
		const float outer_offset = arrow_len * 0.42f;
		const float base_size    = plane_size * 0.60f;

		for(int i = 0; i < NUM_PLANES; ++i)
		{
			const float tgt_size  = (hovered_translate_plane == i) ? base_size * 1.2f : base_size;
			const float tgt_alpha = (hovered_translate_plane == i) ? 1.0f : 0.5f;
			const int   cur_state = (hovered_translate_plane == i) ? 1 : 0;

			if(tp_prev_state[i] < 0)
			{
				tp_src_size[i]  = tp_tgt_size[i]  = tgt_size;
				tp_src_alpha[i] = tp_tgt_alpha[i] = tgt_alpha;
				tp_anim_t[i]    = 1.f;
				tp_prev_state[i] = cur_state;
			}
			else if(cur_state != tp_prev_state[i])
			{
				const float e    = cubicEaseOut(tp_anim_t[i]);
				tp_src_size[i]   = tp_src_size[i]  + (tp_tgt_size[i]  - tp_src_size[i])  * e;
				tp_src_alpha[i]  = tp_src_alpha[i] + (tp_tgt_alpha[i] - tp_src_alpha[i]) * e;
				tp_tgt_size[i]   = tgt_size;
				tp_tgt_alpha[i]  = tgt_alpha;
				tp_anim_t[i]     = 0.f;
				tp_prev_state[i] = cur_state;
			}
			else
			{
				tp_tgt_size[i]  = tgt_size;
				tp_tgt_alpha[i] = tgt_alpha;
			}

			tp_anim_t[i] = myMin(1.f, tp_anim_t[i] + dt / GIZMO_ANIM_DURATION);
			const float e     = cubicEaseOut(tp_anim_t[i]);
			const float size  = tp_src_size[i]  + (tp_tgt_size[i]  - tp_src_size[i])  * e;
			const float alpha = tp_src_alpha[i] + (tp_tgt_alpha[i] - tp_src_alpha[i]) * e;

			const int ai = plane_axes[i][0], bi = plane_axes[i][1];
			const Vec4f dir_a = axis_arrow_segments[ai].b - axis_arrow_segments[ai].a;
			const Vec4f dir_b = axis_arrow_segments[bi].b - axis_arrow_segments[bi].a;
			const Vec4f unit_a = dir_a * (1.f / dir_a.length());
			const Vec4f unit_b = dir_b * (1.f / dir_b.length());
			Vec4f normal = crossProduct(unit_a, unit_b);
			if(dot(normal, cam_pos - gizmo_centre) < 0.f)
				normal = normal * -1.f;

			// Center fixed; corner shifts so plane scales from its center.
			const Vec4f center = gizmo_centre + unit_a * (outer_offset + base_size * 0.5f) + unit_b * (outer_offset + base_size * 0.5f);
			const Vec4f corner = center - unit_a * (size * 0.5f) - unit_b * (size * 0.5f);
			Matrix4f m;
			m.setColumn(0, unit_a * size);
			m.setColumn(1, unit_b * size);
			m.setColumn(2, normal);
			m.setColumn(3, corner);
			translate_plane_objects[i]->ob_to_world_matrix = m;
			translate_plane_objects[i]->materials[0].alpha = alpha;
			engine->updateObjectTransformData(*translate_plane_objects[i]);
			engine->objectMaterialsUpdated(*translate_plane_objects[i]);
		}
	}

	//----------------------- Update center scale cube (morphs into hovered plane shape, animated) -----------------------
	{
		static const int plane_axes[3][2] = {{1,2},{0,2},{0,1}};
		const float cube_side = arrow_len * 0.06f;

		// Cube/slab axes toward camera — same sign convention as arrow shafts.
		const float sx = cam_to_gizmo[0] > 0 ? -1.f : 1.f;
		const float sy = cam_to_gizmo[1] > 0 ? -1.f : 1.f;
		const float sz = cam_to_gizmo[2] > 0 ? -1.f : 1.f;

		// Compute target matrix for the current hover state.
		Matrix4f tgt_matrix;
		if(hovered_scale_plane >= 0)
		{
			const int i = hovered_scale_plane;
			// Unit vectors along each world axis (direction away from camera, same as shafts).
			const Vec4f unit_axes[3] = {
				normalise(axis_arrow_segments[0].b - axis_arrow_segments[0].a),
				normalise(axis_arrow_segments[1].b - axis_arrow_segments[1].a),
				normalise(axis_arrow_segments[2].b - axis_arrow_segments[2].a),
			};
			Vec4f normal = crossProduct(unit_axes[plane_axes[i][0]], unit_axes[plane_axes[i][1]]);
			if(dot(normal, cam_pos - gizmo_centre) < 0.f)
				normal = normal * -1.f;
			const float slab_size = arrow_len * 0.156f * 1.25f;
			const float thickness = cube_side * 0.18f;
			// Column i = thin (normal) axis; other columns = slab axes in their natural world-axis slots.
			// This aligns with the cube matrix (col0=X, col1=Y, col2=Z) so lerp produces a clean squish.
			for(int c = 0; c < 3; ++c)
				tgt_matrix.setColumn(c, (c == i) ? (normal * thickness) : (unit_axes[c] * slab_size));
			tgt_matrix.setColumn(3, gizmo_centre - normal * (thickness * 0.5f));
		}
		else
		{
			const float side = (hovered_cube == 3) ? cube_side * 1.6f : cube_side;
			tgt_matrix.setColumn(0, Vec4f(sx * side, 0, 0, 0));
			tgt_matrix.setColumn(1, Vec4f(0, sy * side, 0, 0));
			tgt_matrix.setColumn(2, Vec4f(0, 0, sz * side, 0));
			tgt_matrix.setColumn(3, gizmo_centre);
		}

		// State: -1=uninit, 0=default cube, 1=white cube, 2+i=slab plane i.
		const int cur_state = (hovered_scale_plane >= 0) ? (2 + hovered_scale_plane) :
		                      (hovered_cube == 3)        ? 1 : 0;

		if(center_cube_prev_state < 0)
		{
			// First frame: snap to target, no animation.
			center_cube_src_matrix = tgt_matrix;
			center_cube_tgt_matrix = tgt_matrix;
			center_cube_anim_t     = 1.0f;
			center_cube_prev_state = cur_state;
		}
		else if(cur_state != center_cube_prev_state)
		{
			// State changed: start transition from current interpolated position.
			center_cube_src_matrix = lerpMatrix(center_cube_src_matrix, center_cube_tgt_matrix, center_cube_anim_t);
			center_cube_tgt_matrix = tgt_matrix;
			center_cube_anim_t     = 0.f;
			center_cube_prev_state = cur_state;
		}
		else
		{
			// Same state but gizmo may have moved — keep target fresh.
			center_cube_tgt_matrix = tgt_matrix;
		}

		center_cube_anim_t = myMin(1.f, center_cube_anim_t + dt / GIZMO_ANIM_DURATION);

		center_scale_cube_object->ob_to_world_matrix = lerpMatrix(center_cube_src_matrix, center_cube_tgt_matrix, cubicEaseOut(center_cube_anim_t));
		engine->updateObjectTransformData(*center_scale_cube_object);
	}

	//----------------------- Update rotation control handle arcs -----------------------
	const float arc_radius = control_scale * 0.7f;

	for(int i=0; i<3; ++i)
	{
		const Vec4f basis_a = basis_vectors[i*2];
		const Vec4f basis_b = basis_vectors[i*2 + 1];

		const Vec4f to_cam = cam_pos - gizmo_centre;
		const float to_cam_angle = safeATan2(dot(basis_b, to_cam), dot(basis_a, to_cam));

		// Position the rotation arc so its oriented towards the camera, unless the user is currently holding and dragging the arc.
		float angle = to_cam_angle;
		if(grabbed_axis >= NUM_AXIS_ARROWS)
		{
			const int grabbed_rot_axis = grabbed_axis - NUM_AXIS_ARROWS;
			if(i == grabbed_rot_axis)
				angle = grabbed_angle + grabbed_arc_angle_offset;
		}

		// Position the arc line segments used for mouse picking.
		const float start_angle = angle - arc_handle_half_angle - 0.1f;; // Extend a little so the arrow heads can be selected
		const float end_angle   = angle + arc_handle_half_angle + 0.1f;

		const size_t N = 32;
		rot_handle_lines[i].resize(N);
		for(size_t z=0; z<N; ++z)
		{
			const float theta_0 = start_angle + (end_angle - start_angle) * (float)z       / (float)N;
			const float theta_1 = start_angle + (end_angle - start_angle) * (float)(z + 1) / (float)N;

			const Vec4f p0 = gizmo_centre + basis_a * (cos(theta_0) * arc_radius) + basis_b * (sin(theta_0) * arc_radius);
			const Vec4f p1 = gizmo_centre + basis_a * (cos(theta_1) * arc_radius) + basis_b * (sin(theta_1) * arc_radius);

			rot_handle_lines[i][z] = LineSegment4f(p0, p1);
		}

		rot_handle_arc_objects[i]->ob_to_world_matrix =
			Matrix4f::translationMatrix(gizmo_centre) *
			Matrix4f::rotationMatrix(crossProduct(basis_a, basis_b), angle - arc_handle_half_angle) *
			Matrix4f(basis_a, basis_b, crossProduct(basis_a, basis_b), Vec4f(0,0,0,1)) *
			Matrix4f::uniformScaleMatrix(arc_radius);

		engine->updateObjectTransformData(*rot_handle_arc_objects[i]);
	}
}


// Returns the axis index (integer in [0, 3)) of the closest axis arrow, or the axis index of the closest rotation arc handle (integer in [3, 6))
// or -1 if no arrow or rotation arc close to pixel coords.
// Also returns world space coords of the closest point.
int TransformGizmo::mouseOverAxisArrowOrRotArc(const Vec2f& px, Vec4f& closest_ws_out)
{
	const Vec4f cam_pos      = engine->getCurrentScene()->cam_to_world.getColumn(3); // = cam_to_world * Vec4f(0,0,0,1);
	const Vec4f cam_forwards = engine->getCurrentScene()->cam_to_world.getColumn(1); // = cam_to_world * Vec4f(0,1,0,0);

	const int viewport_w = engine->getViewPortWidth(); // = gl_w

	const float max_selection_dist = 12.f;
	float closest_dist = 10000.f;
	int closest_axis = -1;

	const Planef cam_front_plane(cam_pos + cam_forwards * 0.01f, cam_forwards);
	const Vec4f ray_dir = engine->pixelToRayDirWS(px);

	// Test translation arrows.
	// Shaft radius is 0.01 * arrow_len — much thinner than the original arrow mesh.
	// Use a tight fixed threshold so hover fires only on the shaft itself.
	const float shaft_max_dist = 6.f;
	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
	{
		const LineSegment4f segment = clipLineSegmentToCameraFrontHalfSpace(axis_arrow_segments[i], cam_front_plane);

		Vec2f start_px, end_px;
		if(!worldToPixel(segment.a, engine, start_px)) continue;
		if(!worldToPixel(segment.b, engine, end_px))   continue;

		const float d = pointLineSegmentDist(px, start_px, end_px);

		const Vec4f closest_pt = closestPointOnLineToRay(segment, cam_pos, ray_dir);
		const float cam_dist = closest_pt.getDist(cam_pos);
		const float approx_radius_px = 0.005f * (float)viewport_w / cam_dist;
		const float use_max_dist = myMax(shaft_max_dist, approx_radius_px);

		if(d <= closest_dist && d < use_max_dist)
		{
			closest_ws_out = closest_pt;
			closest_dist = d;
			closest_axis = i;
		}
	}

	// Test rotation arc line segments.
	for(int i=0; i<3; ++i)
	{
		for(size_t z=0; z<rot_handle_lines[i].size(); ++z)
		{
			const LineSegment4f& segment = rot_handle_lines[i][z];

			Vec2f start_px, end_px;
			if(!worldToPixel(segment.a, engine, start_px)) continue;
			if(!worldToPixel(segment.b, engine, end_px))   continue;

			const float d = pointLineSegmentDist(px, start_px, end_px);

			const Vec4f closest_pt = closestPointOnLineToRay(segment, cam_pos, ray_dir);
			const float cam_dist = closest_pt.getDist(cam_pos);
			const float approx_radius_px = 0.02f * (float)viewport_w / cam_dist;
			const float use_max_dist = myMax(max_selection_dist, approx_radius_px);

			if(d <= closest_dist && d < use_max_dist)
			{
				closest_ws_out = closest_pt;
				closest_dist = d;
				closest_axis = NUM_AXIS_ARROWS + i;
			}
		}
	}

	return closest_axis;
}


bool TransformGizmo::mousePressed(const Vec2f& px, const Vec4f& ob_pos_ws, GizmoDelegateInterface* delegate)
{
	// Center scale cube has highest grab priority (matches hover priority in updateMouseoverHighlight).
	if(hovered_cube == 3)
	{
		grabbed_axis           = GRABBED_CENTER_SCALE;
		ob_origin_at_grab      = ob_pos_ws;
		grabbed_scale_mouse_px = px;
		delegate->onGrabStart(false);
		return true;
	}

	// Two-axis scale plane (only active when center was previously engaged).
	if(hovered_scale_plane >= 0 && center_scale_engaged)
	{
		grabbed_axis           = GRABBED_SCALE_PLANE_BASE + hovered_scale_plane;
		ob_origin_at_grab      = ob_pos_ws;
		grabbed_scale_mouse_px = px;
		delegate->onGrabStart(false);
		return true;
	}

	const int axis = mouseOverAxisArrowOrRotArc(px, grabbed_point_ws);
	if(axis < 0)
		return false;

	grabbed_axis        = axis;
	ob_origin_at_grab   = ob_pos_ws;

	delegate->onGrabStart(grabbed_axis >= NUM_AXIS_ARROWS);

	if(grabbed_axis >= NUM_AXIS_ARROWS)
	{
		// Compute the initial angle on the rotation plane so we can track deltas.
		const Vec4f cam_pos = engine->getCurrentScene()->cam_to_world.getColumn(3); // = cam_to_world * Vec4f(0,0,0,1);
		const int rot_axis = grabbed_axis - NUM_AXIS_ARROWS;
		const Vec4f basis_a = basis_vectors[rot_axis*2];
		const Vec4f basis_b = basis_vectors[rot_axis*2 + 1];
		const Vec4f arc_centre = ob_pos_ws;

		const Vec4f dir = engine->pixelToRayDirWS(px);
		const Planef plane(arc_centre, crossProduct(basis_a, basis_b));
		const float t = plane.rayIntersect(cam_pos, dir);
		const Vec4f plane_p = cam_pos + dir * t;

		const float angle = safeATan2(dot(plane_p - arc_centre, basis_b), dot(plane_p - arc_centre, basis_a));

		const Vec4f to_cam = cam_pos - arc_centre;
		const float to_cam_angle = safeATan2(dot(basis_b, to_cam), dot(basis_a, to_cam));

		grabbed_angle = original_grabbed_angle = angle;
		grabbed_arc_angle_offset = to_cam_angle - original_grabbed_angle;
	}

	return true;
}


bool TransformGizmo::mouseMoved(const Vec2f& px, const Vec4f& ob_pos_ws, GizmoDelegateInterface* delegate, float grid_spacing)
{
	if(grabbed_axis < 0)
		return false;

	const Vec4f cam_pos      = engine->getCurrentScene()->cam_to_world.getColumn(3); // = cam_to_world * Vec4f(0,0,0,1);
	const Vec4f cam_forwards = engine->getCurrentScene()->cam_to_world.getColumn(1); // = cam_to_world * Vec4f(0,1,0,0);

	if(grabbed_axis < NUM_AXIS_ARROWS)
	{
		// Translation drag: project mouse onto the grabbed world-space axis line.
		const float MAX_MOVE_DIST = 100.f;
		const Vec4f line_dir = normalise(axis_arrow_segments[grabbed_axis].b - axis_arrow_segments[grabbed_axis].a);
		Vec4f use_line_start = axis_arrow_segments[grabbed_axis].a - line_dir * MAX_MOVE_DIST;
		Vec4f use_line_end   = axis_arrow_segments[grabbed_axis].a + line_dir * MAX_MOVE_DIST;

		const Planef clip_plane(cam_pos + cam_forwards * 0.1f, cam_forwards * -1.f);
		if(!clipLineToPlaneBackHalfSpace(clip_plane, use_line_start, use_line_end))
			return true;

		Vec2f start_px, end_px;
		const bool sv = worldToPixel(use_line_start, engine, start_px);
		const bool ev = worldToPixel(use_line_end,   engine, end_px);

		if(sv && ev)
		{
			const Vec2f closest_px = closestPointOnLineSegment(px, start_px, end_px);
			Vec4f new_p = pointOnLineWorldSpace(axis_arrow_segments[grabbed_axis].a, axis_arrow_segments[grabbed_axis].b, closest_px);

			Vec4f delta_p = new_p - grabbed_point_ws;
			Vec4f tentative = ob_origin_at_grab + delta_p;

			if(tentative.getDist(ob_origin_at_grab) > MAX_MOVE_DIST)
				tentative = ob_origin_at_grab + (tentative - ob_origin_at_grab) * MAX_MOVE_DIST / (tentative - ob_origin_at_grab).length();

			// Grid snap (grabbed axis only).
			if(grid_spacing > 1.0e-5f)
				tentative[grabbed_axis] = (float)Maths::roundToMultipleFloating((double)tentative[grabbed_axis], (double)grid_spacing);

			updateGizmoDrawTransform(/*new_gizmo_centre=*/tentative);

			const Vec4f total_translation = tentative - ob_origin_at_grab;
			delegate->onTranslationDrag(total_translation, tentative);
		}
	}
	else if(grabbed_axis == GRABBED_CENTER_SCALE)
	{
		// Uniform scale drag: right/up = larger, left/down = smaller.
		// Screen Y increases downward, so negate dy.
		// delta_scale is incremental (relative to last frame), matching the rotation delta_angle pattern.
		const float dx = px.x - grabbed_scale_mouse_px.x;
		const float dy = -(px.y - grabbed_scale_mouse_px.y);
		const float delta_scale = std::exp((dx + dy) * 0.007f);
		grabbed_scale_mouse_px = px;
		updateGizmoDrawTransform(ob_origin_at_grab);
		delegate->onUniformScaleDrag(delta_scale);
	}
	else if(grabbed_axis >= GRABBED_SCALE_PLANE_BASE)
	{
		// Two-axis scale drag: same direction convention as uniform scale.
		const int   plane_index = grabbed_axis - GRABBED_SCALE_PLANE_BASE;
		const float dx          = px.x - grabbed_scale_mouse_px.x;
		const float dy          = -(px.y - grabbed_scale_mouse_px.y);
		const float delta_scale = std::exp((dx + dy) * 0.007f);
		grabbed_scale_mouse_px  = px;
		updateGizmoDrawTransform(ob_origin_at_grab);
		delegate->onTwoAxisScaleDrag(plane_index, delta_scale);
	}
	else
	{
		// Rotation drag: intersect mouse ray with the rotation plane.
		const int rot_axis = grabbed_axis - NUM_AXIS_ARROWS;
		const Vec4f basis_a = basis_vectors[rot_axis*2];
		const Vec4f basis_b = basis_vectors[rot_axis*2 + 1];
		const Vec4f arc_centre = ob_origin_at_grab;

		const Vec4f dir = engine->pixelToRayDirWS(px);
		const Planef plane(arc_centre, crossProduct(basis_a, basis_b));
		const float t = plane.rayIntersect(cam_pos, dir);
		const Vec4f plane_p = cam_pos + dir * t;

		const float angle = safeATan2(dot(plane_p - arc_centre, basis_b), dot(plane_p - arc_centre, basis_a));
		const float delta = angle - grabbed_angle;

		updateGizmoDrawTransform(/*new_gizmo_centre=*/ob_origin_at_grab);

		const float total_angle_change = angle - original_grabbed_angle;
		delegate->onRotationDrag(crossProduct(basis_a, basis_b), total_angle_change, delta);

		grabbed_angle = angle;
	}

	return true;
}


bool TransformGizmo::mouseReleased(GizmoDelegateInterface* delegate)
{
	if(grabbed_axis < 0)
		return false;

	grabbed_axis = -1;
	delegate->onGrabEnd();
	return true;
}


// Returns [0,3) for plane handle hover (0=YZ, 1=XZ, 2=XY), -1 for none.
// Projects the quad corners to screen and does a parallelogram point-in test.
int TransformGizmo::mouseOverScalePlaneHandle(const Vec2f& px) const
{
	static const int plane_axes[3][2] = {{1,2},{0,2},{0,1}};
	const float seg_len   = (axis_arrow_segments[0].b - axis_arrow_segments[0].a).length();
	const float arrow_len = seg_len / 0.80f;
	const float plane_size = arrow_len * 0.156f * 1.69f;
	const Vec4f gizmo_centre = axis_arrow_segments[0].a;

	for(int i = 0; i < NUM_PLANES; ++i)
	{
		const int ai = plane_axes[i][0], bi = plane_axes[i][1];
		const Vec4f dir_a = axis_arrow_segments[ai].b - axis_arrow_segments[ai].a;
		const Vec4f dir_b = axis_arrow_segments[bi].b - axis_arrow_segments[bi].a;
		const Vec4f unit_a = dir_a * (1.f / dir_a.length());
		const Vec4f unit_b = dir_b * (1.f / dir_b.length());

		const Vec4f p00 = gizmo_centre;
		const Vec4f p10 = p00 + unit_a * plane_size;
		const Vec4f p01 = p00 + unit_b * plane_size;

		Vec2f s00, s10, s01;
		if(!worldToPixel(p00, engine, s00)) continue;
		if(!worldToPixel(p10, engine, s10)) continue;
		if(!worldToPixel(p01, engine, s01)) continue;

		// Parallelogram hit test using barycentric-style coordinates.
		const float ux = s10[0]-s00[0], uy = s10[1]-s00[1];
		const float vx = s01[0]-s00[0], vy = s01[1]-s00[1];
		const float wx = px[0] -s00[0], wy = px[1] -s00[1];
		const float uu = ux*ux + uy*uy, uv = ux*vx + uy*vy, vv = vx*vx + vy*vy;
		const float uw = ux*wx + uy*wy, vw = vx*wx + vy*wy;
		const float denom = uu*vv - uv*uv;
		if(std::abs(denom) < 1e-6f) continue;
		const float s = (vv*uw - uv*vw) / denom;
		const float t = (uu*vw - uv*uw) / denom;
		if(s >= 0.f && s <= 1.f && t >= 0.f && t <= 1.f)
			return i;
	}
	return -1;
}


int TransformGizmo::mouseOverTranslatePlaneHandle(const Vec2f& px) const
{
	static const int plane_axes[3][2] = {{1,2},{0,2},{0,1}};
	const float seg_len      = (axis_arrow_segments[0].b - axis_arrow_segments[0].a).length();
	const float arrow_len    = seg_len / 0.80f;
	const float plane_size   = arrow_len * 0.156f * 0.60f; // same 40% reduction as in draw
	const float outer_offset = arrow_len * 0.42f;
	const Vec4f gizmo_centre = axis_arrow_segments[0].a;

	for(int i = 0; i < NUM_PLANES; ++i)
	{
		const int ai = plane_axes[i][0], bi = plane_axes[i][1];
		const Vec4f dir_a = axis_arrow_segments[ai].b - axis_arrow_segments[ai].a;
		const Vec4f dir_b = axis_arrow_segments[bi].b - axis_arrow_segments[bi].a;
		const Vec4f unit_a = dir_a * (1.f / dir_a.length());
		const Vec4f unit_b = dir_b * (1.f / dir_b.length());

		const Vec4f p00 = gizmo_centre + unit_a * outer_offset + unit_b * outer_offset;
		const Vec4f p10 = p00 + unit_a * plane_size;
		const Vec4f p01 = p00 + unit_b * plane_size;

		Vec2f s00, s10, s01;
		if(!worldToPixel(p00, engine, s00)) continue;
		if(!worldToPixel(p10, engine, s10)) continue;
		if(!worldToPixel(p01, engine, s01)) continue;

		const float ux = s10[0]-s00[0], uy = s10[1]-s00[1];
		const float vx = s01[0]-s00[0], vy = s01[1]-s00[1];
		const float wx = px[0] -s00[0], wy = px[1] -s00[1];
		const float uu = ux*ux + uy*uy, uv = ux*vx + uy*vy, vv = vx*vx + vy*vy;
		const float uw = ux*wx + uy*wy, vw = vx*wx + vy*wy;
		const float denom = uu*vv - uv*uv;
		if(std::abs(denom) < 1e-6f) continue;
		const float s = (vv*uw - uv*vw) / denom;
		const float t = (uu*vw - uv*uw) / denom;
		if(s >= 0.f && s <= 1.f && t >= 0.f && t <= 1.f)
			return i;
	}
	return -1;
}


// Returns [0,3) for axis cube tip hover, 3 for center cube hover, -1 for none.
// Uses screen-space projected distance with a fixed pixel threshold.
int TransformGizmo::mouseOverCubeHandle(const Vec2f& px) const
{
	const float threshold_px = 14.f;
	float closest_dist = threshold_px;
	int closest = -1;

	// Axis cube tip centers: 83% along the arrow segment (cx0=0.80 + cs=0.03).
	for(int i = 0; i < NUM_AXIS_ARROWS; ++i)
	{
		// Segment ends at shaft end (0.80*arrow_len); cube center is at 0.83*arrow_len = 0.83/0.80 of segment.
		const Vec4f center_ws = axis_arrow_segments[i].a + (axis_arrow_segments[i].b - axis_arrow_segments[i].a) * (0.83f / 0.80f);
		Vec2f center_px;
		if(!worldToPixel(center_ws, engine, center_px)) continue;
		const float d = (center_px - px).length();
		if(d < closest_dist)
		{
			closest_dist = d;
			closest = i;
		}
	}

	// Center cube — origin == gizmo_centre == axis_arrow_segments[*].a.
	{
		const Vec4f center_ws = axis_arrow_segments[0].a;
		Vec2f center_px;
		if(worldToPixel(center_ws, engine, center_px))
		{
			const float d = (center_px - px).length();
			if(d < closest_dist)
			{
				closest_dist = d;
				closest = 3;
			}
		}
	}

	return closest;
}


void TransformGizmo::updateMouseoverHighlight(const Vec2f& px)
{
	// Reset all to default colours.
	for(int i=0; i<NUM_AXIS_ARROWS; ++i)
	{
		axis_arrow_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i % 3]);
		engine->objectMaterialsUpdated(*axis_arrow_objects[i]);
		axis_scale_cube_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i % 3]);
		engine->objectMaterialsUpdated(*axis_scale_cube_objects[i]);
	}
	for(int i=0; i<3; ++i)
	{
		rot_handle_arc_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
		engine->objectMaterialsUpdated(*rot_handle_arc_objects[i]);
	}

	// Reset center scale cube to default grey and translate planes to default colours.
	center_scale_cube_object->materials[0].albedo_linear_rgb = toLinearSRGB(Colour3f(0.55f, 0.55f, 0.55f));
	engine->objectMaterialsUpdated(*center_scale_cube_object);
	for(int i = 0; i < NUM_PLANES; ++i)
	{
		translate_plane_objects[i]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_default_cols[i]);
		engine->objectMaterialsUpdated(*translate_plane_objects[i]);
	}

	// Priority: virtual-center-zone > cube tips > outer planes > scale planes (only if center was engaged) > shafts/arcs.
	// Exception: when center_scale_engaged is already true, scale-plane slabs take priority over the centre-cube hit
	// zone so the 14-px screen-space threshold doesn't steal clicks from slabs that originate at the gizmo origin.
	hovered_cube          = -1;
	hovered_scale_plane   = -1;
	hovered_translate_plane = -1;

	if(center_scale_engaged)
	{
		hovered_scale_plane = mouseOverScalePlaneHandle(px);
		if(hovered_scale_plane >= 0)
		{
			hovered_axis = -1;
			center_scale_cube_object->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[hovered_scale_plane]);
			engine->objectMaterialsUpdated(*center_scale_cube_object);
			return;
		}
	}

	hovered_cube = mouseOverCubeHandle(px);

	if(hovered_cube == 3)
	{
		// Central zone: engage and show white cube.
		center_scale_engaged = true;
		hovered_axis = -1;
		center_scale_cube_object->materials[0].albedo_linear_rgb = toLinearSRGB(Colour3f(1.f, 1.f, 1.f));
		engine->objectMaterialsUpdated(*center_scale_cube_object);
		return;
	}

	if(hovered_cube >= 0) // axis cube tip
	{
		center_scale_engaged = false;
		hovered_axis = -1;
		axis_scale_cube_objects[hovered_cube]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[hovered_cube]);
		engine->objectMaterialsUpdated(*axis_scale_cube_objects[hovered_cube]);
		return;
	}

	hovered_translate_plane = mouseOverTranslatePlaneHandle(px);
	if(hovered_translate_plane >= 0)
	{
		center_scale_engaged = false;
		hovered_axis = -1;
		translate_plane_objects[hovered_translate_plane]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[hovered_translate_plane]);
		engine->objectMaterialsUpdated(*translate_plane_objects[hovered_translate_plane]);
		return;
	}

	// Scale plane morph only activates if the center cube was hovered first.
	hovered_scale_plane = mouseOverScalePlaneHandle(px);
	if(hovered_scale_plane >= 0 && center_scale_engaged)
	{
		hovered_axis = -1;
		center_scale_cube_object->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[hovered_scale_plane]);
		engine->objectMaterialsUpdated(*center_scale_cube_object);
		return;
	}

	// Mouse is not over center or any scale plane — disengage.
	if(hovered_scale_plane >= 0)
		hovered_scale_plane = -1; // was in zone but not engaged — suppress
	center_scale_engaged = false;

	Vec4f dummy;
	const int axis = mouseOverAxisArrowOrRotArc(px, dummy);
	hovered_axis = axis;

	if(axis < 0)
		return;

	if(axis < NUM_AXIS_ARROWS)
	{
		axis_arrow_objects[axis]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[axis]);
		engine->objectMaterialsUpdated(*axis_arrow_objects[axis]);
	}
	else
	{
		const int rot_axis = axis - NUM_AXIS_ARROWS;
		rot_handle_arc_objects[rot_axis]->materials[0].albedo_linear_rgb = toLinearSRGB(axis_arrows_mouseover_cols[rot_axis]);
		engine->objectMaterialsUpdated(*rot_handle_arc_objects[rot_axis]);
	}
}
