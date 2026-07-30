/*=====================================================================
TransformGizmo.h
----------------
Copyright Glare Technologies Limited 2026 -
=====================================================================*/
#pragma once


#include "../maths/Vec4f.h"
#include "../maths/vec2.h"
#include "../maths/LineSegment4f.h"
#include "../maths/Matrix4f.h"
#include "../utils/Reference.h"
#include <vector>

struct GLObject;
typedef Reference<GLObject> GLObjectRef;
class OpenGLEngine;


/*=====================================================================
IGizmoDelegate
--------------
Implement this to receive transform events from TransformGizmo and to
supply any policy the gizmo cannot determine itself.
=====================================================================*/
class GizmoDelegateInterface
{
public:
	// Called each mouseMoved tick during a translation drag.
	// total_translation is the total translation since the axis was grabbed.
	// desired_new_ob_pos is the new world-space origin of the edited object.
	virtual void onTranslationDrag(const Vec4f& total_translation, const Vec4f& desired_new_ob_pos) = 0;

	// Called each mouseMoved tick during a rotation drag.
	// total_angle_change is the total angle change since the arc was grabbed.
	// delta_angle is the angle change since onRotationDrag was last called.
	// axis is a world-space unit vector; delta_angle is a signed radian increment.
	virtual void onRotationDrag(const Vec4f& axis, float total_angle_change, float delta_angle) = 0;

	// Called when the user starts a drag (mouse down on an arrow/arc).
	// is_rotation: true if grabbing a rotation arc, false if a translation arrow.
	virtual void onGrabStart(bool is_rotation) = 0;

	// Called each mouseMoved tick during a uniform scale drag.
	// delta_scale is the multiplicative scale change since the last call (1.0 = no change).
	virtual void onUniformScaleDrag(float delta_scale) = 0;

	// Called each mouseMoved tick during a two-axis scale drag.
	// plane_index: 0=YZ (X fixed), 1=XZ (Y fixed), 2=XY (Z fixed).
	// delta_scale is the multiplicative scale change since the last call (1.0 = no change).
	virtual void onTwoAxisScaleDrag(int plane_index, float delta_scale) = 0;

	// Called each mouseMoved tick during a per-axis scale drag (axis cube tip).
	// axis_index: 0=X, 1=Y, 2=Z (the other two axes are left unchanged).
	// delta_scale is the multiplicative scale change since the last call (1.0 = no change).
	virtual void onAxisScaleDrag(int axis_index, float delta_scale) = 0;

	// Called when the user releases the mouse after a drag.
	virtual void onGrabEnd() = 0;

	virtual ~GizmoDelegateInterface() {}
};


/*=====================================================================
TransformGizmo
--------------
Renders and handles interaction for a 3-axis translation + 3-arc
rotation gizmo. Owns the OpenGL objects for the arrows and arcs.

Usage:
  Call update() each frame while enabled to reposition the handles.
  Forward mouse events; each returns true when the gizmo consumed them.
=====================================================================*/
class TransformGizmo : public ThreadSafeRefCounted
{
public:
	TransformGizmo(OpenGLEngine* engine, const Vec4f& gizmo_centre);
	~TransformGizmo();

	// Reposition the arrows and arcs based on the object origin and camera.
	// Call each frame while enabled.
	void update(const Vec4f& ob_pos_ws);

	// Forward mouse events.  Each returns true when the gizmo consumed the event.
	// ob_pos_ws: current world-space origin of the thing being edited.
	// grid_spacing: snap increment for translation (0 = no snap).
	bool mousePressed(const Vec2f& px, const Vec4f& ob_pos_ws, GizmoDelegateInterface* delegate);
	bool mouseMoved(const Vec2f& px, const Vec4f& ob_pos_ws, GizmoDelegateInterface* delegate, float grid_spacing = 0.f);
	bool mouseReleased(GizmoDelegateInterface* delegate);

	// Highlight the arrow/arc under the cursor (call from mouseMoved when not dragging).
	void updateMouseoverHighlight(const Vec2f& px);

	bool isGrabbed() const { return grabbed_axis >= 0; }

private:
	void updateGizmoDrawTransform(const Vec4f& new_gizmo_centre);
	// Returns the axis index (integer in [0, 3)) of the closest axis arrow, or the axis index of the closest rotation arc handle (integer in [3, 6))
	// or -1 if no arrow or rotation arc close to pixel coords.
	// Also returns world space coords of the closest point.
	int mouseOverAxisArrowOrRotArc(const Vec2f& px, Vec4f& closest_ws_out);

	// Projection helpers (use GizmoViewState instead of a live camera object).
	static bool worldToPixel(const Vec4f& ws_pos, OpenGLEngine* engine, Vec2f& px_out);
	Vec4f pointOnLineWorldSpace(const Vec4f& p_a_ws, const Vec4f& p_b_ws, const Vec2f& px) const;

	// Returns [0,3) for axis cube tip hover, 3 for center cube hover, -1 for none.
	int mouseOverCubeHandle(const Vec2f& px) const;

	// Returns [0,3) for inner plane handle hover (0=YZ, 1=XZ, 2=XY), -1 for none.
	int mouseOverScalePlaneHandle(const Vec2f& px) const;

	// Returns [0,3) for translate plane handle hover, -1 for none.
	int mouseOverTranslatePlaneHandle(const Vec2f& px) const;

	OpenGLEngine* engine;

	static const int NUM_AXIS_ARROWS = 3;
	LineSegment4f  axis_arrow_segments[NUM_AXIS_ARROWS];
	GLObjectRef    axis_arrow_objects[NUM_AXIS_ARROWS]; // Shaft cylinders (translate handles).
	GLObjectRef    axis_scale_cube_objects[NUM_AXIS_ARROWS];  // Per-axis scale handles (cube tips).
	std::vector<LineSegment4f> rot_handle_lines[3];
	GLObjectRef    rot_handle_arc_objects[3];

	static const int NUM_PLANES = 3;
	GLObjectRef    translate_plane_objects[NUM_PLANES]; // 2-axis translate handles, offset along both axes.
	GLObjectRef    center_scale_cube_object;            // Single cube at gizmo centre; morphs into the hovered scale-plane shape on hover.

	int   grabbed_axis;             // -1 = none, [0,3) = translation, [3,6) = rotation
	int   hovered_axis;             // -1 = none; [0,3) shaft, [3,6) arc; excludes cube/plane hits
	int   hovered_cube;             // -1 = none, [0,3) = axis cube tip, 3 = center cube
	int   hovered_scale_plane;      // -1 = none, [0,3) = inner scale plane handle (0=YZ,1=XZ,2=XY)
	int   hovered_translate_plane;  // -1 = none, [0,3) = outer translate plane handle
	bool  center_scale_engaged;     // true once the center cube has been hovered; required before scale-plane morph activates

	// Center cube animation state
	Matrix4f center_cube_src_matrix;
	Matrix4f center_cube_tgt_matrix;
	float    center_cube_anim_t;       // [0..1]
	int      center_cube_prev_state;   // -1=uninit, 0=default, 1=white, 2+i=slab i

	// Per-axis cube tip animation state
	float axis_cube_src_side[NUM_AXIS_ARROWS];
	float axis_cube_tgt_side[NUM_AXIS_ARROWS];
	float axis_cube_src_cfrac[NUM_AXIS_ARROWS];
	float axis_cube_tgt_cfrac[NUM_AXIS_ARROWS];
	float axis_cube_anim_t[NUM_AXIS_ARROWS];     // [0..1]
	int   axis_cube_prev_state[NUM_AXIS_ARROWS]; // -1=uninit, 0=default, 1=axis-hover, 2=cube-hover

	// Per-axis shaft animation state (scale factor: 1.0 default, 1.07 hovered)
	float shaft_src_scale[NUM_AXIS_ARROWS];
	float shaft_tgt_scale[NUM_AXIS_ARROWS];
	float shaft_anim_t[NUM_AXIS_ARROWS];         // [0..1]
	int   shaft_prev_state[NUM_AXIS_ARROWS];     // -1=uninit, 0=default, 1=hovered

	// Translate plane animation state (size)
	float tp_src_size[NUM_PLANES];
	float tp_tgt_size[NUM_PLANES];
	float tp_anim_t[NUM_PLANES];                 // [0..1]
	int   tp_prev_state[NUM_PLANES];             // -1=uninit, 0=default, 1=hovered

	double last_frame_time;            // steady_clock seconds; -1.0 on first frame

	Vec4f grabbed_point_ws;
	Vec4f ob_origin_at_grab;
	float grabbed_angle;
	float original_grabbed_angle;
	float grabbed_arc_angle_offset;

	Vec2f grabbed_scale_mouse_px;   // mouse position when center scale was grabbed
};
