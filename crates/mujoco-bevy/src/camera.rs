//! The Bevy cameras from `MuJoCo`'s (the view's free camera, and the model's
//! own), and the mouse driving `MuJoCo`'s camera and perturb as `MuJoCo`'s
//! own `simulate` does.

use bevy::camera::ScalingMode;
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use mujoco_rs::prelude::*;

use crate::pick::Selection;
use crate::{MjView, quat_from_mat};

/// A press and release closer than this, in logical pixels, is a click (a
/// pick), not a drag.
const CLICK_SLOP: f32 = 4.0;

/// Pixels of a touchpad's scroll that count as one wheel notch.
const PIXELS_PER_LINE: f32 = 40.0;

/// Marks the camera drawn from [`MjView::camera`]. One per app: with more,
/// Bevy skips the systems that need it.
#[derive(Component, Default)]
pub struct MjCamera;

/// Keeps a Bevy camera on the model's camera of this id (an MJCF `<camera>`),
/// with its field of view, as the passive state has it.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCamera(pub usize);

/// The keys held with a click.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl Modifiers {
    /// Either key of each pair counts.
    #[must_use]
    pub fn held(keys: &ButtonInput<KeyCode>) -> Self {
        Self {
            shift: keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
            ctrl: keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
            alt: keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]),
        }
    }

    #[must_use]
    pub const fn any(self) -> bool {
        self.shift || self.ctrl || self.alt
    }
}

/// What the host tells the plugin about the pointer each frame.
#[derive(Resource, Default)]
pub struct ViewInput {
    /// The pointer belongs to the host's UI (egui wants it): no camera
    /// moves, picks or perturbs start.
    pub blocked: bool,
}

/// A click to pick against the next scene update.
#[derive(Clone, Copy, Debug)]
pub struct Click {
    /// Fractions of the viewport from its left and bottom edges.
    pub at: Vec2,
    /// Viewport width over height.
    pub aspect: f64,
    /// Only a click with none selects; any click is reported as `Picked`.
    pub modifiers: Modifiers,
}

/// The click waiting for this frame's pick. A host may set it too, to pick
/// without the mouse.
#[derive(Resource, Default)]
pub struct PendingClick(pub Option<Click>);

/// Sets the Bevy camera from the scene's: position, forward, up, frustum.
pub fn sync_camera(
    view: Res<MjView>,
    camera: Single<(&mut Transform, &mut Projection), With<MjCamera>>,
) {
    let (mut transform, mut projection) = camera.into_inner();
    let [left, right] = view.scene.camera();
    let gl = left.average_camera(right);
    let forward = Vec3::from_array(gl.forward);
    if forward.length_squared() < 1e-12 || gl.frustum_near <= 0.0 {
        return;
    }
    transform.set_if_neq(
        Transform::from_translation(Vec3::from_array(gl.pos))
            .looking_to(forward, Vec3::from_array(gl.up)),
    );
    let (near, far) = (gl.frustum_near, gl.frustum_far.max(gl.frustum_near * 2.0));
    let wanted = if gl.orthographic == 0 {
        Projection::Perspective(PerspectiveProjection {
            fov: (gl.frustum_top / near).atan() - (gl.frustum_bottom / near).atan(),
            near,
            far,
            ..default()
        })
    } else {
        // `MuJoCo`'s orthographic frustum is in metres at any distance.
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: gl.frustum_top - gl.frustum_bottom,
            },
            near,
            far,
            ..OrthographicProjection::default_3d()
        })
    };
    if !same_frustum(&projection, &wanted) {
        *projection = wanted;
    }
}

/// Each [`ModelCamera`] at its camera's pose: `MuJoCo`'s cameras look down
/// their own -Z with +Y up, as Bevy's do, so the frame is used as it is.
pub fn sync_model_cameras(
    view: Res<MjView>,
    mut cameras: Query<(&ModelCamera, &mut Transform, &mut Projection)>,
) {
    let model = view.model();
    let ncam = usize::try_from(model.ncam()).unwrap_or(0);
    #[allow(clippy::cast_possible_truncation)]
    let extent = model.stat().extent as f32;
    let (near, far) = (
        model.vis().map.znear * extent,
        model.vis().map.zfar * extent,
    );
    for (&ModelCamera(id), mut transform, mut projection) in &mut cameras {
        if id >= ncam {
            continue;
        }
        #[allow(clippy::cast_possible_truncation)]
        let (pos, mat, fovy) = (
            view.data.cam_xpos()[id].map(|v| v as f32),
            view.data.cam_xmat()[id].map(|v| v as f32),
            model.cam_fovy()[id] as f32,
        );
        transform.set_if_neq(Transform {
            translation: Vec3::from_array(pos),
            rotation: quat_from_mat(&mat),
            scale: Vec3::ONE,
        });
        let wanted = if model.cam_projection()[id] == MjtProjection::mjPROJ_ORTHOGRAPHIC {
            // An orthographic camera's fovy is the view's height in metres.
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical {
                    viewport_height: fovy,
                },
                near,
                far,
                ..OrthographicProjection::default_3d()
            })
        } else {
            Projection::Perspective(PerspectiveProjection {
                fov: fovy.to_radians(),
                near,
                far,
                ..default()
            })
        };
        if !same_frustum(&projection, &wanted) {
            *projection = wanted;
        }
    }
}

/// Says once that there is more than one [`MjCamera`]: Bevy then skips the
/// systems that take it, and the view stops following without an error.
pub fn warn_many_cameras(cameras: Query<(), With<MjCamera>>, mut warned: Local<bool>) {
    let n = cameras.iter().len();
    if n > 1 && !*warned {
        warn!("{n} cameras carry MjCamera; the view follows none of them until there is one");
        *warned = true;
    }
}

/// Whether two projections differ only in what Bevy sets itself (the aspect
/// ratio, from the viewport), so an unchanged frustum leaves the camera's
/// change detection quiet.
fn same_frustum(a: &Projection, b: &Projection) -> bool {
    match (a, b) {
        (Projection::Perspective(a), Projection::Perspective(b)) => {
            a.fov.to_bits() == b.fov.to_bits()
                && a.near.to_bits() == b.near.to_bits()
                && a.far.to_bits() == b.far.to_bits()
        }
        (Projection::Orthographic(a), Projection::Orthographic(b)) => {
            let height = |p: &OrthographicProjection| match p.scaling_mode {
                ScalingMode::FixedVertical { viewport_height } => Some(viewport_height.to_bits()),
                _ => None,
            };
            height(a) == height(b)
                && a.near.to_bits() == b.near.to_bits()
                && a.far.to_bits() == b.far.to_bits()
        }
        _ => false,
    }
}

/// The pointer as the 3D view sees it this frame.
#[derive(SystemParam)]
pub struct Pointer<'w, 's> {
    buttons: Res<'w, ButtonInput<MouseButton>>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    scroll: Res<'w, AccumulatedMouseScroll>,
    input: Res<'w, ViewInput>,
    window: Single<'w, 's, &'static Window, With<PrimaryWindow>>,
    camera: Single<'w, 's, &'static Camera, With<MjCamera>>,
}

impl Pointer<'_, '_> {
    fn cursor(&self) -> Option<Vec2> {
        self.window.cursor_position()
    }

    fn viewport(&self) -> Option<Rect> {
        self.camera.logical_viewport_rect()
    }

    /// Over the 3D view and not claimed by the host's UI.
    fn over_view(&self) -> bool {
        !self.input.blocked
            && self
                .cursor()
                .zip(self.viewport())
                .is_some_and(|(c, r)| r.contains(c))
    }

    fn modifiers(&self) -> Modifiers {
        Modifiers::held(&self.keys)
    }

    /// Wheel notches scrolled this frame, up positive.
    fn notches(&self) -> f32 {
        match self.scroll.unit {
            MouseScrollUnit::Line => self.scroll.delta.y,
            MouseScrollUnit::Pixel => self.scroll.delta.y / PIXELS_PER_LINE,
        }
    }
}

/// The drag in progress.
#[derive(Default)]
pub struct Drag {
    button: Option<MouseButton>,
    last: Vec2,
    travelled: f32,
    perturbing: bool,
}

/// `MuJoCo`'s mouse action for a drag with `button`, as `simulate` maps it.
const fn action(button: MouseButton, shift: bool) -> MjtMouse {
    match (button, shift) {
        (MouseButton::Left, false) => MjtMouse::mjMOUSE_ROTATE_V,
        (MouseButton::Left, true) => MjtMouse::mjMOUSE_ROTATE_H,
        (MouseButton::Right, false) => MjtMouse::mjMOUSE_MOVE_V,
        (MouseButton::Right, true) => MjtMouse::mjMOUSE_MOVE_H,
        _ => MjtMouse::mjMOUSE_ZOOM,
    }
}

/// The mouse over the 3D view, as `simulate` maps it.
///
/// Left drag rotates, right drag pans, middle drag and the wheel zoom, all
/// through `mjv_moveCamera`, with Shift for the horizontal variants. Ctrl
/// turns a drag on the selected body into a perturb (left rotates, right
/// pulls). A click without a drag asks for a pick.
pub fn mouse(
    mut view: ResMut<MjView>,
    pointer: Pointer,
    selection: Res<Selection>,
    mut pending: ResMut<PendingClick>,
    mut drag: Local<Drag>,
) {
    let Some(rect) = pointer.viewport() else {
        return;
    };
    let height = f64::from(rect.height().max(1.0));

    if drag.button.is_none()
        && pointer.over_view()
        && let Some(at) = pointer.cursor()
        && let Some(button) = [MouseButton::Left, MouseButton::Right, MouseButton::Middle]
            .into_iter()
            .find(|b| pointer.buttons.just_pressed(*b))
    {
        let selected = selection
            .0
            .as_ref()
            .and_then(|p| p.body)
            .is_some_and(|b| b > 0);
        // Ctrl alone: Ctrl+Shift and Shift+Alt clicks are left to the host.
        let held = pointer.modifiers();
        let only_ctrl = held.ctrl && !held.shift && !held.alt;
        let perturbing =
            only_ctrl && button != MouseButton::Middle && selected && view.scene_ready();
        if perturbing {
            let kind = if button == MouseButton::Left {
                MjtPertBit::mjPERT_ROTATE
            } else {
                MjtPertBit::mjPERT_TRANSLATE
            };
            let view = &mut *view;
            view.perturb.start(kind, &mut view.data, &view.scene);
        }
        *drag = Drag {
            button: Some(button),
            last: at,
            travelled: 0.0,
            perturbing,
        };
    }

    if let Some(button) = drag.button {
        if let Some(at) = pointer.cursor() {
            let delta = at - drag.last;
            drag.last = at;
            drag.travelled += delta.length();
            if delta != Vec2::ZERO && (drag.travelled > CLICK_SLOP || drag.perturbing) {
                let action = action(button, pointer.modifiers().shift);
                let (dx, dy) = (f64::from(delta.x) / height, f64::from(delta.y) / height);
                let view = &mut *view;
                if drag.perturbing {
                    view.perturb.move_(&view.data, action, dx, dy, &view.scene);
                } else {
                    view.camera.move_(action, view.data.model(), dx, dy);
                }
            }
        }
        if pointer.buttons.just_released(button) {
            if drag.perturbing {
                view.perturb.active = 0;
            } else if button == MouseButton::Left
                && drag.travelled <= CLICK_SLOP
                && let Some(at) = pointer.cursor()
            {
                pending.0 = Some(Click {
                    at: Vec2::new(
                        (at.x - rect.min.x) / rect.width(),
                        1.0 - (at.y - rect.min.y) / rect.height(),
                    ),
                    aspect: f64::from(rect.width()) / height,
                    modifiers: pointer.modifiers(),
                });
            }
            *drag = Drag::default();
        }
    }

    let notches = pointer.notches();
    if notches != 0.0 && pointer.over_view() {
        let view = &mut *view;
        view.camera.move_(
            MjtMouse::mjMOUSE_ZOOM,
            view.data.model(),
            0.0,
            -0.05 * f64::from(notches),
        );
    }
}
