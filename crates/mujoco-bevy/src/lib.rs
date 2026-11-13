//! A compiled `MuJoCo` model drawn in Bevy from `MuJoCo`'s own geom list.
//!
//! The host owns the simulation. Each frame it copies the state it wants drawn
//! into [`MjView::data`], a passive `MjData` (`MjData::copy_visual_to`), and may
//! fill [`MjView::user_scene`] with overlay geoms. [`MujocoPlugin`] then runs
//! `mjv_updateScene` and keeps one Bevy entity per geom in the list, so joint
//! and camera markers, frames, contacts and perturb handles come for free. The
//! camera, picking and perturbing go through `MuJoCo`'s own functions, so a pick
//! uses the camera the picture was drawn with.
//!
//! `MuJoCo`'s world is used as it is: Z up, metres. Nothing is swapped.
//!
//! The host spawns one `Camera3d` with [`MjCamera`] on it, sets its viewport,
//! and says through [`ViewInput`] when the pointer belongs to its UI.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::transform::TransformSystems;
use mujoco_rs::prelude::*;

mod camera;
mod lights;
mod pick;
mod scene;

pub use camera::{Click, MjCamera, PendingClick, ViewInput};
pub use lights::{LightSettings, MjLight};
pub use mujoco_bevy_assets::ModelAssets;
pub use pick::{Pick, Picked, Selection, pick};
pub use scene::{GeomSlot, GeomStats, Label, Labels};

/// Room for the geoms of a busy scene: every part, contacts and decor.
pub const DEFAULT_MAX_GEOM: usize = 20_000;

/// What is drawn and how it is looked at: the passive copy of the state, the
/// scenes built from it, and `MuJoCo`'s own option, perturb and camera.
#[derive(Resource)]
pub struct MjView {
    pub(crate) model: Arc<MjModel>,
    /// The state to draw. The host copies into it each frame; the plugin only
    /// reads it, except that `mjv_updateScene` and `mjv_select` take it `mut`.
    pub data: MjData<Arc<MjModel>>,
    /// Rebuilt from [`Self::data`] every frame.
    pub scene: MjvScene,
    /// The host's overlays, drawn after [`Self::scene`]. The plugin never
    /// clears it: the host does, before writing a frame's geoms.
    pub user_scene: MjvScene,
    pub option: MjvOption,
    /// The selected body and the drag on it. The host applies it to the
    /// simulated data (`MjvPerturb::apply`), not to [`Self::data`].
    pub perturb: MjvPerturb,
    pub camera: MjvCamera,
    generation: u64,
    scene_ready: bool,
}

impl MjView {
    /// A view of `model` with `MuJoCo`'s default options and a free camera.
    #[must_use]
    pub fn new(model: Arc<MjModel>) -> Self {
        let mut data = MjData::new(Arc::clone(&model));
        data.forward();
        Self {
            scene: MjvScene::new(&model, DEFAULT_MAX_GEOM),
            user_scene: MjvScene::new(&model, DEFAULT_MAX_GEOM / 4),
            camera: MjvCamera::new_free(&model),
            option: MjvOption::default(),
            perturb: MjvPerturb::default(),
            data,
            model,
            generation: 0,
            scene_ready: false,
        }
    }

    /// Replaces the model, as the app does when it recompiles. Every Bevy
    /// asset made for the old model is dropped on the next frame. The camera
    /// keeps where it looks from; a tracked body that no longer exists, and
    /// any selection, are let go.
    pub fn set_model(&mut self, model: Arc<MjModel>) {
        let camera = self.camera.clone();
        let option = self.option.clone();
        *self = Self {
            generation: self.generation + 1,
            ..Self::new(model)
        };
        self.camera = camera;
        self.option = option;
        let tracked_gone = self.camera.type_ == MjtCamera::mjCAMERA_TRACKING as i32
            && !(0..self.model.nbody()).contains(&i64::from(self.camera.trackbodyid));
        let fixed_gone = self.camera.type_ == MjtCamera::mjCAMERA_FIXED as i32
            && !(0..self.model.ncam()).contains(&i64::from(self.camera.fixedcamid));
        if tracked_gone || fixed_gone {
            self.camera.free();
        }
    }

    #[must_use]
    pub const fn model(&self) -> &Arc<MjModel> {
        &self.model
    }

    /// Bumped by every [`Self::set_model`], so caches keyed on the model know
    /// when to rebuild.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether [`Self::scene`] has been built at least once. `MuJoCo` ends the
    /// process if a pick or a perturb starts on a scene whose cameras were
    /// never filled.
    #[must_use]
    pub const fn scene_ready(&self) -> bool {
        self.scene_ready
    }

    /// Rebuilds [`Self::scene`] from [`Self::data`] with the current option,
    /// perturb and camera.
    pub fn update_scene(&mut self) {
        self.scene.update(
            &mut self.data,
            &self.option,
            &self.perturb,
            &mut self.camera,
        );
        self.scene_ready = true;
    }
}

/// Where the plugin's work sits in `PostUpdate`, so a host can order its own
/// systems against it: copy state and write overlays before [`Self::Scene`],
/// read [`Labels`] and [`Selection`] after [`Self::Draw`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MjSystems {
    /// `mjv_updateScene`, then the pick asked for this frame.
    Scene,
    /// Entities, lights, gizmos, labels and the camera from the scene.
    Draw,
}

/// Draws the [`MjView`] resource. Insert the resource before the first
/// frame; frames without it draw nothing.
pub struct MujocoPlugin;

impl Plugin for MujocoPlugin {
    fn build(&self, app: &mut App) {
        let has_view = resource_exists::<MjView>;
        app.add_plugins(mujoco_bevy_assets::MujocoAssetsPlugin)
            .init_resource::<Labels>()
            .init_resource::<GeomStats>()
            .init_resource::<LightSettings>()
            .init_resource::<ViewInput>()
            .init_resource::<PendingClick>()
            .init_resource::<Selection>()
            .add_message::<Picked>()
            .configure_sets(
                PostUpdate,
                (MjSystems::Scene, MjSystems::Draw)
                    .chain()
                    .run_if(has_view)
                    .before(TransformSystems::Propagate),
            )
            .add_systems(Update, camera::mouse.run_if(has_view))
            .add_systems(
                PostUpdate,
                (
                    (pick::forget_old_model, update_scene, pick::pick_clicked)
                        .chain()
                        .in_set(MjSystems::Scene),
                    (
                        (
                            rebuild_assets,
                            scene::despawn_on_new_model,
                            scene::sync_surfaces,
                        )
                            .chain(),
                        scene::draw_lines,
                        scene::collect_labels,
                        scene::count_geoms,
                        lights::sync_lights,
                        lights::sync_ambient,
                        camera::sync_camera,
                    )
                        .in_set(MjSystems::Draw),
                ),
            );
    }
}

fn update_scene(mut view: ResMut<MjView>) {
    view.update_scene();
}

/// Rebuilds the model's Bevy assets the frame [`MjView`] holds a new model.
fn rebuild_assets(
    view: Res<MjView>,
    mut assets: ResMut<ModelAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut built: Local<Option<u64>>,
) {
    if built.replace(view.generation()) != Some(view.generation()) {
        assets.rebuild(view.model(), &mut meshes, &mut images);
    }
}

/// `MuJoCo`'s row-major rotation as a quaternion.
#[must_use]
pub fn quat_from_mat(m: &[f32; 9]) -> Quat {
    let c = |k: usize| Vec3::new(m[k], m[3 + k], m[6 + k]);
    Quat::from_mat3(&Mat3::from_cols(c(0), c(1), c(2))).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_major_rotation_turns_x_onto_y() {
        // 90 degrees about Z, row-major: the first row is (cos, -sin, 0).
        let q = quat_from_mat(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!((q * Vec3::X - Vec3::Y).length() < 1e-6);
    }
}
