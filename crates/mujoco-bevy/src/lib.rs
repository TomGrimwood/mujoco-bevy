//! A compiled `MuJoCo` model drawn in Bevy from `MuJoCo`'s own geom list.
//!
//! The host owns the simulation. Each frame it copies the state it wants drawn
//! into [`MjView::data`], a passive `MjData` (`MjData::copy_visual_to`), and may
//! fill [`MjView::user_scene`] with overlay geoms. [`MujocoScenePlugin`] then
//! runs `mjv_updateScene` and keeps one Bevy entity per geom in the list, so
//! joint and camera markers, frames, contacts and perturb handles come for
//! free. Picking goes through `MuJoCo`'s own ray cast, against the camera the
//! picture was drawn with.
//!
//! [`MujocoInputPlugin`] adds `simulate`'s mouse: orbit, pan, zoom, click to
//! pick, Ctrl + drag to perturb. A host with its own input leaves it out and
//! drives [`MjView::camera`] and [`PendingClick`] itself. [`MujocoPlugin`]
//! adds both.
//!
//! `MuJoCo`'s world is used as it is: Z up, metres. Nothing is swapped.
//!
//! The host spawns one `Camera3d` with [`MjCamera`] on it and sets its
//! viewport; more cameras can follow the model's own cameras through
//! [`ModelCamera`].

use bevy::asset::AssetEventSystems;
use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::transform::TransformSystems;
use mujoco_rs::prelude::*;

mod camera;
pub mod debug;
mod lights;
mod loading;
mod pick;
mod scene;
pub mod stats;

pub use camera::{Click, MjCamera, ModelCamera, Modifiers, PendingClick, ViewInput};
pub use debug::{GroupKind, MjRenderOptions, Named, VisScale};
pub use lights::{LightSettings, MjLight};
pub use loading::PipelinesPending;
pub use mujoco_bevy_assets::MeshCounts;
pub use mujoco_bevy_assets::{ModelAssets, RebuildStats};
pub use pick::{Pick, Picked, Selection, pick};
pub use scene::{DrawCounts, GeomSlot, GeomStats, Label, Labels};
pub use stats::{
    CollisionCounts, CollisionStats, MjFigures, MjStatsInput, MjStatsPlugin, MjStatsSystems,
    StatsSource,
};

/// How many geoms the scenes hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    /// The scene `mjv_updateScene` fills: every part, contacts and decor.
    pub scene: usize,
    /// The host's overlays.
    pub user: usize,
}

impl Default for Capacity {
    /// Room for a busy scene, and the classic viewer's overlay size.
    fn default() -> Self {
        Self {
            scene: 20_000,
            user: 512,
        }
    }
}

/// What is drawn and how it is looked at: the passive copy of the state, the
/// scenes built from it, and `MuJoCo`'s own option, perturb and camera.
#[derive(Resource)]
pub struct MjView {
    /// The state to draw, on the view's own copy of the model. The host
    /// copies into it each frame; the plugin only reads it, except that
    /// `mjv_updateScene` and `mjv_select` take it `mut`.
    pub data: MjData<Box<MjModel>>,
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
    /// Model geoms left undrawn, by id: ones the host draws itself (a ground
    /// drawn from its own texture over the model's stand-in geom). They still
    /// take part in picks. A new model ([`Self::set_model`]) empties it.
    pub hidden: Vec<usize>,
    capacity: Capacity,
    generation: u64,
    scene_ready: bool,
    /// Meshes rewritten in place, for their Bevy meshes to follow.
    meshes_rewritten: Vec<usize>,
    /// Decor sizes the host set ([`Self::set_vis_scale`]).
    vis_overrides: Vec<debug::VisOverride>,
}

/// What [`MjView::refresh_model`] made of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refresh {
    /// Same layout: values copied, assets and entities kept.
    InPlace,
    /// Another layout: handled as [`MjView::set_model`].
    Replaced,
}

impl MjView {
    /// A view of a copy of `model`, with `MuJoCo`'s default options, a free
    /// camera and [`Capacity::default`].
    #[must_use]
    pub fn new(model: &MjModel) -> Self {
        Self::with_capacity(model, Capacity::default())
    }

    #[must_use]
    pub fn with_capacity(model: &MjModel, capacity: Capacity) -> Self {
        let mut data = MjData::new(Box::new(model.clone()));
        data.forward();
        Self {
            scene: MjvScene::new(model, capacity.scene),
            user_scene: MjvScene::new(model, capacity.user),
            camera: MjvCamera::new_free(model),
            option: MjvOption::default(),
            perturb: MjvPerturb::default(),
            hidden: Vec::new(),
            data,
            capacity,
            generation: 0,
            scene_ready: false,
            meshes_rewritten: Vec::new(),
            vis_overrides: Vec::new(),
        }
    }

    /// Replaces the model, as the app does when it recompiles: the scenes,
    /// the entities and the assets follow on the next frame (assets whose
    /// content is unchanged are kept). The camera keeps where it looks from;
    /// a tracked body or a fixed camera that no longer exists, and any
    /// selection, are let go.
    pub fn set_model(&mut self, model: &MjModel) {
        let camera = self.camera.clone();
        let option = self.option.clone();
        let overrides = std::mem::take(&mut self.vis_overrides);
        *self = Self {
            generation: self.generation + 1,
            ..Self::with_capacity(model, self.capacity)
        };
        self.camera = camera;
        self.option = option;
        self.apply_vis_overrides(overrides);
        let tracked_gone = self.camera.type_ == MjtCamera::mjCAMERA_TRACKING as i32
            && !(0..model.nbody()).contains(&i64::from(self.camera.trackbodyid));
        let fixed_gone = self.camera.type_ == MjtCamera::mjCAMERA_FIXED as i32
            && !(0..model.ncam()).contains(&i64::from(self.camera.fixedcamid));
        if tracked_gone || fixed_gone {
            self.camera.free();
        }
    }

    /// Takes `model`'s values (colours, materials, lights, sizes) where only
    /// values changed, as after a paint edit: no rebuild, no respawn, the new
    /// colours reach the materials through the next scene. A model of another
    /// layout is a new model.
    pub fn refresh_model(&mut self, model: &MjModel) -> Refresh {
        if self.data.try_swap_model(Box::new(model.clone())).is_ok() {
            let overrides = std::mem::take(&mut self.vis_overrides);
            self.apply_vis_overrides(overrides);
            Refresh::InPlace
        } else {
            self.set_model(model);
            Refresh::Replaced
        }
    }

    /// Takes mesh `id`'s vertices, normals, texture coordinates and faces
    /// from `model`, where the app rewrote them in place (a bucket's load, a
    /// design's ground), and has its Bevy mesh rebuilt under the same handle
    /// next frame. Picks see the new shape too. Returns `false`, changing
    /// nothing, when `model` is not laid out as the view's or the mesh's
    /// counts differ.
    pub fn update_mesh(&mut self, model: &MjModel, id: usize) -> bool {
        let ours = self.data.model();
        if !ours.is_compatible_with_model(model) || id >= usize::try_from(ours.nmesh()).unwrap_or(0)
        {
            return false;
        }
        let counts = |m: &MjModel| {
            (
                m.mesh_vertnum()[id],
                m.mesh_normalnum()[id],
                m.mesh_facenum()[id],
                m.mesh_texcoordnum()[id],
            )
        };
        if counts(ours) != counts(model) {
            return false;
        }
        let at = |a: i32| usize::try_from(a).unwrap_or(0);
        let (v0, nv) = (at(model.mesh_vertadr()[id]), at(model.mesh_vertnum()[id]));
        let (n0, nn) = (
            at(model.mesh_normaladr()[id]),
            at(model.mesh_normalnum()[id]),
        );
        let (f0, nf) = (at(model.mesh_faceadr()[id]), at(model.mesh_facenum()[id]));
        let texcoords = usize::try_from(model.mesh_texcoordadr()[id])
            .ok()
            .map(|t0| (t0, at(model.mesh_texcoordnum()[id])));
        let src = model.ffi();
        // SAFETY: both models have one layout and this mesh has the same
        // counts in each, so every range below is inside both buffers; the
        // buffers belong to different models, so they do not overlap; and
        // only the mesh's own values change, no address or count.
        unsafe {
            let dst = self.data.model_mut().ffi_mut();
            std::ptr::copy_nonoverlapping(
                src.mesh_vert.add(3 * v0),
                dst.mesh_vert.add(3 * v0),
                3 * nv,
            );
            std::ptr::copy_nonoverlapping(
                src.mesh_normal.add(3 * n0),
                dst.mesh_normal.add(3 * n0),
                3 * nn,
            );
            std::ptr::copy_nonoverlapping(
                src.mesh_face.add(3 * f0),
                dst.mesh_face.add(3 * f0),
                3 * nf,
            );
            std::ptr::copy_nonoverlapping(
                src.mesh_facenormal.add(3 * f0),
                dst.mesh_facenormal.add(3 * f0),
                3 * nf,
            );
            if let Some((t0, nt)) = texcoords {
                std::ptr::copy_nonoverlapping(
                    src.mesh_texcoord.add(2 * t0),
                    dst.mesh_texcoord.add(2 * t0),
                    2 * nt,
                );
                std::ptr::copy_nonoverlapping(
                    src.mesh_facetexcoord.add(3 * f0),
                    dst.mesh_facetexcoord.add(3 * f0),
                    3 * nf,
                );
            }
        }
        self.meshes_rewritten.push(id);
        true
    }

    /// Takes `overlays`' geoms as this frame's [`Self::user_scene`], as many
    /// as it holds: for a host that draws its overlays into a scene of its
    /// own (one that exists without a view, headless too). Returns `false`,
    /// leaving the user scene empty, when `overlays` was made for a model of
    /// another layout: its geoms' ids would mean nothing here.
    pub fn copy_user_scene(&mut self, overlays: &MjvScene) -> bool {
        self.user_scene.clear_geom();
        if !self.user_scene.is_compatible_with_scene(overlays) {
            return false;
        }
        let from = overlays.ffi();
        // SAFETY: `n` is within both scenes' geom buffers (each holds
        // `maxgeom`); the scenes are different allocations; an `mjvGeom` is
        // plain data, and both scenes were made for one model layout, so the
        // copied ids index the same arrays.
        unsafe {
            let to = self.user_scene.ffi_mut();
            let n = from.ngeom.min(to.maxgeom).max(0);
            std::ptr::copy_nonoverlapping(from.geoms, to.geoms, usize::try_from(n).unwrap_or(0));
            to.ngeom = n;
        }
        true
    }

    /// The view's own copy of the model.
    #[must_use]
    pub fn model(&self) -> &MjModel {
        self.data.model()
    }

    #[must_use]
    pub const fn capacity(&self) -> Capacity {
        self.capacity
    }

    /// Bumped by every [`Self::set_model`]; [`ModelChanged`] says so too.
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

/// Sent the first frame [`MjView`] holds a model, and after every
/// [`MjView::set_model`]: what was made for the old one is no longer valid.
#[derive(Message, Clone, Copy, Debug)]
pub struct ModelChanged {
    pub generation: u64,
}

/// Where the plugin's work sits in `PostUpdate`, so a host can order its own
/// systems against it: copy state and write overlays before [`Self::Scene`],
/// read [`Labels`] and [`Selection`] after [`Self::Draw`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MjSystems {
    /// `mjv_updateScene`, then the pick asked for this frame.
    Scene,
    /// Entities, lights, gizmos, labels and the cameras from the scene.
    Draw,
}

/// Draws [`MjView`].
///
/// Assets, the geom list as entities, lines, labels, lights, the cameras,
/// and picks asked for through [`PendingClick`], as [`MjRenderOptions`]
/// says. Insert the resource before the first frame; frames without it draw
/// nothing.
///
/// With a renderer, it adds Bevy's `WireframePlugin` for
/// [`MjRenderOptions::wireframe`] unless the host added it first: a host
/// that wants it too adds it before this plugin.
pub struct MujocoScenePlugin;

impl Plugin for MujocoScenePlugin {
    fn build(&self, app: &mut App) {
        let model_changed = on_message::<ModelChanged>;
        loading::plugin(app);
        if app.get_sub_app(RenderApp).is_some() && !app.is_plugin_added::<WireframePlugin>() {
            app.add_plugins(WireframePlugin::default());
        }
        app.add_plugins(mujoco_bevy_assets::MujocoAssetsPlugin)
            .init_resource::<MjRenderOptions>()
            .init_resource::<Labels>()
            .init_resource::<GeomStats>()
            .init_resource::<LightSettings>()
            .init_resource::<PendingClick>()
            .init_resource::<Selection>()
            .add_message::<Picked>()
            .add_message::<ModelChanged>()
            .configure_sets(
                PostUpdate,
                (MjSystems::Scene, MjSystems::Draw)
                    .chain()
                    .run_if(resource_exists::<MjView>)
                    .before(TransformSystems::Propagate)
                    // Meshes, images and materials made or rewritten here are
                    // announced the same frame, so the render world extracts
                    // them once, from this frame's data. After the flush, a
                    // rewritten mesh was announced a frame late and its
                    // render-world-only data taken twice.
                    .before(AssetEventSystems),
            )
            .add_systems(
                PostUpdate,
                (
                    (
                        notice_new_model,
                        pick::forget_selection.run_if(model_changed),
                        update_scene,
                        pick::pick_clicked,
                    )
                        .chain()
                        .in_set(MjSystems::Scene),
                    (
                        (
                            rebuild_assets.run_if(model_changed),
                            scene::despawn_slots.run_if(model_changed),
                            rewrite_meshes,
                            scene::sync_surfaces,
                        )
                            .chain(),
                        scene::draw_lines,
                        scene::collect_labels,
                        scene::count_geoms,
                        debug::sync_fog,
                        lights::sync_lights,
                        lights::sync_ambient,
                        camera::sync_camera,
                        camera::sync_model_cameras,
                        camera::warn_many_cameras,
                    )
                        .in_set(MjSystems::Draw),
                ),
            );
    }
}

/// `simulate`'s mouse over the [`MjCamera`]'s viewport: drag to orbit, pan
/// and zoom, click to pick, Ctrl + drag on the selection to perturb. Needs
/// [`MujocoScenePlugin`].
pub struct MujocoInputPlugin;

impl Plugin for MujocoInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewInput>()
            .add_systems(Update, camera::mouse.run_if(resource_exists::<MjView>));
    }
}

/// [`MujocoScenePlugin`] and [`MujocoInputPlugin`].
pub struct MujocoPlugin;

impl Plugin for MujocoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((MujocoScenePlugin, MujocoInputPlugin));
    }
}

/// Says [`ModelChanged`] the frame the view's generation is new to it.
fn notice_new_model(
    view: Res<MjView>,
    mut seen: Local<Option<u64>>,
    mut changed: MessageWriter<ModelChanged>,
) {
    let generation = view.generation();
    if seen.replace(generation) != Some(generation) {
        changed.write(ModelChanged { generation });
    }
}

fn update_scene(mut view: ResMut<MjView>) {
    view.update_scene();
}

fn rebuild_assets(
    view: Res<MjView>,
    mut assets: ResMut<ModelAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
) {
    assets.rebuild(view.model(), &mut meshes, &mut images);
}

/// The Bevy meshes of meshes the host rewrote in place.
fn rewrite_meshes(
    mut view: ResMut<MjView>,
    mut assets: ResMut<ModelAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if view.meshes_rewritten.is_empty() {
        return;
    }
    let ids = std::mem::take(&mut view.meshes_rewritten);
    for id in ids {
        assets.update_mesh(view.model(), id, &mut meshes);
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
