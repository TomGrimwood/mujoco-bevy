//! `MujocoPlugin` in an app with no window and no GPU: the geom list as
//! entities, change detection, model swaps, labels, lights and picking.
//!
//! The app has what the plugin's systems read and nothing that renders: asset
//! stores for meshes, materials and images, input, gizmos (`bevy_gizmos`
//! keeps its render side in another crate) and the ambient light. The mouse system takes the primary
//! window through `Single`, so with no window Bevy skips it rather than
//! failing, which is what a headless host wants.

use bevy::asset::AssetPlugin;
use bevy::gizmos::GizmoPlugin;
use bevy::input::InputPlugin;
use bevy::light::GlobalAmbientLight;
use bevy::mesh::skinning::SkinnedMeshInverseBindposes;
use bevy::prelude::*;
use mujoco_bevy::{
    Click, GeomSlot, Labels, LightSettings, MjCamera, MjLight, MjView, ModelCamera, Modifiers,
    MujocoPlugin, MujocoScenePlugin, PendingClick, Picked, Refresh, Selection,
};
use mujoco_rs::prelude::*;

/// A floor, two parts fixed to the world (a box carrying a capsule and a
/// mesh) and a falling ball, under a directional and a spot light.
const SCENE: &str = r#"<mujoco model="headless">
  <option timestep="0.002" />
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.3" rgb2="0.2 0.2 0.2" width="64" height="64" />
    <material name="grid" texture="grid" texuniform="true" texrepeat="1 1" />
    <mesh name="tet" vertex="0 0 0 0.4 0 0 0 0.4 0 0 0 0.4" />
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 10" dir="0 0 -1" castshadow="true" />
    <light name="spot" pos="2 0 3" dir="0 0 -1" castshadow="false" />
    <camera name="eye" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1" fovy="50" />
    <geom name="floor" type="plane" size="5 5 0.1" material="grid" />
    <body name="post" pos="-1 0 0.5" euler="0 0 30">
      <geom name="post" type="box" size="0.1 0.2 0.5" />
    </body>
    <body name="target" pos="1 1 0.5">
      <geom name="target" type="box" size="0.3 0.3 0.3" />
      <geom name="pin" type="capsule" fromto="0 0 0.3 0 0 0.8" size="0.05" />
      <geom name="tet" type="mesh" mesh="tet" pos="0 0 1" />
    </body>
    <body name="ball" pos="0 -1 1">
      <freejoint />
      <geom name="ball" type="sphere" size="0.2" />
    </body>
  </worldbody>
</mujoco>"#;

/// Another model to swap to: nothing in common with [`SCENE`].
const OTHER: &str = r#"<mujoco model="other">
  <worldbody>
    <geom type="plane" size="3 3 0.1" />
    <body pos="0 0 1"><geom type="cylinder" size="0.2 0.4" /></body>
  </worldbody>
</mujoco>"#;

/// Geom types drawn as meshes, as `mujoco-bevy` sorts them.
const SURFACES: [MjtGeom; 11] = [
    MjtGeom::mjGEOM_PLANE,
    MjtGeom::mjGEOM_SPHERE,
    MjtGeom::mjGEOM_CAPSULE,
    MjtGeom::mjGEOM_ELLIPSOID,
    MjtGeom::mjGEOM_CYLINDER,
    MjtGeom::mjGEOM_BOX,
    MjtGeom::mjGEOM_MESH,
    MjtGeom::mjGEOM_ARROW,
    MjtGeom::mjGEOM_ARROW1,
    MjtGeom::mjGEOM_ARROW2,
    MjtGeom::mjGEOM_TRIANGLE,
];

fn model(xml: &str) -> MjModel {
    MjModel::from_xml_string(xml).expect("the test model compiles")
}

/// Slots whose transform changed this frame, as `Ref::is_changed` sees it
/// from a system: outside systems every change already reads as old.
#[derive(Resource, Default)]
struct Moved(Vec<usize>);

fn record_moved(slots: Query<(&GeomSlot, Ref<Transform>)>, mut moved: ResMut<Moved>) {
    moved.0 = slots
        .iter()
        .filter(|(_, t)| t.is_changed())
        .map(|(slot, _)| slot.index())
        .collect();
}

/// The plugin on [`SCENE`], run for a couple of frames so the assets, the
/// slots and the lights exist.
fn app() -> App {
    app_with(MujocoPlugin)
}

fn app_with(plugin: impl Plugin) -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        InputPlugin,
        GizmoPlugin,
    ))
    // `GizmoPlugin` draws skinned meshes' bounds, so it reads this store;
    // in a full app Bevy's mesh plugin registers it.
    .init_asset::<SkinnedMeshInverseBindposes>()
    .init_asset::<Mesh>()
    .init_asset::<StandardMaterial>()
    .init_asset::<Image>()
    .init_resource::<GlobalAmbientLight>()
    .init_resource::<Moved>()
    .add_plugins(plugin)
    .insert_resource(MjView::new(&model(SCENE)))
    .add_systems(Last, record_moved);
    app.world_mut()
        .spawn((MjCamera, Transform::default(), Projection::default()));
    frames(&mut app, 2);
    app
}

fn frames(app: &mut App, n: usize) {
    for _ in 0..n {
        app.update();
    }
}

fn view(app: &mut App) -> Mut<'_, MjView> {
    app.world_mut().resource_mut::<MjView>()
}

/// Model geom ids of the scene's surface geoms, in list order: what slot `k`
/// should draw.
fn surface_geoms(app: &App) -> Vec<usize> {
    let view = app.world().resource::<MjView>();
    view.scene
        .geoms()
        .iter()
        .filter(|g| SURFACES.iter().any(|&s| g.type_ == s as i32))
        .map(|g| usize::try_from(g.objid).expect("a model geom"))
        .collect()
}

/// Every slot's transform, by slot index; hidden slots left out.
fn slots(app: &mut App) -> Vec<Transform> {
    let mut found: Vec<(usize, Transform)> = app
        .world_mut()
        .query::<(&GeomSlot, &Transform, &Visibility)>()
        .iter(app.world())
        .filter(|(_, _, v)| **v != Visibility::Hidden)
        .map(|(slot, t, _)| (slot.index(), *t))
        .collect();
    found.sort_by_key(|(i, _)| *i);
    found.into_iter().map(|(_, t)| t).collect()
}

fn asset_counts(app: &App) -> (usize, usize, usize) {
    let world = app.world();
    (
        world.resource::<Assets<Mesh>>().len(),
        world.resource::<Assets<StandardMaterial>>().len(),
        world.resource::<Assets<Image>>().len(),
    )
}

fn body(app: &App, name: &str) -> usize {
    let view = app.world().resource::<MjView>();
    view.model()
        .name_to_id(MjtObjBase::mjOBJ_BODY, name)
        .expect("the body exists")
}

#[test]
fn each_surface_geom_has_an_entity_at_its_pose() {
    let mut app = app();
    let geoms = surface_geoms(&app);
    let slots = slots(&mut app);
    assert_eq!(geoms.len(), 6, "floor, post, target, pin, tet and ball");
    assert_eq!(slots.len(), geoms.len());
    let view = app.world().resource::<MjView>();
    for (slot, &g) in slots.iter().zip(&geoms) {
        #[allow(clippy::cast_possible_truncation)]
        let (pos, mat) = (
            view.data.geom_xpos()[g].map(|v| v as f32),
            view.data.geom_xmat()[g].map(|v| v as f32),
        );
        // Row-major in `MuJoCo`, so read as columns it is the transpose.
        let rotation = Quat::from_mat3(&Mat3::from_cols_array(&mat).transpose());
        assert!(
            slot.translation.distance(Vec3::from_array(pos)) < 1e-5,
            "geom {g}"
        );
        assert!(slot.rotation.angle_between(rotation) < 1e-4, "geom {g}");
    }
    let post = view
        .model()
        .name_to_id(MjtObjBase::mjOBJ_GEOM, "post")
        .expect("post");
    let k = geoms
        .iter()
        .position(|&g| g == post)
        .expect("post is drawn");
    assert!(slots[k].scale.distance(Vec3::new(0.1, 0.2, 0.5)) < 1e-6);
}

#[test]
fn only_the_moving_parts_are_marked_changed() {
    let mut app = app();
    let ball_geom = {
        let view = app.world().resource::<MjView>();
        view.model()
            .name_to_id(MjtObjBase::mjOBJ_GEOM, "ball")
            .expect("ball")
    };
    let ball_slot = surface_geoms(&app)
        .iter()
        .position(|&g| g == ball_geom)
        .expect("the ball is drawn");
    frames(&mut app, 1);
    assert!(
        app.world().resource::<Moved>().0.is_empty(),
        "nothing moves while the data stands still"
    );
    for _ in 0..10 {
        view(&mut app).data.step();
    }
    frames(&mut app, 1);
    assert_eq!(app.world().resource::<Moved>().0, vec![ball_slot]);
}

#[test]
fn the_user_scene_is_drawn_after_the_scene() {
    let mut app = app();
    let before = slots(&mut app).len();
    {
        let mut view = view(&mut app);
        // SAFETY: no material, texture or object id: plain decor.
        unsafe {
            view.user_scene.create_geom(
                MjtGeom::mjGEOM_BOX,
                Some([0.5, 0.5, 0.5]),
                Some([3.0, 0.0, 1.0]),
                None,
                None,
            );
        }
    }
    frames(&mut app, 1);
    let after = slots(&mut app);
    assert_eq!(after.len(), before + 1);
    assert_eq!(after[before].translation, Vec3::new(3.0, 0.0, 1.0));

    view(&mut app).user_scene.clear_geom();
    frames(&mut app, 1);
    assert_eq!(
        slots(&mut app).len(),
        before,
        "the overlay's slot is hidden"
    );
}

#[test]
fn a_model_swap_respawns_the_slots_and_frees_the_old_assets() {
    let mut app = app();
    let swap = |app: &mut App, xml: &str| {
        view(app).set_model(&model(xml));
        // A frame to build and despawn, more for Bevy to free what the
        // dropped handles held.
        frames(app, 3);
    };
    swap(&mut app, OTHER);
    assert_eq!(slots(&mut app).len(), 2);
    let other = asset_counts(&app);
    swap(&mut app, SCENE);
    assert_eq!(slots(&mut app).len(), 6);
    let scene = asset_counts(&app);
    for _ in 0..3 {
        swap(&mut app, OTHER);
        assert_eq!(asset_counts(&app), other);
        swap(&mut app, SCENE);
        assert_eq!(asset_counts(&app), scene);
    }
}

#[test]
fn labels_follow_the_label_option() {
    let mut app = app();
    assert!(app.world().resource::<Labels>().0.is_empty());
    view(&mut app).option.label = MjtLabel::mjLABEL_BODY as i32;
    frames(&mut app, 1);
    let labels = &app.world().resource::<Labels>().0;
    for name in ["post", "target", "ball"] {
        assert!(labels.iter().any(|l| l.text == name), "no label {name}");
    }
}

#[test]
fn each_scene_light_is_a_bevy_light_of_its_kind() {
    let mut app = app();
    let lights = app.world().resource::<MjView>().scene.lights().len();
    assert_eq!(lights, 3, "the headlight, the sun and the spot");
    let kinds = |app: &mut App| {
        app.world_mut()
            .query::<(&MjLight, Option<&DirectionalLight>, Option<&SpotLight>)>()
            .iter(app.world())
            .map(|(_, d, s)| {
                (
                    d.map(|d| d.shadow_maps_enabled),
                    s.map(|s| s.shadow_maps_enabled),
                )
            })
            .collect::<Vec<_>>()
    };
    let shadows = |all: &[(Option<bool>, Option<bool>)]| {
        all.iter().filter(|(d, s)| d.or(*s) == Some(true)).count()
    };
    let with = kinds(&mut app);
    assert_eq!(with.len(), 3);
    assert_eq!(with.iter().filter(|(d, _)| d.is_some()).count(), 2);
    assert_eq!(with.iter().filter(|(_, s)| s.is_some()).count(), 1);
    assert_eq!(shadows(&with), 1, "only the sun casts shadows");

    app.world_mut().resource_mut::<LightSettings>().shadows = false;
    frames(&mut app, 1);
    assert_eq!(shadows(&kinds(&mut app)), 0);
}

#[test]
fn the_bevy_camera_sits_where_mujocos_does() {
    let mut app = app();
    let (pos, forward) = {
        let view = app.world().resource::<MjView>();
        let [l, r] = view.scene.camera();
        let gl = l.average_camera(r);
        (Vec3::from_array(gl.pos), Vec3::from_array(gl.forward))
    };
    let camera = app
        .world_mut()
        .query_filtered::<&Transform, With<MjCamera>>()
        .single(app.world())
        .copied()
        .expect("one camera");
    assert!(camera.translation.distance(pos) < 1e-5);
    assert!(camera.forward().angle_between(forward) < 1e-4);
}

/// Aims `MuJoCo`'s free camera at the target box and clicks the middle of
/// the view.
fn click_target(app: &mut App) {
    click_target_with(app, Modifiers::default());
}

fn click_target_with(app: &mut App, modifiers: Modifiers) {
    {
        let mut view = view(app);
        view.camera.lookat = [1.0, 1.0, 0.5];
        view.camera.distance = 4.0;
        view.camera.azimuth = 90.0;
        view.camera.elevation = -30.0;
    }
    app.world_mut().resource_mut::<PendingClick>().0 = Some(Click {
        at: Vec2::splat(0.5),
        aspect: 1.0,
        modifiers,
    });
    frames(app, 1);
}

#[test]
fn a_click_selects_the_body_under_it() {
    let mut app = app();
    click_target(&mut app);
    let target = body(&app, "target");
    let picked = app
        .world()
        .resource::<Selection>()
        .0
        .clone()
        .expect("a pick");
    assert_eq!(picked.body, Some(target));
    assert!(picked.point.distance(bevy::math::DVec3::new(1.0, 1.0, 0.5)) < 0.5);
    let sent: Vec<_> = app
        .world()
        .resource::<Messages<Picked>>()
        .iter_current_update_messages()
        .map(|p| p.pick.body)
        .collect();
    assert_eq!(sent, vec![Some(target)]);
    assert_eq!(
        usize::try_from(app.world().resource::<MjView>().perturb.select).ok(),
        Some(target)
    );
}

#[test]
fn a_model_swap_lets_go_of_the_selection() {
    let mut app = app();
    click_target(&mut app);
    assert!(app.world().resource::<Selection>().0.is_some());
    view(&mut app).set_model(&model(OTHER));
    frames(&mut app, 1);
    assert!(app.world().resource::<Selection>().0.is_none());
}

#[test]
fn a_modified_click_is_reported_but_selects_nothing() {
    let mut app = app();
    let held = Modifiers {
        ctrl: true,
        shift: true,
        alt: false,
    };
    click_target_with(&mut app, held);
    let sent: Vec<_> = app
        .world()
        .resource::<Messages<Picked>>()
        .iter_current_update_messages()
        .map(|p| (p.pick.body, p.modifiers))
        .collect();
    assert_eq!(sent, vec![(Some(body(&app, "target")), held)]);
    assert!(app.world().resource::<Selection>().0.is_none());
    assert_eq!(app.world().resource::<MjView>().perturb.select, 0);
}

/// Every shown slot, by index: what it says it draws, its entity, its mesh
/// and its material.
fn slot_entities(
    app: &mut App,
) -> Vec<(GeomSlot, Entity, AssetId<Mesh>, Handle<StandardMaterial>)> {
    let mut found: Vec<_> = app
        .world_mut()
        .query::<(
            &GeomSlot,
            Entity,
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &Visibility,
        )>()
        .iter(app.world())
        .filter(|(.., v)| **v != Visibility::Hidden)
        .map(|(slot, e, mesh, material, _)| (*slot, e, mesh.0.id(), material.0.clone()))
        .collect();
    found.sort_by_key(|(slot, ..)| slot.index());
    found
}

#[test]
fn a_slot_says_which_geom_it_draws() {
    let mut app = app();
    {
        let mut view = view(&mut app);
        // SAFETY: no material, texture or object id: plain decor.
        unsafe {
            view.user_scene.create_geom(
                MjtGeom::mjGEOM_SPHERE,
                Some([0.1, 0.0, 0.0]),
                None,
                None,
                None,
            );
        }
    }
    frames(&mut app, 1);
    let geoms = surface_geoms(&app);
    let slots = slot_entities(&mut app);
    assert_eq!(slots.len(), geoms.len() + 1);
    for ((slot, ..), &g) in slots.iter().zip(&geoms) {
        assert_eq!(slot.model_geom(), Some(g));
        assert_ne!(slot.category(), MjtCatBit::mjCAT_DECOR as i32);
    }
    let (overlay, ..) = slots.last().expect("the overlay's slot");
    assert_eq!(overlay.model_geom(), None);
    assert_eq!(overlay.objid(), -1);
    assert_eq!(overlay.category(), MjtCatBit::mjCAT_DECOR as i32);
}

#[test]
fn a_model_camera_follows_the_models_camera() {
    let mut app = app();
    let eye = app
        .world_mut()
        .spawn((ModelCamera(0), Transform::default(), Projection::default()))
        .id();
    frames(&mut app, 1);
    let (pos, back) = {
        let view = app.world().resource::<MjView>();
        let m = view.data.cam_xmat()[0];
        #[allow(clippy::cast_possible_truncation)]
        let pos = Vec3::from_array(view.data.cam_xpos()[0].map(|v| v as f32));
        // The camera's own Z, row-major's third column: it looks the other way.
        #[allow(clippy::cast_possible_truncation)]
        let back = Vec3::new(m[2] as f32, m[5] as f32, m[8] as f32);
        (pos, back)
    };
    let world = app.world();
    let transform = world.get::<Transform>(eye).expect("a transform");
    assert!(transform.translation.distance(pos) < 1e-5);
    assert!(transform.forward().angle_between(-back) < 1e-4);
    let Some(Projection::Perspective(p)) = world.get::<Projection>(eye) else {
        panic!("a perspective camera");
    };
    assert!((p.fov - 50f32.to_radians()).abs() < 1e-5);
}

#[test]
fn a_refreshed_model_recolours_without_rebuilding() {
    let mut app = app();
    let before = slot_entities(&mut app);
    let mut painted = model(SCENE);
    let post = painted.geom("post").expect("post").id;
    {
        let info = painted.geom("post").expect("post");
        info.view_mut(&mut painted)
            .rgba
            .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    }
    assert_eq!(view(&mut app).refresh_model(&painted), Refresh::InPlace);
    frames(&mut app, 1);
    let after = slot_entities(&mut app);
    assert_eq!(
        before
            .iter()
            .map(|(_, e, m, _)| (*e, *m))
            .collect::<Vec<_>>(),
        after
            .iter()
            .map(|(_, e, m, _)| (*e, *m))
            .collect::<Vec<_>>(),
        "the same entities, drawing the same meshes"
    );
    let (.., material) = after
        .iter()
        .find(|(slot, ..)| slot.model_geom() == Some(post))
        .expect("the post's slot");
    let colour = app
        .world()
        .resource::<Assets<StandardMaterial>>()
        .get(material)
        .expect("the material")
        .base_color;
    assert_eq!(colour.to_srgba(), Srgba::rgb(1.0, 0.0, 0.0));
}

fn tet_slot(app: &mut App) -> AssetId<Mesh> {
    let tet = app
        .world()
        .resource::<MjView>()
        .model()
        .geom("tet")
        .expect("tet")
        .id;
    slot_entities(app)
        .into_iter()
        .find(|(slot, ..)| slot.model_geom() == Some(tet))
        .map(|(_, _, mesh, _)| mesh)
        .expect("the tet's slot")
}

#[test]
fn an_updated_mesh_keeps_its_handle_and_moves_its_vertices() {
    let mut app = app();
    let handle = tet_slot(&mut app);
    let x_extent = |app: &App| {
        let mesh = app
            .world()
            .resource::<Assets<Mesh>>()
            .get(handle)
            .expect("the tet");
        let xs: Vec<f32> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(bevy::mesh::VertexAttributeValues::as_float3)
            .expect("positions")
            .iter()
            .map(|p| p[0])
            .collect();
        xs.iter().copied().fold(f32::MIN, f32::max) - xs.iter().copied().fold(f32::MAX, f32::min)
    };
    let before = x_extent(&app);
    let mut grown = model(SCENE);
    let id = grown
        .name_to_id(MjtObjBase::mjOBJ_MESH, "tet")
        .expect("tet");
    let (v0, nv) = (grown.mesh_vertadr()[id], grown.mesh_vertnum()[id]);
    for v in v0..v0 + nv {
        for k in 0..3 {
            let at = usize::try_from(3 * v + k).expect("an index");
            // SAFETY: the vertex is inside the tet's own range of the buffer.
            unsafe { *grown.ffi_mut().mesh_vert.add(at) *= 2.0 };
        }
    }
    assert!(view(&mut app).update_mesh(&grown, id));
    frames(&mut app, 1);
    assert_eq!(tet_slot(&mut app), handle);
    assert!((x_extent(&app) / before - 2.0).abs() < 1e-4);
}

#[test]
fn a_recompile_keeps_the_meshes_that_did_not_change() {
    let mut app = app();
    let handle = tet_slot(&mut app);
    view(&mut app).set_model(&model(SCENE));
    frames(&mut app, 1);
    assert_eq!(tet_slot(&mut app), handle);
    let stats = app
        .world()
        .resource::<mujoco_bevy::ModelAssets>()
        .last_rebuild();
    assert_eq!((stats.meshes_built, stats.meshes_reused), (0, 1));
    assert_eq!((stats.textures_built, stats.textures_reused), (0, 1));
}

#[test]
fn the_scene_plugin_works_without_the_input_plugin() {
    let mut app = app_with(MujocoScenePlugin);
    assert_eq!(slots(&mut app).len(), 6);
    click_target(&mut app);
    assert_eq!(
        app.world()
            .resource::<Selection>()
            .0
            .as_ref()
            .and_then(|p| p.body),
        Some(body(&app, "target"))
    );
}
