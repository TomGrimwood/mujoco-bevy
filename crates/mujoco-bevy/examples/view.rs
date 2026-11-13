//! Any MJCF file, stepped in real time and drawn by `mujoco-bevy`.
//!
//! ```text
//! cargo run --example view -- path/to/model.xml   # or no path: a small demo scene
//! SECONDS=10 cargo run --example view             # closes itself after 10 s
//! ```
//!
//! Left drag rotates, right drag pans, the wheel zooms, a click selects a body
//! (its name is printed), Ctrl + drag on it pushes it around. W toggles the
//! wireframe (on at start), F the model's fog, C contact points. `MuJoCo`'s
//! stage timers, the step's counts and what is drawn and collides are logged
//! every 2 s (`mujoco/...` diagnostics).

use std::time::Duration;

use bevy::diagnostic::LogDiagnosticsPlugin;
use bevy::prelude::*;
use mujoco_bevy::stats::{TIMERS, paths};
use mujoco_bevy::{MjCamera, MjRenderOptions, MjStatsPlugin, MjView, MujocoPlugin, Picked};
use mujoco_rs::prelude::*;

const DEMO: &str = r#"<mujoco model="demo">
  <option timestep="0.002" />
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.32 0.35" rgb2="0.24 0.26 0.28" width="256" height="256" />
    <material name="grid" texture="grid" texuniform="true" texrepeat="1 1" />
  </asset>
  <worldbody>
    <light pos="4 3 8" dir="-0.4 -0.3 -1" directional="true" castshadow="true" diffuse="0.7 0.7 0.65" ambient="0.2 0.2 0.22" />
    <geom type="plane" size="20 20 0.1" material="grid" />
    <body pos="0 0 2">
      <joint type="ball" />
      <geom type="capsule" fromto="0 0 0 0.8 0 -0.8" size="0.06" rgba="0.9 0.6 0.1 1" />
      <body pos="0.8 0 -0.8">
        <joint type="hinge" axis="0 1 0" />
        <geom type="box" size="0.25 0.15 0.1" rgba="0.2 0.5 0.8 1" />
      </body>
    </body>
    <body pos="1.5 1 0.5">
      <freejoint />
      <geom type="sphere" size="0.3" rgba="0.8 0.2 0.2 1" />
    </body>
    <body pos="-1.5 -1 0.6">
      <freejoint />
      <geom type="cylinder" size="0.25 0.3" rgba="0.3 0.7 0.3 1" />
    </body>
  </worldbody>
</mujoco>"#;

/// The simulated state; the view draws a copy of it.
#[derive(Resource)]
struct Sim(MjData<Box<MjModel>>);

fn main() {
    let model = std::env::args()
        .nth(1)
        .map_or_else(|| MjModel::from_xml_string(DEMO), MjModel::from_xml)
        .expect("the model compiles");
    let per_call = TIMERS
        .iter()
        .zip(&paths::PER_CALL)
        .filter(|(t, _)| ["step", "forward", "pos_collision", "constraint"].contains(&t.name))
        .map(|(_, p)| p.clone());
    let logged = per_call
        .chain([
            paths::TIME[0].clone(),
            paths::STEPS,
            paths::CONTACTS,
            paths::CONSTRAINTS,
            paths::SOLVER_ITERATIONS,
            paths::DRAW_ENTITIES,
            paths::DRAW_TRIANGLES,
            paths::COLLISION_GEOMS,
            paths::COLLISION_HULL_VERTICES,
        ])
        .collect();
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, MujocoPlugin, MjStatsPlugin::default()))
        .add_plugins(LogDiagnosticsPlugin {
            wait_duration: Duration::from_secs(2),
            ..LogDiagnosticsPlugin::filtered(logged)
        })
        .insert_resource(MjView::new(&model))
        .insert_resource(MjRenderOptions {
            wireframe: true,
            ..default()
        })
        .insert_resource(Sim(MjData::new(Box::new(model))))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn((Camera3d::default(), MjCamera));
        })
        .add_systems(Update, (step, print_picks, toggles));
    if let Some(seconds) = std::env::var("SECONDS").ok().and_then(|s| s.parse().ok()) {
        app.add_systems(
            Update,
            move |time: Res<Time>, mut exit: MessageWriter<AppExit>| {
                if time.elapsed_secs_f64() > seconds {
                    exit.write(AppExit::Success);
                }
            },
        );
    }
    app.run();
}

fn toggles(
    keys: Res<ButtonInput<KeyCode>>,
    mut options: ResMut<MjRenderOptions>,
    mut view: ResMut<MjView>,
) {
    if keys.just_pressed(KeyCode::KeyW) {
        options.wireframe = !options.wireframe;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        options.fog = !options.fog;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        let on = view.vis_flag(MjtVisFlag::mjVIS_CONTACTPOINT);
        view.set_vis_flag(MjtVisFlag::mjVIS_CONTACTPOINT, !on);
    }
}

/// Steps until the sim catches up with the wall clock, applies the view's
/// perturb, and hands the view the result.
fn step(mut sim: ResMut<Sim>, mut view: ResMut<MjView>, time: Res<Time>) {
    let data = &mut sim.0;
    // At most a tenth of a second a frame, so a slow model slows down
    // instead of falling ever further behind.
    let behind = time.elapsed_secs_f64().min(data.time() + 0.1) - data.time();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let steps = (behind / data.model().opt().timestep).ceil().max(0.0) as usize;
    for _ in 0..steps {
        if view.perturb.active != 0 && view.perturb.select > 0 {
            view.perturb.apply(data);
        }
        data.step();
    }
    let _ = data.copy_visual_to(&mut view.data);
}

fn print_picks(mut picked: MessageReader<Picked>, view: Res<MjView>) {
    for Picked { pick, modifiers } in picked.read() {
        let name = pick
            .body
            .and_then(|b| view.model().id_to_name(MjtObjBase::mjOBJ_BODY, b))
            .unwrap_or("nothing");
        println!("picked {name} at {:?} ({modifiers:?})", pick.point);
    }
}
