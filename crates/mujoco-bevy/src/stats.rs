//! `MuJoCo`'s own figures as Bevy diagnostics, under `mujoco/...`.
//!
//! - `mujoco/time/<timer>`: milliseconds spent in each of `MuJoCo`'s stages
//!   this frame (summed over the frame's steps), and
//!   `mujoco/per_call/<timer>` the mean of one call, for `mjTIMER_*` named
//!   as `MuJoCo` names them ([`TIMERS`]: `step`, `forward`, `position`,
//!   `pos_collision`, `col_narrowphase`, `constraint`, ...).
//! - `mujoco/steps` this frame; `mujoco/contacts`, `constraints`,
//!   `solver_iterations`, `islands` of the last step; `mujoco/warnings`
//!   raised this frame; `mujoco/energy/{potential,kinetic}` while the model
//!   enables energy; `mujoco/arena` (% of the arena ever used).
//! - `mujoco/draw/*`: what the bridge draws ([`GeomStats`]): entities,
//!   triangles, vertices, materials, meshes, in total, as `decor`, and per
//!   top-level body subtree (`mujoco/draw/subtree/<root body name>/...`).
//! - `mujoco/collision/*`: what collides, from the model ([`CollisionStats`]):
//!   collision geoms (primitives, meshes, other), the vertices of the mesh
//!   colliders' convex hulls, and per subtree also this frame's contacts
//!   (`mujoco/collision/subtree/<name>/contacts`).
//!
//! Counts, not timings, except `time` and `per_call`. A subtree is a body
//! whose parent is the world, with all below it; `world` is the world
//! body's own geoms. A contact counts for the subtree of each geom, except
//! that the world's side of a contact with a subtree counts for the subtree
//! alone: a machine's contacts with the ground are the machine's.
//!
//! The figures come from an `MjData`. By default the view's
//! ([`StatsSource::View`]): `mjv_copyData` copies the whole `mjData`
//! header, so the timers, counts, warnings and energy, and the contact
//! list, reach [`MjView::data`] with every `copy_visual_to`. A host
//! that copies less often than it steps, or draws another state, hands the
//! figures itself ([`StatsSource::Host`]): `MjStatsInput(Some(MjFigures::read(&data)))`
//! before [`MjStatsSystems::Publish`].
//!
//! # The clock
//!
//! `MuJoCo` times its stages only through `mjcb_time`, one C callback for
//! the whole process. [`install_timer`] sets it (the plugin does, when
//! built), to a clock in milliseconds that reads zero unless armed; the
//! plugin arms it in `First` while any `time` or `per_call` diagnostic is
//! enabled, so a hidden overlay costs one atomic load per timer call.
//! A host that sets `mjcb_time` itself keeps its own: [`install_timer`]
//! does not replace another callback. The timers add up per `MjData` and
//! start again at a reset or a new model; the diagnostics are differences
//! from frame to frame, and a figure that went down is taken as a fresh
//! start.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use bevy::diagnostic::{
    Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, RegisterDiagnostic,
};
use bevy::prelude::*;
use mujoco_rs::mujoco_c;
use mujoco_rs::prelude::*;
use mujoco_rs::wrappers::MjContact;
use mujoco_rs::wrappers::mj_model::traits::ModelType;

use crate::debug::Named;
use crate::{GeomStats, MjSystems, MjView};

/// How many timers `MuJoCo` keeps.
pub const NTIMER: usize = MjtTimer::mjNTIMER as usize;

/// `mjtTimer`, named as `mjTIMERSTRING` names them: the last part of the
/// timer's diagnostic paths.
#[allow(clippy::enum_glob_use)]
pub const TIMERS: [Named<MjtTimer>; NTIMER] = {
    use mujoco_rs::mujoco_c::mjtTimer::*;
    const fn n(value: MjtTimer, name: &'static str) -> Named<MjtTimer> {
        Named {
            value,
            name,
            drawn: true,
        }
    }
    [
        n(mjTIMER_STEP, "step"),
        n(mjTIMER_FORWARD, "forward"),
        n(mjTIMER_INVERSE, "inverse"),
        n(mjTIMER_POSITION, "position"),
        n(mjTIMER_VELOCITY, "velocity"),
        n(mjTIMER_ACTUATION, "actuation"),
        n(mjTIMER_CONSTRAINT, "constraint"),
        n(mjTIMER_ADVANCE, "advance"),
        n(mjTIMER_POS_KINEMATICS, "pos_kinematics"),
        n(mjTIMER_POS_INERTIA, "pos_inertia"),
        n(mjTIMER_POS_COLLISION, "pos_collision"),
        n(mjTIMER_POS_MAKE, "pos_make"),
        n(mjTIMER_POS_PROJECT, "pos_project"),
        n(mjTIMER_COL_BROAD, "col_broadphase"),
        n(mjTIMER_COL_NARROW, "col_narrowphase"),
    ]
};

/// The diagnostics' paths.
pub mod paths {
    use super::NTIMER;
    use bevy::diagnostic::DiagnosticPath;

    const fn p(path: &'static str) -> DiagnosticPath {
        DiagnosticPath::const_new(path)
    }

    /// Milliseconds in each timer this frame, in [`TIMERS`](super::TIMERS) order.
    pub static TIME: [DiagnosticPath; NTIMER] = [
        p("mujoco/time/step"),
        p("mujoco/time/forward"),
        p("mujoco/time/inverse"),
        p("mujoco/time/position"),
        p("mujoco/time/velocity"),
        p("mujoco/time/actuation"),
        p("mujoco/time/constraint"),
        p("mujoco/time/advance"),
        p("mujoco/time/pos_kinematics"),
        p("mujoco/time/pos_inertia"),
        p("mujoco/time/pos_collision"),
        p("mujoco/time/pos_make"),
        p("mujoco/time/pos_project"),
        p("mujoco/time/col_broadphase"),
        p("mujoco/time/col_narrowphase"),
    ];
    /// Milliseconds a call of each timer took on average this frame; no
    /// measurement in a frame without calls.
    pub static PER_CALL: [DiagnosticPath; NTIMER] = [
        p("mujoco/per_call/step"),
        p("mujoco/per_call/forward"),
        p("mujoco/per_call/inverse"),
        p("mujoco/per_call/position"),
        p("mujoco/per_call/velocity"),
        p("mujoco/per_call/actuation"),
        p("mujoco/per_call/constraint"),
        p("mujoco/per_call/advance"),
        p("mujoco/per_call/pos_kinematics"),
        p("mujoco/per_call/pos_inertia"),
        p("mujoco/per_call/pos_collision"),
        p("mujoco/per_call/pos_make"),
        p("mujoco/per_call/pos_project"),
        p("mujoco/per_call/col_broadphase"),
        p("mujoco/per_call/col_narrowphase"),
    ];
    pub const STEPS: DiagnosticPath = p("mujoco/steps");
    pub const CONTACTS: DiagnosticPath = p("mujoco/contacts");
    pub const CONSTRAINTS: DiagnosticPath = p("mujoco/constraints");
    pub const SOLVER_ITERATIONS: DiagnosticPath = p("mujoco/solver_iterations");
    pub const ISLANDS: DiagnosticPath = p("mujoco/islands");
    pub const WARNINGS: DiagnosticPath = p("mujoco/warnings");
    pub const ENERGY_POTENTIAL: DiagnosticPath = p("mujoco/energy/potential");
    pub const ENERGY_KINETIC: DiagnosticPath = p("mujoco/energy/kinetic");
    pub const ARENA: DiagnosticPath = p("mujoco/arena");

    pub const DRAW_ENTITIES: DiagnosticPath = p("mujoco/draw/entities");
    pub const DRAW_TRIANGLES: DiagnosticPath = p("mujoco/draw/triangles");
    pub const DRAW_VERTICES: DiagnosticPath = p("mujoco/draw/vertices");
    pub const DRAW_MATERIALS: DiagnosticPath = p("mujoco/draw/materials");
    pub const DRAW_MESHES: DiagnosticPath = p("mujoco/draw/meshes");
    pub const DECOR_ENTITIES: DiagnosticPath = p("mujoco/draw/decor/entities");
    pub const DECOR_TRIANGLES: DiagnosticPath = p("mujoco/draw/decor/triangles");
    pub const DECOR_VERTICES: DiagnosticPath = p("mujoco/draw/decor/vertices");

    pub const COLLISION_GEOMS: DiagnosticPath = p("mujoco/collision/geoms");
    pub const COLLISION_PRIMITIVES: DiagnosticPath = p("mujoco/collision/primitives");
    pub const COLLISION_MESHES: DiagnosticPath = p("mujoco/collision/meshes");
    pub const COLLISION_OTHER: DiagnosticPath = p("mujoco/collision/other");
    pub const COLLISION_HULL_VERTICES: DiagnosticPath = p("mujoco/collision/hull_vertices");

    /// The last parts of `mujoco/draw/subtree/<name>/...`.
    pub const SUBTREE_DRAW: [&str; 3] = ["entities", "triangles", "vertices"];
    /// The last parts of `mujoco/collision/subtree/<name>/...`.
    pub const SUBTREE_COLLISION: [&str; 6] = [
        "geoms",
        "primitives",
        "meshes",
        "other",
        "hull_vertices",
        "contacts",
    ];
}

static ARMED: AtomicBool = AtomicBool::new(false);
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// `mjcb_time`: milliseconds since the clock was installed, or 0 while
/// disarmed, so `MuJoCo` adds nothing to its durations.
extern "C" fn clock() -> f64 {
    if ARMED.load(Ordering::Relaxed) {
        EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    }
}

/// Sets `MuJoCo`'s `mjcb_time` to this crate's clock, unless another
/// callback is set.
///
/// The clock reads milliseconds while [`arm_timer`] has armed it. Returns
/// whether the clock is ours. Process-wide; call it before any thread steps
/// a model, as [`MjStatsPlugin`] does when built.
#[allow(clippy::option_if_let_else)]
pub fn install_timer() -> bool {
    let ours: unsafe extern "C" fn() -> f64 = clock;
    EPOCH.get_or_init(Instant::now);
    // SAFETY: a plain read and write of a C global holding a function
    // pointer, done while nothing steps (see above); `clock` is a valid
    // `mjfTime` for the life of the process.
    unsafe {
        match mujoco_c::mjcb_time {
            None => {
                mujoco_c::mjcb_time = Some(ours);
                true
            }
            Some(set) => std::ptr::fn_addr_eq(set, ours),
        }
    }
}

/// Starts or stops the clock [`install_timer`] set. A step that runs on
/// another thread while it flips adds one bad duration, which
/// [`MjStatsPlugin`] drops (a frame's time longer than the frame).
pub fn arm_timer(on: bool) {
    ARMED.store(on, Ordering::Relaxed);
}

#[must_use]
pub fn timer_armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// One `mjTimerStat`: what a stage has taken since the data started.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimerReading {
    /// In the clock's unit: milliseconds with [`install_timer`]'s.
    pub ms: f64,
    pub calls: u64,
}

/// `MuJoCo`'s figures in one `MjData` at one moment.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MjFigures {
    /// Each timer's totals since the data started, in [`TIMERS`] order.
    pub timers: [TimerReading; NTIMER],
    /// Of the last step.
    pub contacts: usize,
    pub constraints: usize,
    /// Summed over the islands (`solver_niter`).
    pub solver_iterations: usize,
    pub islands: usize,
    /// Warnings raised since the data started, of every kind.
    pub warnings: u64,
    /// Potential and kinetic, when the step computed them (`mjENBL_ENERGY`).
    pub energy: Option<[f64; 2]>,
    /// The share of the arena the data ever used, 0 to 1.
    pub arena: f64,
    /// The last step's contacts by top-level body subtree (the root body's
    /// id), counted as the module says; in id order, empty ones left out.
    pub contacts_by_subtree: Vec<(usize, usize)>,
}

impl MjFigures {
    /// Reads `data`. Cheap: the header, and one pass over the contacts.
    #[must_use]
    pub fn read<M: ModelType>(data: &MjData<M>) -> Self {
        let model = data.model();
        let timers = std::array::from_fn(|k| {
            let t = &data.timer()[k];
            TimerReading {
                ms: t.duration,
                calls: u64::try_from(t.number).unwrap_or(0),
            }
        });
        let islands = to_usize(data.nisland());
        let solver_iterations = data.solver_niter()
            [..islands.clamp(1, mujoco_c::mjNISLAND as usize)]
            .iter()
            .map(|&n| to_usize(n))
            .sum();
        let narena = data.narena();
        Self {
            timers,
            contacts: to_usize(data.ncon()),
            constraints: to_usize(data.nefc()),
            solver_iterations,
            islands,
            warnings: data
                .warning()
                .iter()
                .map(|w| u64::try_from(w.number).unwrap_or(0))
                .sum(),
            energy: data.flg_energypos().then(|| *data.energy()),
            arena: if narena > 0 {
                data.maxuse_arena() as f64 / narena as f64
            } else {
                0.0
            },
            contacts_by_subtree: contacts_by_subtree(model, data.contact()),
        }
    }
}

fn to_usize<T: TryInto<usize>>(n: T) -> usize {
    n.try_into().unwrap_or(0)
}

/// The subtree (root body) geom `geom` belongs to.
fn root_of_geom(model: &MjModel, geom: i32) -> Option<usize> {
    let body = *model.geom_bodyid().get(usize::try_from(geom).ok()?)?;
    usize::try_from(*model.body_rootid().get(usize::try_from(body).ok()?)?).ok()
}

fn contacts_by_subtree(model: &MjModel, contacts: &[MjContact]) -> Vec<(usize, usize)> {
    let mut by_body = vec![0usize; to_usize(model.nbody())];
    for contact in contacts {
        let mut count = |root: Option<usize>| {
            if let Some(n) = root.and_then(|r| by_body.get_mut(r)) {
                *n += 1;
            }
        };
        match contact.geom.map(|g| root_of_geom(model, g)) {
            [Some(a), Some(b)] if a == b => count(Some(a)),
            [Some(0), other] | [other, Some(0)] => count(other),
            [a, b] => {
                count(a);
                count(b);
            }
        }
    }
    nonzero(&by_body)
}

fn nonzero<T: Copy + PartialEq + Default>(by_body: &[T]) -> Vec<(usize, T)> {
    by_body
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != T::default())
        .map(|(b, c)| (b, *c))
        .collect()
}

/// What collides in a part of the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollisionCounts {
    /// Geoms with a non-zero `contype` or `conaffinity`.
    pub geoms: usize,
    /// Of those: spheres, capsules, boxes, cylinders, ellipsoids, planes.
    pub primitives: usize,
    pub meshes: usize,
    /// Height fields and SDFs.
    pub other: usize,
    /// Vertices of the mesh colliders' convex hulls (`MuJoCo` collides a
    /// mesh by its hull; a mesh without one, by all its vertices).
    pub hull_vertices: usize,
    /// This frame's contacts.
    pub contacts: usize,
}

/// What collides, from the view's model, with this frame's contacts.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct CollisionStats {
    pub total: CollisionCounts,
    /// By top-level body subtree (the root body's id), in id order; ones
    /// with no collision geom and no contact left out.
    pub by_subtree: Vec<(usize, CollisionCounts)>,
}

impl CollisionStats {
    /// The model's collision geoms; no contacts.
    #[must_use]
    pub fn of_model(model: &MjModel) -> Self {
        let mut by_body = vec![CollisionCounts::default(); to_usize(model.nbody())];
        let mut total = CollisionCounts::default();
        let (contype, conaffinity) = (model.geom_contype(), model.geom_conaffinity());
        for g in 0..to_usize(model.ngeom()) {
            if contype[g] == 0 && conaffinity[g] == 0 {
                continue;
            }
            let ty = model.geom_type()[g] as i32;
            let is = |t: MjtGeom| ty == t as i32;
            let add = |c: &mut CollisionCounts| {
                c.geoms += 1;
                if is(MjtGeom::mjGEOM_MESH) {
                    c.meshes += 1;
                    c.hull_vertices += hull_vertices(model, model.geom_dataid()[g]);
                } else if is(MjtGeom::mjGEOM_HFIELD) || is(MjtGeom::mjGEOM_SDF) {
                    c.other += 1;
                } else {
                    c.primitives += 1;
                }
            };
            add(&mut total);
            if let Some(c) =
                root_of_geom(model, i32::try_from(g).unwrap_or(-1)).and_then(|r| by_body.get_mut(r))
            {
                add(c);
            }
        }
        Self {
            total,
            by_subtree: nonzero(&by_body),
        }
    }

    /// Takes this frame's contacts, keeping the model's counts.
    pub fn set_contacts(&mut self, total: usize, by_subtree: &[(usize, usize)]) {
        self.total.contacts = total;
        for (_, c) in &mut self.by_subtree {
            c.contacts = 0;
        }
        for &(body, n) in by_subtree {
            match self.by_subtree.binary_search_by_key(&body, |(b, _)| *b) {
                Ok(k) => self.by_subtree[k].1.contacts = n,
                Err(k) => self.by_subtree.insert(
                    k,
                    (
                        body,
                        CollisionCounts {
                            contacts: n,
                            ..default()
                        },
                    ),
                ),
            }
        }
    }
}

fn hull_vertices(model: &MjModel, mesh: i32) -> usize {
    let Ok(mesh) = usize::try_from(mesh) else {
        return 0;
    };
    usize::try_from(model.mesh_graphadr()[mesh])
        .ok()
        .and_then(|at| model.mesh_graph().get(at).copied())
        .map_or_else(|| to_usize(model.mesh_vertnum()[mesh]), to_usize)
}

/// A subtree's name in diagnostic paths: its root body's name with empty
/// parts dropped (`m0/base` stays `m0/base`), `world` for the world body,
/// `body<id>` for a body without a name.
#[must_use]
pub fn subtree_name(model: &MjModel, body: usize) -> String {
    if body == 0 {
        return "world".to_owned();
    }
    let name = model
        .id_to_name(MjtObjBase::mjOBJ_BODY, body)
        .unwrap_or_default();
    let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        format!("body{body}")
    } else {
        parts.join("/")
    }
}

/// Where the figures come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StatsSource {
    /// [`MjView::data`], read in [`MjStatsSystems::Read`].
    #[default]
    View,
    /// [`MjStatsInput`], written by the host.
    Host,
}

/// The figures to publish this frame.
///
/// With [`StatsSource::Host`] the host writes it before
/// [`MjStatsSystems::Publish`]; the last figures written stand until the
/// next (unchanged figures publish as a frame without steps).
#[derive(Resource, Clone, Debug, Default)]
pub struct MjStatsInput(pub Option<MjFigures>);

/// Where the stats systems sit in `PostUpdate`, after [`MjSystems::Draw`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MjStatsSystems {
    /// The view's figures into [`MjStatsInput`] ([`StatsSource::View`]).
    Read,
    /// Diagnostics from [`MjStatsInput`], [`GeomStats`] and
    /// [`CollisionStats`].
    Publish,
}

/// `MuJoCo`'s figures, the drawn geometry and the colliders as Bevy
/// diagnostics (see the module). Needs an [`MjView`]; the draw rows need
/// [`MujocoScenePlugin`](crate::MujocoScenePlugin). Installs the timer
/// clock when built.
#[derive(Default)]
pub struct MjStatsPlugin {
    pub source: StatsSource,
}

impl Plugin for MjStatsPlugin {
    fn build(&self, app: &mut App) {
        if !install_timer() {
            warn!("mjcb_time is set by someone else: MuJoCo's timers follow that clock");
        }
        let ms = |path: &DiagnosticPath| Diagnostic::new(path.clone()).with_suffix("ms");
        let plain = |path: &DiagnosticPath| Diagnostic::new(path.clone());
        for k in 0..NTIMER {
            app.register_diagnostic(ms(&paths::TIME[k]))
                .register_diagnostic(ms(&paths::PER_CALL[k]));
        }
        for path in [
            &paths::STEPS,
            &paths::CONTACTS,
            &paths::CONSTRAINTS,
            &paths::SOLVER_ITERATIONS,
            &paths::ISLANDS,
            &paths::WARNINGS,
            &paths::ENERGY_POTENTIAL,
            &paths::ENERGY_KINETIC,
            &paths::DRAW_ENTITIES,
            &paths::DRAW_TRIANGLES,
            &paths::DRAW_VERTICES,
            &paths::DRAW_MATERIALS,
            &paths::DRAW_MESHES,
            &paths::DECOR_ENTITIES,
            &paths::DECOR_TRIANGLES,
            &paths::DECOR_VERTICES,
            &paths::COLLISION_GEOMS,
            &paths::COLLISION_PRIMITIVES,
            &paths::COLLISION_MESHES,
            &paths::COLLISION_OTHER,
            &paths::COLLISION_HULL_VERTICES,
        ] {
            app.register_diagnostic(plain(path));
        }
        app.register_diagnostic(plain(&paths::ARENA).with_suffix("%"));

        let has_view = resource_exists::<MjView>;
        app.init_resource::<MjStatsInput>()
            .init_resource::<CollisionStats>()
            .init_resource::<Subtrees>()
            .init_resource::<Previous>()
            .configure_sets(
                PostUpdate,
                (MjStatsSystems::Read, MjStatsSystems::Publish)
                    .chain()
                    .after(MjSystems::Draw)
                    .run_if(has_view),
            )
            .add_systems(First, arm)
            .add_systems(
                PostUpdate,
                (
                    (read_view.run_if(source_is(StatsSource::View)), subtrees)
                        .in_set(MjStatsSystems::Read),
                    publish.in_set(MjStatsSystems::Publish),
                ),
            )
            .insert_resource(Source(self.source));
    }
}

#[derive(Resource, Clone, Copy)]
struct Source(StatsSource);

fn source_is(wanted: StatsSource) -> impl Fn(Res<Source>) -> bool + Clone {
    move |source: Res<Source>| source.0 == wanted
}

/// Arms the clock while any timer diagnostic is enabled.
fn arm(store: Res<DiagnosticsStore>) {
    let enabled = |path: &DiagnosticPath| store.get(path).is_some_and(|d| d.is_enabled);
    arm_timer(paths::TIME.iter().chain(&paths::PER_CALL).any(enabled));
}

fn read_view(view: Res<MjView>, mut input: ResMut<MjStatsInput>) {
    input.0 = Some(MjFigures::read(&view.data));
}

/// One subtree's diagnostic paths.
struct SubtreePaths {
    body: usize,
    draw: [DiagnosticPath; 3],
    collision: [DiagnosticPath; 6],
}

/// The model's subtrees and their paths, made once per model.
#[derive(Resource, Default)]
struct Subtrees {
    generation: Option<u64>,
    paths: Vec<SubtreePaths>,
}

/// On a new model: its collision counts, and a diagnostic per subtree,
/// registered once (a path seen before keeps its history; one of an old
/// model stays, unfed). A new subtree's rows start enabled as their
/// totals are, so a host that hid the totals is not handed new rows.
fn subtrees(
    view: Res<MjView>,
    mut subtrees: ResMut<Subtrees>,
    mut collision: ResMut<CollisionStats>,
    mut store: ResMut<DiagnosticsStore>,
) {
    if subtrees.generation == Some(view.generation()) {
        return;
    }
    subtrees.generation = Some(view.generation());
    let model = view.model();
    *collision = CollisionStats::of_model(model);
    let rootid = model.body_rootid();
    subtrees.paths = (0..to_usize(model.nbody()))
        .filter(|&b| to_usize(rootid[b]) == b)
        .map(|body| {
            let name = subtree_name(model, body);
            let path = |kind: &str, what: &str| {
                DiagnosticPath::new(format!("mujoco/{kind}/subtree/{name}/{what}"))
            };
            SubtreePaths {
                body,
                draw: paths::SUBTREE_DRAW.map(|w| path("draw", w)),
                collision: paths::SUBTREE_COLLISION.map(|w| path("collision", w)),
            }
        })
        .collect();
    let enabled = |store: &DiagnosticsStore, path: &DiagnosticPath| {
        store.get(path).is_none_or(|d| d.is_enabled)
    };
    let draw_on = enabled(&store, &paths::DRAW_TRIANGLES);
    let collision_on = enabled(&store, &paths::COLLISION_GEOMS);
    for subtree in &subtrees.paths {
        let draw = subtree.draw.iter().map(|p| (p, draw_on));
        let collision = subtree.collision.iter().map(|p| (p, collision_on));
        for (path, on) in draw.chain(collision) {
            if store.get(path).is_none() {
                let mut diagnostic = Diagnostic::new(path.clone());
                diagnostic.is_enabled = on;
                store.add(diagnostic);
            }
        }
    }
}

/// Last frame's totals, for the differences.
#[derive(Resource, Default)]
struct Previous {
    figures: Option<([TimerReading; NTIMER], u64)>,
    at: Option<Instant>,
}

/// `now - before`, or `now` where the figure went down: the data started
/// again.
fn since(now: f64, before: f64) -> f64 {
    if now >= before { now - before } else { now }
}

#[allow(clippy::cast_precision_loss)]
fn publish(
    input: Res<MjStatsInput>,
    geoms: Option<Res<GeomStats>>,
    subtrees: Res<Subtrees>,
    mut collision: ResMut<CollisionStats>,
    mut previous: ResMut<Previous>,
    mut diagnostics: Diagnostics,
) {
    let now = Instant::now();
    let wall_ms = previous
        .at
        .replace(now)
        .map(|at| (now - at).as_secs_f64() * 1000.0);

    if let Some(figures) = &input.0 {
        let before = previous.figures.replace((figures.timers, figures.warnings));
        if let Some((timers, warnings)) = before {
            for (k, (&t, &b)) in figures.timers.iter().zip(&timers).enumerate() {
                let fresh = t.calls < b.calls || t.ms < b.ms;
                let (ms, calls) = if fresh {
                    (t.ms, t.calls)
                } else {
                    (t.ms - b.ms, t.calls - b.calls)
                };
                // More time in one stage than the frame had: a clock armed
                // or disarmed in the middle of a step.
                if wall_ms.is_some_and(|w| ms > 1.5 * w + 1.0) {
                    continue;
                }
                diagnostics.add_measurement(&paths::TIME[k], || ms);
                if calls > 0 {
                    diagnostics.add_measurement(&paths::PER_CALL[k], || ms / calls as f64);
                }
                if k == MjtTimer::mjTIMER_STEP as usize {
                    diagnostics.add_measurement(&paths::STEPS, || calls as f64);
                }
            }
            diagnostics.add_measurement(&paths::WARNINGS, || {
                since(figures.warnings as f64, warnings as f64)
            });
        }
        diagnostics.add_measurement(&paths::CONTACTS, || figures.contacts as f64);
        diagnostics.add_measurement(&paths::CONSTRAINTS, || figures.constraints as f64);
        diagnostics.add_measurement(&paths::SOLVER_ITERATIONS, || {
            figures.solver_iterations as f64
        });
        diagnostics.add_measurement(&paths::ISLANDS, || figures.islands as f64);
        diagnostics.add_measurement(&paths::ARENA, || 100.0 * figures.arena);
        if let Some([potential, kinetic]) = figures.energy {
            diagnostics.add_measurement(&paths::ENERGY_POTENTIAL, || potential);
            diagnostics.add_measurement(&paths::ENERGY_KINETIC, || kinetic);
        }
        collision.set_contacts(figures.contacts, &figures.contacts_by_subtree);
    }

    let n = |v: usize| move || v as f64;
    if let Some(geoms) = geoms {
        diagnostics.add_measurement(&paths::DRAW_ENTITIES, n(geoms.surfaces));
        diagnostics.add_measurement(&paths::DRAW_TRIANGLES, n(geoms.triangles));
        diagnostics.add_measurement(&paths::DRAW_VERTICES, n(geoms.vertices));
        diagnostics.add_measurement(&paths::DRAW_MATERIALS, n(geoms.materials));
        diagnostics.add_measurement(&paths::DRAW_MESHES, n(geoms.meshes));
        diagnostics.add_measurement(&paths::DECOR_ENTITIES, n(geoms.decor.entities));
        diagnostics.add_measurement(&paths::DECOR_TRIANGLES, n(geoms.decor.triangles));
        diagnostics.add_measurement(&paths::DECOR_VERTICES, n(geoms.decor.vertices));
        for subtree in &subtrees.paths {
            let counts = geoms
                .by_subtree
                .iter()
                .find(|(b, _)| *b == subtree.body)
                .map(|(_, c)| *c)
                .unwrap_or_default();
            for (path, value) in
                subtree
                    .draw
                    .iter()
                    .zip([counts.entities, counts.triangles, counts.vertices])
            {
                diagnostics.add_measurement(path, n(value));
            }
        }
    }

    let total = collision.total;
    diagnostics.add_measurement(&paths::COLLISION_GEOMS, n(total.geoms));
    diagnostics.add_measurement(&paths::COLLISION_PRIMITIVES, n(total.primitives));
    diagnostics.add_measurement(&paths::COLLISION_MESHES, n(total.meshes));
    diagnostics.add_measurement(&paths::COLLISION_OTHER, n(total.other));
    diagnostics.add_measurement(&paths::COLLISION_HULL_VERTICES, n(total.hull_vertices));
    for subtree in &subtrees.paths {
        let c = collision
            .by_subtree
            .iter()
            .find(|(b, _)| *b == subtree.body)
            .map(|(_, c)| *c)
            .unwrap_or_default();
        for (path, value) in subtree.collision.iter().zip([
            c.geoms,
            c.primitives,
            c.meshes,
            c.other,
            c.hull_vertices,
            c.contacts,
        ]) {
            diagnostics.add_measurement(path, n(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn timer_names_and_paths_are_mujocos() {
        // SAFETY: a read of `MuJoCo`'s constant string table.
        let names = unsafe { mujoco_c::mjTIMERSTRING };
        for (k, timer) in TIMERS.iter().enumerate() {
            assert_eq!(timer.value as usize, k);
            // SAFETY: static NUL-terminated strings.
            let name = unsafe { CStr::from_ptr(names[k]) }.to_str().expect("ASCII");
            assert_eq!(timer.name, name);
            assert_eq!(paths::TIME[k].as_str(), format!("mujoco/time/{name}"));
            assert_eq!(
                paths::PER_CALL[k].as_str(),
                format!("mujoco/per_call/{name}")
            );
        }
    }

    /// Two machines (`m0/`, `m1/`) on a world floor: a box chassis with a
    /// mesh bucket, and a capsule; a visual-only geom that does not collide.
    const FLEET: &str = r#"<mujoco>
      <asset><mesh name="tet" vertex="0 0 0 1 0 0 0 1 0 0 0 1" /></asset>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" />
        <body name="m0/base" pos="0 0 0.5">
          <freejoint />
          <geom type="box" size="0.5 0.3 0.2" />
          <body name="m0/arm" pos="0.6 0 0">
            <joint type="hinge" />
            <geom type="mesh" mesh="tet" />
            <geom type="sphere" size="0.1" contype="0" conaffinity="0" />
          </body>
        </body>
        <body name="m1/" pos="3 0 0.3">
          <freejoint />
          <geom type="capsule" size="0.1 0.3" />
        </body>
      </worldbody>
    </mujoco>"#;

    #[test]
    fn collision_counts_by_subtree() {
        let model = MjModel::from_xml_string(FLEET).expect("compiles");
        let stats = CollisionStats::of_model(&model);
        assert_eq!(
            stats.total,
            CollisionCounts {
                geoms: 4,
                primitives: 3,
                meshes: 1,
                hull_vertices: 4,
                ..default()
            }
        );
        let id = |name| model.name_to_id(MjtObjBase::mjOBJ_BODY, name).expect(name);
        let of = |body| {
            stats
                .by_subtree
                .iter()
                .find(|(b, _)| *b == body)
                .map(|(_, c)| *c)
                .unwrap_or_default()
        };
        assert_eq!((of(0).geoms, of(0).primitives), (1, 1));
        let m0 = of(id("m0/base"));
        assert_eq!(
            (m0.geoms, m0.primitives, m0.meshes, m0.hull_vertices),
            (2, 1, 1, 4)
        );
        assert_eq!(of(id("m1/")).geoms, 1);
        assert_eq!(subtree_name(&model, id("m0/base")), "m0/base");
        assert_eq!(subtree_name(&model, id("m1/")), "m1");
        assert_eq!(subtree_name(&model, 0), "world");
    }

    #[test]
    fn a_ground_contact_counts_for_the_machine_alone() {
        let model = MjModel::from_xml_string(FLEET).expect("compiles");
        let mut data = MjData::new(Box::new(model));
        for _ in 0..500 {
            data.step();
        }
        let figures = MjFigures::read(&data);
        assert!(figures.contacts > 0);
        assert!(
            figures.contacts_by_subtree.iter().all(|(b, _)| *b != 0),
            "the floor's side is the machine's"
        );
        assert_eq!(
            figures
                .contacts_by_subtree
                .iter()
                .map(|(_, n)| n)
                .sum::<usize>(),
            figures.contacts,
            "every contact is between the floor and one machine"
        );
        assert_eq!(figures.timers[0].calls, 500);
    }
}
