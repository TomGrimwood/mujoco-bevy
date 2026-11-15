//! The geom list as Bevy entities, kept from frame to frame.
//!
//! The list's order is stable while the model and the options are, so the
//! entity with [`GeomSlot`] `k` draws the `k`-th surface geom of the list.
//! Updating it in place keeps Bevy's change detection quiet for parts that do
//! not move, and spawning nothing per frame keeps the frame rate. Lines are
//! drawn as gizmos and labels are handed to the host as text.

use std::ffi::CStr;

use std::collections::HashSet;

use bevy::ecs::system::SystemParam;
use bevy::light::NotShadowCaster;
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;
use mujoco_rs::prelude::*;

use crate::debug::MjRenderOptions;
use crate::{MjView, quat_from_mat};
use mujoco_bevy_assets::{GeomLook, MeshCounts, ModelAssets};

/// A plane of size 0 is infinite in `MuJoCo`; it is drawn this far out.
const INFINITE_PLANE: f32 = 1000.0;

/// Text `MuJoCo` put on a geom, at its world position, for the host to draw
/// (with egui at the projected point; no text is drawn in 3D).
pub struct Label {
    pub text: String,
    pub position: Vec3,
}

/// This frame's labels.
#[derive(Resource, Default)]
pub struct Labels(pub Vec<Label>);

/// What the last frame drew, for a host's panel and for
/// [`MjStatsPlugin`](crate::MjStatsPlugin)'s `mujoco/draw/*` rows. Counts of
/// what is drawn, not timings: they change when the model or what is shown
/// does.
#[derive(Resource, Default, Clone)]
pub struct GeomStats {
    /// Geoms in the scene list and the user scene.
    pub listed: usize,
    /// Drawn as Bevy meshes: one entity each.
    pub surfaces: usize,
    /// Drawn as gizmo lines.
    pub lines: usize,
    /// Text only, handed over in [`Labels`].
    pub labels_only: usize,
    /// `(mjtGeom, count)` of what is not drawn yet: height fields, skins,
    /// flex, SDFs.
    pub unsupported: Vec<(i32, usize)>,
    pub triangles: usize,
    pub vertices: usize,
    /// Distinct materials and meshes among the drawn surfaces.
    pub materials: usize,
    pub meshes: usize,
    /// What each top-level body's subtree draws (its geoms and sites), by
    /// the body's id (`body_rootid`; 0 is the world's own geoms), in id
    /// order, subtrees that draw nothing left out.
    pub by_subtree: Vec<(usize, DrawCounts)>,
    /// What no body owns: frames, contacts, perturb handles, the host's
    /// overlays.
    pub decor: DrawCounts,
}

/// The size of what a part of the scene draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawCounts {
    pub entities: usize,
    pub triangles: usize,
    pub vertices: usize,
}

impl DrawCounts {
    const fn add(&mut self, counts: MeshCounts) {
        self.entities += 1;
        self.triangles += counts.triangles;
        self.vertices += counts.vertices;
    }
}

/// Marks the entity that draws the `n`-th surface geom of the list, and says
/// what that geom is this frame. A slot is reused for whatever geom is `n`-th,
/// so its identity can change when the list does.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeomSlot {
    index: usize,
    category: i32,
    objtype: i32,
    objid: i32,
}

impl GeomSlot {
    const fn of(index: usize, geom: &MjvGeom) -> Self {
        Self {
            index,
            category: geom.category,
            objtype: geom.objtype,
            objid: geom.objid,
        }
    }

    /// Which surface geom of the scene list (then the user scene) this entity
    /// draws, counting only geoms drawn as meshes.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// `mjtCatBit`: static, dynamic or decor (frames, contacts, perturb
    /// handles, the host's overlays).
    #[must_use]
    pub const fn category(&self) -> i32 {
        self.category
    }

    /// `mjtObj` of what the geom stands for (`mjOBJ_GEOM` for a model geom),
    /// or `mjOBJ_UNKNOWN` for an overlay.
    #[must_use]
    pub const fn objtype(&self) -> i32 {
        self.objtype
    }

    /// The id of [`Self::objtype`]'s object, or -1 for none.
    #[must_use]
    pub const fn objid(&self) -> i32 {
        self.objid
    }

    /// The model geom drawn, if the slot draws one.
    #[must_use]
    pub fn model_geom(&self) -> Option<usize> {
        (self.objtype == MjtObj::mjOBJ_GEOM as i32)
            .then(|| usize::try_from(self.objid).ok())
            .flatten()
    }
}

/// How a geom type is drawn.
enum Kind {
    Surface,
    Line,
    LineBox,
    Label,
    Unsupported,
}

fn kind(geom: &MjvGeom) -> Kind {
    const fn is(ty: i32, geom: MjtGeom) -> bool {
        ty == geom as i32
    }
    let ty = geom.type_;
    if [
        MjtGeom::mjGEOM_MESH,
        MjtGeom::mjGEOM_SPHERE,
        MjtGeom::mjGEOM_ELLIPSOID,
        MjtGeom::mjGEOM_BOX,
        MjtGeom::mjGEOM_CYLINDER,
        MjtGeom::mjGEOM_CAPSULE,
        MjtGeom::mjGEOM_PLANE,
        MjtGeom::mjGEOM_ARROW,
        MjtGeom::mjGEOM_ARROW1,
        MjtGeom::mjGEOM_ARROW2,
        MjtGeom::mjGEOM_TRIANGLE,
    ]
    .into_iter()
    .any(|g| is(ty, g))
    {
        Kind::Surface
    } else if is(ty, MjtGeom::mjGEOM_LINE) {
        Kind::Line
    } else if is(ty, MjtGeom::mjGEOM_LINEBOX) {
        Kind::LineBox
    } else if is(ty, MjtGeom::mjGEOM_LABEL) {
        Kind::Label
    } else {
        Kind::Unsupported
    }
}

/// Every geom drawn this frame: the scene's, then the host's overlays,
/// less the categories the render options hide.
fn geoms<'a>(view: &'a MjView, options: &MjRenderOptions) -> impl Iterator<Item = &'a MjvGeom> {
    let shown = options.category_mask();
    view.scene
        .geoms()
        .iter()
        .chain(view.user_scene.geoms())
        .filter(move |g| g.category & shown != 0)
}

/// The top-level body whose subtree a geom belongs to, for a model geom or
/// site; `None` for decor.
fn owner(model: &MjModel, geom: &MjvGeom) -> Option<usize> {
    if geom.category == MjtCatBit::mjCAT_DECOR as i32 {
        return None;
    }
    let id = usize::try_from(geom.objid).ok()?;
    let body = if geom.objtype == MjtObj::mjOBJ_GEOM as i32 {
        *model.geom_bodyid().get(id)?
    } else if geom.objtype == MjtObj::mjOBJ_SITE as i32 {
        *model.site_bodyid().get(id)?
    } else {
        return None;
    };
    usize::try_from(*model.body_rootid().get(usize::try_from(body).ok()?)?).ok()
}

fn frame_of(geom: &MjvGeom, scale: Vec3) -> Transform {
    Transform {
        translation: Vec3::from_array(geom.pos),
        rotation: quat_from_mat(&geom.mat),
        scale,
    }
}

/// One surface geom, ready to draw.
pub struct Surface {
    slot: GeomSlot,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    /// The colour its wires are drawn in: its own, or its texture's.
    wire: Color,
    transform: Transform,
    casts_shadow: bool,
    counts: MeshCounts,
    owner: Option<usize>,
}

/// The stores a surface's mesh and material come from.
#[derive(SystemParam)]
pub struct SurfaceAssets<'w> {
    model: ResMut<'w, ModelAssets>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
}

impl SurfaceAssets<'_> {
    /// The material of a geom shown as its wires alone.
    fn hidden_material(&mut self) -> Handle<StandardMaterial> {
        self.model.hidden_material(&mut self.materials)
    }

    /// The mesh, material and transform for a [`Kind::Surface`] geom: one
    /// shared unit mesh per primitive scaled by the geom's size (`MuJoCo`
    /// gives sizes as scales: radius, radius, half-length for a cylinder),
    /// the model's own mesh for a mesh geom.
    fn surface(&mut self, model: &MjModel, geom: &MjvGeom) -> Option<Surface> {
        let size = Vec3::from_array(geom.size);
        let mut uv_scale = Vec2::from_array(geom.texrepeat);
        let ty = geom.type_;
        let is = |g: MjtGeom| ty == g as i32;

        let primitives = self.model.primitives()?;
        let (mesh, transform, has_uv) = if is(MjtGeom::mjGEOM_MESH) {
            // Even ids are the mesh, odd ones its convex hull (the convex
            // hull flag, on a collision geom).
            let id = usize::try_from(geom.dataid / 2).ok()?;
            let hull = if geom.dataid % 2 == 1 {
                self.model.hull(model, id, &mut self.meshes)
            } else {
                None
            };
            let (mesh, has_uv) = match hull {
                Some(hull) => (hull, false),
                None => self.model.mesh(id).map(|(m, uv)| (m.clone(), uv))?,
            };
            (mesh, frame_of(geom, Vec3::ONE), has_uv)
        } else if is(MjtGeom::mjGEOM_SPHERE) {
            (
                primitives.sphere.clone(),
                frame_of(geom, Vec3::splat(size.x)),
                true,
            )
        } else if is(MjtGeom::mjGEOM_ELLIPSOID) {
            (primitives.sphere.clone(), frame_of(geom, size), true)
        } else if is(MjtGeom::mjGEOM_BOX) {
            (primitives.cuboid.clone(), frame_of(geom, size), true)
        } else if is(MjtGeom::mjGEOM_CYLINDER) {
            (primitives.cylinder.clone(), frame_of(geom, size), true)
        } else if is(MjtGeom::mjGEOM_CAPSULE) {
            let radius = size.x.max(1e-6);
            let mesh = self.model.capsule(&mut self.meshes, size.z / radius);
            (mesh, frame_of(geom, Vec3::splat(radius)), true)
        } else if is(MjtGeom::mjGEOM_PLANE) {
            let half = Vec2::new(
                if size.x > 0.0 { size.x } else { INFINITE_PLANE },
                if size.y > 0.0 { size.y } else { INFINITE_PLANE },
            );
            if geom.texuniform != 0 {
                // `texuniform`: the texture repeats `texrepeat` times a metre.
                uv_scale *= 2.0 * half;
            }
            (
                primitives.plane.clone(),
                frame_of(geom, half.extend(1.0)),
                true,
            )
        } else if is(MjtGeom::mjGEOM_ARROW2) {
            (primitives.arrow2.clone(), frame_of(geom, size), true)
        } else if is(MjtGeom::mjGEOM_ARROW) || is(MjtGeom::mjGEOM_ARROW1) {
            (primitives.arrow.clone(), frame_of(geom, size), true)
        } else if is(MjtGeom::mjGEOM_TRIANGLE) {
            (
                primitives.triangle.clone(),
                frame_of(geom, size.with_z(1.0)),
                true,
            )
        } else {
            return None;
        };

        let look = GeomLook {
            rgba: geom.rgba,
            emission: geom.emission,
            specular: geom.specular,
            shininess: geom.shininess,
            pbr: usize::try_from(geom.matid)
                .ok()
                .filter(|&m| m < model.mat_metallic().len())
                .map(|m| [model.mat_metallic()[m], model.mat_roughness()[m]]),
            // A texture on a mesh without coordinates would not draw at all.
            texture: self.model.texture(geom.texid).filter(|_| has_uv),
            uv_scale,
        };
        let tint = look
            .texture
            .and_then(|t| self.model.texture_tint(t))
            .unwrap_or([1.0; 3]);
        let [r, g, b, _] = geom.rgba;
        Some(Surface {
            slot: GeomSlot::of(0, geom),
            counts: self.model.counts(mesh.id()),
            mesh,
            wire: Color::srgb(r * tint[0], g * tint[1], b * tint[2]),
            material: self.model.material(&mut self.materials, &look),
            transform,
            casts_shadow: geom.category != MjtCatBit::mjCAT_DECOR as i32 && geom.rgba[3] >= 0.999,
            owner: owner(model, geom),
        })
    }
}

/// Despawns every geom entity: a new model's list is another.
pub fn despawn_slots(mut commands: Commands, slots: Query<Entity, With<GeomSlot>>) {
    for entity in &slots {
        commands.entity(entity).despawn();
    }
}

type Slots<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut GeomSlot,
        &'static mut Mesh3d,
        &'static mut MeshMaterial3d<StandardMaterial>,
        &'static mut Transform,
        &'static mut Visibility,
        Has<NotShadowCaster>,
        Option<&'static mut WireframeColor>,
    ),
>;

/// Scratch for counting what is drawn, kept between frames.
#[derive(Default)]
pub struct Tally {
    by_body: Vec<DrawCounts>,
    materials: HashSet<AssetId<StandardMaterial>>,
    meshes: HashSet<AssetId<Mesh>>,
}

/// One entity per surface geom, reused in list order; slots past the end of
/// this frame's list are hidden, not despawned.
#[allow(clippy::too_many_arguments)]
pub fn sync_surfaces(
    mut commands: Commands,
    view: Res<MjView>,
    options: Res<MjRenderOptions>,
    mut assets: SurfaceAssets,
    mut slots: Slots,
    mut stats: ResMut<GeomStats>,
    mut surfaces: Local<Vec<Surface>>,
    mut tally: Local<Tally>,
) {
    surfaces.clear();
    let shown = |g: &&MjvGeom| {
        g.objtype != MjtObj::mjOBJ_GEOM as i32
            || usize::try_from(g.objid).map_or(true, |id| !view.hidden.contains(&id))
    };
    let model = view.model();
    surfaces.extend(
        geoms(&view, &options)
            .filter(|g| matches!(kind(g), Kind::Surface))
            .filter(shown)
            .filter_map(|g| assets.surface(model, g)),
    );
    for (index, surface) in surfaces.iter_mut().enumerate() {
        surface.slot.index = index;
        surface.casts_shadow &= options.shadows;
    }
    let hidden = options.wireframe.then(|| assets.hidden_material());

    for (
        entity,
        mut slot,
        mut mesh,
        mut material,
        mut transform,
        mut visibility,
        shadowless,
        wire,
    ) in &mut slots
    {
        let Some(surface) = surfaces.get(slot.index) else {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        };
        slot.set_if_neq(surface.slot);
        if mesh.0 != surface.mesh {
            mesh.0 = surface.mesh.clone();
        }
        // As wires, the geom itself draws nothing: `MuJoCo`'s wireframe
        // is the wires alone, each in its geom's colour.
        let fill = hidden.as_ref().unwrap_or(&surface.material);
        if material.0 != *fill {
            material.0 = fill.clone();
        }
        transform.set_if_neq(surface.transform);
        visibility.set_if_neq(Visibility::Inherited);
        if shadowless == surface.casts_shadow {
            if surface.casts_shadow {
                commands.entity(entity).remove::<NotShadowCaster>();
            } else {
                commands.entity(entity).insert(NotShadowCaster);
            }
        }
        let colour = WireframeColor {
            color: surface.wire,
        };
        match (options.wireframe, wire) {
            (true, Some(mut wire)) => {
                if wire.color != colour.color {
                    *wire = colour;
                }
            }
            (true, None) => {
                commands.entity(entity).insert((Wireframe, colour));
            }
            (false, Some(_)) => {
                commands
                    .entity(entity)
                    .remove::<(Wireframe, WireframeColor)>();
            }
            (false, None) => {}
        }
    }

    let existing = slots.iter().len();
    for surface in surfaces.iter().skip(existing) {
        // `Visibility` spelled out: `Mesh3d` gains it only through a render
        // plugin's required components, and without it the slot query above
        // never matches and every frame would spawn the list again.
        let mut entity = commands.spawn((
            surface.slot,
            Mesh3d(surface.mesh.clone()),
            MeshMaterial3d(hidden.clone().unwrap_or_else(|| surface.material.clone())),
            surface.transform,
            Visibility::Inherited,
        ));
        if !surface.casts_shadow {
            entity.insert(NotShadowCaster);
        }
        if options.wireframe {
            entity.insert((
                Wireframe,
                WireframeColor {
                    color: surface.wire,
                },
            ));
        }
    }

    count_surfaces(&surfaces, view.model().nbody(), &mut tally, &mut stats);
}

fn count_surfaces(surfaces: &[Surface], nbody: i64, tally: &mut Tally, stats: &mut GeomStats) {
    let Tally {
        by_body,
        materials,
        meshes,
    } = tally;
    by_body.clear();
    by_body.resize(usize::try_from(nbody).unwrap_or(0), DrawCounts::default());
    materials.clear();
    meshes.clear();
    let (mut total, mut decor) = (DrawCounts::default(), DrawCounts::default());
    for surface in surfaces {
        total.add(surface.counts);
        match surface.owner.and_then(|b| by_body.get_mut(b)) {
            Some(body) => body.add(surface.counts),
            None => decor.add(surface.counts),
        }
        materials.insert(surface.material.id());
        meshes.insert(surface.mesh.id());
    }
    stats.surfaces = total.entities;
    stats.triangles = total.triangles;
    stats.vertices = total.vertices;
    stats.materials = materials.len();
    stats.meshes = meshes.len();
    stats.decor = decor;
    stats.by_subtree.clear();
    stats.by_subtree.extend(
        by_body
            .iter()
            .enumerate()
            .filter(|(_, c)| c.entities > 0)
            .map(|(b, c)| (b, *c)),
    );
}

/// Lines and line boxes as gizmos, redrawn each frame.
pub fn draw_lines(view: Res<MjView>, options: Res<MjRenderOptions>, mut gizmos: Gizmos) {
    if !options.lines {
        return;
    }
    for geom in geoms(&view, &options) {
        let [r, g, b, a] = geom.rgba;
        let color = Color::srgba(r, g, b, a);
        match kind(geom) {
            Kind::Line => {
                // `MuJoCo`'s line runs `size[2]` along the geom's own Z.
                let from = Vec3::from_array(geom.pos);
                let along = quat_from_mat(&geom.mat) * Vec3::Z * geom.size[2];
                gizmos.line(from, from + along, color);
            }
            Kind::LineBox => {
                gizmos.cube(frame_of(geom, 2.0 * Vec3::from_array(geom.size)), color);
            }
            _ => {}
        }
    }
}

/// Every geom's label, for the host to draw.
pub fn collect_labels(
    view: Res<MjView>,
    options: Res<MjRenderOptions>,
    mut labels: ResMut<Labels>,
) {
    labels.0.clear();
    labels
        .0
        .extend(geoms(&view, &options).filter(|g| g.label[0] != 0).map(|g| {
            // SAFETY: `MuJoCo` writes the label NUL-terminated within its 100
            // bytes, and `mjv_initGeom` starts it empty.
            let text = unsafe { CStr::from_ptr(g.label.as_ptr()) };
            Label {
                text: text.to_string_lossy().into_owned(),
                position: Vec3::from_array(g.pos),
            }
        }));
}

/// What the list holds by how it is drawn.
pub fn count_geoms(view: Res<MjView>, options: Res<MjRenderOptions>, mut stats: ResMut<GeomStats>) {
    let (mut listed, mut lines, mut labels_only) = (0, 0, 0);
    let mut unsupported: Vec<(i32, usize)> = Vec::new();
    for geom in geoms(&view, &options) {
        listed += 1;
        match kind(geom) {
            Kind::Line | Kind::LineBox => lines += 1,
            Kind::Label => labels_only += 1,
            Kind::Unsupported => match unsupported.iter_mut().find(|(t, _)| *t == geom.type_) {
                Some((_, n)) => *n += 1,
                None => unsupported.push((geom.type_, 1)),
            },
            Kind::Surface => {}
        }
    }
    stats.listed = listed;
    stats.lines = lines;
    stats.labels_only = labels_only;
    stats.unsupported = unsupported;
}
