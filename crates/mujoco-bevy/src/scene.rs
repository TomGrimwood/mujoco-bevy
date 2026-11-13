//! The geom list as Bevy entities, kept from frame to frame.
//!
//! The list's order is stable while the model and the options are, so the
//! entity with [`GeomSlot`] `k` draws the `k`-th surface geom of the list.
//! Updating it in place keeps Bevy's change detection quiet for parts that do
//! not move, and spawning nothing per frame keeps the frame rate. Lines are
//! drawn as gizmos and labels are handed to the host as text.

use std::ffi::CStr;

use bevy::ecs::system::SystemParam;
use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use mujoco_rs::prelude::*;

use crate::{MjView, quat_from_mat};
use mujoco_bevy_assets::{GeomLook, ModelAssets};

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

/// What the last frame drew, for a host's panel.
#[derive(Resource, Default, Clone)]
pub struct GeomStats {
    /// Geoms in the scene list and the user scene.
    pub listed: usize,
    /// Drawn as Bevy meshes.
    pub surfaces: usize,
    /// Drawn as gizmo lines.
    pub lines: usize,
    /// Text only, handed over in [`Labels`].
    pub labels_only: usize,
    /// `(mjtGeom, count)` of what is not drawn yet: height fields, skins,
    /// flex, SDFs.
    pub unsupported: Vec<(i32, usize)>,
    pub triangles: usize,
}

/// Marks the entity that draws the `n`-th surface geom of the list.
#[derive(Component)]
pub struct GeomSlot(usize);

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

/// Every geom drawn this frame: the scene's, then the host's overlays.
fn geoms(view: &MjView) -> impl Iterator<Item = &MjvGeom> {
    view.scene.geoms().iter().chain(view.user_scene.geoms())
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
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    transform: Transform,
    casts_shadow: bool,
    triangles: usize,
}

/// The stores a surface's mesh and material come from.
#[derive(SystemParam)]
pub struct SurfaceAssets<'w> {
    model: ResMut<'w, ModelAssets>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
}

impl SurfaceAssets<'_> {
    /// The mesh, material and transform for a [`Kind::Surface`] geom: one
    /// shared unit mesh per primitive scaled by the geom's size (`MuJoCo`
    /// gives sizes as scales: radius, radius, half-length for a cylinder),
    /// the model's own mesh for a mesh geom.
    fn surface(&mut self, geom: &MjvGeom) -> Option<Surface> {
        let size = Vec3::from_array(geom.size);
        let mut uv_scale = Vec2::from_array(geom.texrepeat);
        let ty = geom.type_;
        let is = |g: MjtGeom| ty == g as i32;

        let primitives = self.model.primitives()?;
        let (mesh, transform, has_uv, triangles) = if is(MjtGeom::mjGEOM_MESH) {
            // Even ids are the mesh, odd ones its convex hull; the hull is
            // drawn as the mesh itself.
            let id = usize::try_from(geom.dataid / 2).ok()?;
            let (mesh, has_uv) = self.model.mesh(id)?;
            let triangles = self.model.mesh_triangles(id);
            (mesh.clone(), frame_of(geom, Vec3::ONE), has_uv, triangles)
        } else if is(MjtGeom::mjGEOM_SPHERE) {
            (
                primitives.sphere.clone(),
                frame_of(geom, Vec3::splat(size.x)),
                true,
                1024,
            )
        } else if is(MjtGeom::mjGEOM_ELLIPSOID) {
            (primitives.sphere.clone(), frame_of(geom, size), true, 1024)
        } else if is(MjtGeom::mjGEOM_BOX) {
            (primitives.cuboid.clone(), frame_of(geom, size), true, 12)
        } else if is(MjtGeom::mjGEOM_CYLINDER) {
            (primitives.cylinder.clone(), frame_of(geom, size), true, 128)
        } else if is(MjtGeom::mjGEOM_CAPSULE) {
            let radius = size.x.max(1e-6);
            let mesh = self.model.capsule(&mut self.meshes, size.z / radius);
            (mesh, frame_of(geom, Vec3::splat(radius)), true, 600)
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
                2,
            )
        } else if is(MjtGeom::mjGEOM_ARROW2) {
            (primitives.arrow2.clone(), frame_of(geom, size), true, 128)
        } else if is(MjtGeom::mjGEOM_ARROW) || is(MjtGeom::mjGEOM_ARROW1) {
            (primitives.arrow.clone(), frame_of(geom, size), true, 96)
        } else if is(MjtGeom::mjGEOM_TRIANGLE) {
            (
                primitives.triangle.clone(),
                frame_of(geom, size.with_z(1.0)),
                true,
                1,
            )
        } else {
            return None;
        };

        let look = GeomLook {
            rgba: geom.rgba,
            emission: geom.emission,
            specular: geom.specular,
            shininess: geom.shininess,
            // A texture on a mesh without coordinates would not draw at all.
            texture: self.model.texture(geom.texid).filter(|_| has_uv),
            uv_scale,
        };
        Some(Surface {
            mesh,
            material: self.model.material(&mut self.materials, &look),
            transform,
            casts_shadow: geom.category != MjtCatBit::mjCAT_DECOR as i32 && geom.rgba[3] >= 0.999,
            triangles,
        })
    }
}

/// Despawns every geom entity when the model changes: its list is another.
pub fn despawn_on_new_model(
    mut commands: Commands,
    view: Res<MjView>,
    slots: Query<Entity, With<GeomSlot>>,
    mut seen: Local<Option<u64>>,
) {
    if seen.replace(view.generation()) != Some(view.generation()) {
        for entity in &slots {
            commands.entity(entity).despawn();
        }
    }
}

type Slots<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static GeomSlot,
        &'static mut Mesh3d,
        &'static mut MeshMaterial3d<StandardMaterial>,
        &'static mut Transform,
        &'static mut Visibility,
        Has<NotShadowCaster>,
    ),
>;

/// One entity per surface geom, reused in list order; slots past the end of
/// this frame's list are hidden, not despawned.
pub fn sync_surfaces(
    mut commands: Commands,
    view: Res<MjView>,
    mut assets: SurfaceAssets,
    mut slots: Slots,
    mut stats: ResMut<GeomStats>,
    mut surfaces: Local<Vec<Surface>>,
) {
    surfaces.clear();
    surfaces.extend(
        geoms(&view)
            .filter(|g| matches!(kind(g), Kind::Surface))
            .filter_map(|g| assets.surface(g)),
    );

    for (entity, slot, mut mesh, mut material, mut transform, mut visibility, shadowless) in
        &mut slots
    {
        let Some(surface) = surfaces.get(slot.0) else {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        };
        if mesh.0 != surface.mesh {
            mesh.0 = surface.mesh.clone();
        }
        if material.0 != surface.material {
            material.0 = surface.material.clone();
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
    }

    let existing = slots.iter().len();
    for (index, surface) in surfaces.iter().enumerate().skip(existing) {
        let mut entity = commands.spawn((
            GeomSlot(index),
            Mesh3d(surface.mesh.clone()),
            MeshMaterial3d(surface.material.clone()),
            surface.transform,
        ));
        if !surface.casts_shadow {
            entity.insert(NotShadowCaster);
        }
    }

    stats.surfaces = surfaces.len();
    stats.triangles = surfaces.iter().map(|s| s.triangles).sum();
}

/// Lines and line boxes as gizmos, redrawn each frame.
pub fn draw_lines(view: Res<MjView>, mut gizmos: Gizmos) {
    for geom in geoms(&view) {
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
pub fn collect_labels(view: Res<MjView>, mut labels: ResMut<Labels>) {
    labels.0.clear();
    labels
        .0
        .extend(geoms(&view).filter(|g| g.label[0] != 0).map(|g| {
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
pub fn count_geoms(view: Res<MjView>, mut stats: ResMut<GeomStats>) {
    let (mut listed, mut lines, mut labels_only) = (0, 0, 0);
    let mut unsupported: Vec<(i32, usize)> = Vec::new();
    for geom in geoms(&view) {
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
