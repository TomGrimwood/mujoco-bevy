//! Visual geoms drawn from a model of their own, on the bodies of the model
//! [`MjView`] draws.
//!
//! `MuJoCo` processes every mesh on every compile, and nothing in the
//! physics touches one that meets nothing, so a host may leave the visual
//! meshes out of the model it simulates and keep them in a second one,
//! compiled once: a look. Put [`MjLook`] on an entity. Its visual geoms are
//! built off the main thread, once per look however many entities show it,
//! then spawned as one entity per body ([`MjBody`], related to the
//! [`MjLook`] entity by [`VisualsOf`]) with the geoms as its children
//! ([`MjVisualGeom`]); [`VisualsSpawned`] is triggered on the [`MjLook`]
//! entity then. Each frame a body entity takes the pose of the body of the
//! same name, under the look's prefix, in [`MjView`]'s data.
//!
//! A visual geom is a mesh `MuJoCo` would not collide (`contype` and
//! `conaffinity` 0) in a drawn group (0-2). A texture with texels below
//! half alpha (a decal on a quad) is drawn alpha-tested: only its outline
//! shows.
//!
//! What a part looks like is the host's to change: [`MjVisualGeom::styled`]
//! (the look's own material, shared by every entity showing it, until the
//! host gives the part another) and [`MjHiddenGeom`]. Its material,
//! `Visibility` and wires are this module's, from those and the render
//! options: in wireframe a part is its wires alone, in its colour, as the
//! geoms `MuJoCo` draws are.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use mujoco_bevy_assets::{
    BuiltMesh, BuiltTexture, GeomLook, MeshCounts, ModelAssets, build_model_mesh,
    build_model_texture, standard_material,
};
use mujoco_rs::prelude::*;

use crate::debug::MjRenderOptions;
use crate::{GeomStats, MjSystems, MjView, ModelChanged};

/// The highest group a visual geom is drawn in; 3 and up are colliders,
/// markers and debug shapes.
const LAST_DRAWN_GROUP: i32 = 2;
/// `mat_texid`'s colour role (`mjTEXROLE_RGB`).
const ALBEDO: usize = 1;
/// `MuJoCo`'s geom colour when the MJCF sets none: a geom with a material
/// and this colour takes the material's.
const DEFAULT_RGBA: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

pub(crate) fn plugin(app: &mut App) {
    app.init_asset::<MjVisuals>()
        .init_resource::<Looks>()
        .init_resource::<Tints>()
        .add_observer(start_build)
        .add_observer(respawn)
        .add_systems(
            PostUpdate,
            (
                (finish_builds, spawn_visuals, sync_bodies, sync_parts).chain(),
                count_visuals
                    .after(sync_parts)
                    .after(crate::scene::sync_surfaces),
            )
                .in_set(MjSystems::Draw),
        );
}

/// A look to draw: a compiled model holding visual geoms, and the prefix
/// its bodies' names have in [`MjView`]'s model.
#[derive(Component, Clone)]
pub struct MjLook {
    pub model: Arc<MjModel>,
    pub prefix: String,
}

/// A body of a look, on the [`MjLook`] entity it belongs to.
#[derive(Component, Debug)]
#[relationship(relationship_target = Visuals)]
pub struct VisualsOf(pub Entity);

/// The bodies drawn for an [`MjLook`]; despawned with it.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = VisualsOf, linked_spawn)]
pub struct Visuals(Vec<Entity>);

impl Visuals {
    /// The body entities.
    pub fn bodies(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
}

/// A body of a look, posed each frame as the body of this name in
/// [`MjView`]'s model; hidden while that model has none.
#[derive(Component, Debug)]
pub struct MjBody {
    /// Prefixed, as in [`MjView`]'s model.
    pub name: String,
    id: Option<usize>,
}

/// A visual geom of a look, a child of its [`MjBody`].
#[derive(Component, Clone, Debug)]
pub struct MjVisualGeom {
    /// The geom's and its material's names in the look (unprefixed).
    pub name: Option<String>,
    pub material: Option<String>,
    /// Shown while [`MjView`]'s option shows this geom group (and no
    /// [`MjHiddenGeom`] is on it).
    pub group: i32,
    /// The look's own material for it, shared by every entity showing the
    /// look: what a host that restyled the part goes back to.
    pub drawn_as: Handle<StandardMaterial>,
    /// The material the host draws it in instead (painted, in a livery).
    pub styled: Option<Handle<StandardMaterial>>,
    /// The size of its mesh, for [`GeomStats`].
    pub counts: MeshCounts,
    /// A skin (a hose bent by bones): shown by the option's skin groups,
    /// not its geom groups.
    pub skin: bool,
}

impl MjVisualGeom {
    /// The material it is filled with when not in wireframe.
    #[must_use]
    pub fn fill(&self) -> &Handle<StandardMaterial> {
        self.styled.as_ref().unwrap_or(&self.drawn_as)
    }
}

/// Each look texture's mean colour, by image: a part's wire colour is its
/// material's times its texture's.
#[derive(Resource, Default)]
struct Tints(HashMap<AssetId<Image>, [f32; 3]>);

/// A visual geom the host hides (a livery's other body, a tool not
/// carried), whatever its group.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct MjHiddenGeom;

/// On an [`MjLook`] entity once its visuals are spawned: their assets, for
/// a host to find the look's materials and textures by name.
#[derive(Component, Clone, Debug)]
pub struct MjLookVisuals(pub Handle<MjVisuals>);

/// Triggered on an [`MjLook`] entity once its visuals are spawned.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct VisualsSpawned {
    pub entity: Entity,
}

/// A look's visual geoms as assets, shared by every entity showing it.
#[derive(Asset, TypePath)]
pub struct MjVisuals {
    pub parts: Vec<VisualPart>,
    /// The look's skins, drawn on the bodies of their bones.
    pub skins: Vec<VisualSkin>,
    /// The look's materials as drawn, by name.
    pub materials: HashMap<String, Handle<StandardMaterial>>,
    /// The look's 2D textures, by name.
    pub textures: HashMap<String, Handle<Image>>,
    /// All the parts' meshes together.
    pub counts: MeshCounts,
}

impl MjVisuals {
    /// 2D texture `name` of `model` (this look's), built on first ask. A
    /// look builds only the textures its materials draw with; a host that
    /// restyles a part with another (a livery) gets it here, built on the
    /// calling thread.
    pub fn texture(
        &mut self,
        model: &MjModel,
        name: &str,
        images: &mut Assets<Image>,
    ) -> Option<Handle<Image>> {
        if let Some(handle) = self.textures.get(name) {
            return Some(handle.clone());
        }
        let id = model.name_to_id(MjtObjBase::mjOBJ_TEXTURE, name)?;
        let handle = images.add(build_model_texture(model, id)?.image);
        self.textures.insert(name.to_owned(), handle.clone());
        Some(handle)
    }
}

/// One visual geom of a look.
pub struct VisualPart {
    /// Its body's name in the look (unprefixed).
    pub body: String,
    pub geom: MjVisualGeom,
    /// In its body's frame.
    pub transform: Transform,
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}

/// One skin of a look: a mesh bent by its bones, which are bodies.
pub struct VisualSkin {
    /// Its bones' bodies' names in the look (unprefixed), in joint order.
    pub bones: Vec<String>,
    pub inverse_bindposes: Handle<SkinnedMeshInverseBindposes>,
    pub geom: MjVisualGeom,
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}

/// Every look built or being built, by its model.
#[derive(Resource, Default)]
struct Looks(HashMap<usize, Look>);

/// Each holds its model so the model's address, the key, is not reused.
enum Look {
    Building(Task<Source>, Arc<MjModel>),
    Built {
        visuals: Handle<MjVisuals>,
        _model: Arc<MjModel>,
    },
}

fn key(model: &Arc<MjModel>) -> usize {
    Arc::as_ptr(model) as usize
}

/// What a build makes off the main thread.
struct Source {
    meshes: Vec<BuiltMesh>,
    /// By the texture's id in the model, with its name.
    textures: Vec<(usize, String, Option<BuiltTexture>)>,
    materials: Vec<MaterialSource>,
    parts: Vec<PartSource>,
    skins: Vec<SkinSource>,
}

struct SkinSource {
    bones: Vec<String>,
    inverse_bindposes: Vec<Mat4>,
    mesh: Mesh,
    geom: MjVisualGeom,
    material: usize,
}

struct MaterialSource {
    name: Option<String>,
    look: GeomLook,
}

struct PartSource {
    body: String,
    geom: MjVisualGeom,
    transform: Transform,
    mesh: usize,
    material: usize,
}

fn start_build(add: On<Add<MjLook>>, looks: Query<&MjLook>, mut cache: ResMut<Looks>) {
    let Ok(look) = looks.get(add.entity) else {
        return;
    };
    let model = Arc::clone(&look.model);
    cache.0.entry(key(&model)).or_insert_with(|| {
        let built = Arc::clone(&model);
        let task = AsyncComputeTaskPool::get().spawn(async move { Source::of(&built) });
        Look::Building(task, model)
    });
}

/// A look replaced on its entity is drawn anew.
fn respawn(insert: On<Insert<MjLook>>, mut commands: Commands, spawned: Query<(), With<Spawned>>) {
    if spawned.contains(insert.entity) {
        commands
            .entity(insert.entity)
            .remove::<(Spawned, MjLookVisuals)>()
            .despawn_related::<Visuals>();
    }
}

/// The [`MjLook`] entity's visuals are spawned.
#[derive(Component)]
struct Spawned;

fn finish_builds(
    mut cache: ResMut<Looks>,
    mut tints: ResMut<Tints>,
    mut visuals: ResMut<Assets<MjVisuals>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    for look in cache.0.values_mut() {
        let Look::Building(task, model) = look else {
            continue;
        };
        let Some(source) = check_ready(task) else {
            continue;
        };
        let built = source.into_assets(
            &mut meshes,
            &mut images,
            &mut materials,
            &mut bindposes,
            &mut tints,
        );
        *look = Look::Built {
            visuals: visuals.add(built),
            _model: Arc::clone(model),
        };
    }
}

fn spawn_visuals(
    mut commands: Commands,
    looks: Query<(Entity, &MjLook), Without<Spawned>>,
    cache: Res<Looks>,
    visuals: Res<Assets<MjVisuals>>,
) {
    for (root, look) in &looks {
        let Some(Look::Built {
            visuals: handle, ..
        }) = cache.0.get(&key(&look.model))
        else {
            continue;
        };
        let Some(built) = visuals.get(handle) else {
            continue;
        };
        let mut bodies: HashMap<String, Entity> = HashMap::new();
        let mut body_of = |name: &str, commands: &mut Commands| -> Entity {
            *bodies.entry(name.to_owned()).or_insert_with(|| {
                commands
                    .spawn((
                        MjBody {
                            name: format!("{}{name}", look.prefix),
                            id: None,
                        },
                        VisualsOf(root),
                        Transform::default(),
                        Visibility::Hidden,
                    ))
                    .id()
            })
        };
        for part in &built.parts {
            let body = body_of(&part.body, &mut commands);
            commands.spawn((
                ChildOf(body),
                part.geom.clone(),
                Mesh3d(part.mesh.clone()),
                MeshMaterial3d(part.material.clone()),
                part.transform,
                Visibility::Inherited,
            ));
        }
        for skin in &built.skins {
            let joints: Vec<Entity> = skin
                .bones
                .iter()
                .map(|bone| body_of(bone, &mut commands))
                .collect();
            let Some(&first) = joints.first() else {
                continue;
            };
            // Skinned, a mesh is placed by its joints alone; under its first
            // bone it is counted, styled and shown with that body. Its box
            // is its rest pose's, wherever the machine stands: never culled.
            commands.spawn((
                ChildOf(first),
                skin.geom.clone(),
                Mesh3d(skin.mesh.clone()),
                MeshMaterial3d(skin.material.clone()),
                SkinnedMesh {
                    inverse_bindposes: skin.inverse_bindposes.clone(),
                    joints,
                },
                NoFrustumCulling,
                Transform::default(),
                Visibility::Inherited,
            ));
        }
        commands
            .entity(root)
            .insert((Spawned, MjLookVisuals(handle.clone())));
        commands.trigger(VisualsSpawned { entity: root });
    }
}

/// Each body at its pose in [`MjView`]'s data, its id found by name when
/// it is new or the view's model is.
fn sync_bodies(
    view: Res<MjView>,
    mut changed: MessageReader<ModelChanged>,
    mut bodies: Query<(&mut MjBody, &mut Transform, &mut Visibility)>,
) {
    let model = view.model();
    let renamed = changed.read().count() > 0;
    let (xpos, xquat) = (view.data.xpos(), view.data.xquat());
    for (mut body, mut transform, mut visibility) in &mut bodies {
        if renamed || body.id.is_none() {
            let id = model.name_to_id(MjtObjBase::mjOBJ_BODY, &body.name);
            if body.id != id {
                body.id = id;
            }
        }
        let Some(id) = body.id else {
            visibility.set_if_neq(Visibility::Hidden);
            continue;
        };
        let [x, y, z] = xpos[id];
        let [w, i, j, k] = xquat[id];
        #[allow(clippy::cast_possible_truncation)] // world poses fit an f32
        let pose = Transform {
            translation: Vec3::new(x as f32, y as f32, z as f32),
            rotation: Quat::from_xyzw(i as f32, j as f32, k as f32, w as f32),
            scale: Vec3::ONE,
        };
        transform.set_if_neq(pose);
        visibility.set_if_neq(Visibility::Inherited);
    }
}

/// Each part shown as its group and the render options say, filled with
/// its material, or in wireframe its wires alone in its colour.
#[allow(clippy::type_complexity)]
fn sync_parts(
    mut commands: Commands,
    view: Res<MjView>,
    options: Res<MjRenderOptions>,
    mut assets: ResMut<ModelAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    tints: Res<Tints>,
    mut parts: Query<(
        Entity,
        &MjVisualGeom,
        Option<&mut MeshMaterial3d<StandardMaterial>>,
        &mut Visibility,
        Has<MjHiddenGeom>,
        Has<NotShadowCaster>,
        Option<&mut WireframeColor>,
    )>,
) {
    let (groups, skin_groups) = (view.option.geomgroup, view.option.skingroup);
    let hidden_fill = options
        .wireframe
        .then(|| assets.hidden_material(&mut materials));
    for (entity, geom, material, mut visibility, hidden, shadowless, wire) in &mut parts {
        let shown = !hidden
            && usize::try_from(geom.group)
                .ok()
                .and_then(|g| {
                    if geom.skin {
                        skin_groups.get(g)
                    } else {
                        groups.get(g)
                    }
                })
                .is_some_and(|&on| on != 0);
        visibility.set_if_neq(if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
        // A host may draw a part in a material of its own (one extended
        // from its fill); it is still shown, shadowed and wired here.
        let fill = hidden_fill.as_ref().unwrap_or_else(|| geom.fill());
        if let Some(mut material) = material
            && material.0 != *fill
        {
            material.0 = fill.clone();
        }
        if shadowless == options.shadows {
            if options.shadows {
                commands.entity(entity).remove::<NotShadowCaster>();
            } else {
                commands.entity(entity).insert(NotShadowCaster);
            }
        }
        match (options.wireframe, wire) {
            (true, wire) => {
                let colour = WireframeColor {
                    color: wire_colour(materials.get(geom.fill()), &tints),
                };
                match wire {
                    Some(mut wire) => {
                        if wire.color != colour.color {
                            *wire = colour;
                        }
                    }
                    None => {
                        commands.entity(entity).insert((Wireframe, colour));
                    }
                }
            }
            (false, Some(_)) => {
                commands
                    .entity(entity)
                    .remove::<(Wireframe, WireframeColor)>();
            }
            (false, None) => {}
        }
    }
}

/// The parts shown, added to [`GeomStats`] after the geom list's own count
/// (which starts it again each frame): per top-level body, so a machine's
/// draw counts include its look.
fn count_visuals(
    view: Res<MjView>,
    mut stats: ResMut<GeomStats>,
    bodies: Query<(&MjBody, &Visibility, &Children)>,
    parts: Query<(
        &MjVisualGeom,
        &Visibility,
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
    )>,
    mut seen: Local<(HashSet<AssetId<Mesh>>, HashSet<AssetId<StandardMaterial>>)>,
) {
    let roots = view.model().body_rootid();
    let (meshes, materials) = &mut *seen;
    meshes.clear();
    materials.clear();
    for (body, visibility, children) in &bodies {
        let (Some(id), false) = (body.id, *visibility == Visibility::Hidden) else {
            continue;
        };
        let root = roots
            .get(id)
            .and_then(|&r| usize::try_from(r).ok())
            .unwrap_or(0);
        let mut counts = crate::DrawCounts::default();
        for (geom, visibility, mesh, material) in parts.iter_many(children).flatten() {
            if *visibility == Visibility::Hidden {
                continue;
            }
            counts.entities += 1;
            counts.triangles += geom.counts.triangles;
            counts.vertices += geom.counts.vertices;
            meshes.insert(mesh.id());
            materials.insert(material.id());
        }
        if counts.entities == 0 {
            continue;
        }
        stats.surfaces += counts.entities;
        stats.triangles += counts.triangles;
        stats.vertices += counts.vertices;
        match stats.by_subtree.binary_search_by_key(&root, |(b, _)| *b) {
            Ok(k) => {
                let c = &mut stats.by_subtree[k].1;
                c.entities += counts.entities;
                c.triangles += counts.triangles;
                c.vertices += counts.vertices;
            }
            Err(k) => stats.by_subtree.insert(k, (root, counts)),
        }
    }
    stats.meshes += meshes.len();
    stats.materials += materials.len();
}

/// A part's wire colour: its material's, times its texture's mean colour.
fn wire_colour(material: Option<&StandardMaterial>, tints: &Tints) -> Color {
    let Some(material) = material else {
        return Color::WHITE;
    };
    let tint = material
        .base_color_texture
        .as_ref()
        .and_then(|t| tints.0.get(&t.id()))
        .copied()
        .unwrap_or([1.0; 3]);
    let base = material.base_color.to_srgba();
    Color::srgb(
        base.red * tint[0],
        base.green * tint[1],
        base.blue * tint[2],
    )
}

impl Source {
    /// `model`'s visual geoms, their meshes, textures and materials, built.
    fn of(model: &MjModel) -> Self {
        let name = |kind, id: usize| model.id_to_name(kind, id).map(str::to_owned);
        let visual = |g: usize| {
            model.geom_type()[g] == MjtGeomBase::mjGEOM_MESH
                && model.geom_contype()[g] == 0
                && model.geom_conaffinity()[g] == 0
                && model.geom_group()[g] <= LAST_DRAWN_GROUP
        };
        let geoms: Vec<usize> = (0..model.geom_type().len())
            .filter(|&g| visual(g))
            .collect();

        let mut mesh_index: HashMap<usize, usize> = HashMap::new();
        let mut meshes = Vec::new();
        let mut materials: Vec<MaterialSource> = Vec::new();
        let mut material_index: HashMap<(i32, [u32; 4], bool), usize> = HashMap::new();
        let mut used_textures: Vec<bool> = vec![false; to_usize(model.ntex())];
        let mut parts = Vec::with_capacity(geoms.len());
        for g in geoms {
            let Ok(mesh_id) = usize::try_from(model.geom_dataid()[g]) else {
                continue;
            };
            let mesh = *mesh_index.entry(mesh_id).or_insert_with(|| {
                meshes.push(build_model_mesh(model, mesh_id));
                meshes.len() - 1
            });
            let has_uv = meshes[mesh].has_uv;
            let matid = model.geom_matid()[g];
            let rgba = model.geom_rgba()[g];
            let material = *material_index
                .entry((matid, rgba.map(f32::to_bits), has_uv))
                .or_insert_with(|| {
                    let look = geom_look(model, matid, rgba, has_uv);
                    if let Some(t) = look.texture {
                        used_textures[t] = true;
                    }
                    materials.push(MaterialSource {
                        name: usize::try_from(matid)
                            .ok()
                            .and_then(|m| name(MjtObjBase::mjOBJ_MATERIAL, m)),
                        look,
                    });
                    materials.len() - 1
                });
            let body = usize::try_from(model.geom_bodyid()[g]).unwrap_or(0);
            let [x, y, z] = model.geom_pos()[g];
            let [w, i, j, k] = model.geom_quat()[g];
            #[allow(clippy::cast_possible_truncation)] // a part's pose fits an f32
            let transform = Transform {
                translation: Vec3::new(x as f32, y as f32, z as f32),
                rotation: Quat::from_xyzw(i as f32, j as f32, k as f32, w as f32),
                scale: Vec3::ONE,
            };
            parts.push(PartSource {
                body: name(MjtObjBase::mjOBJ_BODY, body).unwrap_or_default(),
                geom: MjVisualGeom {
                    name: name(MjtObjBase::mjOBJ_GEOM, g),
                    material: materials[material].name.clone(),
                    group: model.geom_group()[g],
                    drawn_as: Handle::default(),
                    styled: None,
                    counts: MeshCounts::default(),
                    skin: false,
                },
                transform,
                mesh,
                material,
            });
        }
        // Every 2D texture a material of the model names, not only those the
        // visuals draw now: a host may swap one in (a livery); a skin draws
        // its material's.
        for m in 0..to_usize(model.nmat()) {
            if let Ok(t) = usize::try_from(model.mat_texid()[m][ALBEDO]) {
                used_textures[t] = true;
            }
        }
        let textures = used_textures
            .iter()
            .enumerate()
            .filter(|&(_, &used)| used)
            .map(|(t, _)| {
                (
                    t,
                    name(MjtObjBase::mjOBJ_TEXTURE, t).unwrap_or_default(),
                    build_model_texture(model, t),
                )
            })
            .collect();
        let skins = (0..to_usize(model.nskin()))
            .map(|k| {
                let matid = model.skin_matid()[k];
                let has_uv = model.skin_texcoordadr()[k] >= 0;
                let look = geom_look(model, matid, model.skin_rgba()[k], has_uv);
                materials.push(MaterialSource {
                    name: usize::try_from(matid)
                        .ok()
                        .and_then(|m| name(MjtObjBase::mjOBJ_MATERIAL, m)),
                    look,
                });
                skin_of(model, k, materials.len() - 1, &name)
            })
            .collect();
        Self {
            meshes,
            textures,
            materials,
            parts,
            skins,
        }
    }

    fn into_assets(
        self,
        meshes: &mut Assets<Mesh>,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
        bindposes: &mut Assets<SkinnedMeshInverseBindposes>,
        tints: &mut Tints,
    ) -> MjVisuals {
        let mut counts = MeshCounts::default();
        let mesh_handles: Vec<(Handle<Mesh>, MeshCounts)> = self
            .meshes
            .into_iter()
            .map(|m| (meshes.add(m.mesh), m.counts))
            .collect();
        // By id in the model, for the materials to find.
        let mut by_id: HashMap<usize, (Handle<Image>, bool)> = HashMap::new();
        let mut textures = HashMap::new();
        for (id, name, built) in self.textures {
            let Some(built) = built else { continue };
            let handle = images.add(built.image);
            tints.0.insert(handle.id(), built.tint);
            textures.insert(name, handle.clone());
            by_id.insert(id, (handle, built.cutout));
        }
        let mut named = HashMap::new();
        let material_handles: Vec<Handle<StandardMaterial>> = self
            .materials
            .iter()
            .map(|m| {
                let (image, cutout) = m.look.texture.and_then(|t| by_id.get(&t).cloned()).unzip();
                let handle =
                    materials.add(standard_material(&m.look, image, cutout.unwrap_or(false)));
                if let Some(name) = &m.name {
                    named.entry(name.clone()).or_insert_with(|| handle.clone());
                }
                handle
            })
            .collect();
        let parts = self
            .parts
            .into_iter()
            .map(|p| {
                let (mesh, c) = &mesh_handles[p.mesh];
                counts.vertices += c.vertices;
                counts.triangles += c.triangles;
                let material = material_handles[p.material].clone();
                VisualPart {
                    body: p.body,
                    geom: MjVisualGeom {
                        drawn_as: material.clone(),
                        counts: *c,
                        ..p.geom
                    },
                    transform: p.transform,
                    mesh: mesh.clone(),
                    material,
                }
            })
            .collect();
        let skins = self
            .skins
            .into_iter()
            .map(|skin| {
                let material = material_handles[skin.material].clone();
                counts.vertices += skin.geom.counts.vertices;
                counts.triangles += skin.geom.counts.triangles;
                VisualSkin {
                    bones: skin.bones,
                    inverse_bindposes: bindposes.add(skin.inverse_bindposes),
                    geom: MjVisualGeom {
                        drawn_as: material.clone(),
                        ..skin.geom
                    },
                    mesh: meshes.add(skin.mesh),
                    material,
                }
            })
            .collect();
        MjVisuals {
            parts,
            skins,
            materials: named,
            textures,
            counts,
        }
    }
}

/// Skin `k` of `model` as a mesh skinned by its bones: each vertex bent by
/// its four heaviest bones (renormalised), each bone's inverse bind pose
/// from `MuJoCo`'s bind pose, so a vertex goes where `MuJoCo` puts it:
/// the body's pose times the bind pose's inverse.
fn skin_of(
    model: &MjModel,
    k: usize,
    material: usize,
    name: &impl Fn(MjtObjBase, usize) -> Option<String>,
) -> SkinSource {
    let (v0, nv) = (
        to_usize(model.skin_vertadr()[k]),
        to_usize(model.skin_vertnum()[k]),
    );
    let (f0, nf) = (
        to_usize(model.skin_faceadr()[k]),
        to_usize(model.skin_facenum()[k]),
    );
    let (b0, nb) = (
        to_usize(model.skin_boneadr()[k]),
        to_usize(model.skin_bonenum()[k]),
    );
    let positions: Vec<[f32; 3]> = model.skin_vert()[v0..v0 + nv].to_vec();
    let mut weights: Vec<Vec<(u16, f32)>> = vec![Vec::new(); nv];
    let mut bones = Vec::with_capacity(nb);
    let mut inverse_bindposes = Vec::with_capacity(nb);
    for (j, b) in (b0..b0 + nb).enumerate() {
        let body = to_usize(model.skin_bonebodyid()[b]);
        bones.push(name(MjtObjBase::mjOBJ_BODY, body).unwrap_or_default());
        let [x, y, z] = model.skin_bonebindpos()[b];
        let [w, i, jq, kq] = model.skin_bonebindquat()[b];
        let bind = Mat4::from_rotation_translation(
            Quat::from_xyzw(i, jq, kq, w).normalize(),
            Vec3::new(x, y, z),
        );
        inverse_bindposes.push(bind.inverse());
        let (a, n) = (
            to_usize(model.skin_bonevertadr()[b]),
            to_usize(model.skin_bonevertnum()[b]),
        );
        for v in a..a + n {
            let vertex = to_usize(model.skin_bonevertid()[v]);
            if let Some(list) = weights.get_mut(vertex) {
                list.push((
                    u16::try_from(j).unwrap_or(0),
                    model.skin_bonevertweight()[v],
                ));
            }
        }
    }
    let (indices, joint_weights): (Vec<[u16; 4]>, Vec<[f32; 4]>) = weights
        .into_iter()
        .map(|mut list| {
            list.sort_by(|a, b| b.1.total_cmp(&a.1));
            list.truncate(4);
            let total: f32 = list.iter().map(|(_, w)| w).sum();
            let mut joints = [0u16; 4];
            let mut ws = [0.0f32; 4];
            for (slot, (joint, w)) in list.into_iter().enumerate() {
                joints[slot] = joint;
                ws[slot] = if total > 0.0 { w / total } else { 0.0 };
            }
            (joints, ws)
        })
        .unzip();
    let faces: Vec<u32> = model.skin_face()[f0..f0 + nf]
        .iter()
        .flat_map(|f| f.map(|c| u32::try_from(c).unwrap_or(0)))
        .collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(indices),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, joint_weights)
    .with_inserted_indices(Indices::U32(faces));
    if let Ok(t0) = usize::try_from(model.skin_texcoordadr()[k]) {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            model.skin_texcoord()[t0..t0 + nv].to_vec(),
        );
    }
    mesh.compute_smooth_normals();
    SkinSource {
        bones,
        inverse_bindposes,
        mesh,
        geom: MjVisualGeom {
            name: name(MjtObjBase::mjOBJ_SKIN, k),
            material: None,
            group: model.skin_group()[k],
            drawn_as: Handle::default(),
            styled: None,
            counts: MeshCounts {
                vertices: nv,
                triangles: nf,
            },
            skin: true,
        },
        material,
    }
}

/// How `MuJoCo` draws a geom of material `matid` and colour `rgba`.
fn geom_look(model: &MjModel, matid: i32, rgba: [f32; 4], has_uv: bool) -> GeomLook {
    let Ok(m) = usize::try_from(matid) else {
        return GeomLook {
            rgba,
            emission: 0.0,
            specular: 0.5,
            shininess: 0.5,
            pbr: None,
            texture: None,
            uv_scale: Vec2::ONE,
        };
    };
    GeomLook {
        rgba: if rgba == DEFAULT_RGBA {
            model.mat_rgba()[m]
        } else {
            rgba
        },
        emission: model.mat_emission()[m],
        specular: model.mat_specular()[m],
        shininess: model.mat_shininess()[m],
        pbr: Some([model.mat_metallic()[m], model.mat_roughness()[m]]),
        // A texture on a mesh without coordinates would not draw at all.
        texture: usize::try_from(model.mat_texid()[m][ALBEDO])
            .ok()
            .filter(|_| has_uv),
        uv_scale: Vec2::from_array(model.mat_texrepeat()[m]),
    }
}

fn to_usize<T: TryInto<usize>>(n: T) -> usize {
    n.try_into().unwrap_or(0)
}
