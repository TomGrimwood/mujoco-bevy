//! A compiled `MuJoCo` model's meshes, textures and materials as Bevy assets,
//! built once per model, and the unit primitive meshes every geom scales.
//!
//! Add [`MujocoAssetsPlugin`], then call [`ModelAssets::rebuild`] whenever the
//! model changes, and [`ModelAssets::update_mesh`] when one mesh is rewritten
//! in place. Drawing the model is `mujoco-bevy`'s business, not this crate's.
//!
//! Meshes and textures are keyed by their content: a recompile that leaves a
//! part unchanged keeps its Bevy asset, and two identical meshes in one model
//! share one.

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::hash::{DefaultHasher, Hasher};
use std::time::{Duration, Instant};

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Affine2;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use mujoco_rs::prelude::*;

pub mod shape;

/// Past this many cached materials the cache starts again: geoms that change
/// colour every frame (activation colours, a fading highlight) would
/// otherwise grow it without end.
const MATERIAL_CACHE_CAP: usize = 4096;

/// Everything made for one model. A new model replaces it; a handle the new
/// model does not reuse drops, and Bevy frees the asset once no entity holds
/// it.
#[derive(Resource, Default)]
pub struct ModelAssets {
    meshes: Vec<ModelMesh>,
    textures: Vec<Option<Handle<Image>>>,
    /// Each texture's mean colour, as stored: what a wire of a textured
    /// geom is drawn in.
    tints: Vec<Option<[f32; 3]>>,
    /// A material that draws nothing: a geom shown as its wires alone.
    hidden: Option<Handle<StandardMaterial>>,
    /// The meshes and textures of the current model by content, for the next
    /// rebuild to reuse.
    mesh_by_content: HashMap<u64, ModelMesh>,
    texture_by_content: HashMap<u64, Handle<Image>>,
    materials: HashMap<MaterialKey, Handle<StandardMaterial>>,
    primitives: Option<Primitives>,
    capsules: HashMap<u32, Handle<Mesh>>,
    /// Convex hulls by `MuJoCo` mesh id, made the first time one is drawn.
    hulls: HashMap<usize, Handle<Mesh>>,
    /// The size of every mesh made here, for counting what is drawn.
    counts: HashMap<AssetId<Mesh>, MeshCounts>,
    last_rebuild: RebuildStats,
}

/// How big a Bevy mesh is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshCounts {
    pub vertices: usize,
    pub triangles: usize,
}

impl MeshCounts {
    #[must_use]
    pub fn of(mesh: &Mesh) -> Self {
        let vertices = mesh.count_vertices();
        let corners = mesh.indices().map_or(vertices, Indices::len);
        Self {
            vertices,
            triangles: corners / 3,
        }
    }
}

/// What the last [`ModelAssets::rebuild`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RebuildStats {
    pub took: Duration,
    pub meshes_built: usize,
    /// Kept from the model before, or shared with an identical mesh.
    pub meshes_reused: usize,
    pub textures_built: usize,
    pub textures_reused: usize,
}

#[derive(Clone)]
struct ModelMesh {
    handle: Handle<Mesh>,
    has_uv: bool,
    vertices: usize,
    triangles: usize,
}

/// One unit mesh per primitive type, scaled per geom by its size: radius 1,
/// half-length 1, centred, long axis on Z as `MuJoCo` has it.
#[derive(Clone)]
pub struct Primitives {
    pub sphere: Handle<Mesh>,
    pub cuboid: Handle<Mesh>,
    pub cylinder: Handle<Mesh>,
    pub plane: Handle<Mesh>,
    /// From the origin to +Z 1, one head.
    pub arrow: Handle<Mesh>,
    /// From the origin to +Z 1, a head at each end.
    pub arrow2: Handle<Mesh>,
    /// Corners at the origin, +X 1 and +Y 1, both faces.
    pub triangle: Handle<Mesh>,
}

impl Primitives {
    fn all(&self) -> impl Iterator<Item = &Handle<Mesh>> {
        [
            &self.sphere,
            &self.cuboid,
            &self.cylinder,
            &self.plane,
            &self.arrow,
            &self.arrow2,
            &self.triangle,
        ]
        .into_iter()
    }
}

/// What a geom's look is made of, quantised so near-equal looks share one
/// material.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialKey {
    texture: i32,
    rgba: [u8; 4],
    emission: u8,
    specular: u8,
    shininess: u8,
    /// `metallic` and `roughness`, 255 for one left unset.
    pbr: [u8; 2],
    uv_scale: [u32; 2],
}

/// The parts of an `mjvGeom` that pick its material.
pub struct GeomLook {
    pub rgba: [f32; 4],
    pub emission: f32,
    pub specular: f32,
    pub shininess: f32,
    /// The material's `metallic` and `roughness`, which `MuJoCo` carries for
    /// a physically based renderer and leaves negative when the MJCF sets
    /// neither; `None` for a geom with no material.
    pub pbr: Option<[f32; 2]>,
    /// A 2D texture of the model, already checked to exist.
    pub texture: Option<usize>,
    pub uv_scale: Vec2,
}

impl ModelAssets {
    /// The Bevy mesh for `MuJoCo` mesh `id`, and whether it has texture
    /// coordinates.
    #[must_use]
    pub fn mesh(&self, id: usize) -> Option<(&Handle<Mesh>, bool)> {
        self.meshes.get(id).map(|m| (&m.handle, m.has_uv))
    }

    #[must_use]
    pub const fn primitives(&self) -> Option<&Primitives> {
        self.primitives.as_ref()
    }

    /// The model's 2D texture `id`, if it was loaded.
    #[must_use]
    pub fn texture(&self, id: i32) -> Option<usize> {
        let id = usize::try_from(id).ok()?;
        self.textures.get(id)?.as_ref().map(|_| id)
    }

    /// (meshes, unique vertices, triangles) of the model's meshes.
    #[must_use]
    pub fn mesh_stats(&self) -> (usize, usize, usize) {
        let (v, t) = self
            .meshes
            .iter()
            .fold((0, 0), |(v, t), m| (v + m.vertices, t + m.triangles));
        (self.meshes.len(), v, t)
    }

    /// Triangles in `MuJoCo` mesh `id`.
    #[must_use]
    pub fn mesh_triangles(&self, id: usize) -> usize {
        self.meshes.get(id).map_or(0, |m| m.triangles)
    }

    /// The size of a mesh made here (a model mesh, a primitive, a capsule or
    /// a hull); zero for any other.
    #[must_use]
    pub fn counts(&self, mesh: AssetId<Mesh>) -> MeshCounts {
        self.counts.get(&mesh).copied().unwrap_or_default()
    }

    fn add_counted(&mut self, meshes: &mut Assets<Mesh>, mesh: Mesh) -> Handle<Mesh> {
        let counts = MeshCounts::of(&mesh);
        let handle = meshes.add(mesh);
        self.counts.insert(handle.id(), counts);
        handle
    }

    /// The convex hull `MuJoCo` made of mesh `id` for collisions, as a flat
    /// shaded mesh, built the first time it is asked for: what the convex
    /// hull flag draws. `None` for a mesh without one (one no collision geom
    /// uses).
    pub fn hull(
        &mut self,
        model: &MjModel,
        id: usize,
        meshes: &mut Assets<Mesh>,
    ) -> Option<Handle<Mesh>> {
        if let Some(handle) = self.hulls.get(&id) {
            return Some(handle.clone());
        }
        let mesh = hull_of(model, id)?;
        let handle = self.add_counted(meshes, mesh);
        self.hulls.insert(id, handle.clone());
        Some(handle)
    }

    /// A capsule of radius 1 whose straight part is `2 * ratio` long, for a
    /// geom scaled uniformly by its radius: a capsule cannot be scaled
    /// unevenly without squashing its ends. Ratios are bucketed on a log
    /// scale, within about 2.5 % of the length: a perturb handle stretches
    /// every frame, and even buckets made a mesh for each new length.
    pub fn capsule(&mut self, meshes: &mut Assets<Mesh>, ratio: f32) -> Handle<Mesh> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bucket = (ratio.clamp(0.0, 100.0).ln_1p() * 20.0).round() as u32;
        if let Some(handle) = self.capsules.get(&bucket) {
            return handle.clone();
        }
        let half = (bucket as f32 / 20.0).exp_m1();
        let handle = self.add_counted(
            meshes,
            Capsule3d::new(1.0, 2.0 * half)
                .mesh()
                .longitudes(24)
                .latitudes(12)
                .build()
                .rotated_by(Quat::from_rotation_x(FRAC_PI_2)),
        );
        self.capsules.insert(bucket, handle.clone());
        handle
    }

    /// The material for a geom's look, made on first use.
    pub fn material(
        &mut self,
        materials: &mut Assets<StandardMaterial>,
        look: &GeomLook,
    ) -> Handle<StandardMaterial> {
        let q = |x: f32| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let b = (x.clamp(0.0, 1.0) * 255.0).round() as u8;
            b
        };
        let key = MaterialKey {
            texture: look.texture.map_or(-1, |t| i32::try_from(t).unwrap_or(-1)),
            rgba: look.rgba.map(q),
            emission: q(look.emission),
            specular: q(look.specular),
            shininess: q(look.shininess),
            pbr: look
                .pbr
                .filter(|pbr| pbr.iter().any(|&x| x >= 0.0))
                .map_or([255; 2], |pbr| pbr.map(|x| q(x).min(254))),
            uv_scale: [look.uv_scale.x.to_bits(), look.uv_scale.y.to_bits()],
        };
        if let Some(handle) = self.materials.get(&key) {
            return handle.clone();
        }
        if self.materials.len() >= MATERIAL_CACHE_CAP {
            self.materials.clear();
        }
        let texture = look
            .texture
            .and_then(|t| self.textures.get(t).cloned().flatten());
        let material = standard_material(look, texture, false);
        let handle = materials.add(material);
        self.materials.insert(key, handle.clone());
        handle
    }

    /// The mean colour of texture `id`, as its texels are stored (sRGB for
    /// a colour map).
    #[must_use]
    pub fn texture_tint(&self, id: usize) -> Option<[f32; 3]> {
        self.tints.get(id).copied().flatten()
    }

    /// A material that draws nothing and casts no shadow (every fragment
    /// masked out): for a geom shown as its wires alone, which Bevy draws
    /// apart from the material.
    pub fn hidden_material(
        &mut self,
        materials: &mut Assets<StandardMaterial>,
    ) -> Handle<StandardMaterial> {
        self.hidden
            .get_or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: Color::NONE,
                    alpha_mode: AlphaMode::Mask(0.5),
                    unlit: true,
                    ..default()
                })
            })
            .clone()
    }

    /// Materials cached now, to show that a model swap lets go of them.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// What the last [`Self::rebuild`] cost and reused.
    #[must_use]
    pub const fn last_rebuild(&self) -> RebuildStats {
        self.last_rebuild
    }

    /// Replaces every mesh, texture and cached material with `model`'s,
    /// keeping the Bevy asset of any mesh or texture whose content the model
    /// before had too. Handles not kept drop here, so the old model's assets
    /// are freed once no entity holds them.
    pub fn rebuild(
        &mut self,
        model: &MjModel,
        meshes: &mut Assets<Mesh>,
        images: &mut Assets<Image>,
    ) {
        let started = Instant::now();
        let mut stats = RebuildStats::default();
        self.materials.clear();

        let mut before = std::mem::take(&mut self.mesh_by_content);
        self.meshes = (0..to_usize(model.nmesh()))
            .map(|id| {
                let key = mesh_key(model, id);
                let reused = self
                    .mesh_by_content
                    .get(&key)
                    .cloned()
                    .or_else(|| before.remove(&key));
                let mesh = reused.map_or_else(
                    || {
                        stats.meshes_built += 1;
                        build_mesh(model, id, meshes)
                    },
                    |mesh| {
                        stats.meshes_reused += 1;
                        mesh
                    },
                );
                self.mesh_by_content.insert(key, mesh.clone());
                mesh
            })
            .collect();

        let mut before = std::mem::take(&mut self.texture_by_content);
        let mut tints = Vec::new();
        self.textures = (0..to_usize(model.ntex()))
            .map(|id| {
                tints.push(None);
                let (key, image) = texture_key(model, id)?;
                tints[id] = Some(mean_colour(&image.rgba));
                let reused = self
                    .texture_by_content
                    .get(&key)
                    .cloned()
                    .or_else(|| before.remove(&key));
                let handle = if let Some(handle) = reused {
                    stats.textures_reused += 1;
                    handle
                } else {
                    stats.textures_built += 1;
                    images.add(build_texture(image))
                };
                self.texture_by_content.insert(key, handle.clone());
                Some(handle)
            })
            .collect();
        self.tints = tints;

        self.hulls.clear();
        self.recount(meshes);
        stats.took = started.elapsed();
        self.last_rebuild = stats;
    }

    /// The counts of the meshes held now, and only those.
    fn recount(&mut self, meshes: &Assets<Mesh>) {
        let held: Vec<AssetId<Mesh>> = self
            .meshes
            .iter()
            .map(|m| m.handle.id())
            .chain(self.capsules.values().map(Handle::id))
            .chain(self.hulls.values().map(Handle::id))
            .chain(self.primitives.iter().flat_map(|p| p.all().map(Handle::id)))
            .collect();
        self.counts.retain(|id, _| held.contains(id));
        for id in held {
            if !self.counts.contains_key(&id)
                && let Some(mesh) = meshes.get(id)
            {
                self.counts.insert(id, MeshCounts::of(mesh));
            }
        }
    }

    /// Rewrites `MuJoCo` mesh `id` from `model` under the handle it has, so
    /// every entity drawing it shows the new shape. For a mesh rewritten in
    /// place between recompiles; the topology may change, the mesh id may
    /// not. Returns whether there was such a mesh.
    pub fn update_mesh(&mut self, model: &MjModel, id: usize, meshes: &mut Assets<Mesh>) -> bool {
        let Some(slot) = self.meshes.get_mut(id) else {
            return false;
        };
        let (mesh, built) = mesh_of(model, id);
        // Its old content no longer describes it: keep it out of the reuse
        // map, or a later model with the old shape would get the new one.
        self.mesh_by_content.retain(|_, m| m.handle != slot.handle);
        slot.has_uv = built.has_uv;
        slot.vertices = built.vertices;
        slot.triangles = built.triangles;
        let handle = slot.handle.id();
        self.counts.insert(handle, MeshCounts::of(&mesh));
        // Its hull is the old shape's: built again when next asked for.
        self.hulls.remove(&id);
        // Same handle: the entities holding it need no change.
        meshes.insert(handle, mesh).is_ok()
    }
}

fn to_usize<T: TryInto<usize>>(n: T) -> usize {
    n.try_into().unwrap_or(0)
}

/// The [`ModelAssets`] resource, with its primitives built at start-up.
pub struct MujocoAssetsPlugin;

impl Plugin for MujocoAssetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ModelAssets>()
            .add_systems(Startup, init_primitives);
    }
}

/// Builds the shared primitives once.
fn init_primitives(mut assets: ResMut<ModelAssets>, mut meshes: ResMut<Assets<Mesh>>) {
    let z_up = Quat::from_rotation_x(FRAC_PI_2);
    let arrow = |two_heads: bool| {
        let head = 0.25;
        let shaft_from = if two_heads { head } else { 0.0 };
        let shaft_len = 1.0 - head - shaft_from;
        let mut mesh = Cylinder::new(0.5, shaft_len)
            .mesh()
            .resolution(16)
            .build()
            .rotated_by(z_up)
            .translated_by(Vec3::Z * (shaft_from + shaft_len / 2.0));
        let cone = || {
            Cone::new(1.0, head)
                .mesh()
                .resolution(16)
                .build()
                .rotated_by(z_up)
        };
        // The merge cannot fail: Bevy's primitives share one attribute set.
        let _ = mesh.merge(&cone().translated_by(Vec3::Z * (1.0 - head / 2.0)));
        if two_heads {
            let back = cone()
                .rotated_by(Quat::from_rotation_x(std::f32::consts::PI))
                .translated_by(Vec3::Z * (head / 2.0));
            let _ = mesh.merge(&back);
        }
        mesh
    };
    let triangle = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3])
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 1]));
    let mut add = |mesh: Mesh| assets.add_counted(&mut meshes, mesh);
    let primitives = Primitives {
        sphere: add(Sphere::new(1.0).mesh().uv(32, 18)),
        cuboid: add(Cuboid::new(2.0, 2.0, 2.0).into()),
        cylinder: add(Cylinder::new(1.0, 2.0)
            .mesh()
            .resolution(32)
            .build()
            .rotated_by(z_up)),
        plane: add(Plane3d::new(Vec3::Z, Vec2::ONE).into()),
        arrow: add(arrow(false)),
        arrow2: add(arrow(true)),
        triangle: add(triangle),
    };
    assets.primitives = Some(primitives);
}

/// A hash of everything a mesh is built from: its vertices, normals, texture
/// coordinates and faces.
fn mesh_key(model: &MjModel, id: usize) -> u64 {
    let v0 = to_usize(model.mesh_vertadr()[id]);
    let nv = to_usize(model.mesh_vertnum()[id]);
    let n0 = to_usize(model.mesh_normaladr()[id]);
    let nn = to_usize(model.mesh_normalnum()[id]);
    let f0 = to_usize(model.mesh_faceadr()[id]);
    let nf = to_usize(model.mesh_facenum()[id]);
    let mut hasher = DefaultHasher::new();
    hasher.write(as_bytes(&model.mesh_vert()[v0..v0 + nv]));
    hasher.write(as_bytes(&model.mesh_normal()[n0..n0 + nn]));
    hasher.write(as_bytes(&model.mesh_face()[f0..f0 + nf]));
    hasher.write(as_bytes(&model.mesh_facenormal()[f0..f0 + nf]));
    if let Ok(t0) = usize::try_from(model.mesh_texcoordadr()[id]) {
        let nt = to_usize(model.mesh_texcoordnum()[id]);
        hasher.write(as_bytes(&model.mesh_texcoord()[t0..t0 + nt]));
        hasher.write(as_bytes(&model.mesh_facetexcoord()[f0..f0 + nf]));
    }
    hasher.finish()
}

/// The bytes of plain number arrays, for hashing.
const fn as_bytes<T: Plain>(values: &[T]) -> &[u8] {
    // SAFETY: `Plain` types are arrays of `f32` or `i32`: no padding, every
    // byte initialised, and `u8` has no alignment to keep.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Arrays of `f32` or `i32`, with no padding.
trait Plain: Copy {}
impl Plain for [f32; 2] {}
impl Plain for [f32; 3] {}
impl Plain for [i32; 3] {}
impl Plain for u8 {}

fn build_mesh(model: &MjModel, id: usize, meshes: &mut Assets<Mesh>) -> ModelMesh {
    let (mesh, mut built) = mesh_of(model, id);
    built.handle = meshes.add(mesh);
    built
}

/// Mesh `id`'s convex hull from `MuJoCo`'s graph of it: `numvert`,
/// `numface`, then per vertex an edge address and a mesh vertex id, the
/// edges, and per face three mesh vertex ids. Three corners per face, so
/// each face has its own normal, as `MuJoCo`'s own renderer draws it.
fn hull_of(model: &MjModel, id: usize) -> Option<Mesh> {
    let at = usize::try_from(*model.mesh_graphadr().get(id)?).ok()?;
    let graph = model.mesh_graph();
    let numvert = to_usize(*graph.get(at)?);
    let numface = to_usize(*graph.get(at + 1)?);
    let faces = graph
        .get(at + 2 + 3 * numvert + 3 * numface..)?
        .get(..3 * numface)?;
    let v0 = to_usize(model.mesh_vertadr()[id]);
    let nv = to_usize(model.mesh_vertnum()[id]);
    let verts = &model.mesh_vert()[v0..v0 + nv];
    let (mut pos, mut nor) = (
        Vec::with_capacity(faces.len()),
        Vec::with_capacity(faces.len()),
    );
    for face in faces.as_chunks::<3>().0 {
        let corner = |k: usize| verts.get(to_usize(face[k])).copied().map(Vec3::from_array);
        let (a, b, c) = (corner(0)?, corner(1)?, corner(2)?);
        let normal = (b - a).cross(c - a).normalize_or_zero().to_array();
        pos.extend([a.to_array(), b.to_array(), c.to_array()]);
        nor.extend([normal; 3]);
    }
    let indices = (0..u32::try_from(pos.len()).ok()?).collect();
    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

/// The material `MuJoCo`'s look of a geom stands for, drawing `texture`.
/// A `cutout` texture (texels below half alpha: a decal on a quad) is drawn
/// alpha-tested, so only the decal's outline shows; `MuJoCo`'s own renderer
/// draws the quad whole.
#[must_use]
pub fn standard_material(
    look: &GeomLook,
    texture: Option<Handle<Image>>,
    cutout: bool,
) -> StandardMaterial {
    let [r, g, b, a] = look.rgba;
    let base = Color::srgba(r, g, b, a);
    // Bevy's emissive is a luminance that the camera's exposure scales
    // down by about a thousand, where `MuJoCo`'s is a fraction of the
    // colour; this makes emission 1 about as bright as a sunlit face.
    let emissive = LinearRgba::from(Color::srgb(r, g, b)) * (look.emission * 1000.0);
    // A material that says what it is made of is drawn as that. One that
    // does not, by eye: `MuJoCo`'s Phong shininess and specular weight
    // to a roughness and a reflectance that look about as glossy.
    let said = look.pbr.is_some_and(|pbr| pbr.iter().any(|&x| x >= 0.0));
    let [metallic, roughness] = look.pbr.unwrap_or_default();
    let (metallic, perceptual_roughness, reflectance) = if said {
        (
            metallic.clamp(0.0, 1.0),
            // Bevy's floor: a mirror aliases under a sun.
            if roughness < 0.0 {
                0.5
            } else {
                roughness.clamp(0.045, 1.0)
            },
            StandardMaterial::default().reflectance,
        )
    } else {
        (
            0.0,
            (1.0 - 0.75 * look.shininess).clamp(0.25, 1.0),
            (0.15 + 0.7 * look.specular).clamp(0.0, 1.0),
        )
    };
    StandardMaterial {
        base_color: base,
        base_color_texture: texture,
        uv_transform: Affine2::from_scale(look.uv_scale),
        emissive,
        metallic,
        perceptual_roughness,
        reflectance,
        alpha_mode: if a < 0.999 {
            AlphaMode::Blend
        } else if cutout {
            AlphaMode::Mask(0.5)
        } else {
            AlphaMode::Opaque
        },
        ..default()
    }
}

/// One indexed mesh, and what it holds (with a placeholder handle):
/// `MuJoCo` indexes positions, normals and texture coordinates apart, so
/// corners that agree on all three are merged.
fn mesh_of(model: &MjModel, id: usize) -> (Mesh, ModelMesh) {
    let v0 = to_usize(model.mesh_vertadr()[id]);
    let n0 = to_usize(model.mesh_normaladr()[id]);
    let f0 = to_usize(model.mesh_faceadr()[id]);
    let nf = to_usize(model.mesh_facenum()[id]);
    let t0 = usize::try_from(model.mesh_texcoordadr()[id]).ok();

    let (verts, normals) = (model.mesh_vert(), model.mesh_normal());
    let (faces, face_normals) = (model.mesh_face(), model.mesh_facenormal());
    let texcoords = t0.map(|_| (model.mesh_texcoord(), model.mesh_facetexcoord()));

    let mut corner_of: HashMap<(i32, i32, i32), u32> = HashMap::with_capacity(nf * 2);
    let (mut pos, mut nor, mut uv) = (Vec::new(), Vec::new(), Vec::new());
    let mut indices = Vec::with_capacity(nf * 3);
    for f in f0..f0 + nf {
        for c in 0..3 {
            let t = texcoords.map_or(0, |(_, ft)| ft[f][c]);
            let key = (faces[f][c], face_normals[f][c], t);
            let index = *corner_of.entry(key).or_insert_with(|| {
                pos.push(verts[v0 + to_usize(key.0)]);
                nor.push(normals[n0 + to_usize(key.1)]);
                if let (Some(t0), Some((tc, _))) = (t0, texcoords) {
                    uv.push(tc[t0 + to_usize(t)]);
                }
                u32::try_from(pos.len() - 1).unwrap_or(u32::MAX)
            });
            indices.push(index);
        }
    }
    let has_uv = !uv.is_empty();
    let shape = shape::distances(shape::Corners {
        pos: &mut pos,
        nor: &mut nor,
        uv: &mut uv,
        indices: &mut indices,
    });
    let (vertices, triangles) = (pos.len(), indices.len() / 3);
    // Bevy's fragment shader takes a second set of coordinates to need a
    // first.
    let uv = if has_uv { uv } else { vec![[0.0f32; 2]; vertices] };
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
    // Each corner's distance to the nearest outside edge and inside
    // corner ([`shape`]), for a material that weathers; Bevy's standard
    // material reads a second set only for a lightmap or a texture told to.
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, shape)
    .with_inserted_indices(Indices::U32(indices));
    let built = ModelMesh {
        handle: Handle::default(),
        has_uv,
        vertices,
        triangles,
    };
    (mesh, built)
}

/// A model mesh built as a Bevy mesh, not yet added to the assets.
pub struct BuiltMesh {
    pub mesh: Mesh,
    /// Whether it has texture coordinates: a texture on a mesh without
    /// them would not draw at all.
    pub has_uv: bool,
    pub counts: MeshCounts,
}

/// `MuJoCo` mesh `id` of `model` as a Bevy mesh: slow for a big mesh, so it
/// may run off the main thread.
#[must_use]
pub fn build_model_mesh(model: &MjModel, id: usize) -> BuiltMesh {
    let (mesh, built) = mesh_of(model, id);
    BuiltMesh {
        mesh,
        has_uv: built.has_uv,
        counts: MeshCounts {
            vertices: built.vertices,
            triangles: built.triangles,
        },
    }
}

/// A model texture built as a Bevy image, not yet added to the assets.
pub struct BuiltTexture {
    pub image: Image,
    /// Some texels are below half alpha: drawn alpha-tested
    /// ([`standard_material`]).
    pub cutout: bool,
    /// Its mean colour as stored: what a wire of a geom drawing it is
    /// drawn in.
    pub tint: [f32; 3],
}

/// 2D texture `id` of `model` as a Bevy image with its mips; `None` for a
/// texture of another kind. May run off the main thread.
#[must_use]
pub fn build_model_texture(model: &MjModel, id: usize) -> Option<BuiltTexture> {
    let source = decode_texture(model, id)?;
    let cutout = source.rgba.as_chunks::<4>().0.iter().any(|t| t[3] < 128);
    let tint = mean_colour(&source.rgba);
    Some(BuiltTexture {
        image: build_texture(source),
        cutout,
        tint,
    })
}

/// A 2D texture's pixels as RGBA, and its size and colour space.
struct TextureSource {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    format: TextureFormat,
}

/// The mean colour of RGBA texels, of a few thousand of them spread over
/// the image: a hue to name the texture by, not a measure of it.
fn mean_colour(rgba: &[u8]) -> [f32; 3] {
    let texels = rgba.as_chunks::<4>().0;
    let step = (texels.len() / 4096).max(1);
    let (mut sum, mut n) = ([0u64; 3], 0u64);
    for texel in texels.iter().step_by(step) {
        for (s, v) in sum.iter_mut().zip(texel) {
            *s += u64::from(*v);
        }
        n += 1;
    }
    #[allow(clippy::cast_precision_loss)] // a mean of bytes
    sum.map(|s| s as f32 / (255.0 * n.max(1) as f32))
}

/// A hash of a 2D texture's pixels, size and colour space, and the decoded
/// texture. Cube maps and the skybox are left out.
fn texture_key(model: &MjModel, id: usize) -> Option<(u64, TextureSource)> {
    let source = decode_texture(model, id)?;
    let mut hasher = DefaultHasher::new();
    hasher.write(as_bytes(&source.rgba));
    hasher.write_u32(source.width);
    hasher.write_u32(source.height);
    hasher.write_u8(u8::from(source.format == TextureFormat::Rgba8Unorm));
    Some((hasher.finish(), source))
}

fn decode_texture(model: &MjModel, id: usize) -> Option<TextureSource> {
    if model.tex_type()[id] != MjtTexture::mjTEXTURE_2D {
        return None;
    }
    let (w, h) = (
        u32::try_from(model.tex_width()[id]).ok()?,
        u32::try_from(model.tex_height()[id]).ok()?,
    );
    let channels = to_usize(model.tex_nchannel()[id]);
    let bytes = model.load_texture(id).ok()?;
    let rgba: Vec<u8> = match channels {
        4 => bytes.into_owned(),
        3 => bytes
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        1 => bytes.iter().flat_map(|&l| [l, l, l, 255]).collect(),
        _ => return None,
    };
    // A model texture is a colour (a material's base colour, all this
    // crate draws with), as the screen shows it: `MuJoCo`'s renderer puts
    // its bytes on the screen as they are. Its `colorspace="auto"` reads a
    // PNG without an sRGB chunk as linear, and taken at its word a texture
    // came out pale and washed toward yellow (Hitachi orange as amber).
    let format = TextureFormat::Rgba8UnormSrgb;
    Some(TextureSource {
        rgba,
        width: w,
        height: h,
        format,
    })
}

/// RGBA pixels as a repeating image with a full mip chain: a tiling
/// texture of a host's own (sRGB colour, or `linear` data).
#[must_use]
pub fn repeating_image(rgba: Vec<u8>, width: u32, height: u32, linear: bool) -> Image {
    build_texture(TextureSource {
        rgba,
        width,
        height,
        format: if linear {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba8UnormSrgb
        },
    })
}

/// An RGBA image (sRGB colour, or linear data) with a full mip chain,
/// repeating, as `MuJoCo` samples it.
fn build_texture(source: TextureSource) -> Image {
    let TextureSource {
        rgba,
        width: w,
        height: h,
        format,
    } = source;
    let base = rgba.len();
    let (data, levels) = mip_chain(rgba, w, h);
    // `Image::new` checks the data against the base level alone.
    let mut image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data[..base].to_vec(),
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..default()
    });
    image
}

/// Box-filtered mip levels appended to `base`. A ground texture repeated a
/// hundred times across the floor shimmers without them, and Bevy does not
/// make them.
fn mip_chain(base: Vec<u8>, mut w: u32, mut h: u32) -> (Vec<u8>, u32) {
    let mut data = base;
    let mut start = 0;
    let mut levels = 1;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let at = |xx: u32, yy: u32| {
                        u32::from(
                            data[start + ((yy.min(h - 1) * w + xx.min(w - 1)) * 4 + c) as usize],
                        )
                    };
                    let sum = at(2 * x, 2 * y)
                        + at(2 * x + 1, 2 * y)
                        + at(2 * x, 2 * y + 1)
                        + at(2 * x + 1, 2 * y + 1);
                    #[allow(clippy::cast_possible_truncation)]
                    next.push(((sum + 2) / 4) as u8);
                }
            }
        }
        start = data.len();
        data.extend_from_slice(&next);
        (w, h) = (nw, nh);
        levels += 1;
    }
    (data, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube given by its 8 corners and 12 triangles: every corner is
    /// shared by several faces, so indexing has something to merge.
    fn cube(texcoords: bool) -> MjModel {
        let texcoord = if texcoords {
            r#"texcoord="0 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1""#
        } else {
            ""
        };
        let xml = format!(
            r#"<mujoco><asset><mesh name="cube" smoothnormal="true"
                vertex="0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1"
                face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"
                {texcoord} /></asset>
              <worldbody><geom type="mesh" mesh="cube" /></worldbody></mujoco>"#
        );
        MjModel::from_xml_string(&xml).expect("the cube compiles")
    }

    fn built(model: &MjModel) -> (ModelAssets, Assets<Mesh>) {
        let (mut assets, mut meshes) = (ModelAssets::default(), Assets::<Mesh>::default());
        assets.rebuild(model, &mut meshes, &mut Assets::<Image>::default());
        (assets, meshes)
    }

    #[test]
    fn shared_corners_are_merged_and_every_triangle_kept() {
        let model = cube(false);
        let (assets, meshes) = built(&model);
        let (count, vertices, triangles) = assets.mesh_stats();
        assert_eq!((count, triangles), (1, 12));
        assert!(
            vertices < 3 * triangles,
            "{vertices} vertices for {triangles} triangles"
        );
        let (handle, has_uv) = assets.mesh(0).expect("mesh 0");
        assert!(!has_uv);
        let mesh = meshes.get(handle).expect("the mesh is in the store");
        assert_eq!(mesh.count_vertices(), vertices);
        assert_eq!(mesh.indices().map(bevy::mesh::Indices::len), Some(36));
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
                .map(bevy::mesh::VertexAttributeValues::len),
            Some(vertices)
        );
    }

    #[test]
    fn texture_coordinates_come_through_with_their_corners() {
        let model = cube(true);
        let (assets, meshes) = built(&model);
        let (handle, has_uv) = assets.mesh(0).expect("mesh 0");
        assert!(has_uv);
        let mesh = meshes.get(handle).expect("the mesh is in the store");
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_UV_0)
                .map(bevy::mesh::VertexAttributeValues::len),
            Some(mesh.count_vertices())
        );
    }

    /// The cube, scaled: same faces, other vertices.
    fn big_cube() -> MjModel {
        let xml = r#"<mujoco><asset><mesh name="cube" scale="2 2 2"
                vertex="0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1"
                face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7" /></asset>
              <worldbody><geom type="mesh" mesh="cube" /></worldbody></mujoco>"#;
        MjModel::from_xml_string(xml).expect("the cube compiles")
    }

    #[test]
    fn a_rebuild_keeps_what_did_not_change_and_rebuilds_what_did() {
        let model = cube(false);
        let (mut assets, mut meshes) = built(&model);
        let first = assets.mesh(0).expect("mesh 0").0.id();
        assets.rebuild(&model, &mut meshes, &mut Assets::<Image>::default());
        assert_eq!(assets.mesh(0).expect("mesh 0").0.id(), first);
        assert_eq!(
            (
                assets.last_rebuild().meshes_built,
                assets.last_rebuild().meshes_reused
            ),
            (0, 1)
        );

        assets.rebuild(&big_cube(), &mut meshes, &mut Assets::<Image>::default());
        assert_ne!(assets.mesh(0).expect("mesh 0").0.id(), first);
        assert_eq!(assets.last_rebuild().meshes_built, 1);
    }

    #[test]
    fn an_updated_mesh_keeps_its_handle_and_takes_the_new_shape() {
        let (mut assets, mut meshes) = built(&cube(false));
        let handle = assets.mesh(0).expect("mesh 0").0.clone();
        let positions = |meshes: &Assets<Mesh>| {
            meshes
                .get(&handle)
                .and_then(|m| m.attribute(Mesh::ATTRIBUTE_POSITION))
                .and_then(bevy::mesh::VertexAttributeValues::as_float3)
                .map(<[[f32; 3]]>::to_vec)
                .expect("positions")
        };
        let small = positions(&meshes);
        assert!(assets.update_mesh(&big_cube(), 0, &mut meshes));
        assert_eq!(assets.mesh(0).expect("mesh 0").0.id(), handle.id());
        let big = positions(&meshes);
        let extent = |p: &[[f32; 3]]| {
            p.iter().map(|v| v[0]).fold(f32::MIN, f32::max)
                - p.iter().map(|v| v[0]).fold(f32::MAX, f32::min)
        };
        assert!((extent(&big) / extent(&small) - 2.0).abs() < 1e-5);
        assert!(
            !assets.update_mesh(&big_cube(), 1, &mut meshes),
            "no mesh 1"
        );
    }

    #[test]
    fn a_hull_is_built_once_with_flat_faces_and_counted() {
        let model = cube(false);
        let (mut assets, mut meshes) = built(&model);
        let hull = assets
            .hull(&model, 0, &mut meshes)
            .expect("a collision mesh has a hull");
        assert_eq!(assets.hull(&model, 0, &mut meshes), Some(hull.clone()));
        assert_eq!(
            assets.counts(hull.id()),
            MeshCounts {
                vertices: 36,
                triangles: 12
            }
        );
        let mesh = meshes.get(&hull).expect("the hull is in the store");
        assert_eq!(MeshCounts::of(mesh), assets.counts(hull.id()));
        assert_eq!(
            assets
                .counts(assets.mesh(0).expect("mesh 0").0.id())
                .triangles,
            12
        );
    }

    #[test]
    fn a_four_texel_texture_gets_two_levels_and_its_mean() {
        let base = [
            0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
        ]
        .to_vec();
        let (data, levels) = mip_chain(base, 2, 2);
        assert_eq!(levels, 2);
        assert_eq!(&data[16..], &[128, 128, 128, 255]);
    }
}
