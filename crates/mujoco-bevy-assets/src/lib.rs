//! A compiled `MuJoCo` model's meshes, textures and materials as Bevy assets,
//! built once per model, and the unit primitive meshes every geom scales.
//!
//! Add [`MujocoAssetsPlugin`], then call [`ModelAssets::rebuild`] whenever the
//! model changes. Drawing the model is `mujoco-bevy`'s business, not this
//! crate's.

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Affine2;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use mujoco_rs::prelude::*;

/// Past this many cached materials the cache starts again: geoms that change
/// colour every frame (activation colours, a fading highlight) would
/// otherwise grow it without end.
const MATERIAL_CACHE_CAP: usize = 4096;

/// Everything made for one model. A new model replaces all of it; the old
/// handles drop and Bevy frees the assets once no entity holds them.
#[derive(Resource, Default)]
pub struct ModelAssets {
    meshes: Vec<ModelMesh>,
    textures: Vec<Option<Handle<Image>>>,
    materials: HashMap<MaterialKey, Handle<StandardMaterial>>,
    primitives: Option<Primitives>,
    capsules: HashMap<u32, Handle<Mesh>>,
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

/// What a geom's look is made of, quantised so near-equal looks share one
/// material.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialKey {
    texture: i32,
    rgba: [u8; 4],
    emission: u8,
    specular: u8,
    shininess: u8,
    uv_scale: [u32; 2],
}

/// The parts of an `mjvGeom` that pick its material.
pub struct GeomLook {
    pub rgba: [f32; 4],
    pub emission: f32,
    pub specular: f32,
    pub shininess: f32,
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

    /// A capsule of radius 1 whose straight part is `2 * ratio` long, for a
    /// geom scaled uniformly by its radius: a capsule cannot be scaled
    /// unevenly without squashing its ends. Ratios are bucketed on a log
    /// scale, within about 2.5 % of the length: a perturb handle stretches
    /// every frame, and even buckets made a mesh for each new length.
    pub fn capsule(&mut self, meshes: &mut Assets<Mesh>, ratio: f32) -> Handle<Mesh> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bucket = (ratio.clamp(0.0, 100.0).ln_1p() * 20.0).round() as u32;
        self.capsules
            .entry(bucket)
            .or_insert_with(|| {
                let half = (bucket as f32 / 20.0).exp_m1();
                meshes.add(
                    Capsule3d::new(1.0, 2.0 * half)
                        .mesh()
                        .longitudes(24)
                        .latitudes(12)
                        .build()
                        .rotated_by(Quat::from_rotation_x(FRAC_PI_2)),
                )
            })
            .clone()
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
            uv_scale: [look.uv_scale.x.to_bits(), look.uv_scale.y.to_bits()],
        };
        if let Some(handle) = self.materials.get(&key) {
            return handle.clone();
        }
        if self.materials.len() >= MATERIAL_CACHE_CAP {
            self.materials.clear();
        }
        let [r, g, b, a] = look.rgba;
        let base = Color::srgba(r, g, b, a);
        // Bevy's emissive is a luminance that the camera's exposure scales
        // down by about a thousand, where `MuJoCo`'s is a fraction of the
        // colour; this makes emission 1 about as bright as a sunlit face.
        let emissive = LinearRgba::from(Color::srgb(r, g, b)) * (look.emission * 1000.0);
        let material = StandardMaterial {
            base_color: base,
            base_color_texture: look
                .texture
                .and_then(|t| self.textures.get(t).cloned().flatten()),
            uv_transform: Affine2::from_scale(look.uv_scale),
            emissive,
            // By eye: `MuJoCo`'s Phong shininess and specular weight to a
            // roughness and a reflectance that look about as glossy.
            perceptual_roughness: (1.0 - 0.75 * look.shininess).clamp(0.25, 1.0),
            reflectance: (0.15 + 0.7 * look.specular).clamp(0.0, 1.0),
            alpha_mode: if a < 0.999 {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            ..default()
        };
        let handle = materials.add(material);
        self.materials.insert(key, handle.clone());
        handle
    }

    /// Materials cached now, to show that a model swap lets go of them.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Replaces every mesh, texture and cached material with `model`'s. The
    /// old handles drop here, so the old model's assets are freed once no
    /// entity holds them.
    pub fn rebuild(
        &mut self,
        model: &MjModel,
        meshes: &mut Assets<Mesh>,
        images: &mut Assets<Image>,
    ) {
        self.materials.clear();
        self.meshes = (0..to_usize(model.nmesh()))
            .map(|id| build_mesh(model, id, meshes))
            .collect();
        self.textures = (0..to_usize(model.ntex()))
            .map(|id| build_texture(model, id).map(|image| images.add(image)))
            .collect();
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
    assets.primitives = Some(Primitives {
        sphere: meshes.add(Sphere::new(1.0).mesh().uv(32, 18)),
        cuboid: meshes.add(Cuboid::new(2.0, 2.0, 2.0)),
        cylinder: meshes.add(
            Cylinder::new(1.0, 2.0)
                .mesh()
                .resolution(32)
                .build()
                .rotated_by(z_up),
        ),
        plane: meshes.add(Plane3d::new(Vec3::Z, Vec2::ONE)),
        arrow: meshes.add(arrow(false)),
        arrow2: meshes.add(arrow(true)),
        triangle: meshes.add(triangle),
    });
}

/// One indexed mesh: `MuJoCo` indexes positions, normals and texture
/// coordinates apart, so corners that agree on all three are merged.
fn build_mesh(model: &MjModel, id: usize, meshes: &mut Assets<Mesh>) -> ModelMesh {
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
    let vertices = pos.len();
    let has_uv = !uv.is_empty();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
    .with_inserted_indices(Indices::U32(indices));
    if has_uv {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    }
    ModelMesh {
        handle: meshes.add(mesh),
        has_uv,
        vertices,
        triangles: nf,
    }
}

/// A 2D texture as an sRGB (or linear, if the model says so) RGBA image with
/// a full mip chain, repeating, as `MuJoCo` samples it. Cube maps and the
/// skybox are left out.
fn build_texture(model: &MjModel, id: usize) -> Option<Image> {
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
    let format = if model.tex_colorspace()[id] == MjtColorSpace::mjCOLORSPACE_LINEAR {
        TextureFormat::Rgba8Unorm
    } else {
        TextureFormat::Rgba8UnormSrgb
    };
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
    Some(image)
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
