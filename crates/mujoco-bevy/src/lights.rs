//! The scene's lights as Bevy lights. Intensities are picked to look right,
//! not to match `MuJoCo`'s fixed-function lighting.

use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::prelude::*;
use mujoco_rs::prelude::*;

use crate::MjView;

/// The host's say over the lights.
#[derive(Resource, Clone, Copy)]
pub struct LightSettings {
    /// Shadows from every light `MuJoCo` marks `castshadow`.
    pub shadows: bool,
    /// Lux for a directional light of diffuse 1.
    pub lux_per_diffuse: f32,
    /// Lumens for a spot or point light of diffuse 1.
    pub lumens_per_diffuse: f32,
    /// Ambient brightness for an ambient sum of 1.
    pub ambient_per_unit: f32,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            shadows: true,
            lux_per_diffuse: 30_000.0,
            lumens_per_diffuse: 2_000_000.0,
            ambient_per_unit: 2_000.0,
        }
    }
}

/// Marks the entity drawing the scene's `n`-th light.
#[derive(Component)]
pub struct MjLight(usize);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Directional,
    Spot,
    Point,
}

/// One scene light as Bevy wants it.
struct Wanted {
    kind: Kind,
    color: Color,
    /// Lux for a directional light, lumens otherwise.
    intensity: f32,
    shadows: bool,
    transform: Transform,
    range: f32,
    cutoff: f32,
}

impl Wanted {
    fn new(light: &MjvLight, settings: &LightSettings) -> Self {
        let kind =
            if light.headlight != 0 || light.type_ == MjtLightType::mjLIGHT_DIRECTIONAL as i32 {
                Kind::Directional
            } else if light.type_ == MjtLightType::mjLIGHT_SPOT as i32 {
                Kind::Spot
            } else {
                Kind::Point
            };
        let (color, level) = colour_and_level(light.diffuse);
        let per_unit = if kind == Kind::Directional {
            settings.lux_per_diffuse
        } else {
            settings.lumens_per_diffuse
        };
        Self {
            kind,
            color,
            intensity: level * per_unit,
            shadows: settings.shadows && light.castshadow != 0,
            transform: aim(Vec3::from_array(light.pos), Vec3::from_array(light.dir)),
            range: if light.range > 0.0 { light.range } else { 50.0 },
            cutoff: light.cutoff.to_radians().min(1.5),
        }
    }

    fn spawn(&self, index: usize, commands: &mut Commands) {
        let mut entity = commands.spawn((MjLight(index), self.transform));
        match self.kind {
            Kind::Directional => entity.insert((
                DirectionalLight {
                    color: self.color,
                    illuminance: self.intensity,
                    shadow_maps_enabled: self.shadows,
                    ..default()
                },
                cascades(),
            )),
            Kind::Spot => entity.insert(SpotLight {
                color: self.color,
                intensity: self.intensity,
                shadow_maps_enabled: self.shadows,
                range: self.range,
                outer_angle: self.cutoff,
                inner_angle: self.cutoff * 0.8,
                ..default()
            }),
            Kind::Point => entity.insert(PointLight {
                color: self.color,
                intensity: self.intensity,
                shadow_maps_enabled: self.shadows,
                range: self.range,
                ..default()
            }),
        };
    }
}

/// Cascades for a work area about 50 m across: one over what the camera
/// usually orbits at, a machine and the ground round it, and one beyond,
/// out to where a machine is still large on screen. Two, not Bevy's four:
/// every cascade is the scene drawn again into the shadow map.
fn cascades() -> CascadeShadowConfig {
    CascadeShadowConfigBuilder {
        num_cascades: 2,
        minimum_distance: 0.1,
        first_cascade_far_bound: 16.0,
        maximum_distance: 60.0,
        overlap_proportion: 0.2,
    }
    .build()
}

/// `MuJoCo` colours are display values; Bevy wants a colour and a separate
/// brightness, so the colour is normalised to its largest channel.
fn colour_and_level(rgb: [f32; 3]) -> (Color, f32) {
    let level = rgb.iter().copied().fold(0.0, f32::max);
    if level <= 0.0 {
        return (Color::WHITE, 0.0);
    }
    (
        Color::srgb(rgb[0] / level, rgb[1] / level, rgb[2] / level),
        level,
    )
}

fn aim(pos: Vec3, dir: Vec3) -> Transform {
    let up = if dir.normalize_or_zero().z.abs() > 0.99 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    Transform::from_translation(pos).looking_to(dir, up)
}

/// Assigns only a different value, so change detection stays quiet.
fn set<T: PartialEq>(slot: &mut T, value: T) {
    if *slot != value {
        *slot = value;
    }
}

type LightQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static MjLight,
        &'static mut Transform,
        Option<&'static mut DirectionalLight>,
        Option<&'static mut SpotLight>,
        Option<&'static mut PointLight>,
    ),
>;

/// One Bevy light per scene light. The set changes only with the model, so a
/// change of count or kind respawns them all; otherwise each is updated in
/// place (the headlight follows the camera every frame).
pub fn sync_lights(
    mut commands: Commands,
    view: Res<MjView>,
    settings: Res<LightSettings>,
    mut lights: LightQuery,
) {
    let wanted: Vec<Wanted> = view
        .scene
        .lights()
        .iter()
        .map(|l| Wanted::new(l, &settings))
        .collect();
    let same_set = lights.iter().len() == wanted.len()
        && lights.iter().all(|(_, slot, _, directional, spot, _)| {
            let have = if directional.is_some() {
                Kind::Directional
            } else if spot.is_some() {
                Kind::Spot
            } else {
                Kind::Point
            };
            wanted.get(slot.0).is_some_and(|w| w.kind == have)
        });
    if !same_set {
        for (entity, ..) in &lights {
            commands.entity(entity).despawn();
        }
        for (index, w) in wanted.iter().enumerate() {
            w.spawn(index, &mut commands);
        }
        return;
    }
    for (_, slot, mut transform, directional, spot, point) in &mut lights {
        let w = &wanted[slot.0];
        transform.set_if_neq(w.transform);
        if let Some(mut l) = directional {
            set(&mut l.color, w.color);
            set(&mut l.illuminance, w.intensity);
            set(&mut l.shadow_maps_enabled, w.shadows);
        } else if let Some(mut l) = spot {
            set(&mut l.color, w.color);
            set(&mut l.intensity, w.intensity);
            set(&mut l.shadow_maps_enabled, w.shadows);
        } else if let Some(mut l) = point {
            set(&mut l.color, w.color);
            set(&mut l.intensity, w.intensity);
            set(&mut l.shadow_maps_enabled, w.shadows);
        }
    }
}

/// The ambient terms of all scene lights, summed, as Bevy's one ambient light.
pub fn sync_ambient(
    view: Res<MjView>,
    settings: Res<LightSettings>,
    mut ambient: ResMut<GlobalAmbientLight>,
) {
    let sum = view.scene.lights().iter().fold([0.0_f32; 3], |mut sum, l| {
        for (s, a) in sum.iter_mut().zip(l.ambient) {
            *s += a;
        }
        sum
    });
    let (color, level) = colour_and_level(sum);
    let brightness = level * settings.ambient_per_unit;
    // Compared without marking the resource changed: an unchanged write
    // would still flag it to every reader.
    let current = ambient.bypass_change_detection();
    if current.color != color || current.brightness.to_bits() != brightness.to_bits() {
        ambient.color = color;
        ambient.brightness = brightness;
    }
}
