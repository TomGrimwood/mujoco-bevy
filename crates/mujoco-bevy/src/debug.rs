//! `MuJoCo`'s visualisation switches, named, for a host's debug panel, and
//! the render options Bevy has to carry out itself.
//!
//! The switches are `MuJoCo`'s own ([`MjView::option`]): a panel lists
//! [`VIS_FLAGS`], [`LABELS`], [`FRAMES`] and [`GROUP_KINDS`] with `MuJoCo`'s
//! names and writes the option. Sizes live in the view's copy of the model
//! (`vis.scale`, `vis.map`), which a new model replaces, so they go through
//! [`MjView::set_vis_scale`] and outlive [`MjView::set_model`].
//!
//! `MuJoCo`'s renderer flags ([`RENDER_FLAGS`]) mostly mean GL passes Bevy
//! does not have. The ones kept are carried by [`MjRenderOptions`]
//! (wireframe, fog) and [`LightSettings`](crate::LightSettings) (shadows),
//! with a few isolations of Bevy's own: geom categories, shadow casting,
//! gizmo lines.
//!
//! The physics option (`MjModel::opt`) has names too, for a panel that
//! edits it: [`DISABLE_FLAGS`], [`ENABLE_FLAGS`], [`INTEGRATORS`],
//! [`CONES`], [`JACOBIANS`], [`SOLVERS`].

use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use mujoco_rs::mujoco_c::mjVisual;
use mujoco_rs::prelude::*;

use crate::{MjCamera, MjView, ModelCamera, ModelChanged};

/// One of `MuJoCo`'s enumerated switches with the name `MuJoCo`'s own
/// viewer shows for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Named<T> {
    pub value: T,
    pub name: &'static str,
    /// Whether the bridge draws what it switches. Flex, skins and SDF
    /// iterations are not drawn, so their switches change nothing; a render
    /// flag is drawn when Bevy carries it (shadow, wireframe, fog).
    pub drawn: bool,
}

const fn named<T>(value: T, name: &'static str, drawn: bool) -> Named<T> {
    Named { value, name, drawn }
}

/// `mjtVisFlag`, in `MjvOption::flags` order: index `k` is flag `k`.
#[allow(clippy::enum_glob_use)]
pub const VIS_FLAGS: [Named<MjtVisFlag>; MjtVisFlag::mjNVISFLAG as usize] = {
    use mujoco_rs::mujoco_c::mjtVisFlag_::*;
    [
        named(mjVIS_CONVEXHULL, "Convex Hull", true),
        named(mjVIS_TEXTURE, "Texture", true),
        named(mjVIS_JOINT, "Joint", true),
        named(mjVIS_CAMERA, "Camera", true),
        named(mjVIS_ACTUATOR, "Actuator", true),
        named(mjVIS_ACTIVATION, "Activation", true),
        named(mjVIS_LIGHT, "Light", true),
        named(mjVIS_TENDON, "Tendon", true),
        named(mjVIS_RANGEFINDER, "Range Finder", true),
        named(mjVIS_CONSTRAINT, "Equality", true),
        named(mjVIS_INERTIA, "Inertia", true),
        named(mjVIS_SCLINERTIA, "Scale Inertia", true),
        named(mjVIS_PERTFORCE, "Perturb Force", true),
        named(mjVIS_PERTOBJ, "Perturb Object", true),
        named(mjVIS_CONTACTPOINT, "Contact Point", true),
        named(mjVIS_ISLAND, "Island", true),
        named(mjVIS_CONTACTFORCE, "Contact Force", true),
        named(mjVIS_CONTACTSPLIT, "Contact Split", true),
        named(mjVIS_TRANSPARENT, "Transparent", true),
        named(mjVIS_AUTOCONNECT, "Auto Connect", true),
        named(mjVIS_COM, "Center of Mass", true),
        named(mjVIS_SELECT, "Select Point", true),
        named(mjVIS_STATIC, "Static Body", true),
        named(mjVIS_SKIN, "Skin", false),
        named(mjVIS_FLEXVERT, "Flex Vert", false),
        named(mjVIS_FLEXEDGE, "Flex Edge", false),
        named(mjVIS_FLEXFACE, "Flex Face", false),
        named(mjVIS_FLEXSKIN, "Flex Skin", false),
        named(mjVIS_BODYBVH, "Body Tree", true),
        named(mjVIS_MESHBVH, "Mesh Tree", true),
        named(mjVIS_SDFITER, "SDF iters", false),
    ]
};

/// `mjtRndFlag`. Drawn: shadow ([`LightSettings::shadows`](crate::LightSettings)),
/// wireframe and fog ([`MjRenderOptions`]); the rest are passes of
/// `MuJoCo`'s GL renderer with no meaning here.
#[allow(clippy::enum_glob_use)]
pub const RENDER_FLAGS: [Named<MjtRndFlag>; MjtRndFlag::mjNRNDFLAG as usize] = {
    use mujoco_rs::mujoco_c::mjtRndFlag_::*;
    [
        named(mjRND_SHADOW, "Shadow", true),
        named(mjRND_WIREFRAME, "Wireframe", true),
        named(mjRND_REFLECTION, "Reflection", false),
        named(mjRND_ADDITIVE, "Additive", false),
        named(mjRND_SKYBOX, "Skybox", false),
        named(mjRND_FOG, "Fog", true),
        named(mjRND_HAZE, "Haze", false),
        named(mjRND_DEPTH, "Depth", false),
        named(mjRND_SEGMENT, "Segment", false),
        named(mjRND_IDCOLOR, "Id Color", false),
        named(mjRND_CULL_FACE, "Cull Face", false),
    ]
};

/// `mjtLabel`, for `MjvOption::label`.
#[allow(clippy::enum_glob_use)]
pub const LABELS: [Named<MjtLabel>; MjtLabel::mjNLABEL as usize] = {
    use mujoco_rs::mujoco_c::mjtLabel_::*;
    [
        named(mjLABEL_NONE, "None", true),
        named(mjLABEL_BODY, "Body", true),
        named(mjLABEL_JOINT, "Joint", true),
        named(mjLABEL_GEOM, "Geom", true),
        named(mjLABEL_SITE, "Site", true),
        named(mjLABEL_CAMERA, "Camera", true),
        named(mjLABEL_LIGHT, "Light", true),
        named(mjLABEL_TENDON, "Tendon", true),
        named(mjLABEL_ACTUATOR, "Actuator", true),
        named(mjLABEL_CONSTRAINT, "Constraint", true),
        named(mjLABEL_FLEX, "Flex", false),
        named(mjLABEL_SKIN, "Skin", false),
        named(mjLABEL_SELECTION, "Selection", true),
        named(mjLABEL_SELPNT, "SelPoint", true),
        named(mjLABEL_CONTACTPOINT, "Contact", true),
        named(mjLABEL_CONTACTFORCE, "ContactForce", true),
        named(mjLABEL_ISLAND, "Island", true),
    ]
};

/// `mjtFrame`, for `MjvOption::frame`.
#[allow(clippy::enum_glob_use)]
pub const FRAMES: [Named<MjtFrame>; MjtFrame::mjNFRAME as usize] = {
    use mujoco_rs::mujoco_c::mjtFrame_::*;
    [
        named(mjFRAME_NONE, "None", true),
        named(mjFRAME_BODY, "Body", true),
        named(mjFRAME_GEOM, "Geom", true),
        named(mjFRAME_SITE, "Site", true),
        named(mjFRAME_CAMERA, "Camera", true),
        named(mjFRAME_LIGHT, "Light", true),
        named(mjFRAME_CONTACT, "Contact", true),
        named(mjFRAME_WORLD, "World", true),
    ]
};

/// The kinds of element `MjvOption` shows or hides by group (0 to 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupKind {
    Geom,
    Site,
    Joint,
    Tendon,
    Actuator,
    Flex,
    Skin,
}

/// Every [`GroupKind`], with the name `MuJoCo`'s viewer gives its section.
pub const GROUP_KINDS: [Named<GroupKind>; 7] = [
    named(GroupKind::Geom, "Geom groups", true),
    named(GroupKind::Site, "Site groups", true),
    named(GroupKind::Joint, "Joint groups", true),
    named(GroupKind::Tendon, "Tendon groups", true),
    named(GroupKind::Actuator, "Actuator groups", true),
    named(GroupKind::Flex, "Flex groups", false),
    named(GroupKind::Skin, "Skin groups", false),
];

/// How many groups each kind has.
pub const GROUPS: usize = 6;

impl GroupKind {
    /// The kind's six switches in `option`.
    #[must_use]
    pub const fn groups(self, option: &MjvOption) -> &[u8; GROUPS] {
        match self {
            Self::Geom => &option.geomgroup,
            Self::Site => &option.sitegroup,
            Self::Joint => &option.jointgroup,
            Self::Tendon => &option.tendongroup,
            Self::Actuator => &option.actuatorgroup,
            Self::Flex => &option.flexgroup,
            Self::Skin => &option.skingroup,
        }
    }

    pub const fn groups_mut(self, option: &mut MjvOption) -> &mut [u8; GROUPS] {
        match self {
            Self::Geom => &mut option.geomgroup,
            Self::Site => &mut option.sitegroup,
            Self::Joint => &mut option.jointgroup,
            Self::Tendon => &mut option.tendongroup,
            Self::Actuator => &mut option.actuatorgroup,
            Self::Flex => &mut option.flexgroup,
            Self::Skin => &mut option.skingroup,
        }
    }
}

/// `mjtDisableBit` as `MuJoCo`'s viewer names them: entry `k` is bit
/// `1 << k` of `MjOption::disableflags`.
pub const DISABLE_FLAGS: [&str; MjtDisableBit::mjNDISABLE as usize] = [
    "Constraint",
    "Equality",
    "Frictionloss",
    "Limit",
    "Contact",
    "Spring",
    "Damper",
    "Gravity",
    "Clampctrl",
    "Warmstart",
    "Filterparent",
    "Actuation",
    "Refsafe",
    "Sensor",
    "Midphase",
    "Eulerdamp",
    "AutoReset",
    "NativeCCD",
    "Island",
    "MultiCCD",
];

/// `mjtEnableBit`: entry `k` is bit `1 << k` of `MjOption::enableflags`.
pub const ENABLE_FLAGS: [&str; MjtEnableBit::mjNENABLE as usize] = [
    "Override",
    "Energy",
    "Fwdinv",
    "InvDiscrete",
    "Sleep",
    "DiagExact",
    "IPC",
];

/// `mjtIntegrator`, for `MjOption::integrator`: entry `k` is value `k`.
pub const INTEGRATORS: [&str; 5] = ["Euler", "RK4", "implicit", "implicitfast", "discrete"];

/// `mjtCone`, for `MjOption::cone`.
pub const CONES: [&str; 2] = ["Pyramidal", "Elliptic"];

/// `mjtJacobian`, for `MjOption::jacobian`.
pub const JACOBIANS: [&str; 3] = ["Dense", "Sparse", "Auto"];

/// `mjtSolver`, for `MjOption::solver`.
pub const SOLVERS: [&str; 3] = ["PGS", "CG", "Newton"];

/// A size or scale in the model's `vis` that decor is drawn with. Sizes are
/// fractions of the model's mean body size (`stat.meansize`); the map
/// entries are `MuJoCo`'s force-to-length scale and transparency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisScale {
    ForceWidth,
    ContactWidth,
    ContactHeight,
    Connect,
    Com,
    Camera,
    Light,
    SelectPoint,
    JointLength,
    JointWidth,
    ActuatorLength,
    ActuatorWidth,
    FrameLength,
    FrameWidth,
    Constraint,
    SliderCrank,
    Frustum,
    /// `vis.map.force`: arrow length per newton of contact or perturb force.
    MapForce,
    /// `vis.map.torque`: the same for torques.
    MapTorque,
    /// `vis.map.alpha`: what the transparent flag multiplies alpha by.
    MapAlpha,
}

/// Every [`VisScale`] with its MJCF attribute name (`<visual><scale .../>`,
/// `<visual><map .../>`).
#[allow(clippy::enum_glob_use)]
pub const VIS_SCALES: [Named<VisScale>; 20] = {
    use VisScale::*;
    [
        named(ForceWidth, "forcewidth", true),
        named(ContactWidth, "contactwidth", true),
        named(ContactHeight, "contactheight", true),
        named(Connect, "connect", true),
        named(Com, "com", true),
        named(Camera, "camera", true),
        named(Light, "light", true),
        named(SelectPoint, "selectpoint", true),
        named(JointLength, "jointlength", true),
        named(JointWidth, "jointwidth", true),
        named(ActuatorLength, "actuatorlength", true),
        named(ActuatorWidth, "actuatorwidth", true),
        named(FrameLength, "framelength", true),
        named(FrameWidth, "framewidth", true),
        named(Constraint, "constraint", true),
        named(SliderCrank, "slidercrank", true),
        named(Frustum, "frustum", true),
        named(MapForce, "force", true),
        named(MapTorque, "torque", true),
        named(MapAlpha, "alpha", true),
    ]
};

impl VisScale {
    const fn field(self, vis: &mut mjVisual) -> &mut f32 {
        let (s, m) = (&mut vis.scale, &mut vis.map);
        match self {
            Self::ForceWidth => &mut s.forcewidth,
            Self::ContactWidth => &mut s.contactwidth,
            Self::ContactHeight => &mut s.contactheight,
            Self::Connect => &mut s.connect,
            Self::Com => &mut s.com,
            Self::Camera => &mut s.camera,
            Self::Light => &mut s.light,
            Self::SelectPoint => &mut s.selectpoint,
            Self::JointLength => &mut s.jointlength,
            Self::JointWidth => &mut s.jointwidth,
            Self::ActuatorLength => &mut s.actuatorlength,
            Self::ActuatorWidth => &mut s.actuatorwidth,
            Self::FrameLength => &mut s.framelength,
            Self::FrameWidth => &mut s.framewidth,
            Self::Constraint => &mut s.constraint,
            Self::SliderCrank => &mut s.slidercrank,
            Self::Frustum => &mut s.frustum,
            Self::MapForce => &mut m.force,
            Self::MapTorque => &mut m.torque,
            Self::MapAlpha => &mut m.alpha,
        }
    }

    fn get(self, model: &MjModel) -> f32 {
        let mut vis = model.ffi().vis.clone();
        *self.field(&mut vis)
    }
}

/// A [`VisScale`] the host set, and the model's own value under it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VisOverride {
    scale: VisScale,
    value: f32,
    model: f32,
}

impl MjView {
    /// Whether `MuJoCo`'s visualisation flag is on.
    #[must_use]
    pub const fn vis_flag(&self, flag: MjtVisFlag) -> bool {
        self.option.flags[flag as usize] != 0
    }

    pub fn set_vis_flag(&mut self, flag: MjtVisFlag, on: bool) {
        self.option.flags[flag as usize] = u8::from(on);
    }

    /// Whether group `group` (0 to 5) of `kind` is shown; `false` past 5.
    #[must_use]
    pub fn group(&self, kind: GroupKind, group: usize) -> bool {
        kind.groups(&self.option)
            .get(group)
            .is_some_and(|&g| g != 0)
    }

    /// Shows or hides group `group` (0 to 5) of `kind`; nothing past 5.
    pub fn set_group(&mut self, kind: GroupKind, group: usize, on: bool) {
        if let Some(g) = kind.groups_mut(&mut self.option).get_mut(group) {
            *g = u8::from(on);
        }
    }

    /// The size decor is drawn with now: the host's, else the model's.
    #[must_use]
    pub fn vis_scale(&self, scale: VisScale) -> f32 {
        scale.get(self.model())
    }

    /// The model's own value of `scale`, whatever the host set.
    #[must_use]
    pub fn model_vis_scale(&self, scale: VisScale) -> f32 {
        self.vis_overrides
            .iter()
            .find(|o| o.scale == scale)
            .map_or_else(|| scale.get(self.model()), |o| o.model)
    }

    /// Draws decor with `value` for `scale` from the next scene on, and
    /// keeps it across [`Self::set_model`] and [`Self::refresh_model`]
    /// (the model's own `vis` is what a recompile brings back).
    pub fn set_vis_scale(&mut self, scale: VisScale, value: f32) {
        let model = self.model_vis_scale(scale);
        self.vis_overrides.retain(|o| o.scale != scale);
        self.vis_overrides.push(VisOverride {
            scale,
            value,
            model,
        });
        self.write_vis(scale, value);
    }

    /// Goes back to the model's own value of `scale`.
    pub fn reset_vis_scale(&mut self, scale: VisScale) {
        let model = self.model_vis_scale(scale);
        self.vis_overrides.retain(|o| o.scale != scale);
        self.write_vis(scale, model);
    }

    /// The scales the host set, with their values.
    pub fn vis_scales_set(&self) -> impl Iterator<Item = (VisScale, f32)> + '_ {
        self.vis_overrides.iter().map(|o| (o.scale, o.value))
    }

    /// Writes the host's scales into a model just taken, noting its own.
    pub(crate) fn apply_vis_overrides(&mut self, overrides: Vec<VisOverride>) {
        self.vis_overrides = overrides;
        for k in 0..self.vis_overrides.len() {
            let VisOverride { scale, value, .. } = self.vis_overrides[k];
            self.vis_overrides[k].model = scale.get(self.model());
            self.write_vis(scale, value);
        }
    }

    fn write_vis(&mut self, scale: VisScale, value: f32) {
        // SAFETY: a plain number in `vis`; no address or count changes.
        let vis = unsafe { &mut self.data.model_mut().ffi_mut().vis };
        *scale.field(vis) = value;
    }
}

/// How the bridge's geoms are drawn, beyond `MuJoCo`'s option: the render
/// flags Bevy carries out itself and isolations for finding what a frame
/// costs. Read every frame by [`MujocoScenePlugin`](crate::MujocoScenePlugin).
// Independent switches, as `MuJoCo`'s own flags are.
#[allow(clippy::struct_excessive_bools)]
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MjRenderOptions {
    /// Every geom as its wires alone, each in the geom's colour (its
    /// texture's mean, for a textured one), as `MuJoCo`'s own wireframe:
    /// Bevy's [`Wireframe`](bevy::pbr::wireframe::Wireframe) in a pass of
    /// its own (polygon mode line), the geom's material one that draws
    /// nothing. Needs `WireframePlugin`, which the scene plugin
    /// adds when there is a renderer, and the GPU features
    /// `POLYGON_MODE_LINE | IMMEDIATES` (Vulkan, DX12, Metal; Bevy logs a
    /// warning and draws none without them). Costs about one more draw of
    /// every geom's triangles.
    pub wireframe: bool,
    /// Bevy's [`DistanceFog`] on the [`MjCamera`] and [`ModelCamera`]s, with
    /// the model's fog: linear from `vis.map.fogstart` to `fogend` times
    /// `stat.extent`, coloured `vis.rgba.fog`. A few instructions a pixel.
    pub fog: bool,
    /// Geoms are drawn into the lights' shadow maps. Off, the maps still
    /// exist (the host's own meshes still cast) but the model is left out
    /// of them, which isolates its shadow pass cost. For no maps at all, see
    /// [`LightSettings::shadows`](crate::LightSettings).
    pub shadows: bool,
    /// Draw static geoms (the world body's and those of bodies welded to
    /// it), dynamic ones (every other body's) and decor (frames, contacts, perturb handles, the host's
    /// overlays): `mjtCatBit`. Hidden ones are not drawn, labelled or
    /// counted; picks still see them.
    pub show_static: bool,
    pub show_dynamic: bool,
    pub show_decor: bool,
    /// Lines and line boxes as gizmos.
    pub lines: bool,
}

impl Default for MjRenderOptions {
    fn default() -> Self {
        Self {
            wireframe: false,
            fog: false,
            shadows: true,
            show_static: true,
            show_dynamic: true,
            show_decor: true,
            lines: true,
        }
    }
}

impl MjRenderOptions {
    /// The `mjtCatBit` mask of the categories shown.
    #[must_use]
    pub const fn category_mask(&self) -> i32 {
        const fn bit(on: bool, cat: MjtCatBit) -> i32 {
            if on { cat as i32 } else { 0 }
        }
        bit(self.show_static, MjtCatBit::mjCAT_STATIC)
            | bit(self.show_dynamic, MjtCatBit::mjCAT_DYNAMIC)
            | bit(self.show_decor, MjtCatBit::mjCAT_DECOR)
    }

    /// A render flag these options carry (wireframe, fog); `None` for the
    /// others, shadow included (that is [`LightSettings`](crate::LightSettings)).
    #[must_use]
    pub const fn render_flag(&self, flag: MjtRndFlag) -> Option<bool> {
        match flag {
            MjtRndFlag::mjRND_WIREFRAME => Some(self.wireframe),
            MjtRndFlag::mjRND_FOG => Some(self.fog),
            _ => None,
        }
    }

    /// Sets a render flag these options carry; `false`, changing nothing,
    /// for the others.
    pub const fn set_render_flag(&mut self, flag: MjtRndFlag, on: bool) -> bool {
        match flag {
            MjtRndFlag::mjRND_WIREFRAME => self.wireframe = on,
            MjtRndFlag::mjRND_FOG => self.fog = on,
            _ => return false,
        }
        true
    }
}

/// The model's fog, as Bevy's.
#[must_use]
pub fn model_fog(model: &MjModel) -> DistanceFog {
    let vis = &model.ffi().vis;
    #[allow(clippy::cast_possible_truncation)]
    let extent = model.ffi().stat.extent as f32;
    let [r, g, b, a] = vis.rgba.fog;
    DistanceFog {
        color: Color::srgba(r, g, b, a),
        falloff: FogFalloff::Linear {
            start: vis.map.fogstart * extent,
            end: vis.map.fogend * extent,
        },
        ..default()
    }
}

type FogCameras<'w, 's> =
    Query<'w, 's, (Entity, Has<DistanceFog>), Or<(With<MjCamera>, With<ModelCamera>)>>;

/// Fog on the view's cameras while [`MjRenderOptions::fog`] is on, made
/// again from a new model's.
pub fn sync_fog(
    mut commands: Commands,
    view: Res<MjView>,
    options: Res<MjRenderOptions>,
    cameras: FogCameras,
    mut changed: MessageReader<ModelChanged>,
) {
    let new_model = changed.read().count() > 0;
    for (camera, fogged) in &cameras {
        if options.fog && (!fogged || new_model) {
            commands.entity(camera).insert(model_fog(view.model()));
        } else if !options.fog && fogged {
            commands.entity(camera).remove::<DistanceFog>();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, c_char};

    use super::*;

    fn c_name(ptr: *const c_char) -> &'static str {
        // SAFETY: `MuJoCo`'s name tables are static NUL-terminated strings.
        unsafe { CStr::from_ptr(ptr) }.to_str().expect("ASCII")
    }

    #[test]
    fn the_names_are_mujocos_and_in_its_order() {
        // SAFETY: reads of `MuJoCo`'s constant string tables.
        let (vis, rnd, label, frame) = unsafe {
            (
                crate::dll::mj_vis_string(),
                crate::dll::mj_rnd_string(),
                crate::dll::mj_label_string(),
                crate::dll::mj_frame_string(),
            )
        };
        for (k, flag) in VIS_FLAGS.iter().enumerate() {
            assert_eq!(flag.value as usize, k);
            assert_eq!(flag.name, c_name(vis[k][0]));
        }
        for (k, flag) in RENDER_FLAGS.iter().enumerate() {
            assert_eq!(flag.value as usize, k);
            assert_eq!(flag.name, c_name(rnd[k][0]));
        }
        for (k, l) in LABELS.iter().enumerate() {
            assert_eq!(l.value as usize, k);
            assert_eq!(l.name, c_name(label[k]));
        }
        for (k, f) in FRAMES.iter().enumerate() {
            assert_eq!(f.value as usize, k);
            assert_eq!(f.name, c_name(frame[k]));
        }
    }

    #[test]
    fn the_physics_flags_are_named_as_mujoco_names_them() {
        // SAFETY: reads of `MuJoCo`'s constant string tables.
        let (disable, enable) = unsafe {
            (
                crate::dll::mj_disable_string(),
                crate::dll::mj_enable_string(),
            )
        };
        for (k, name) in DISABLE_FLAGS.iter().enumerate() {
            assert_eq!(*name, c_name(disable[k]));
        }
        for (k, name) in ENABLE_FLAGS.iter().enumerate() {
            assert_eq!(*name, c_name(enable[k]));
        }
        assert_eq!(
            MjtIntegrator::mjINT_DISCRETE as usize,
            INTEGRATORS.len() - 1
        );
        assert_eq!(MjtCone::mjCONE_ELLIPTIC as usize, CONES.len() - 1);
        assert_eq!(MjtJacobian::mjJAC_AUTO as usize, JACOBIANS.len() - 1);
        assert_eq!(MjtSolver::mjSOL_NEWTON as usize, SOLVERS.len() - 1);
    }

    #[test]
    fn render_flags_reach_only_what_the_options_carry() {
        let mut options = MjRenderOptions::default();
        for flag in RENDER_FLAGS {
            let carried = options.set_render_flag(flag.value, true);
            assert_eq!(carried, options.render_flag(flag.value).is_some());
            assert_eq!(
                carried,
                flag.drawn && flag.value != MjtRndFlag::mjRND_SHADOW,
                "{}",
                flag.name
            );
        }
        assert!(options.wireframe && options.fog);
        assert_eq!(options.category_mask(), MjtCatBit::mjCAT_ALL as i32);
        options.show_decor = false;
        assert_eq!(
            options.category_mask(),
            MjtCatBit::mjCAT_STATIC as i32 | MjtCatBit::mjCAT_DYNAMIC as i32
        );
    }
}
