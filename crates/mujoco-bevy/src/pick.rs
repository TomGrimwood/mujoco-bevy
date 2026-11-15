//! Picking through `MuJoCo`'s own ray cast (`mjv_select`), against the scene
//! the picture was drawn from.

use bevy::math::DVec3;
use bevy::prelude::*;

use crate::camera::{Click, Modifiers, PendingClick};
use crate::{MjBody, MjView};

/// The geom group of colliders: what a click finds a body by when its
/// visuals are drawn apart from the model ([`crate::MjLook`]).
const COLLIDER_GROUP: usize = 3;

/// What a click hit.
#[derive(Clone, Debug)]
pub struct Pick {
    /// `None` when the ray hit nothing.
    pub body: Option<usize>,
    pub geom: Option<usize>,
    pub point: DVec3,
}

/// Sent once per click on the 3D view, with the keys held: a host tells its
/// own picks (a ruler, placing a machine) from a selection by them.
#[derive(Message, Clone, Debug)]
pub struct Picked {
    pub pick: Pick,
    pub modifiers: Modifiers,
}

/// The last unmodified click's pick; `MuJoCo`'s highlight and the perturb
/// follow it.
#[derive(Resource, Default)]
pub struct Selection(pub Option<Pick>);

/// Lets go of the selection on a new model: its body and geom ids mean
/// nothing there.
pub fn forget_selection(mut selection: ResMut<Selection>) {
    selection.0 = None;
}

/// Casts the ray for this frame's click, if there was one, and reports it.
/// An unmodified click also selects what it hit, as `simulate` does: the body
/// becomes the perturb's selection. While bodies are drawn from looks, the
/// ray meets their colliders: the model has no visual geoms to meet.
pub fn pick_clicked(
    mut view: ResMut<MjView>,
    mut pending: ResMut<PendingClick>,
    mut selection: ResMut<Selection>,
    mut picked: MessageWriter<Picked>,
    looks: Query<(), With<MjBody>>,
) {
    let Some(click) = pending.0.take() else {
        return;
    };
    let pick = if looks.is_empty() {
        pick(&mut view, click)
    } else {
        let shown = view.option.geomgroup[COLLIDER_GROUP];
        view.option.geomgroup[COLLIDER_GROUP] = 1;
        let pick = pick(&mut view, click);
        view.option.geomgroup[COLLIDER_GROUP] = shown;
        pick
    };
    if click.modifiers.any() {
        picked.write(Picked {
            pick,
            modifiers: click.modifiers,
        });
        return;
    }
    let body = pick.body.filter(|&b| b > 0);
    view.perturb.select = body.and_then(|b| i32::try_from(b).ok()).unwrap_or(0);
    view.perturb.skinselect = -1;
    view.perturb.flexselect = -1;
    if let Some(b) = body {
        let point = pick.point.to_array();
        let view = &mut *view;
        if b < usize::try_from(view.model().nbody()).unwrap_or(0) {
            view.perturb.update_local_pos(&point, &view.data);
        }
    }
    selection.0 = Some(pick.clone());
    picked.write(Picked {
        pick,
        modifiers: click.modifiers,
    });
}

/// `mjv_select` for `click`. The scene must have been updated at least once
/// ([`MjView::scene_ready`]); before that nothing is hit.
#[must_use]
pub fn pick(view: &mut MjView, click: Click) -> Pick {
    let nothing = Pick {
        body: None,
        geom: None,
        point: DVec3::ZERO,
    };
    if !view.scene_ready() {
        return nothing;
    }
    let view = &mut *view;
    let hit = view.scene.find_selection(
        &mut view.data,
        &view.option,
        click.aspect,
        f64::from(click.at.x),
        f64::from(click.at.y),
    );
    if hit.body_id.is_none() {
        return nothing;
    }
    Pick {
        body: hit.body_id,
        geom: hit.geom_id,
        point: DVec3::from_array(hit.point),
    }
}
