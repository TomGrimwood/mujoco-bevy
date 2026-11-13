//! Whether the renderer is still compiling pipelines, for a host to say so
//! while the view draws little more than its clear colour: Bevy compiles
//! them in the background and leaves out whatever waits on one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::render_resource::PipelineCache;
use bevy::render::{Render, RenderApp, RenderSystems};

/// True while any render pipeline is queued or being compiled, as the render
/// world last saw it (a frame behind the main world).
#[derive(Resource, Clone, Default)]
pub struct PipelinesPending(Arc<AtomicBool>);

impl PipelinesPending {
    #[must_use]
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

pub fn plugin(app: &mut App) {
    let pending = PipelinesPending::default();
    app.insert_resource(pending.clone());
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render
            .insert_resource(pending)
            .add_systems(Render, track.in_set(RenderSystems::Cleanup));
    }
}

fn track(cache: Res<PipelineCache>, pending: Res<PipelinesPending>) {
    pending.0.store(
        cache.waiting_pipelines().next().is_some(),
        Ordering::Relaxed,
    );
}
