# mujoco-bevy

A compiled MuJoCo model drawn in Bevy from MuJoCo's own geom list
(`mjv_updateScene`): one reused entity per geom, lines as gizmos, labels for
the host to draw, lights, a camera driven like MuJoCo's `simulate`, and
picking and perturbing through `mjv_select` / `MjvPerturb`. Z up, metres.

| Crate | |
|---|---|
| `mujoco-bevy-assets` | An `MjModel`'s meshes (indexed), 2D textures (mipped) and materials as Bevy assets |
| `mujoco-bevy` | `MjView` + `MujocoPlugin` |

Debugging: `mujoco_bevy::debug` names `MuJoCo`'s visualisation flags, labels,
frames, groups and render flags for a panel, keeps decor sizes
(`MjView::set_vis_scale`) across recompiles, and `MjRenderOptions` draws
wireframe, fog, shadow casting and geom categories. `MjStatsPlugin` publishes
`MuJoCo`'s stage timers (it installs `mjcb_time`), the step's counts, what is
drawn and what collides, per top-level body subtree, as `mujoco/...` Bevy
diagnostics.

```text
MUJOCO_DYNAMIC_LINK_DIR=<dir with libmujoco> LD_LIBRARY_PATH=<same> \
  cargo run --example view -- path/to/model.xml
```

Depends on mujoco-rs with default features off, by path
(`../mujoco-rs-gl-accessors`), and Bevy 0.20.
