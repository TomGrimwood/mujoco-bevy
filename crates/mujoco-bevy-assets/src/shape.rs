//! Where on a mesh its edges and corners are, for a material that weathers
//! (paint worn off the edges, dirt gathered in the corners): each vertex's
//! distance to the nearest sharp outside edge and to the nearest inside
//! corner, in metres.
//!
//! A distance, not a flag, so it holds between vertices: across a triangle
//! with an edge on the feature, the distance to that edge is linear and the
//! interpolated value is exact. Where it is not (a big panel triangle whose
//! corners all touch features, a long diagonal from one feature to
//! another), the triangles are split until the interpolation is close:
//! each such edge at its midpoint, in every triangle that has it, so no
//! crack opens. Game meshes have big triangles, and a flag per vertex
//! smeared wear across whole panels.

use std::collections::VecDeque;

use bevy::math::Vec3;
use bevy::platform::collections::HashMap;

/// How far each distance is followed (outside edges, inside corners);
/// past it a vertex is just far.
pub const REACH: [f32; 2] = [0.05, 0.2];
/// Faces meeting at more than this bend (25 degrees) meet at a feature.
const SHARP_COS: f32 = 0.906;
/// How far off an interpolated distance may be before its edge is split.
const TOLERANCE: [f32; 2] = [0.008, 0.04];
/// Rounds of splitting at most.
const ROUNDS: usize = 3;
/// An edge this short is not split.
const SHORTEST: f32 = 0.02;

const FAR: u32 = u32::MAX;

/// A mesh's corners, by index: positions, normals, and texture coordinates
/// when it has them; split in place.
pub struct Corners<'a> {
    pub pos: &'a mut Vec<[f32; 3]>,
    pub nor: &'a mut Vec<[f32; 3]>,
    pub uv: &'a mut Vec<[f32; 2]>,
    pub indices: &'a mut Vec<u32>,
}

/// The distances for each corner (outside edge, inside corner), the mesh
/// refined first where they would not interpolate.
pub fn distances(mesh: Corners<'_>) -> Vec<[f32; 2]> {
    let Corners {
        pos,
        nor,
        uv,
        indices,
    } = mesh;
    let mut round = 0;
    loop {
        let field = Field::of(pos, indices);
        let splits = if round < ROUNDS {
            field.splits(pos, indices)
        } else {
            HashMap::default()
        };
        if splits.is_empty() {
            return field.of.iter().map(|&w| field.dist[w as usize]).collect();
        }
        split(pos, nor, uv, indices, &field, &splits);
        round += 1;
    }
}

/// Positions welded, the feature segments, and each welded point's
/// distance to the nearest of each kind.
struct Field {
    /// Each corner's welded point.
    of: Vec<u32>,
    point: Vec<Vec3>,
    /// Feature segments: (from, to) welded points, and which kind.
    segments: Vec<(u32, u32, usize)>,
    /// Each welded point's distance to each kind, and the segment.
    dist: Vec<[f32; 2]>,
    nearest: Vec<[u32; 2]>,
    /// Welded edges that are features.
    feature: HashMap<(u32, u32), usize>,
}

const fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

impl Field {
    fn of(pos: &[[f32; 3]], indices: &[u32]) -> Self {
        let mut welded: HashMap<[i64; 3], u32> = HashMap::with_capacity_and_hasher(pos.len(), Default::default());
        let mut point = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        let of: Vec<u32> = pos
            .iter()
            .map(|&p| {
                let k = p.map(|x| (f64::from(x) * 1e5).round() as i64);
                *welded.entry(k).or_insert_with(|| {
                    point.push(Vec3::from(p));
                    u32::try_from(point.len() - 1).unwrap_or(FAR)
                })
            })
            .collect();
        // Each welded edge's faces: (face normal, the far point) of up to
        // two; a third makes it no feature.
        let mut edges: HashMap<(u32, u32), (u8, [(Vec3, u32); 2])> =
            HashMap::with_capacity_and_hasher(indices.len(), Default::default());
        for t in indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| of[i as usize]);
            if a == b || b == c || a == c {
                continue;
            }
            let (pa, pb, pc) = (point[a as usize], point[b as usize], point[c as usize]);
            let n = (pb - pa).cross(pc - pa).normalize_or_zero();
            if n == Vec3::ZERO {
                continue;
            }
            for (e, far) in [((a, b), c), ((b, c), a), ((c, a), b)] {
                let entry = edges
                    .entry(key(e.0, e.1))
                    .or_insert((0, [(Vec3::ZERO, FAR); 2]));
                if entry.0 < 2 {
                    entry.1[entry.0 as usize] = (n, far);
                }
                entry.0 = entry.0.saturating_add(1);
            }
        }
        let mut segments = Vec::new();
        let mut feature = HashMap::default();
        for (&(a, b), &(count, [(n1, _), (n2, far2)])) in &edges {
            if count != 2 || n1.dot(n2) > SHARP_COS {
                continue;
            }
            // Outside if the second face falls away below the first.
            let below = n1.dot(point[far2 as usize] - point[a as usize]) < 0.0;
            let kind = usize::from(!below);
            feature.insert((a, b), kind);
            segments.push((a, b, kind));
        }
        let n = point.len();
        let mut dist = vec![REACH; n];
        let mut nearest = vec![[FAR; 2]; n];
        for (s, &(a, b, kind)) in segments.iter().enumerate() {
            let s = u32::try_from(s).unwrap_or(FAR);
            for v in [a, b] {
                dist[v as usize][kind] = 0.0;
                nearest[v as usize][kind] = s;
            }
        }
        // Each point takes the nearest segment its neighbours know of,
        // from the features outward as far as the reach: a queue of the
        // points that changed, so the far ones are never visited twice.
        let mut start = vec![0u32; n + 1];
        for &(a, b) in edges.keys() {
            start[a as usize + 1] += 1;
            start[b as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut next = vec![0u32; start[n] as usize];
        for &(a, b) in edges.keys() {
            for (from, to) in [(a, b), (b, a)] {
                next[fill[from as usize] as usize] = to;
                fill[from as usize] += 1;
            }
        }
        let mut queued = vec![false; n];
        let mut queue: VecDeque<u32> = VecDeque::new();
        for &(a, b, _) in &segments {
            for v in [a, b] {
                if !queued[v as usize] {
                    queued[v as usize] = true;
                    queue.push_back(v);
                }
            }
        }
        while let Some(from) = queue.pop_front() {
            queued[from as usize] = false;
            let (lo, hi) = (start[from as usize] as usize, start[from as usize + 1] as usize);
            for &to in &next[lo..hi] {
                for kind in 0..2 {
                    let s = nearest[from as usize][kind];
                    if s == FAR || nearest[to as usize][kind] == s {
                        continue;
                    }
                    let (sa, sb, _) = segments[s as usize];
                    let d = segment_distance(
                        point[to as usize],
                        point[sa as usize],
                        point[sb as usize],
                    );
                    if d < dist[to as usize][kind] {
                        dist[to as usize][kind] = d;
                        nearest[to as usize][kind] = s;
                        if !queued[to as usize] {
                            queued[to as usize] = true;
                            queue.push_back(to);
                        }
                    }
                }
            }
        }
        Self {
            of,
            point,
            segments,
            dist,
            nearest,
            feature,
        }
    }

    /// The distance at `p` by the segments points `a` and `b` know of.
    fn at(&self, p: Vec3, a: u32, b: u32, kind: usize) -> f32 {
        [a, b]
            .iter()
            .map(|&v| self.nearest[v as usize][kind])
            .filter(|&s| s != FAR)
            .map(|s| {
                let (sa, sb, _) = self.segments[s as usize];
                segment_distance(p, self.point[sa as usize], self.point[sb as usize])
            })
            .fold(REACH[kind], f32::min)
    }

    /// Welded edges whose midpoint the interpolation misses, each with its
    /// midpoint's distances.
    fn splits(&self, pos: &[[f32; 3]], indices: &[u32]) -> HashMap<(u32, u32), [f32; 2]> {
        let mut splits = HashMap::default();
        for t in indices.as_chunks::<3>().0 {
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = (self.of[t[i] as usize], self.of[t[j] as usize]);
                let e = key(a, b);
                if a == b || self.feature.contains_key(&e) || splits.contains_key(&e) {
                    continue;
                }
                let (da, db) = (self.dist[a as usize], self.dist[b as usize]);
                if (0..2).all(|k| da[k] + db[k] >= 2.0 * REACH[k]) {
                    continue;
                }
                let (pa, pb) = (Vec3::from(pos[t[i] as usize]), Vec3::from(pos[t[j] as usize]));
                if pa.distance(pb) < SHORTEST {
                    continue;
                }
                let mid = (pa + pb) * 0.5;
                let truth = [self.at(mid, a, b, 0), self.at(mid, a, b, 1)];
                let off = (0..2).any(|k| {
                    let lerp = (da[k] + db[k]) * 0.5;
                    lerp < REACH[k] && truth[k] - lerp > TOLERANCE[k]
                });
                if off {
                    splits.insert(e, truth);
                }
            }
        }
        splits
    }
}

/// Every triangle with an edge in `splits` split at its midpoints.
fn split(
    pos: &mut Vec<[f32; 3]>,
    nor: &mut Vec<[f32; 3]>,
    uv: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    field: &Field,
    splits: &HashMap<(u32, u32), [f32; 2]>,
) {
    let has_uv = !uv.is_empty();
    // A midpoint per pair of corners: a hard edge splits on both sides.
    let mut mids: HashMap<(u32, u32), u32> = HashMap::default();
    let mut mid = |a: u32, b: u32| -> Option<u32> {
        let e = key(field.of[a as usize], field.of[b as usize]);
        if !splits.contains_key(&e) {
            return None;
        }
        Some(*mids.entry(key(a, b)).or_insert_with(|| {
            let (a, b) = (a as usize, b as usize);
            pos.push(((Vec3::from(pos[a]) + Vec3::from(pos[b])) * 0.5).into());
            nor.push(
                (Vec3::from(nor[a]) + Vec3::from(nor[b]))
                    .normalize_or_zero()
                    .into(),
            );
            if has_uv {
                uv.push([(uv[a][0] + uv[b][0]) * 0.5, (uv[a][1] + uv[b][1]) * 0.5]);
            }
            u32::try_from(pos.len() - 1).unwrap_or(FAR)
        }))
    };
    let mut out = Vec::with_capacity(indices.len() * 2);
    for t in indices.as_chunks::<3>().0 {
        let [a, b, c] = *t;
        let m = [mid(a, b), mid(b, c), mid(c, a)];
        match m {
            [None, None, None] => out.extend([a, b, c]),
            [Some(ab), Some(bc), Some(ca)] => {
                out.extend([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]);
            }
            _ => {
                // Turned so the first split edge is a-b.
                let turn = (0..3)
                    .find(|&r| m[r].is_some() && m[(r + 2) % 3].is_none())
                    .unwrap_or(0);
                let v = [t[turn], t[(turn + 1) % 3], t[(turn + 2) % 3]];
                let (ab, bc) = (m[turn], m[(turn + 1) % 3]);
                match (ab, bc) {
                    (Some(ab), None) => out.extend([v[0], ab, v[2], ab, v[1], v[2]]),
                    (Some(ab), Some(bc)) => {
                        out.extend([ab, v[1], bc, v[0], ab, bc, v[0], bc, v[2]]);
                    }
                    _ => out.extend([a, b, c]),
                }
            }
        }
    }
    *indices = out;
}
