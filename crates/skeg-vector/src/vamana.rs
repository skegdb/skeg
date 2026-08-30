//! The Vamana graph index (DiskANN-style ANN).
//!
//! Vamana builds a single directed graph over `N` points where every node has
//! at most `R` out-edges, navigable from one entry point (the medoid). It is
//! the algorithmic core of the vector tier beyond flat scan; this chunk is the
//! in-memory, single-threaded build + search. On-disk format, streaming
//! insert, and parallel build come later.
//!
//! Reference: the `DiskANN` paper (Subramanya et al., 2019).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ahash::{AHashMap, AHashSet};
use crc32c::crc32c;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use skeg_platform::advise_sequential_file;
use skeg_simd::{cosine_f32, dot_int8};
use smallvec::SmallVec;

use crate::ivf_router::IvfRouter;
use crate::quant::{QuantKind, QuantizedVectors, Tq1ProxyMode};
use crate::source::{InMemoryVectorSource, VectorSource};
use crate::tq1_control::Tq1ProxyController;
use crate::visited::VisitedBitset;

/// Internal dense vector id (0..n).
pub type VecId = u32;

/// Maximum out-degree the `Node` can physically hold.
const MAX_R: usize = 64;

/// Vamana distance: `1 - cosine`, so smaller means closer.
fn dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_f32(a, b)
}

/// Unit-normalised copy of `v`. The int8 proxy used by the on-disk graph walk
/// is a dot product, which only tracks the cosine ordering on unit vectors.
fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

// -- graph node ----------------------------------------------------------------

/// One graph node: a bounded out-edge list.
///
/// `#[repr(C)]` + `Pod` guarantee the in-memory layout matches the
/// `graph.vmn` file layout exactly (one little-endian `u32` for `degree`
/// then `MAX_R` little-endian `u32` neighbour ids = 260 bytes per Node on
/// little-endian targets). This lets `--graph-mmap` reinterpret the
/// mmap'd file bytes as `&[Node]` via `bytemuck::cast_slice` without copy.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    degree: u32,
    neighbors: [VecId; MAX_R],
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("degree", &self.degree)
            .finish()
    }
}

impl Node {
    fn new() -> Node {
        Node {
            degree: 0,
            neighbors: [0; MAX_R],
        }
    }

    fn slice(&self) -> &[VecId] {
        &self.neighbors[..self.degree as usize]
    }

    #[allow(clippy::cast_possible_truncation)] // n <= MAX_R = 64
    fn set(&mut self, ids: &[VecId]) {
        let n = ids.len().min(MAX_R);
        self.neighbors[..n].copy_from_slice(&ids[..n]);
        self.degree = n as u32;
    }

    fn has(&self, id: VecId) -> bool {
        self.slice().contains(&id)
    }

    /// Append `id` if there is room and it is not already present.
    fn try_push(&mut self, id: VecId, max_degree: usize) -> bool {
        if self.has(id) {
            return true;
        }
        if (self.degree as usize) < max_degree {
            self.neighbors[self.degree as usize] = id;
            self.degree += 1;
            true
        } else {
            false
        }
    }
}

// -- search list ---------------------------------------------------------------

/// Bounded sorted candidate list for `GreedySearch`: keeps the `capacity`
/// entries closest to the target, ascending by distance, with a cursor over
/// the not-yet-expanded ones.
struct SearchList {
    items: SmallVec<[(f32, VecId, bool); 256]>,
    capacity: usize,
    next_unvisited: usize,
}

impl SearchList {
    fn new(capacity: usize) -> SearchList {
        SearchList {
            items: SmallVec::with_capacity(capacity + 1),
            capacity,
            next_unvisited: 0,
        }
    }

    /// Insert `(dist, id)` if it improves the list. Caller dedups ids.
    fn insert(&mut self, dist: f32, id: VecId) {
        let pos = self.items.partition_point(|&(d, _, _)| d < dist);
        if pos >= self.capacity {
            return;
        }
        self.items.insert(pos, (dist, id, false));
        if self.items.len() > self.capacity {
            self.items.truncate(self.capacity);
        }
        if pos <= self.next_unvisited {
            self.next_unvisited = pos;
        }
    }

    /// Closest not-yet-expanded entry; marks it expanded.
    fn pop_next_unvisited(&mut self) -> Option<(f32, VecId)> {
        while self.next_unvisited < self.items.len() {
            let i = self.next_unvisited;
            self.next_unvisited += 1;
            if !self.items[i].2 {
                self.items[i].2 = true;
                return Some((self.items[i].0, self.items[i].1));
            }
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = (f32, VecId)> + '_ {
        self.items.iter().map(|&(d, id, _)| (d, id))
    }
}

// -- greedy search -------------------------------------------------------------

/// Early-termination policy for the greedy walk. When the top-`k` of the
/// search list does not
/// change for `window` consecutive expansions, the walk stops short of
/// `list_size`. The list-size cap stays as the hard upper bound; this only
/// trims the tail when convergence has already happened.
///
/// Search paths (`VamanaIndex::search`, `DiskVamanaIndex::search_with_l`)
/// opt in. The build path (`insert_point_concurrent`) does NOT: a truncated
/// walk would shrink the candidate pool fed to `robust_prune` and degrade
/// the graph quality.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EarlyTerm {
    pub k: usize,
    pub window: usize,
}

/// Module-level switch for the opt-in early-termination behaviour. Set
/// once (typically by the binary at startup) and cached for the process
/// lifetime; subsequent calls to [`set_speed_enabled`] are silently
/// ignored. If the binary never sets the flag, the value is initialised
/// from the `SKEG_SPEED` environment variable on first read.
static SPEED_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Error returned by [`set_speed_enabled`] when the flag was already latched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeedAlreadySet;

impl std::fmt::Display for SpeedAlreadySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SPEED_FLAG already latched")
    }
}

impl std::error::Error for SpeedAlreadySet {}

/// Programmatic toggle for `--speed`. Call once from the server binary
/// before any search runs (`Server::bind*` is the natural point). The
/// value is latched on first read; calls after that point have no
/// effect and return [`SpeedAlreadySet`] so the caller can log a warning.
///
/// # Errors
///
/// Returns [`SpeedAlreadySet`] if the flag has already been read or set.
pub fn set_speed_enabled(enable: bool) -> Result<(), SpeedAlreadySet> {
    SPEED_FLAG.set(enable).map_err(|_| SpeedAlreadySet)
}

/// Opt-in early-termination toggle. Trades 0.3-0.7% recall@10 /
/// 1.3-2.8% recall@100 for +40-60% QPS (dual-distribution gate
/// 2026-05-21). Off by default. The CLI sets the flag via
/// [`set_speed_enabled`]; `SKEG_SPEED` env var is a fallback for tests
/// and ad-hoc invocations that have no Rust API access (e.g. running a
/// bench harness against an externally built server).
/// Stability window for the early-terminated walk: stop after this many
/// consecutive expansions leave the top-`k` signature unchanged. The recall
/// vs walk-time knob: 5 measured -1,4pt recall for walk/3 at 150k mxbai;
/// wider windows trade time back for recall. `SKEG_SPEED_WINDOW` overrides.
fn speed_window() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("SKEG_SPEED_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&w| w >= 1)
            .unwrap_or(5)
    })
}

fn speed_enabled() -> bool {
    *SPEED_FLAG.get_or_init(|| {
        matches!(
            std::env::var("SKEG_SPEED").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}

/// Hash the top-`k` ids from the (distance-sorted) search list. The hash
/// is intentionally weak (mul-add over u64): cheap to compute per iteration,
/// and collisions are tolerable because a false-positive stability claim
/// triggers at most one more iteration before the next signature.
fn top_k_signature(list: &SearchList, k: usize) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325; // FNV offset
    for (_, id) in list.iter().take(k) {
        h = h.wrapping_mul(0x100000001B3).wrapping_add(u64::from(id));
    }
    h
}

/// Greedy graph walk from `entry`. `dist_to_query(id)` is the distance of node
/// `id` to the implicit query (f32-exact in the build and the in-memory
/// search, quantized on disk); `neighbors(id)` yields a node's out-edges
/// (a direct slice read for a finished graph, a brief lock during the
/// concurrent build).
///
/// `visited` and `seen` are caller-owned scratch sets: the build reuses one
/// pair across every point a worker inserts (see [`BuildScratch`]) so a
/// rebuild does not allocate them per node. Both are cleared on entry.
/// Returns the bounded result list (ascending by distance); on return
/// `visited` holds the expanded nodes - the candidate pool for `robust_prune`.
///
/// If `trace` is `Some`, each node id is pushed to it in expansion order -
/// the graph access sequence, used by the cache-locality analysis.
///
/// Primitive gate passed 6.20x vs AHashSet: `visited`/`seen` are
/// [`VisitedBitset`] - bit-packed `N/64` bytes. The walk's access pattern
/// (~6400 test_and_set per query) is ~6x faster than AHashSet with mirrored
/// semantics (insert -> test_and_set; `true` means "already present").
#[allow(clippy::too_many_arguments)] // 8 args is the price of generic closures + scratch buffers
fn greedy_search<D, N>(
    seeds: &[VecId],
    list_size: usize,
    early_term: Option<EarlyTerm>,
    dist_to_query: D,
    neighbors: N,
    admit: Option<&dyn Fn(VecId) -> bool>,
    visited: &mut VisitedBitset,
    seen: &mut VisitedBitset,
    mut trace: Option<&mut Vec<VecId>>,
) -> SearchList
where
    D: Fn(VecId) -> f32,
    N: Fn(VecId) -> SmallVec<[VecId; MAX_R]>,
{
    let mut list = SearchList::new(list_size.max(1));
    visited.clear();
    seen.clear();

    // Seed the frontier from every entry point. A plain search passes one
    // (the medoid); a filtered search passes points drawn from the matching set
    // so the walk starts inside the matching region, not only at the centre.
    //
    // `admit` (filtered search only) gates which nodes may enter the list: only
    // matching nodes are kept and expanded, so the walk explores the matching
    // subgraph and a far matching cluster is not evicted by near non-matching
    // nodes. Edges are still followed through `neighbors`; non-matching nodes
    // are simply never admitted as candidates.
    for &s in seeds {
        if !seen.test_and_set(s) && admit.is_none_or(|a| a(s)) {
            list.insert(dist_to_query(s), s);
        }
    }

    // Early-termination: track top-k signature stability across expansions.
    let mut last_sig: u64 = 0;
    let mut stable_count: usize = 0;

    while let Some((_, cur)) = list.pop_next_unvisited() {
        if visited.test_and_set(cur) {
            continue;
        }
        if let Some(t) = trace.as_deref_mut() {
            t.push(cur);
        }
        for nbr in neighbors(cur) {
            if seen.test_and_set(nbr) {
                continue;
            }
            if admit.is_none_or(|a| a(nbr)) {
                list.insert(dist_to_query(nbr), nbr);
            }
        }
        if let Some(et) = early_term {
            let sig = top_k_signature(&list, et.k);
            if sig == last_sig {
                stable_count += 1;
                if stable_count >= et.window {
                    break;
                }
            } else {
                stable_count = 0;
                last_sig = sig;
            }
        }
    }
    list
}

// -- robust prune --------------------------------------------------------------

/// `RobustPrune`: from `candidates` (distances measured to `p`), select up to
/// `r` out-neighbors for `p`. A candidate `v` is dropped once a closer-picked
/// neighbour `p*` satisfies `alpha * d(p*, v) <= d(p, v)` - then the edge
/// `p -> v` is redundant because the walk reaches `v` through `p*`.
fn robust_prune(
    p: VecId,
    candidates: &mut Vec<(f32, VecId)>,
    alpha: f32,
    r: usize,
    source: &dyn VectorSource,
) -> SmallVec<[VecId; MAX_R]> {
    candidates.retain(|&(_, id)| id != p);
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let mut result: SmallVec<[VecId; MAX_R]> = SmallVec::new();
    let mut cursor = 0;
    while cursor < candidates.len() && result.len() < r {
        let (_, p_star) = candidates[cursor];
        result.push(p_star);
        cursor += 1;

        let p_star_vec = source.row(p_star);
        let mut write = cursor;
        for read in cursor..candidates.len() {
            let (d_pv, v) = candidates[read];
            if v == p_star {
                continue;
            }
            let d_star = dist(p_star_vec, source.row(v));
            if alpha * d_star > d_pv {
                candidates[write] = (d_pv, v);
                write += 1;
            }
        }
        candidates.truncate(write);
    }
    result
}

// -- build ---------------------------------------------------------------------

/// Tunables for [`VamanaIndex::build`].
#[derive(Debug, Clone, Copy)]
pub struct VamanaConfig {
    /// Max out-degree `R`.
    pub r: usize,
    /// Search-list size during the build. 64 is validated recall- and
    /// latency-neutral vs the old 125 across 100d-3072d and 60k-1.18M (8
    /// datasets), at ~2.5x faster builds: 125 was over-provisioned.
    pub l_build: usize,
    /// Search-list size at query time.
    pub l_search: usize,
    /// Pruning relaxation for pass 1 (aggressive, ~1.0).
    pub alpha1: f32,
    /// Pruning relaxation for pass 2 (relaxed, ~1.2).
    pub alpha2: f32,
    /// Sample size for the approximate medoid.
    pub medoid_sample: usize,
    /// RNG seed - the build is deterministic given the seed.
    pub seed: u64,
}

impl Default for VamanaConfig {
    fn default() -> VamanaConfig {
        VamanaConfig {
            r: 64,
            l_build: 64,
            l_search: 100,
            alpha1: 1.0,
            alpha2: 1.2,
            medoid_sample: 1000,
            seed: 0x42,
        }
    }
}

/// Build config for DiskVamana's internal rebuilds (flush, consolidate). The
/// Disk-path build width. Default 48 (down from the in-RAM default 64): the
/// l_build sweep (benches/l_build_sweep.rs, mxbai 100k tq2) showed 48 holds
/// recall@10 (0.994 vs 0.9955 at 64) while cutting the graph build ~24% - and
/// the build dominates consolidate/ingest (~90%, measured). `SKEG_L_BUILD`
/// overrides it (32 -> 0.991 recall, ~40% faster; for build-critical loads).
/// DIAGNOSTIC ONLY - do NOT ship. Navigates the walk with EXACT f32 distances
/// (read from vectors.bin) instead of the tq1 proxy. It PROVED the true
/// neighbours are reachable in the graph (glove r@100 0.58->0.95) - the 1-bit
/// proxy just can't steer to them at low dim. But it VIOLATES the whole point of
/// tq1 (minimal RAM): navigating on f32 means either holding f32 in RAM or one
/// positioned read per walked node (the ~5-6x latency measured). The
/// RAM-preserving fix for low-dim is `wide` (deep walk - costs latency, not
/// RAM), not this. Kept flag-gated (`SKEG_TQ1_NAV_F32`) purely as a measurement.
/// Opt-in adaptive rerank (efficiency knob, off by default). In best-by-proxy
/// order, skip a candidate's disk read when its proxy-estimated cosine plus a
/// per-vector RaBitQ bound `C*sqrt(1-g^2)` (g = code reconstruction quality, 1/scale)
/// can't beat the current k-th exact cosine. Measured +20-100% QPS at ~0 recall
/// loss on the median; it does NOT cut the 1M p99 tail (hard queries have nothing
/// to prune), so it stays opt-in, not a serving default. Env
/// `SKEG_TQ1_ADAPTIVE_RR=on`, coefficient from `SKEG_ADAPTIVE_MARGIN` (default 0.02).
fn adaptive_rr() -> Option<f32> {
    static M: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_ADAPTIVE_RR").ok().as_deref(),
            Some("on" | "1" | "true")
        )
        .then(|| {
            std::env::var("SKEG_ADAPTIVE_MARGIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.02f32)
        })
    })
}

/// Adaptive re-rank for the 2-/4-bit TurboQuant tiers, whose proxy is a
/// cosine estimate: a candidate whose estimate plus this flat margin cannot
/// reach the current k-th exact cosine stops the disk reads (best-first
/// order, so neither can the rest). Unlike tq1 there is no per-vector
/// reconstruction quality, so the margin is flat and conservative.
/// OFF by default: A/B under a 12-user storm on the 441k demo measured it
/// neutral (server p99 27,97 vs 28,24 ms) - the re-rank row cache and the
/// page cache already absorb the reads it would skip. It stays as an opt-in
/// (`SKEG_TQ24_ADAPTIVE_RR=1`) for regimes where re-rank reads actually
/// miss: cold opens, indexes far beyond RAM. `SKEG_ADAPTIVE_MARGIN`
/// overrides the margin (shared with the tq1 path); the toy sweep at 150k
/// said 0,02 buys p50 -27% warm for -0,1pt recall, 0,05 is recall-safer.
fn adaptive_rr_tq24() -> Option<f32> {
    static M: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ24_ADAPTIVE_RR").ok().as_deref(),
            Some("on" | "1" | "true")
        )
        .then(|| {
            std::env::var("SKEG_ADAPTIVE_MARGIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.05f32)
        })
    })
}

/// Diagnostic: rank by the quantized proxy alone, no f32 rerank (blog-comparable).
fn no_rerank() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_NO_RERANK").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

fn nav_f32_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_NAV_F32").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

/// QuIVer opt-in: build the 1-bit tq1 graph on the popcount metric. Off by
/// default (exact-f32 build). Env `SKEG_TQ1_QUIVER=on|1`. Read once.
///
/// MEASURED NULL RESULT (2026-07-04, mxbai-100k / qwen3-20k, hybrid default):
/// recall@100 moved 0.8934->0.8915 and 0.9383->0.9388 - within noise, no gain.
/// skeg's f32 graph is already popcount-navigable; the recall@100 limiter is
/// the popcount metric mis-RANKING candidates in the beam (fixed by asymmetric
/// navigation), not the topology (which QuIVer co-designs). Kept flag-gated for
/// reproducibility; do not enable expecting a win.
fn quiver_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("SKEG_TQ1_QUIVER").ok().as_deref(),
            Some("on" | "1" | "true")
        )
    })
}

/// Pick the graph builder for a disk rebuild: QuIVer popcount-metric build when
/// enabled and the tier is 1-bit tq1, else the exact-f32 build.
fn build_disk_graph(
    tier: QuantKind,
    vectors: Vec<f32>,
    ids: Vec<u64>,
    dim: usize,
    config: &VamanaConfig,
) -> VamanaIndex {
    if quiver_enabled() && matches!(tier, QuantKind::TurboQuant { bits: 1 }) {
        VamanaIndex::build_quiver_tq1(vectors, ids, dim, config)
    } else {
        VamanaIndex::build(vectors, ids, dim, config)
    }
}

/// Fold the base, runs, and delta into one graph WITHOUT rebuilding the base.
///
/// The from-scratch build runs two full passes over every node, each a greedy
/// search from the medoid, which is why a fold costs O(live) and grows
/// superlinearly with the index. But the base already has a good graph over the
/// very same points, and this reuses it, the way `patch_graph` already does for
/// deletes in production:
///
///   - a surviving base row whose neighbours all survived keeps its edges
///     verbatim, just remapped to new rows: zero distance computations;
///   - one that lost a neighbour bridges through the dead neighbour's own
///     surviving edges, then re-prunes, exactly the delete-patch repair;
///   - a genuinely new row (delta or run) is inserted with the same
///     greedy+prune+back-edge primitive the build uses per point, entering from
///     the medoid.
///
/// Cost tracks what changed: O(kept) is a remap, O(bridged) a local re-prune,
/// O(new) an insert each. The single pass at `alpha2` for repairs and inserts
/// is the shape the delete-patch verdict and the June incremental-insert work
/// both validated at full recall; alpha below 1 under churn is the documented
/// way to degrade a graph slowly.
///
/// `patch_connectivity` runs at the end, unconditionally: an insert-only graph
/// mutation cannot strand a node, but a bridge can, and a stranded node is a
/// silent recall loss.
fn build_patched_graph(
    patch: PatchBase,
    vectors: Vec<f32>,
    ids: Vec<u64>,
    base_origin: &[u32],
    dim: usize,
    cfg: &VamanaConfig,
) -> VamanaIndex {
    let n_new = ids.len() as u32;
    let src = InMemoryVectorSource::new(vectors, dim);

    // Old base row -> new row (u32::MAX = did not survive).
    let mut remap = vec![u32::MAX; patch.adj.len()];
    for (new_row, &orow) in base_origin.iter().enumerate() {
        if orow != u32::MAX {
            remap[orow as usize] = new_row as u32;
        }
    }
    let medoid = if (patch.medoid as usize) < remap.len()
        && remap[patch.medoid as usize] != u32::MAX
    {
        remap[patch.medoid as usize]
    } else {
        approximate_medoid(&src, n_new, cfg.medoid_sample, cfg.seed)
    };

    // Surviving base rows first: verbatim remap, or bridge+re-prune where a
    // neighbour died. Each row writes only its own node, so this is a plain
    // parallel map, no locks.
    let repaired: Vec<(Node, bool)> = (0..n_new as usize)
        .into_par_iter()
        .map(|new_row| {
            let orow = base_origin[new_row];
            if orow == u32::MAX {
                return (Node::new(), false);
            }
            let out = patch.adj[orow as usize].slice();
            let all_live = out.iter().all(|&w| remap[w as usize] != u32::MAX);
            let mut node = Node::new();
            if all_live {
                let mapped: SmallVec<[VecId; MAX_R]> =
                    out.iter().map(|&w| remap[w as usize]).collect();
                node.set(&mapped);
                return (node, false);
            }
            let mut cand: AHashSet<VecId> = AHashSet::new();
            for &w in out {
                if remap[w as usize] != u32::MAX {
                    cand.insert(remap[w as usize]);
                } else {
                    for &x in patch.adj[w as usize].slice() {
                        if x != orow && remap[x as usize] != u32::MAX {
                            cand.insert(remap[x as usize]);
                        }
                    }
                }
            }
            let pv = src.row(new_row as u32);
            let mut scored: Vec<(f32, VecId)> =
                cand.iter().map(|&c| (dist(pv, src.row(c)), c)).collect();
            let picked = robust_prune(new_row as u32, &mut scored, cfg.alpha2, cfg.r, &src);
            node.set(&picked);
            (node, true)
        })
        .collect();
    let bridged = repaired.iter().filter(|&&(_, b)| b).count();
    let graph: Vec<Mutex<Node>> = repaired.into_iter().map(|(n, _)| Mutex::new(n)).collect();

    // New rows: the build's own per-point insert, shuffled so concurrent
    // inserts spread across the per-node locks instead of marching in order.
    let mut new_points: Vec<VecId> = base_origin
        .iter()
        .enumerate()
        .filter(|&(_, &o)| o == u32::MAX)
        .map(|(i, _)| i as u32)
        .collect();
    let inserted = new_points.len();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    new_points.shuffle(&mut rng);
    let cap = n_new as usize;
    new_points.par_iter().for_each_init(
        || BuildScratch::with_capacity(cap),
        |scratch, &pt| {
            insert_point_concurrent(
                &graph,
                &src,
                &[medoid],
                pt,
                cfg.alpha2,
                cfg.r,
                cfg.l_build,
                scratch,
                None,
            );
        },
    );

    let mut nodes: Vec<Node> = graph.into_iter().map(Mutex::into_inner).collect();
    patch_connectivity(&mut nodes, &src, n_new, medoid, cfg.r);
    tracing::info!(
        "patched fold: {} rows kept verbatim, {bridged} bridged, {inserted} inserted",
        n_new as usize - bridged - inserted,
    );
    VamanaIndex {
        dim,
        n: n_new,
        vectors: Box::new(src),
        ids,
        nodes,
        medoid,
        r: cfg.r,
        l_search: cfg.l_search,
    }
}

fn disk_build_config() -> VamanaConfig {
    let mut cfg = VamanaConfig {
        l_build: 48,
        ..Default::default()
    };
    if let Some(l) = std::env::var("SKEG_L_BUILD")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        cfg.l_build = l;
    }
    cfg
}

/// Random `R`-regular directed graph - the build's starting point.
fn init_random_graph(nodes: &mut [Node], n: u32, r: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let target = r.min(n.saturating_sub(1) as usize);
    for i in 0..n {
        let mut chosen: SmallVec<[VecId; MAX_R]> = SmallVec::new();
        while chosen.len() < target {
            let c = rng.random_range(0..n);
            if c != i && !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        nodes[i as usize].set(&chosen);
    }
}

/// Approximate medoid: the sampled point with the smallest summed distance to
/// the rest of the sample. Exact medoid is O(N^2).
fn approximate_medoid(source: &dyn VectorSource, n: u32, sample_size: usize, seed: u64) -> VecId {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9);
    let mut all: Vec<VecId> = (0..n).collect();
    all.shuffle(&mut rng);
    let sample = &all[..sample_size.min(n as usize)];

    let mut best = sample[0];
    let mut best_sum = f32::INFINITY;
    for &cand in sample {
        let cv = source.row(cand);
        let mut sum = 0.0f32;
        for &other in sample {
            if other != cand {
                sum += dist(cv, source.row(other));
            }
        }
        if sum < best_sum {
            best_sum = sum;
            best = cand;
        }
    }
    best
}

/// Copy a node's out-edges out from under its lock.
fn locked_neighbors(graph: &[Mutex<Node>], id: VecId) -> SmallVec<[VecId; MAX_R]> {
    graph[id as usize].lock().slice().iter().copied().collect()
}

// -- build profiling -----------------------------------------------------------
//
// Cumulative nanoseconds per build phase, summed across worker threads. Each
// worker accumulates into its `BuildScratch` (no contention) and flushes to
// these atomics when the scratch is dropped (a handful of times per build).

static BUILD_WALK_NS: AtomicU64 = AtomicU64::new(0);
static BUILD_PRUNE_NS: AtomicU64 = AtomicU64::new(0);
static BUILD_BACKEDGE_NS: AtomicU64 = AtomicU64::new(0);

/// Cumulative `(greedy walk, robust-prune, back-edge)` nanoseconds across all
/// worker threads since the last [`reset_build_phase_times`]. A build
/// profiling hook; the counters are process-global.
#[must_use]
pub fn build_phase_times_ns() -> (u64, u64, u64) {
    (
        BUILD_WALK_NS.load(Ordering::Relaxed),
        BUILD_PRUNE_NS.load(Ordering::Relaxed),
        BUILD_BACKEDGE_NS.load(Ordering::Relaxed),
    )
}

/// Reset the build-phase counters read by [`build_phase_times_ns`].
pub fn reset_build_phase_times() {
    BUILD_WALK_NS.store(0, Ordering::Relaxed);
    BUILD_PRUNE_NS.store(0, Ordering::Relaxed);
    BUILD_BACKEDGE_NS.store(0, Ordering::Relaxed);
}

/// Per-worker scratch for the parallel build, reused across every point a
/// rayon task inserts so a rebuild does not allocate fresh sets and vectors
/// per node. `rayon::for_each_init` hands one to each task. Every field is
/// cleared at its point of use, so reuse is bit-identical to a fresh
/// allocation - this is a pure allocation-churn optimisation.
///
/// It also carries per-worker phase timers, flushed to the global counters
/// on drop (build profiling, build-optimization gate).
struct BuildScratch {
    /// Nodes expanded by the greedy walk - the candidate pool for pruning.
    visited: VisitedBitset,
    /// Nodes ever added to the search list - the walk's dedup set.
    seen: VisitedBitset,
    /// `(distance, id)` candidates passed to `robust_prune` for `p`.
    candidates: Vec<(f32, VecId)>,
    /// `(distance, id)` scratch for a back-edge re-prune.
    back: Vec<(f32, VecId)>,
    /// Nanoseconds this worker spent in the greedy walk / prune / back-edge.
    walk_ns: u64,
    prune_ns: u64,
    backedge_ns: u64,
}

impl BuildScratch {
    fn with_capacity(n: usize) -> BuildScratch {
        BuildScratch {
            visited: VisitedBitset::new(n),
            seen: VisitedBitset::new(n),
            candidates: Vec::new(),
            back: Vec::new(),
            walk_ns: 0,
            prune_ns: 0,
            backedge_ns: 0,
        }
    }
}

impl Drop for BuildScratch {
    fn drop(&mut self) {
        BUILD_WALK_NS.fetch_add(self.walk_ns, Ordering::Relaxed);
        BUILD_PRUNE_NS.fetch_add(self.prune_ns, Ordering::Relaxed);
        BUILD_BACKEDGE_NS.fetch_add(self.backedge_ns, Ordering::Relaxed);
    }
}

/// Insert one point concurrently: greedy-search for candidates, prune to the
/// out-neighbour set, propagate back-edges. Each graph access takes a brief
/// per-node lock; `robust_prune` touches only the (immutable) vectors. A
/// greedy walk may see a slightly stale graph - accepted by the Vamana paper,
/// it does not break the invariants.
///
/// `scratch` is reused across calls on the same worker; it is cleared as it
/// is filled, so the result is identical to a fresh allocation per point.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn insert_point_concurrent(
    graph: &[Mutex<Node>],
    source: &dyn VectorSource,
    entry: &[VecId],
    p: VecId,
    alpha: f32,
    r: usize,
    l_build: usize,
    scratch: &mut BuildScratch,
    proxy: Option<&Int8WalkProxy>,
) {
    let p_vec = source.row(p);
    let t_walk = Instant::now();
    // The walk only steers which candidates reach the prune; with the proxy
    // it ranks by the int8 dot, and the prune below re-scores in f32 either
    // way, so edge quality is decided on exact distances.
    match proxy {
        Some(px) => greedy_search(
            entry,
            l_build,
            None, // build: never early-terminate (full candidate pool for prune)
            |id| px.dist(p, id),
            |id| locked_neighbors(graph, id),
            None, // build: no filter admission
            &mut scratch.visited,
            &mut scratch.seen,
            None,
        ),
        None => greedy_search(
            entry,
            l_build,
            None, // build: never early-terminate (full candidate pool for prune)
            |id| dist(p_vec, source.row(id)),
            |id| locked_neighbors(graph, id),
            None, // build: no filter admission
            &mut scratch.visited,
            &mut scratch.seen,
            None,
        ),
    };
    scratch.walk_ns += t_walk.elapsed().as_nanos() as u64;

    let t_prune = Instant::now();
    scratch.candidates.clear();
    scratch.candidates.extend(
        scratch
            .visited
            .iter()
            .filter(|&id| id != p)
            .map(|id| (dist(p_vec, source.row(id)), id)),
    );
    for nbr in locked_neighbors(graph, p) {
        if nbr != p && !scratch.candidates.iter().any(|&(_, id)| id == nbr) {
            scratch.candidates.push((dist(p_vec, source.row(nbr)), nbr));
        }
    }

    let new_neighbors = robust_prune(p, &mut scratch.candidates, alpha, r, source);
    graph[p as usize].lock().set(&new_neighbors);
    scratch.prune_ns += t_prune.elapsed().as_nanos() as u64;

    let t_back = Instant::now();
    for &j in &new_neighbors {
        // Fast path: append p under j's lock if there is room.
        if graph[j as usize].lock().try_push(p, r) {
            continue;
        }
        // j is full: re-prune with p included. Read, prune unlocked, write.
        let j_vec = source.row(j);
        scratch.back.clear();
        scratch.back.extend(
            locked_neighbors(graph, j)
                .iter()
                .copied()
                .chain(std::iter::once(p))
                .map(|id| (dist(j_vec, source.row(id)), id)),
        );
        let new_j = robust_prune(j, &mut scratch.back, alpha, r, source);
        graph[j as usize].lock().set(&new_j);
    }
    scratch.backedge_ns += t_back.elapsed().as_nanos() as u64;
}

/// One build pass over a random permutation of all points, run in parallel
/// across the rayon thread pool. Inserts touch disjoint locks most of the
/// time, so contention is low.
/// Per-row int8 proxy for the build walk, behind `SKEG_BUILD_INT8_WALK=1`.
///
/// The walk only needs the cosine *ordering*, so each row is quantized to i8
/// with a symmetric per-row scale, and the scale is folded together with the
/// row's inverse norm into one factor: `-(dot_i8) * factor[v]` ranks rows the
/// way `1 - cosine` does. The query side's own factor is constant across one
/// walk and positive, so it drops out of the ordering. Only the walk uses
/// this; the prune re-scores every candidate in f32, which is mandatory (an
/// int8 prune measured recall 0,31 against 1,00 for f32).
struct Int8WalkProxy {
    data: Vec<i8>,
    factor: Vec<f32>,
    dim: usize,
}

impl Int8WalkProxy {
    fn build(source: &dyn VectorSource, n: u32, dim: usize) -> Int8WalkProxy {
        let mut data = vec![0i8; n as usize * dim];
        let mut factor = vec![0f32; n as usize];
        data.par_chunks_mut(dim)
            .zip(factor.par_iter_mut())
            .enumerate()
            .for_each(|(v, (row_q, f))| {
                let row = source.row(v as u32);
                let max_abs = row.iter().fold(0f32, |m, x| m.max(x.abs()));
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if max_abs == 0.0 || norm == 0.0 {
                    return;
                }
                let scale = max_abs / 127.0;
                for (q, x) in row_q.iter_mut().zip(row) {
                    *q = (x / scale).round() as i8;
                }
                *f = scale / norm;
            });
        Int8WalkProxy { data, factor, dim }
    }

    #[inline]
    fn dist(&self, p: VecId, v: VecId) -> f32 {
        let pr = &self.data[p as usize * self.dim..(p as usize + 1) * self.dim];
        let vr = &self.data[v as usize * self.dim..(v as usize + 1) * self.dim];
        -(dot_int8(pr, vr) as f32) * self.factor[v as usize]
    }
}

/// The int8 build walk is the default: gated at 150k real mxbai rows, fold
/// 28,3s -> 14,9s (1,90x) with recall 0,9878 -> 0,9880. (An earlier "1,09x"
/// verdict was measured on a binary that lacked the flag entirely - two
/// identical runs and their noise.) `SKEG_BUILD_INT8_WALK=0` restores the
/// f32 walk.
fn build_int8_walk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("SKEG_BUILD_INT8_WALK").is_ok_and(|v| v == "0"))
}

#[allow(clippy::too_many_arguments)] // mirrors insert_point_concurrent's parameters
fn run_pass_parallel(
    graph: &[Mutex<Node>],
    source: &dyn VectorSource,
    n: u32,
    medoid: VecId,
    alpha: f32,
    r: usize,
    l_build: usize,
    seed: u64,
    proxy: Option<&Int8WalkProxy>,
) {
    let mut order: Vec<VecId> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ u64::from(alpha.to_bits()));
    order.shuffle(&mut rng);
    // `for_each_init` builds one BuildScratch per rayon task and reuses it
    // across every point that task inserts: the build's scratch sets and
    // vectors are allocated once per worker, not once per node.
    let cap = n as usize;
    order.par_iter().for_each_init(
        || BuildScratch::with_capacity(cap),
        |scratch, &p| {
            insert_point_concurrent(graph, source, &[medoid], p, alpha, r, l_build, scratch, proxy);
        },
    );
}

/// BFS from the medoid; returns the reachable bitmap and the reachable count.
fn reachable_from_medoid(nodes: &[Node], n: u32, medoid: VecId) -> (Vec<bool>, u32) {
    let mut reachable = vec![false; n as usize];
    let mut queue = std::collections::VecDeque::new();
    reachable[medoid as usize] = true;
    queue.push_back(medoid);
    let mut count = 1;
    while let Some(cur) = queue.pop_front() {
        for &nbr in nodes[cur as usize].slice() {
            if !reachable[nbr as usize] {
                reachable[nbr as usize] = true;
                count += 1;
                queue.push_back(nbr);
            }
        }
    }
    (reachable, count)
}

/// Give every node unreachable from the medoid an inbound edge from its
/// nearest reachable node, so greedy search can find it.
fn patch_connectivity(
    nodes: &mut [Node],
    source: &dyn VectorSource,
    n: u32,
    medoid: VecId,
    r: usize,
) {
    let (reachable, count) = reachable_from_medoid(nodes, n, medoid);
    if count == n {
        return;
    }
    // The walk from the medoid can only ever visit reachable nodes, so a
    // greedy search over the graph finds each stranded node's nearest
    // reachable neighbour without touching the rest of the index. The exact
    // scan this replaces cost O(stranded x n) distances and was the whole
    // graph phase of an otherwise no-op fold: ~0,5s per stranded node at
    // 460k rows, 50s per shard with about a hundred of them.
    let stranded = n - count;
    let t = std::time::Instant::now();
    let mut visited = VisitedBitset::new(n as usize);
    let mut seen = VisitedBitset::new(n as usize);
    for u in 0..n {
        if reachable[u as usize] {
            continue;
        }
        let u_vec = source.row(u);
        let list = greedy_search(
            &[medoid],
            r.max(64),
            None,
            |v| dist(u_vec, source.row(v)),
            |v| SmallVec::from_slice(nodes[v as usize].slice()),
            None,
            &mut visited,
            &mut seen,
            None,
        );
        let best = list.iter().next().map_or(medoid, |(_, v)| v);
        nodes[best as usize].try_push(u, r);
    }
    tracing::info!(
        "patch_connectivity: {stranded} stranded of {n}, attached in {:?}",
        t.elapsed()
    );
}

// -- public index --------------------------------------------------------------

/// A Vamana graph index. The graph lives in RAM; the f32 vectors are drawn
/// from a [`VectorSource`] - an owned `Vec` or a memory-mapped file.
pub struct VamanaIndex {
    dim: usize,
    n: u32,
    vectors: Box<dyn VectorSource>,
    ids: Vec<u64>,
    nodes: Vec<Node>,
    medoid: VecId,
    r: usize,
    l_search: usize,
}

impl VamanaIndex {
    /// Build a Vamana index over `n` row-major f32 vectors held in memory,
    /// each labelled by the matching entry of `ids`. A thin wrapper over
    /// [`build_from_source`](Self::build_from_source).
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, if `vectors.len()` is not `ids.len() * dim`, or if
    /// `ids` is empty.
    #[must_use]
    pub fn build(
        vectors: Vec<f32>,
        ids: Vec<u64>,
        dim: usize,
        config: &VamanaConfig,
    ) -> VamanaIndex {
        let source = InMemoryVectorSource::new(vectors, dim);
        VamanaIndex::build_from_source(Box::new(source), ids, config)
    }

    /// QuIVer build for the 1-bit TurboQuant tier: construct the graph on the
    /// popcount metric (rotated sign vectors) so the cheap popcount walk can
    /// follow its edges, while keeping the f32 `vectors` for rerank. Falls back
    /// to identical topology as [`build`](Self::build) only in the sense that
    /// storage is unchanged; the edges differ (that is the point).
    #[must_use]
    pub fn build_quiver_tq1(
        vectors: Vec<f32>,
        ids: Vec<u64>,
        dim: usize,
        config: &VamanaConfig,
    ) -> VamanaIndex {
        let metric = crate::quant::tq1_sign_metric_vectors(&vectors, ids.len(), dim);
        let storage: Box<dyn VectorSource> = Box::new(InMemoryVectorSource::new(vectors, dim));
        let metric_src: Box<dyn VectorSource> = Box::new(InMemoryVectorSource::new(metric, dim));
        VamanaIndex::build_from_source_with_metric(storage, Some(metric_src), ids, config)
    }

    /// Build a Vamana index, drawing the f32 vectors from `source`. The source
    /// is kept by the index so [`save`](Self::save) and [`search`](Self::search)
    /// can read vectors after the build, without ever copying the dataset into
    /// the heap when `source` is memory-mapped.
    ///
    /// # Panics
    ///
    /// Panics if `ids` is empty or `source.len()` does not equal `ids.len()`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // a >4-billion-vector index is out of scope
    pub fn build_from_source(
        vectors: Box<dyn VectorSource>,
        ids: Vec<u64>,
        config: &VamanaConfig,
    ) -> VamanaIndex {
        Self::build_from_source_with_metric(vectors, None, ids, config)
    }

    /// Like [`build_from_source`](Self::build_from_source) but the graph is
    /// constructed using distances from `metric` (when `Some`) rather than the
    /// stored `vectors`. `metric` must have the same length/order as `vectors`.
    /// The index still holds `vectors` for `save`/rerank; only edge selection
    /// uses `metric`. This is the hook for QuIVer (build on the 1-bit popcount
    /// metric via [`build_quiver_tq1`](Self::build_quiver_tq1)); passing `None`
    /// reproduces the exact-f32 build.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn build_from_source_with_metric(
        vectors: Box<dyn VectorSource>,
        metric: Option<Box<dyn VectorSource>>,
        ids: Vec<u64>,
        config: &VamanaConfig,
    ) -> VamanaIndex {
        assert!(!ids.is_empty(), "Vamana needs at least one vector");
        assert_eq!(vectors.len(), ids.len(), "source/ids length mismatch");
        if let Some(m) = &metric {
            assert_eq!(m.len(), ids.len(), "metric/ids length mismatch");
        }
        let dim = vectors.dim();
        let n = ids.len() as u32;
        // Edge selection reads `metric_src`; the index keeps `vectors` for rerank.
        let metric_src: &dyn VectorSource = metric.as_deref().unwrap_or(&*vectors);

        let mut plain = vec![Node::new(); n as usize];
        init_random_graph(&mut plain, n, config.r, config.seed);
        let medoid = approximate_medoid(metric_src, n, config.medoid_sample, config.seed);

        // Both passes run in parallel across the rayon pool; the graph is a
        // Vec<Mutex<Node>> for the duration of the build, then unwrapped.
        let graph: Vec<Mutex<Node>> = plain.into_iter().map(Mutex::new).collect();
        let proxy = build_int8_walk_enabled()
            .then(|| Int8WalkProxy::build(metric_src, n, metric_src.dim()));
        run_pass_parallel(
            &graph,
            metric_src,
            n,
            medoid,
            config.alpha1,
            config.r,
            config.l_build,
            config.seed,
            proxy.as_ref(),
        );
        run_pass_parallel(
            &graph,
            metric_src,
            n,
            medoid,
            config.alpha2,
            config.r,
            config.l_build,
            config.seed.wrapping_add(1),
            proxy.as_ref(),
        );
        let mut nodes: Vec<Node> = graph.into_iter().map(Mutex::into_inner).collect();
        patch_connectivity(&mut nodes, metric_src, n, medoid, config.r);
        drop(metric);

        VamanaIndex {
            dim,
            n,
            vectors,
            ids,
            nodes,
            medoid,
            r: config.r,
            l_search: config.l_search,
        }
    }

    /// Vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n as usize
    }

    /// True if the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes held in RAM: the vector source's heap (zero for a memory-mapped
    /// source) plus ids and graph nodes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.vectors.heap_bytes()
            + self.ids.len() * std::mem::size_of::<u64>()
            + self.nodes.len() * std::mem::size_of::<Node>()
    }

    /// Out-degree histogram of the built graph: `hist[d]` is the number of
    /// nodes with exactly `d` out-edges. Length is `MAX_R + 1`. Used by the
    /// graph-layout-compaction gate (does the fixed-width node waste bytes?).
    #[must_use]
    pub fn degree_histogram(&self) -> Vec<u32> {
        let mut hist = vec![0u32; MAX_R + 1];
        for node in &self.nodes {
            hist[node.degree as usize] += 1;
        }
        hist
    }

    /// Graph entry point (the approximate medoid). Used by an external walk
    /// that drives the graph with its own proxy distance (the PQ-tier gate).
    #[must_use]
    pub fn medoid(&self) -> VecId {
        self.medoid
    }

    /// Out-edges of node `id`. Used by an external walk that drives the graph
    /// with its own proxy distance (the PQ-tier gate).
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    #[must_use]
    pub fn neighbors(&self, id: VecId) -> &[VecId] {
        self.nodes[id as usize].slice()
    }

    /// Approximate top-`k` `(id, cosine)` for `query`, highest cosine first.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let list_size = self.l_search.max(k);
        let mut visited = VisitedBitset::new(self.n as usize);
        let mut seen = VisitedBitset::new(self.n as usize);
        // Track top-(k*4) in the signature, not top-k: candidates just below
        // the top-k can still reshuffle while the head is stable. Mirrors
        // the disk path which gates on the re-rank pool (also k*4).
        let sig_k = (k * 4).max(32).min(list_size);
        let early = speed_enabled().then_some(EarlyTerm {
            k: sig_k,
            window: speed_window(),
        });
        let list = greedy_search(
            &[self.medoid],
            list_size,
            early,
            |id| dist(query, self.vectors.row(id)),
            |id| self.nodes[id as usize].slice().iter().copied().collect(),
            None, // in-RAM search: no filter admission
            &mut visited,
            &mut seen,
            None,
        );
        list.iter()
            .take(k)
            .map(|(d, id)| (self.ids[id as usize], 1.0 - d))
            .collect()
    }

    /// Serialise the index to `dir`: `graph.vmn` (graph + ids) and
    /// `vectors.bin` (f32 vectors). The directory is created if missing.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory or files cannot be written.
    pub fn save(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        // Write into the live base slot (or dir itself for the legacy flat
        // layout), so `save` and `open` agree on where the base lives.
        let dir = &base_dir(dir);
        std::fs::create_dir_all(dir)?;
        write_graph_vmn(
            &dir.join(GRAPH_FILE),
            self.n,
            self.dim,
            self.medoid,
            self.r,
            self.l_search,
            &self.ids,
            &self.nodes,
        )?;
        write_vectors_bin(&dir.join(VECTORS_FILE), &*self.vectors)?;
        Ok(())
    }
}

// -- on-disk format ------------------------------------------------------------
//
// graph.vmn   : 64-byte header, then n u64 ids, then n nodes
//               (each: degree u32 + MAX_R neighbour u32s).
// vectors.bin : 64-byte header, then n*dim f32 row-major.
//
// The graph and an int8 tier-1 quantisation live in RAM; the f32 vectors stay
// on disk and are read (one positioned `read_exact_at` per candidate) only to
// re-rank the survivors of the graph walk.

const GRAPH_FILE: &str = "graph.vmn";
/// Persisted quantised tier. Opening an index used to stream all of
/// `vectors.bin` (f32) and rebuild the tier every time: measured at ~16 s per
/// index for 223k x 1024 vectors, 914 MB read to produce ~26 MB of codes, and
/// over three minutes projected at 3M. That cost is paid on every restart,
/// crash recovery and deploy, which is exactly when a service can least afford
/// it. The codes are deterministic from the parent index (fixed rotation seed),
/// so they are written once and reloaded.
const TIER_CACHE_FILE: &str = "tier.cache.bin";
/// Header: magic, version, n, dim, tier tag, source len, source mtime.
/// Size alone is not enough: two indexes with the same n/dim produce caches of
/// the same length, so a stale or foreign file would be trusted.
const TIER_CACHE_MAGIC: u32 = 0x5449_4552; // "TIER"
const TIER_CACHE_VERSION: u32 = 1;
const TIER_CACHE_HEADER: usize = 4 + 4 + 4 + 4 + 4 + 8 + 8;

/// Read a tier cache whose fingerprint matches this index, or `None`.
///
/// The file is untrusted: every field is length-checked before use, so a
/// truncated, foreign or crafted cache returns `None` and the caller rebuilds.
/// Never panics, never trusts.
fn read_tier_cache(
    path: &Path,
    n: u32,
    dim: usize,
    tier_tag: u8,
    src_len: u64,
    src_mtime: u64,
) -> Option<Vec<u8>> {
    let buf = std::fs::read(path).ok()?;
    if buf.len() < TIER_CACHE_HEADER {
        return None;
    }
    if read_u32(&buf, 0) != TIER_CACHE_MAGIC || read_u32(&buf, 4) != TIER_CACHE_VERSION {
        return None;
    }
    if read_u32(&buf, 8) != n || read_u32(&buf, 12) as usize != dim {
        return None;
    }
    if read_u32(&buf, 16) != u32::from(tier_tag) {
        return None;
    }
    let len = u64::from_le_bytes(buf.get(20..28)?.try_into().ok()?);
    let mtime = u64::from_le_bytes(buf.get(28..36)?.try_into().ok()?);
    // vectors.bin changed under us (a consolidate rewrites it): the codes no
    // longer describe the vectors, and length alone would not catch it.
    if len != src_len || mtime != src_mtime {
        return None;
    }
    Some(buf[TIER_CACHE_HEADER..].to_vec())
}

/// Write the cache atomically: a half-written file must never be readable as a
/// valid one, or the next open would load a truncated tier.
fn write_tier_cache(
    path: &Path,
    n: u32,
    dim: usize,
    tier_tag: u8,
    src_len: u64,
    src_mtime: u64,
    body: &[u8],
) -> io::Result<()> {
    let mut out = Vec::with_capacity(TIER_CACHE_HEADER + body.len());
    out.extend_from_slice(&TIER_CACHE_MAGIC.to_le_bytes());
    out.extend_from_slice(&TIER_CACHE_VERSION.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&u32::from(tier_tag).to_le_bytes());
    out.extend_from_slice(&src_len.to_le_bytes());
    out.extend_from_slice(&src_mtime.to_le_bytes());
    out.extend_from_slice(body);
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &out)?;
    std::fs::rename(&tmp, path)
}
const VECTORS_FILE: &str = "vectors.bin";
/// Append-only WAL of delta inserts/deletes (replayed on open).
const DELTA_LOG_FILE: &str = "delta.log";
/// V2 vector WAL header.
const DELTA_WAL_V2_MAGIC: &[u8] = b"SKWL\x02";

/// V1 is headerless. V2 has per-record CRC32C.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeltaWalFormat {
    LegacyV1,
    FramedV2,
}

#[derive(Debug)]
enum DeltaWalOp {
    Insert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
}

impl DeltaWalFormat {
    fn detect(bytes: &[u8]) -> io::Result<(Self, &[u8])> {
        if bytes.starts_with(DELTA_WAL_V2_MAGIC) {
            return Ok((Self::FramedV2, &bytes[DELTA_WAL_V2_MAGIC.len()..]));
        }
        // Never parse a malformed V2 header as V1.
        if bytes.starts_with(b"SKWL") || bytes.first().is_some_and(|b| *b > 1) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid vector delta WAL header",
            ));
        }
        Ok((Self::LegacyV1, bytes))
    }
}

fn wal_record_body_len(op: u8, dim: usize) -> io::Result<usize> {
    match op {
        0 => 1usize
            .checked_add(8)
            .and_then(|n| dim.checked_mul(4).and_then(|v| n.checked_add(v)))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "vector WAL record too large")
            }),
        1 => Ok(1 + 8),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown vector WAL operation {op}"),
        )),
    }
}

fn decode_wal_op(body: &[u8]) -> DeltaWalOp {
    let id = u64::from_le_bytes(body[1..9].try_into().expect("record body length checked"));
    match body[0] {
        0 => DeltaWalOp::Insert {
            id,
            vector: body[9..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        },
        1 => DeltaWalOp::Delete { id },
        _ => unreachable!("operation byte validated before decoding"),
    }
}

/// A short tail is ignored. A bad V2 checksum fails recovery.
fn decode_wal_payload(
    format: DeltaWalFormat,
    payload: &[u8],
    dim: usize,
) -> io::Result<Vec<DeltaWalOp>> {
    let mut ops = Vec::new();
    let mut pos = 0;
    while pos < payload.len() {
        let op = payload[pos];
        let body_len = wal_record_body_len(op, dim)?;
        let record_len = match format {
            DeltaWalFormat::LegacyV1 => body_len,
            DeltaWalFormat::FramedV2 => body_len.checked_add(4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "vector WAL record too large")
            })?,
        };
        if payload.len() - pos < record_len {
            break; // crash during the final append: no complete record to apply
        }
        let body = &payload[pos..pos + body_len];
        if format == DeltaWalFormat::FramedV2 {
            let stored = u32::from_le_bytes(
                payload[pos + body_len..pos + record_len]
                    .try_into()
                    .expect("checksum window length checked"),
            );
            let actual = crc32c(body);
            if stored != actual {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("vector WAL checksum mismatch at byte {pos}"),
                ));
            }
        }
        ops.push(decode_wal_op(body));
        pos += record_len;
    }
    Ok(ops)
}

fn decode_wal(bytes: &[u8], dim: usize) -> io::Result<(DeltaWalFormat, Vec<DeltaWalOp>)> {
    let (format, payload) = DeltaWalFormat::detect(bytes)?;
    Ok((format, decode_wal_payload(format, payload, dim)?))
}

fn encode_wal_body(op: &DeltaWalOp) -> Vec<u8> {
    match op {
        DeltaWalOp::Insert { id, vector } => {
            let mut body = Vec::with_capacity(1 + 8 + vector.len() * 4);
            body.push(0);
            body.extend_from_slice(&id.to_le_bytes());
            for &x in vector {
                body.extend_from_slice(&x.to_le_bytes());
            }
            body
        }
        DeltaWalOp::Delete { id } => {
            let mut body = Vec::with_capacity(9);
            body.push(1);
            body.extend_from_slice(&id.to_le_bytes());
            body
        }
    }
}

fn encode_wal_record(format: DeltaWalFormat, op: &DeltaWalOp) -> Vec<u8> {
    let mut record = encode_wal_body(op);
    if format == DeltaWalFormat::FramedV2 {
        record.extend_from_slice(&crc32c(&record).to_le_bytes());
    }
    record
}

fn write_framed_wal(path: &Path, ops: &[DeltaWalOp]) -> io::Result<()> {
    let mut bytes = DELTA_WAL_V2_MAGIC.to_vec();
    for op in ops {
        bytes.extend_from_slice(&encode_wal_record(DeltaWalFormat::FramedV2, op));
    }
    std::fs::write(path, bytes)
}
/// Persisted IVF router sidecar (centroids + cell assignment).
const IVF_FILE: &str = "ivf.bin";
/// Optional per-base-row u64 attribute column (little-endian), for range-filtered
/// search. Absent unless [`DiskVamanaIndex::set_attr`] was called.
const ATTR_FILE: &str = "attr.bin";
/// Persists which tier-1 quantiser a read-write disk index rebuilds at `open`
/// and `consolidate`. Absent => `Int8` (the historical default). Only the KIND
/// is stored, not codes: every tier here is deterministic from `vectors.bin`
/// (int8 calibrates a scale; TurboQuant is data-oblivious, seed-derived).
const TIER_FILE: &str = "tier.kind";

/// Read the persisted RW tier kind (`Int8` if the sidecar is absent).
fn read_tier(dir: &Path) -> QuantKind {
    match std::fs::read_to_string(dir.join(TIER_FILE)) {
        Ok(s) => match s.trim() {
            "tq1" => QuantKind::TurboQuant { bits: 1 },
            "tq2" => QuantKind::TurboQuant { bits: 2 },
            "tq4" => QuantKind::TurboQuant { bits: 4 },
            _ => QuantKind::Int8,
        },
        Err(_) => QuantKind::Int8,
    }
}

/// Wire string for a RW tier kind. Non-RW tiers fall back to `int8`.
fn tier_str(t: QuantKind) -> &'static str {
    match t {
        QuantKind::TurboQuant { bits: 1 } => "tq1",
        QuantKind::TurboQuant { bits: 2 } => "tq2",
        QuantKind::TurboQuant { bits: 4 } => "tq4",
        _ => "int8",
    }
}

fn write_tier(dir: &Path, t: QuantKind) -> io::Result<()> {
    std::fs::write(dir.join(TIER_FILE), tier_str(t))
}
const GRAPH_MAGIC: u32 = 0x4E_4D_56_47; // "GVMN"
const VEC_MAGIC: u32 = 0x4E_49_42_56; // "VBIN"
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
/// Rows read per `vectors.bin` chunk while building the int8 tier on `open`.
/// Bounds peak open-path RAM to one chunk (`TIER_CHUNK_ROWS * dim * 4` bytes)
/// plus the tier itself, instead of a transient the size of the f32 set.
const TIER_CHUNK_ROWS: usize = 4096;
/// Buffered sequential write size for `vectors.bin`. This bounds save-path
/// staging while avoiding the 8 KiB default buffer on large persistent volumes.
const VECTOR_WRITE_BUFFER_BYTES: usize = 1 << 20;
/// Upper bound for one positioned read during a sequential maintenance scan.
/// The output vectors are retained by the caller, but this staging buffer stays
/// bounded so a large index does not transiently double its f32 footprint.
const SEQUENTIAL_VECTOR_READ_BYTES: usize = 1 << 20;

/// Read contiguous f32 rows from `vectors.bin` in bounded positioned-read
/// blocks. Used only by maintenance jobs that consume a full row range in order.
fn read_f32_rows_sequential(
    file: &File,
    start_row: usize,
    rows: usize,
    dim: usize,
) -> io::Result<Vec<f32>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if dim == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector dimension must be positive",
        ));
    }
    let row_bytes = dim
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "vector row size overflow"))?;
    let value_count = rows
        .checked_mul(dim)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "vector range size overflow"))?;
    let start_bytes = start_row
        .checked_mul(row_bytes)
        .and_then(|n| n.checked_add(HEADER_LEN))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "vector range offset overflow")
        })?;
    let total_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "vector range size overflow"))?;
    let start_bytes = u64::try_from(start_bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector range offset is too large",
        )
    })?;
    let total_bytes_u64 = u64::try_from(total_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "vector range is too large"))?;
    advise_sequential_file(file, start_bytes, total_bytes_u64)?;

    let rows_per_read = (SEQUENTIAL_VECTOR_READ_BYTES / row_bytes).max(1);
    let mut vectors = vec![0.0f32; value_count];
    let mut read_rows = 0usize;
    #[cfg(target_endian = "little")]
    // `f32` is Pod, so the initialized result allocation can safely receive
    // the little-endian bytes written by `write_vectors_bin` without decoding
    // every element into a second buffer.
    let bytes = bytemuck::cast_slice_mut(&mut vectors);
    #[cfg(target_endian = "big")]
    let mut raw = Vec::with_capacity(SEQUENTIAL_VECTOR_READ_BYTES);
    while read_rows < rows {
        let chunk_rows = (rows - read_rows).min(rows_per_read);
        let chunk_bytes = chunk_rows.checked_mul(row_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "vector chunk size overflow")
        })?;
        let chunk_offset = read_rows.checked_mul(row_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "vector range offset overflow")
        })?;
        let offset = start_bytes
            + u64::try_from(chunk_offset).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "vector range offset is too large",
                )
            })?;
        #[cfg(target_endian = "little")]
        file.read_exact_at(&mut bytes[chunk_offset..chunk_offset + chunk_bytes], offset)?;
        #[cfg(target_endian = "big")]
        {
            raw.resize(chunk_bytes, 0);
            file.read_exact_at(&mut raw, offset)?;
            let output_offset = read_rows.checked_mul(dim).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "vector output offset overflow")
            })?;
            vectors[output_offset..output_offset + chunk_rows * dim].copy_from_slice(
                &raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
            );
        }
        read_rows += chunk_rows;
    }
    Ok(vectors)
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn write_graph_vmn(
    path: &Path,
    n: u32,
    dim: usize,
    medoid: VecId,
    r: usize,
    l_search: usize,
    ids: &[u64],
    nodes: &[Node],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0..4].copy_from_slice(&GRAPH_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    hdr[8..12].copy_from_slice(&n.to_le_bytes());
    hdr[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
    hdr[16..20].copy_from_slice(&medoid.to_le_bytes());
    hdr[20..24].copy_from_slice(&(r as u32).to_le_bytes());
    hdr[24..28].copy_from_slice(&(l_search as u32).to_le_bytes());
    f.write_all(&hdr)?;
    for &id in ids {
        f.write_all(&id.to_le_bytes())?;
    }
    for node in nodes {
        f.write_all(&node.degree.to_le_bytes())?;
        for &nb in &node.neighbors {
            f.write_all(&nb.to_le_bytes())?;
        }
    }
    f.flush()
}

#[allow(clippy::cast_possible_truncation)]
fn write_vectors_bin(path: &Path, source: &dyn VectorSource) -> io::Result<()> {
    let n = source.len() as u32;
    let dim = source.dim() as u32;
    let mut f = BufWriter::with_capacity(VECTOR_WRITE_BUFFER_BYTES, File::create(path)?);
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0..4].copy_from_slice(&VEC_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    hdr[8..12].copy_from_slice(&n.to_le_bytes());
    hdr[12..16].copy_from_slice(&dim.to_le_bytes());
    f.write_all(&hdr)?;
    // One row at a time: an mmap source never materialises the whole dataset.
    for id in 0..n {
        f.write_all(bytemuck::cast_slice(source.row(id)))?;
    }
    f.flush()
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Storage backing for the Vamana graph's `Node` array. `Owned` is the
/// default (a heap `Vec<Node>`, parsed at open from `graph.vmn`). `Mapped`
/// is the opt-in `--graph-mmap` path: hold the `graph.vmn` mmap and
/// reinterpret the on-disk Node region as `&[Node]` via `bytemuck::cast_slice`
/// (Node is `#[repr(C)]` + `Pod`, with file layout = in-memory layout). The
/// OS page cache can then reclaim graph pages under memory pressure.
#[derive(Debug)]
enum NodeBacking {
    Owned(Vec<Node>),
    Mapped {
        file: skeg_platform::MappedFile,
        /// Byte offset into the mmap where the Node array starts (= header
        /// + ids region).
        offset: usize,
        /// Node count - validated against the graph header at open.
        count: usize,
    },
}

impl std::ops::Deref for NodeBacking {
    type Target = [Node];

    fn deref(&self) -> &[Node] {
        match self {
            NodeBacking::Owned(v) => v.as_slice(),
            NodeBacking::Mapped {
                file,
                offset,
                count,
            } => {
                let n_bytes = *count * std::mem::size_of::<Node>();
                let bytes = &file.as_bytes()[*offset..*offset + n_bytes];
                bytemuck::cast_slice(bytes)
            }
        }
    }
}

/// A Vamana index served from disk: the immutable main graph and an int8
/// tier-1 quantisation are held in RAM, the full-precision f32 vectors stay
/// in `vectors.bin`. Live inserts land in an in-RAM `delta` (a small flat
/// buffer); a search merges a graph walk over the main with a flat scan of
/// the delta. `consolidate` folds the delta back into a fresh on-disk graph.
/// An immutable on-disk Vamana segment: the graph, its in-RAM quantized tier,
/// the external-id mapping, and the f32 `vectors.bin` it re-ranks against. The
/// index holds one base segment today; the LSM design folds streaming writes
/// into additional segments (runs) without ever mutating an existing one.
struct Segment {
    main_n: u32,
    nodes: NodeBacking,
    ids: Vec<u64>,
    id_to_main_row: AHashMap<u64, VecId>,
    medoid: VecId,
    quant: QuantizedVectors,
    vectors_file: File,
    /// Re-rank row cache: the p99 at rest is dominated by cold positioned
    /// reads of candidate rows, and popular rows (hubs, dense regions) recur
    /// across queries. Segments are immutable - a fold replaces them
    /// wholesale - so the cache dies with its segment and can never serve a
    /// stale row. `None` when the budget is zero.
    row_cache: Option<std::sync::Mutex<RowSlotCache>>,
    /// Semantic entry cache: query sketch -> the base rows where similar
    /// queries landed, used as extra walk seeds so a repeated or similar
    /// query starts near its answer instead of at the medoid. Extra seeds
    /// only ADD candidates, so recall cannot drop; rows are segment-local
    /// and the cache dies with its segment, so a fold can never leave a
    /// stale row behind. `None` under `SKEG_ENTRY_CACHE=0`.
    entry_cache: Option<std::sync::Mutex<EntrySlotCache>>,
}

/// Direct-mapped sketch -> seed-rows cache, 4096 slots, overwrite eviction.
struct EntrySlotCache {
    slots: Vec<Option<(u16, SmallVec<[VecId; 8]>)>>,
}

impl EntrySlotCache {
    fn new() -> EntrySlotCache {
        EntrySlotCache { slots: vec![None; 4096] }
    }

    fn get(&self, sketch: u16) -> Option<SmallVec<[VecId; 8]>> {
        match &self.slots[sketch as usize % self.slots.len()] {
            Some((s, rows)) if *s == sketch => Some(rows.clone()),
            _ => None,
        }
    }

    fn put(&mut self, sketch: u16, rows: SmallVec<[VecId; 8]>) {
        let n = self.slots.len();
        self.slots[sketch as usize % n] = Some((sketch, rows));
    }
}

/// 16-bit sign sketch of a query, dimension-agnostic: every coordinate's
/// sign folds (xor) into one of 16 bits, so two queries share a slot only
/// when their broad sign structure matches. Identical queries always
/// collide into the same slot; near-duplicates usually do.
fn query_sketch(q: &[f32]) -> u16 {
    let mut key = 0u16;
    for (i, &x) in q.iter().enumerate() {
        key ^= u16::from(x > 0.0) << (i & 15);
    }
    key
}

/// `SKEG_IVF_SEEDS=1` seeds unfiltered walks from the query's nearest IVF
/// cell. Off until the hop/recall gate clears it.
fn ivf_seeds_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SKEG_IVF_SEEDS").is_ok_and(|v| v == "1"))
}

fn entry_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("SKEG_ENTRY_CACHE").is_ok_and(|v| v == "0"))
}

fn new_entry_cache() -> Option<std::sync::Mutex<EntrySlotCache>> {
    entry_cache_enabled().then(|| std::sync::Mutex::new(EntrySlotCache::new()))
}

/// The base of a vindex (`graph.vmn` + `vectors.bin` + `tier.cache.bin`) lives
/// in a generation slot `g0`/`g1`, and a single `CURRENT` pointer file names
/// the live slot. A consolidate builds the new base in the OTHER slot, fsyncs
/// it, then flips `CURRENT` with one atomic rename - so a crash mid-swap
/// leaves the old base whole under the old pointer, never a torn mix of two
/// generations' files (review P0). Indexes written before this scheme have no
/// `CURRENT` and keep their base flat in the vindex dir; that stays readable.
const CURRENT_FILE: &str = "CURRENT";

/// The directory holding the LIVE base files for `dir`: `dir/gN` per the
/// `CURRENT` pointer, or `dir` itself for a legacy flat layout.
fn base_dir(dir: &Path) -> std::path::PathBuf {
    match std::fs::read_to_string(dir.join(CURRENT_FILE)) {
        Ok(s) => dir.join(format!("g{}", s.trim())),
        Err(_) => dir.to_path_buf(),
    }
}

/// The current live slot (`Some(0|1)`), or `None` for a legacy flat layout.
fn current_slot(dir: &Path) -> Option<u8> {
    std::fs::read_to_string(dir.join(CURRENT_FILE))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Atomically point `CURRENT` at `slot`: write a temp file, fsync it, rename
/// over `CURRENT`, fsync the directory so the rename is durable.
fn set_current_slot(dir: &Path, slot: u8) -> io::Result<()> {
    let tmp = dir.join("CURRENT.tmp");
    std::fs::write(&tmp, slot.to_string())?;
    File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, dir.join(CURRENT_FILE))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Install `built_tmp` (a directory holding the new base files) as the live
/// base for `dir`, atomically: fsync its files, rename it into the inactive
/// slot, flip `CURRENT`, then remove the old slot (or the legacy flat files).
/// The caller's already-open fds into `built_tmp` follow the inode across the
/// rename.
fn install_base_generation(dir: &Path, built_tmp: &Path) -> io::Result<()> {
    // Durability: every file in the new base must be on disk before it can
    // become live.
    for entry in std::fs::read_dir(built_tmp)? {
        let path = entry?.path();
        if path.is_file() {
            File::open(&path)?.sync_all()?;
        }
    }
    let old = current_slot(dir);
    let next = match old {
        Some(0) => 1u8,
        _ => 0u8,
    };
    let slot_dir = dir.join(format!("g{next}"));
    let _ = std::fs::remove_dir_all(&slot_dir); // a torn prior attempt, if any
    std::fs::rename(built_tmp, &slot_dir)?;
    File::open(dir)?.sync_all()?;
    set_current_slot(dir, next)?;
    // Reclaim the superseded generation. A crash before this leaves a dead
    // slot the next install overwrites; never the live one.
    match old {
        Some(s) => {
            let _ = std::fs::remove_dir_all(dir.join(format!("g{s}")));
        }
        None => {
            // Legacy flat files are now dead - the pointer names g{next}.
            let _ = std::fs::remove_file(dir.join(GRAPH_FILE));
            let _ = std::fs::remove_file(dir.join(VECTORS_FILE));
            let _ = std::fs::remove_file(dir.join(TIER_CACHE_FILE));
        }
    }
    Ok(())
}

/// Marker written (and fsynced) inside a run directory once every file of
/// the run is durable and the run is installed: only marked runs reopen.
const RUN_OK_FILE: &str = "run.ok";

/// Open one segment directory (the base, or a `run-N`): parse and validate
/// `graph.vmn`, open `vectors.bin`, build or reload the quantised tier.
/// Returns the segment plus the `(dim, l_search)` its header declares.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
fn open_segment(
    dir: &Path,
    tier: QuantKind,
    mmap_tier: bool,
    mmap_graph: bool,
) -> io::Result<(Segment, usize, usize)> {
        // graph.vmn
    // The live base files live in dir's current generation slot (or dir
    // itself for a legacy flat layout).
    let bdir = base_dir(dir);
    let graph_bytes = std::fs::read(bdir.join(GRAPH_FILE))?;
    if graph_bytes.len() < HEADER_LEN
        || read_u32(&graph_bytes, 0) != GRAPH_MAGIC
        || read_u32(&graph_bytes, 4) != FORMAT_VERSION
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad graph.vmn header",
        ));
    }
    let n = read_u32(&graph_bytes, 8);
    let dim = read_u32(&graph_bytes, 12) as usize;
    let medoid = read_u32(&graph_bytes, 16);
    let l_search = read_u32(&graph_bytes, 24) as usize;

    // Every field below is read straight off disk. Validate up front so a
    // truncated or crafted file becomes a clean `InvalidData`, not an
    // out-of-bounds slice/index panic that (panic=abort) kills the process
    // on open or on the first search. The mmap path already checks its
    // length; the owned path did not.
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "corrupt graph.vmn");
    let node_len = std::mem::size_of::<Node>();
    let need = (n as usize)
        .checked_mul(8)
        .and_then(|ids| {
            (n as usize)
                .checked_mul(node_len)
                .and_then(|nd| ids.checked_add(nd))
        })
        .and_then(|body| body.checked_add(HEADER_LEN))
        .ok_or_else(bad)?;
    if graph_bytes.len() < need {
        return Err(bad());
    }
    if n != 0 && medoid >= n {
        return Err(bad());
    }

    let mut pos = HEADER_LEN;
    let mut ids = Vec::with_capacity(n as usize);
    for _ in 0..n {
        ids.push(u64::from_le_bytes(
            graph_bytes[pos..pos + 8]
                .try_into()
                .expect("8-byte window by construction"),
        ));
        pos += 8;
    }
    debug_assert_eq!(
        node_len,
        4 + MAX_R * 4,
        "Node layout drifted from file format"
    );
    let nodes_offset = pos;
    let nodes = if mmap_graph {
        // Whole-file mmap, cast the Node region as `&[Node]` on access.
        // Skip the per-Node parsing - the file IS the in-memory layout
        // (Node is `#[repr(C)] + Pod`, little-endian u32 fields).
        let file = skeg_platform::MappedFile::open(&bdir.join(GRAPH_FILE))?;
        // Sanity-check the mapped region covers all `n` nodes.
        let need = nodes_offset + (n as usize) * node_len;
        if file.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "graph.vmn truncated: nodes region beyond file length",
            ));
        }
        // Greedy walk follows arbitrary out-edges - access is random
        // across the Node array. `MADV_RANDOM` tells the kernel to
        // skip read-ahead for pages we won't touch. The call is a
        // hint, so a failure (sandbox, unusual fs) is only logged.
        if let Err(e) = file.advise_random() {
            tracing::debug!("graph mmap MADV_RANDOM failed: {e}");
        }
        NodeBacking::Mapped {
            file,
            offset: nodes_offset,
            count: n as usize,
        }
    } else {
        let mut v = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let degree = read_u32(&graph_bytes, pos);
            // `degree` indexes `neighbors[..degree]` (a `[u32; MAX_R]`) on
            // every walk; a disk value > MAX_R is an out-of-range slice.
            if degree as usize > MAX_R {
                return Err(bad());
            }
            let mut neighbors = [0u32; MAX_R];
            for (k, slot) in neighbors.iter_mut().enumerate() {
                *slot = read_u32(&graph_bytes, pos + 4 + k * 4);
            }
            // Neighbor ids index `nodes[id]` / `ids[id]` during a walk.
            if neighbors[..degree as usize].iter().any(|&nb| nb >= n) {
                return Err(bad());
            }
            v.push(Node { degree, neighbors });
            pos += node_len;
        }
        NodeBacking::Owned(v)
    };
    // `pos` is consumed by the in-RAM path; the mmap path skips ahead.
    let _ = pos;

    // vectors.bin: verify header, stream f32 to build the int8 tier.
    let vectors_file = File::open(bdir.join(VECTORS_FILE))?;
    let mut vhdr = [0u8; HEADER_LEN];
    vectors_file.read_exact_at(&mut vhdr, 0)?;
    if read_u32(&vhdr, 0) != VEC_MAGIC || read_u32(&vhdr, 4) != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad vectors.bin header",
        ));
    }
    if read_u32(&vhdr, 8) != n || read_u32(&vhdr, 12) as usize != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "graph.vmn and vectors.bin disagree on n/dim",
        ));
    }

    // Build the int8 tier from unit-normalised vectors (so its dot-product
    // proxy tracks the cosine ordering the graph was built with). The file
    // is read in fixed chunks and quantised on the fly: peak open-path RAM
    // is one chunk plus the int8 tier, never a transient the size of the
    // f32 set (at 1M x 1024 that balloon was ~8 GiB and inflated serve RSS
    // long after the buffers were freed).
    let n_usize = n as usize;
    let read_rows = |emit: &mut dyn FnMut(&[f32])| -> io::Result<()> {
        let mut buf = vec![0u8; TIER_CHUNK_ROWS.min(n_usize.max(1)) * dim * 4];
        let mut row = vec![0f32; dim];
        let mut done = 0usize;
        while done < n_usize {
            let rows = TIER_CHUNK_ROWS.min(n_usize - done);
            let span = &mut buf[..rows * dim * 4];
            vectors_file.read_exact_at(span, HEADER_LEN as u64 + (done * dim * 4) as u64)?;
            for chunk in span.chunks_exact(dim * 4) {
                for (slot, c) in row.iter_mut().zip(chunk.chunks_exact(4)) {
                    *slot = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                emit(&normalized(&row));
            }
            done += rows;
        }
        Ok(())
    };
    // Fast path: a valid cached tier skips streaming and re-quantising the
    // whole of vectors.bin. The fingerprint ties the file to THIS index
    // (n, dim, tier, and the source's length + mtime), because size alone
    // would accept a cache built from different vectors of the same shape.
    let cache_path = bdir.join(TIER_CACHE_FILE);
    let src_meta = std::fs::metadata(bdir.join(VECTORS_FILE)).ok();
    let fingerprint = |meta: Option<&std::fs::Metadata>| -> (u64, u64) {
        match meta {
            Some(m) => (
                m.len(),
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos() as u64),
            ),
            None => (0, 0),
        }
    };
    let (src_len, src_mtime) = fingerprint(src_meta.as_ref());
    let tier_tag = tier.to_wire().unwrap_or(0);
    let cached = if let QuantKind::TurboQuant { bits } = tier {
        read_tier_cache(&cache_path, n, dim, tier_tag, src_len, src_mtime)
            .and_then(|body| QuantizedVectors::from_tier_payload(&body, dim, bits, n_usize))
    } else {
        None
    };

    let mut quant = match cached {
        Some(q) => q,
        None => match tier {
            QuantKind::Int8 => QuantizedVectors::build_int8_streaming(n_usize, dim, read_rows)?,
            QuantKind::Pq { m, k } => {
                QuantizedVectors::build_pq_streaming(n_usize, dim, m, k, read_rows)?
            }
            QuantKind::TurboQuant { bits } => {
                QuantizedVectors::build_turboquant_streaming(n_usize, dim, bits, read_rows)?
            }
            QuantKind::F32 | QuantKind::Binary => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "disk tier supports Int8, Pq, or TurboQuant only",
                ));
            }
        },
    };
    // Persist for the next open. Best-effort: a read-only directory or a
    // full disk must not stop the index from opening, it only means the
    // next open pays the rebuild again.
    if matches!(tier, QuantKind::TurboQuant { .. })
        && !cache_path.exists()
        && let Some(body) = quant.tier_payload()
    {
        let _ = write_tier_cache(&cache_path, n, dim, tier_tag, src_len, src_mtime, &body);
    }
    // TurboQuant tier only, opt-in. Persist the codes buffer to
    // `tier.cache.bin` and swap
    // the in-RAM `Vec<u8>` for a `MappedFile`; the OS page cache then
    // decides which pages stay resident. Other tiers (int8, pq) keep
    // their `Vec<u8>` representation - the experiment runs on
    // TurboQuant only.
    if mmap_tier && matches!(tier, QuantKind::TurboQuant { .. }) {
        quant.swap_turboquant_codes_to_mmap(&bdir.join(TIER_CACHE_FILE))?;
    }

    let id_to_main_row: AHashMap<u64, VecId> = ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row as VecId))
        .collect();
    Ok((
        Segment {
            main_n: n,
            nodes,
            ids,
            id_to_main_row,
            medoid,
            quant,
            vectors_file,
            row_cache: new_row_cache(dim),
            entry_cache: new_entry_cache(),
        },
        dim,
        l_search,
    ))
}

/// One-probe direct-mapped row cache: slot = row % capacity, eviction is
/// overwrite. No recency bookkeeping on purpose - the hit pattern this
/// exists for is hub rows recurring across queries, which a direct map
/// captures, and the miss path is exactly one probe over the disk read it
/// was already going to do.
struct RowSlotCache {
    slots: Vec<Option<(VecId, Vec<f32>)>>,
}

impl RowSlotCache {
    fn new(budget_bytes: usize, dim: usize) -> RowSlotCache {
        let per_row = dim * 4 + std::mem::size_of::<Option<(VecId, Vec<f32>)>>();
        let n = (budget_bytes / per_row).max(16);
        RowSlotCache { slots: vec![None; n] }
    }

    fn get(&self, row: VecId) -> Option<Vec<f32>> {
        match &self.slots[row as usize % self.slots.len()] {
            Some((r, v)) if *r == row => Some(v.clone()),
            _ => None,
        }
    }

    fn put(&mut self, row: VecId, v: Vec<f32>) {
        let n = self.slots.len();
        self.slots[row as usize % n] = Some((row, v));
    }
}

/// Per-segment re-rank cache budget. `SKEG_RERANK_CACHE_MB` (default 8) is
/// megabytes per open segment; at dim 1024 the default holds ~2k rows per
/// shard segment. `0` disables.
fn rerank_cache_budget() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SKEG_RERANK_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            * 1024
            * 1024
    })
}

fn new_row_cache(dim: usize) -> Option<std::sync::Mutex<RowSlotCache>> {
    let b = rerank_cache_budget();
    (b > 0).then(|| std::sync::Mutex::new(RowSlotCache::new(b, dim)))
}

/// The snapshot a background consolidate builds from. Produced by
/// [`DiskVamanaIndex::consolidate_begin`] (short, exclusive), consumed by
/// [`ConsolidateJob::build`] on any thread (long, no lock on the index),
/// finished by [`DiskVamanaIndex::consolidate_finish`] (short, exclusive).
/// Owns everything it needs: the surviving vectors, ids, and the WAL offset
/// separating pre-snapshot records (folded) from post-snapshot ones (replayed).
/// The base graph as it stood at `begin`, captured only when the fold will
/// reuse it instead of rebuilding from scratch: the adjacency (a memcpy of the
/// Node array, ~260 B per row) and the old medoid, both in the OLD base row
/// space. Liveness per old row is not captured here; `build` derives it from
/// the survivor list it already holds.
struct PatchBase {
    adj: Vec<Node>,
    medoid: VecId,
}

/// Which route `consolidate_begin` picks for the coming fold.
///
/// The full rebuild costs O(live) greedy searches and is the reason a fold at
/// scale takes minutes; the patched route keeps the surviving base edges and
/// only inserts what is new, so its cost tracks what changed. The full route
/// stays for the regimes where reuse loses, and both bounds are measured, not
/// guessed:
///
/// - an empty base, or new mass above base size: nothing left worth reusing;
/// - a base whose DEAD fraction is high. This one bit in production before it
///   was guarded: a demo workload that had rewritten ~39% of the base ids took
///   the patched route and spent 407-475s per shard, slower than the full
///   rebuild would have been. With R=64 nearly every surviving row touches a
///   dead neighbour once the dead fraction is large (1-(1-f)^64 saturates
///   fast), the verbatim fast path never fires, and every row pays the O(R^2)
///   bridge+re-prune. The delete-patch verdict measured the same cliff: 10,6x
///   at 1% dead, 0,3x at 40%, crossover at 20-25%. The guard sits at 20%,
///   inside the winning region.
///
/// `SKEG_PATCH_FOLD=off|force` overrides for A/B runs.
fn patch_fold_route(new_rows: usize, base_live: usize, base_rows: usize) -> bool {
    match std::env::var("SKEG_PATCH_FOLD").as_deref() {
        Ok("off") => return false,
        Ok("force") => return base_live > 0,
        _ => {}
    }
    route_from_shape(new_rows, base_live, base_rows)
}

/// The route decision itself, env-free so tests can pin the boundary cases.
fn route_from_shape(new_rows: usize, base_live: usize, base_rows: usize) -> bool {
    if base_live == 0 {
        return false;
    }
    let base_dead = base_rows.saturating_sub(base_live);
    // dead/base_rows <= 20%, in integers.
    base_dead * 5 <= base_rows && new_rows <= base_live
}

pub struct ConsolidateJob {
    /// Newest layer: the delta captured directly at `begin` (no flush, no graph
    /// build on the caller). Row-major, paired with `delta_ids`.
    delta_vectors: Vec<f32>,
    delta_ids: Vec<u64>,
    /// Base/run survivors, read OFF-THREAD in `build` from `seg_files`.
    /// `(id, seg_index, row)`; seg_index 0 = base, 1.. = runs.
    survivors: Vec<(u64, usize, u32)>,
    /// Duplicated `vectors.bin` handles [base, run0, run1, ...]. `try_clone`
    /// dups the fd (O(1)); the reads happen off the caller in `build`. A
    /// concurrent consolidate/flush may rename these files, but an open fd keeps
    /// the inode, so the snapshot stays readable.
    seg_files: Vec<File>,
    dim: usize,
    tier: QuantKind,
    run_seq_high: u64,
    wal_offset: u64,
    /// `Some` when this fold will keep the base edges and insert only the new
    /// rows; `None` for the from-scratch rebuild.
    patch: Option<PatchBase>,
}

/// The output of [`ConsolidateJob::build`]: a freshly built base segment,
/// already OPENED off-thread (graph parsed + quant tier built), plus its sidecar
/// dir for the file rename. [`DiskVamanaIndex::consolidate_finish`] swaps the
/// segment in without a reopen, so the O(live) tier rebuild never lands on the
/// shard thread.
pub struct ConsolidateBuilt {
    base: Segment,
    tmp: PathBuf,
    run_seq_high: u64,
    wal_offset: u64,
    /// Which route built this: `true` when the base edges were reused. Read by
    /// the correctness tests so they cannot silently pass against the wrong
    /// route, and worth reporting either way.
    patched: bool,
}

impl ConsolidateJob {
    /// Build the consolidated graph. CPU-heavy; run it on a background thread.
    /// Writes into `<index dir>/consolidating/`, never touching the files the
    /// live index serves from.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing the sidecar files fails.
    pub fn build(self, index_dir: &Path) -> io::Result<ConsolidateBuilt> {
        self.build_with_threads(index_dir, ConsolidatePace::Serving)
    }

    /// Build, saying how much of the machine the rebuild may take.
    ///
    /// The two cases are genuinely different and the engine knows which it is
    /// in. A fold triggered because writes went quiet has nothing to protect:
    /// there is no traffic, and finishing sooner is strictly better. A fold
    /// triggered by churn is running against live queries, and every core it
    /// takes is one they do not get.
    ///
    /// Measured on 223.135 real vectors, searches running throughout:
    ///
    /// ```text
    ///   threads   search p50   search p99   consolidate
    ///      6        72,3 ms     237,0 ms      ~25,8 s
    ///      4        61,8 ms     175,3 ms      ~25,6 s
    ///      2        42,4 ms     103,0 ms      ~31,0 s
    /// ```
    ///
    /// The build barely speeds up with more threads, so the extra ones buy
    /// little and cost the tail a lot. Which is only true while something is
    /// querying: idle, the same threads are free.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure while reading vectors or writing the
    /// rebuilt graph.
    pub fn build_with_threads(
        self,
        index_dir: &Path,
        pace: ConsolidatePace,
    ) -> io::Result<ConsolidateBuilt> {
        let phase_start = std::time::Instant::now();
        let tmp = index_dir.join("consolidating");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        let ConsolidateJob {
            delta_vectors,
            delta_ids,
            survivors,
            seg_files,
            dim,
            tier,
            run_seq_high,
            wal_offset,
            patch,
        } = self;
        // Assemble the survivor set in id order (near-neighbours land at nearby
        // rows so the re-rank stays cache-local). Reads happen HERE, off-thread:
        // delta from RAM, base/run survivors from the duped `vectors.bin` fds.
        enum Src {
            Delta(usize),
            Seg(usize, u32),
        }
        let total = delta_ids.len() + survivors.len();
        let mut items: Vec<(u64, Src)> = Vec::with_capacity(total);
        for (i, &id) in delta_ids.iter().enumerate() {
            items.push((id, Src::Delta(i)));
        }
        for (id, seg, row) in survivors {
            items.push((id, Src::Seg(seg, row)));
        }
        items.sort_unstable_by_key(|&(id, _)| id);
        let mut vectors: Vec<f32> = Vec::with_capacity(total * dim);
        let mut ids: Vec<u64> = Vec::with_capacity(total);
        // Per new row: the OLD base row it came from, or `u32::MAX` for a row
        // that is new to the base (delta or run). This is what lets the patched
        // route tell "keep your edges" from "insert yourself".
        let mut base_origin: Vec<u32> = Vec::with_capacity(total);
        let mut buf = vec![0u8; dim * 4];
        for (id, src) in items {
            match src {
                Src::Delta(i) => {
                    vectors.extend_from_slice(&delta_vectors[i * dim..(i + 1) * dim]);
                    base_origin.push(u32::MAX);
                }
                Src::Seg(seg, row) => {
                    let off = HEADER_LEN as u64 + u64::from(row) * dim as u64 * 4;
                    seg_files[seg].read_exact_at(&mut buf, off)?;
                    vectors.extend(
                        buf.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                    base_origin.push(if seg == 0 { row } else { u32::MAX });
                }
            }
            ids.push(id);
        }
        let t_read = phase_start.elapsed();
        let cfg = disk_build_config();
        // Cap the build's rayon parallelism. The consolidate runs off the request
        // path on a background thread, but `build_disk_graph` fans out over the
        // GLOBAL rayon pool (all cores), so while it runs it starves the
        // foreground - streaming inserts and queries - of every core, collapsing
        // sustained ingest throughput (measured: churn drops ~10x during a fold).
        // Confine it to a bounded pool so the foreground keeps making progress.
        let was_patched = patch.is_some();
        let build = move || match patch {
            // Keep the surviving base edges, insert only what is new. The
            // QuIVer tq1 variant applies only to the from-scratch route: edge
            // repair on the patched route stays on f32, the choice the recall
            // measurements forced (int8 pruning scored 0.31 on real data).
            Some(pb) => build_patched_graph(pb, vectors, ids, &base_origin, dim, &cfg),
            None => build_disk_graph(tier, vectors, ids, dim, &cfg),
        };
        let rebuilt = match consolidate_thread_cap(pace)
            .and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
        {
            Some(pool) => pool.install(build),
            None => build(),
        };
        let t_graph = phase_start.elapsed();
        rebuilt.save(&tmp)?;
        // Open the freshly-saved base HERE (still off-thread): this is where the
        // O(live) quant-tier build happens now - not on the shard thread in
        // finish. The vectors.bin fd survives the finish rename (inode), so the
        // returned segment stays valid after the file moves into place.
        let base = DiskVamanaIndex::open_with_tier(&tmp, tier)?.base;
        // Where a fold's seconds actually go. Without this the only lever anyone
        // can reason about is the graph build, which may not be the biggest part.
        tracing::info!(
            "consolidate phases ({}): read {:?}, graph {:?}, save+tier {:?}, total {:?}",
            if was_patched { "patched" } else { "full" },
            t_read,
            t_graph - t_read,
            phase_start.elapsed() - t_graph,
            phase_start.elapsed(),
        );
        Ok(ConsolidateBuilt {
            base,
            tmp,
            run_seq_high,
            wal_offset,
            patched: was_patched,
        })
    }
}

/// Rayon thread cap for a background consolidate build, or `None` to use every
/// core. Default: a fixed FRACTION of the machine's cores (three quarters),
/// reserving the rest for the foreground (streaming writes and queries) so a
/// background rebuild does not collapse ingest throughput while it runs. Being
/// proportional means it scales with any core count (4 cores -> 3 build / 1
/// foreground; 64 -> 48 / 16) rather than a fixed reserve that starves either
/// side at the extremes. Override the fraction's numerator with
/// `SKEG_CONSOLIDATE_THREADS` (an absolute thread count; 0 or >= parallelism
/// means "all cores"). Machines with 1-2 cores are left uncapped.
/// How much of the machine a rebuild may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsolidatePace {
    /// Something is querying: leave the machine room to answer.
    Serving,
    /// Writes went quiet and this fold was triggered by that quiet. There is
    /// nothing to protect, and a shorter fold is a shorter window in which
    /// traffic could return and find the machine busy.
    Idle,
}

fn consolidate_thread_cap(pace: ConsolidatePace) -> Option<usize> {
    /// Cores a rebuild takes while queries are being served, out of every 4.
    /// Low on purpose: the build scales poorly with threads, so the ones above
    /// this buy little build speed and cost the query tail a lot.
    const SERVING_NUM: usize = 1;
    const SERVING_DEN: usize = 4;
    let avail = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    // The override, when set, applies to the serving case: it exists to protect
    // queries, and there are none to protect when idle.
    let n = match pace {
        ConsolidatePace::Idle => avail,
        ConsolidatePace::Serving => std::env::var("SKEG_CONSOLIDATE_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| (avail * SERVING_NUM / SERVING_DEN).max(1)),
    };
    (n >= 1 && n < avail).then_some(n)
}

/// Snapshot for a background runs-only merge. Holds the live vectors folded out
/// of the front `n_merged` runs; an off-thread [`RunMergeJob::build`] turns them
/// into one replacement run. Unlike [`ConsolidateJob`] it never touches the base
/// or the WAL: runs are immutable, so the merge is a read-only fold whose result
/// the search precedence (delta > newer runs > this merged run > base) slots in
/// correctly. Cheap (O(runs), not O(live-set)), so it can run often enough to
/// keep the run count bounded under fast churn.
pub struct RunMergeJob {
    /// (seg_index, row) survivor locations, id-sorted; read OFF-THREAD in build
    /// from the duped fds. seg_index parallels `seg_files`.
    survivors: Vec<(usize, u32)>,
    ids: Vec<u64>,
    /// Duped vectors.bin fds for the folded runs (O(1) each at begin).
    seg_files: Vec<File>,
    dim: usize,
    tier: QuantKind,
    merged_seq: u64,
    n_merged: usize,
    old_dirs: Vec<u64>,
}

/// Output of [`RunMergeJob::build`]: the merged run already OPENED off-thread
/// (graph + quant tier built), ready for [`DiskVamanaIndex::merge_runs_finish`]
/// to splice in without a reopen on the shard thread.
pub struct RunMergeBuilt {
    merged: Segment,
    merged_seq: u64,
    n_merged: usize,
    old_dirs: Vec<u64>,
}

impl RunMergeJob {
    /// Build the merged run graph. CPU-bounded but O(runs), far cheaper than a
    /// full consolidate; run it on a background thread. Writes `run-<seq>` under
    /// the index dir, never touching the files the live index serves from.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing the run fails.
    pub fn build(self, index_dir: &Path) -> io::Result<RunMergeBuilt> {
        let RunMergeJob {
            survivors,
            ids,
            seg_files,
            dim,
            tier,
            merged_seq,
            n_merged,
            old_dirs,
        } = self;
        let dir = index_dir.join(format!("run-{merged_seq}"));
        let _ = std::fs::remove_dir_all(&dir);
        // Read survivor vectors OFF-THREAD from the duped run fds.
        let mut vectors: Vec<f32> = Vec::with_capacity(survivors.len() * dim);
        let mut buf = vec![0u8; dim * 4];
        for (seg, row) in survivors {
            let off = HEADER_LEN as u64 + u64::from(row) * dim as u64 * 4;
            seg_files[seg].read_exact_at(&mut buf, off)?;
            vectors.extend(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        let cfg = disk_build_config();
        // Always cautious: these two are O(runs) and short, and the window
        // they would open by taking the machine is not worth the seconds saved.
        let rebuilt = match consolidate_thread_cap(ConsolidatePace::Serving)
            .and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
        {
            Some(pool) => pool.install(move || build_disk_graph(tier, vectors, ids, dim, &cfg)),
            None => build_disk_graph(tier, vectors, ids, dim, &cfg),
        };
        rebuilt.save(&dir)?;
        // Open the merged run HERE (off-thread): the quant-tier build for the
        // merged run happens now, not on the shard thread in finish.
        let merged = DiskVamanaIndex::open_with_tier(&dir, tier)?.base;
        Ok(RunMergeBuilt {
            merged,
            merged_seq,
            n_merged,
            old_dirs,
        })
    }
}

/// Snapshot for an OFF-THREAD flush of the delta into a run. `flush_begin` moves
/// the delta aside (into the index's `flushing` staging buffer, still searched)
/// and hands the vectors here; `build` builds the run graph + opens it, all off
/// the caller; `flush_finish` splices the run and clears the staging. The delta
/// batch never blocks the caller on a graph build.
pub struct FlushJob {
    vectors: Vec<f32>,
    ids: Vec<u64>,
    dim: usize,
    tier: QuantKind,
    seq: u64,
}

/// Output of [`FlushJob::build`]: the run already opened off-thread, ready for
/// [`DiskVamanaIndex::flush_finish`] to splice in.
pub struct FlushBuilt {
    run: Segment,
    seq: u64,
}

/// Snapshot for an OFF-THREAD IVF-router rebuild. `ivf_begin` dups the base
/// vectors fd (O(1)); `build` reads them and runs k-means off the caller;
/// `ivf_finish` swaps the router in under a short lock. The k-means (O(live))
/// no longer runs on the shard thread.
pub struct IvfJob {
    seg_file: File,
    n: u32,
    dim: usize,
    n_cells: usize,
    iters: usize,
}

/// Output of [`IvfJob::build`]: the built router, ready for
/// [`DiskVamanaIndex::ivf_finish`] to install.
pub struct IvfBuilt {
    router: IvfRouter,
}

impl IvfJob {
    /// Read the base vectors and run k-means - off the caller.
    ///
    /// # Errors
    ///
    /// I/O error if a base-vector read fails.
    pub fn build(self) -> io::Result<IvfBuilt> {
        let IvfJob {
            seg_file,
            n,
            dim,
            n_cells,
            iters,
        } = self;
        let all = read_f32_rows_sequential(&seg_file, 0, n as usize, dim)?;
        let n_cells = if n_cells == 0 {
            IvfRouter::cells_for(n as usize)
        } else {
            n_cells
        };
        let router = IvfRouter::build(&all, n, dim, n_cells, iters);
        Ok(IvfBuilt { router })
    }
}

impl FlushJob {
    /// Build the run graph and open it - off the caller's thread.
    ///
    /// # Errors
    ///
    /// I/O error if writing or opening the run fails.
    pub fn build(self, index_dir: &Path) -> io::Result<FlushBuilt> {
        let FlushJob {
            vectors,
            ids,
            dim,
            tier,
            seq,
        } = self;
        let dir = index_dir.join(format!("run-{seq}"));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = disk_build_config();
        // Always cautious: these two are O(runs) and short, and the window
        // they would open by taking the machine is not worth the seconds saved.
        let rebuilt = match consolidate_thread_cap(ConsolidatePace::Serving)
            .and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
        {
            Some(pool) => pool.install(move || build_disk_graph(tier, vectors, ids, dim, &cfg)),
            None => build_disk_graph(tier, vectors, ids, dim, &cfg),
        };
        rebuilt.save(&dir)?;
        let run = DiskVamanaIndex::open_with_tier(&dir, tier)?.base;
        Ok(FlushBuilt { run, seq })
    }
}

/// Snapshot for a background delete-patch of the base graph. Removes the base
/// rows that are dead at `begin` (tombstoned, or shadowed by a newer copy in a
/// run) by re-pruning only the nodes that pointed at a removed node - the
/// FreshDiskANN lazy-deletion rule - instead of rebuilding the whole graph
/// from scratch. Cost is O(#nodes touching a deleted node), not O(live-set), so
/// it reclaims dead base rows far cheaper than a full consolidate. Runs are left
/// intact: the insert side is handled by a run-merge / flush, not here.
pub struct DeletePatchJob {
    /// Duped base `vectors.bin` fd: the O(live) row reads happen OFF-THREAD in
    /// `build`, not on the caller. `n_rows` rows, row-major.
    seg_file: File,
    n_rows: usize,
    /// Base id by old row.
    ids: Vec<u64>,
    /// Base adjacency by old row (owned copy of the served graph).
    adj: Vec<Node>,
    /// old row -> removed at this patch.
    dead: Vec<bool>,
    /// Old-row medoid.
    medoid: VecId,
    dim: usize,
    l_search: usize,
    tier: QuantKind,
}

/// Output of [`DeletePatchJob::build`]: the patched base graph in a sidecar
/// dir, ready for [`DiskVamanaIndex::delete_patch_finish`] to swap in.
pub struct DeletePatchBuilt {
    base: Segment,
    tmp: PathBuf,
}

impl DeletePatchJob {
    /// Apply the delete-patch. CPU-bounded but O(affected), far cheaper than a
    /// full consolidate's greedy rebuild; run it on a background thread. Writes
    /// into `<index dir>/patching/`, never touching the served files.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing the sidecar files fails.
    pub fn build(self, index_dir: &Path) -> io::Result<DeletePatchBuilt> {
        let tmp = index_dir.join("patching");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        let DeletePatchJob {
            seg_file,
            n_rows,
            ids,
            adj,
            dead,
            medoid,
            dim,
            l_search,
            tier,
        } = self;
        // O(live) vector read, OFF-THREAD (was on the shard thread in begin).
        let vectors = read_f32_rows_sequential(&seg_file, 0, n_rows, dim)?;
        let cfg = disk_build_config();
        // Same rayon cap as the consolidate: keep a background patch from
        // starving the foreground of every core while it runs.
        let patched = match consolidate_thread_cap(ConsolidatePace::Serving)
            .and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok())
        {
            Some(pool) => {
                pool.install(|| patch_graph(vectors, ids, adj, &dead, medoid, dim, l_search, &cfg))
            }
            None => patch_graph(vectors, ids, adj, &dead, medoid, dim, l_search, &cfg),
        };
        patched.save(&tmp)?;
        // Open the patched base HERE (off-thread): the O(live) tier rebuild
        // happens off the shard thread, not in finish.
        let base = DiskVamanaIndex::open_with_tier(&tmp, tier)?.base;
        Ok(DeletePatchBuilt { base, tmp })
    }
}

/// FreshDiskANN lazy-delete patch: drop the `dead` rows and re-wire only the
/// survivors that pointed at one. A node whose out-list held no dead neighbour
/// keeps its edges verbatim (just row-remapped); a node that lost a neighbour
/// gets a fresh candidate set - its live neighbours plus the live neighbours of
/// its dead ones (bridging the gap) - and a single `robust_prune` back down to
/// `R`. No greedy search, so the cost is the pruning of affected nodes only.
fn patch_graph(
    vectors: Vec<f32>,
    ids: Vec<u64>,
    adj: Vec<Node>,
    dead: &[bool],
    old_medoid: VecId,
    dim: usize,
    l_search: usize,
    cfg: &VamanaConfig,
) -> VamanaIndex {
    let n_old = ids.len();
    // old row -> new row (u32::MAX = removed).
    let mut remap = vec![u32::MAX; n_old];
    let mut new_ids: Vec<u64> = Vec::new();
    let mut new_vectors: Vec<f32> = Vec::new();
    let mut survivors: Vec<u32> = Vec::new();
    for row in 0..n_old {
        if !dead[row] {
            remap[row] = new_ids.len() as u32;
            new_ids.push(ids[row]);
            new_vectors.extend_from_slice(&vectors[row * dim..(row + 1) * dim]);
            survivors.push(row as u32);
        }
    }
    let n_new = new_ids.len();
    assert!(n_new > 0, "delete-patch left no survivors");
    // Distances during pruning are measured over the OLD row space (candidates
    // are old rows), so prune against a source indexed by old row.
    let old_src = InMemoryVectorSource::new(vectors, dim);
    let new_nodes: Vec<Node> = survivors
        .par_iter()
        .map(|&p| {
            let out = adj[p as usize].slice();
            let touches_dead = out.iter().any(|&w| dead[w as usize]);
            let mut node = Node::new();
            if !touches_dead {
                // Fast path: edges intact, just remap to new rows.
                let kept: SmallVec<[VecId; MAX_R]> =
                    out.iter().map(|&w| remap[w as usize]).collect();
                node.set(&kept);
                return node;
            }
            // Bridge over dead neighbours, then prune back to R.
            let mut cand: AHashSet<VecId> = AHashSet::new();
            for &w in out {
                if !dead[w as usize] {
                    cand.insert(w);
                } else {
                    for &x in adj[w as usize].slice() {
                        if x != p && !dead[x as usize] {
                            cand.insert(x);
                        }
                    }
                }
            }
            let pv = old_src.row(p);
            let mut scored: Vec<(f32, VecId)> = cand
                .iter()
                .map(|&c| (dist(pv, old_src.row(c)), c))
                .collect();
            let picked = robust_prune(p, &mut scored, cfg.alpha2, cfg.r, &old_src);
            let remapped: SmallVec<[VecId; MAX_R]> =
                picked.iter().map(|&w| remap[w as usize]).collect();
            node.set(&remapped);
            node
        })
        .collect();
    drop(old_src);

    let new_src = InMemoryVectorSource::new(new_vectors, dim);
    let medoid = if old_medoid != u32::MAX && !dead[old_medoid as usize] {
        remap[old_medoid as usize]
    } else {
        approximate_medoid(&new_src, n_new as u32, cfg.medoid_sample, cfg.seed)
    };
    VamanaIndex {
        dim,
        n: n_new as u32,
        vectors: Box::new(new_src),
        ids: new_ids,
        nodes: new_nodes,
        medoid,
        r: cfg.r,
        l_search,
    }
}

pub struct DiskVamanaIndex {
    dim: usize,
    l_search: usize,
    /// The immutable base segment. Future LSM runs join it as more segments.
    base: Segment,
    /// Additional immutable LSM runs, searched alongside `base`. Streaming
    /// writes flush from the delta into runs; `consolidate` folds them back.
    runs: Vec<Segment>,
    /// The dir seq (`run-<seq>`) backing each entry of `runs`, kept parallel so
    /// a runs-only merge can delete exactly the dirs it folded. Flush appends,
    /// the merge rewrites the front, discard/consolidate clears.
    run_dirs: Vec<u64>,
    /// Tier-1 quantiser, kept so a `flush` builds runs with the same tier as
    /// the base (and so `consolidate` reopens with it).
    tier: QuantKind,
    /// Monotonic run-directory counter, so flushed run dirs never collide.
    run_seq: u64,
    dir: PathBuf,
    /// Streaming inserts since open / last consolidation: external id -> f32.
    delta: AHashMap<u64, Vec<f32>>,
    /// Staging buffer for an in-flight OFF-THREAD flush: the delta entries moved
    /// aside by `flush_begin` while their run is built off-thread. Searched with
    /// precedence `delta > flushing > runs > base`, so the batch stays visible
    /// during the build; `flush_finish` clears it once the run is spliced in.
    /// Empty except during an off-thread flush.
    flushing: AHashMap<u64, Vec<f32>>,
    /// When true (default), `insert` flushes the delta into a run inline once it
    /// fills. A server turns this off and flushes off-thread instead.
    auto_flush: bool,
    /// Tombstoned external ids (covers both main and delta).
    tombstones: AHashSet<u64>,
    live_count: usize,
    /// Append-only log of delta mutations, replayed on `open` so streaming
    /// inserts/deletes survive a restart. `consolidate` truncates it.
    delta_log: File,
    /// Encoding used by `delta_log`.
    wal_format: DeltaWalFormat,
    /// Online tq1 proxy controller, `None` unless enabled via
    /// [`enable_tq1_controller`](Self::enable_tq1_controller). Behind a mutex
    /// because `search` is `&self`; only touched on shadow queries.
    // Box the whole tq1 runtime (mutex + counter) so this cold state
    // stays off DiskVamanaIndex as one pointer - else the pthread mutex inline
    // bloats skeg-server's VectorBackend enum past large_enum_variant.
    tq1: Box<Tq1Runtime>,
    /// Coarse IVF router: the "cells" branch of hybrid filtered search (sparse /
    /// medium filters). `None` until built via [`build_ivf`](Self::build_ivf).
    /// Boxed to keep DiskVamanaIndex small (VectorBackend large_enum_variant).
    ivf: Option<Box<IvfRouter>>,
    /// Optional u64 attribute per BASE row (index = base row), for range-filtered
    /// search ([`search_range`](Self::search_range)). `None` unless
    /// [`set_attr`](Self::set_attr) was called. Covers base rows only; streaming
    /// (delta/run) inserts are not range-filterable until the next consolidate.
    attr: Option<Vec<u64>>,
}

/// Cold per-index tq1 online-controller state, kept behind a single `Box` on
/// `DiskVamanaIndex`. `ctl` is `None` unless
/// [`enable_tq1_controller`](DiskVamanaIndex::enable_tq1_controller) is called.
#[derive(Default)]
struct Tq1Runtime {
    ctl: std::sync::Mutex<Option<Tq1ProxyController>>,
    ctr: std::sync::atomic::AtomicU64,
}

impl DiskVamanaIndex {
    /// Open an index previously written by [`VamanaIndex::save`]. The graph is
    /// loaded into RAM; `vectors.bin` is streamed once to build the int8
    /// tier-1 quantisation, then left on disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the files are missing, truncated, or carry a
    /// bad magic/version.
    ///
    /// # Panics
    ///
    /// Panics if the two files disagree on `n` or `dim`.
    #[allow(clippy::cast_possible_truncation)] // row < n, and n was read as a u32
    pub fn open(dir: &Path) -> io::Result<DiskVamanaIndex> {
        Self::open_with_tier(dir, read_tier(dir))
    }

    /// Like [`open`](Self::open) but with an explicit tier-1 quantisation:
    /// `QuantKind::Int8` (default) or `QuantKind::Pq { m, k }`. The tier is
    /// rebuilt from `vectors.bin` at open and is deterministic, so no codebook
    /// is persisted on disk.
    ///
    /// # Errors
    ///
    /// I/O errors as [`open`](Self::open); rejects `F32` and `Binary` (the
    /// disk-graph walk needs an int8 or PQ proxy).
    ///
    /// # Panics
    ///
    /// Panics if the two files disagree on `n` or `dim`.
    #[allow(clippy::cast_possible_truncation)] // row < n, and n was read as a u32
    pub fn open_with_tier(dir: &Path, tier: QuantKind) -> io::Result<DiskVamanaIndex> {
        Self::open_with_tier_mmap(dir, tier, false)
    }

    /// Like [`open_with_tier`](Self::open_with_tier) but with an opt-in
    /// memory-mapped TurboQuant tier. With `mmap_tier == true` the tier
    /// codes are persisted to `tier.cache.bin` after build and the in-RAM
    /// `Vec<u8>` is replaced by a memory-mapped view of that file: the OS
    /// page cache decides which pages stay resident, freeing anonymous
    /// memory under pressure. `int8` and `pq` tiers are unaffected by
    /// this flag for now; the experiment runs on TurboQuant only.
    ///
    /// # Errors
    ///
    /// As [`open_with_tier`](Self::open_with_tier), plus any I/O error from
    /// the optional `tier.cache.bin` write/mmap.
    ///
    /// # Panics
    ///
    /// Panics if the two files disagree on `n` or `dim`.
    pub fn open_with_tier_mmap(
        dir: &Path,
        tier: QuantKind,
        mmap_tier: bool,
    ) -> io::Result<DiskVamanaIndex> {
        Self::open_with_tier_full(dir, tier, mmap_tier, false)
    }

    /// Like [`open_with_tier_mmap`](Self::open_with_tier_mmap) plus an
    /// opt-in `mmap_graph` flag: open `graph.vmn` as a `MappedFile` and
    /// reinterpret the Node region as `&[Node]` (Node is `#[repr(C)] + Pod`,
    /// file layout = in-memory layout). The OS page cache can then reclaim
    /// graph pages under memory pressure - the same property as
    /// `mmap_tier`, extended to the larger graph buffer (~260 MB at 1M
    /// dim=1024 vs ~26 MB tier).
    ///
    /// Combined with `mmap_tier`, the whole index becomes paginable under
    /// pressure with zero penalty in steady state.
    ///
    /// # Errors
    ///
    /// As [`open_with_tier`](Self::open_with_tier), plus I/O errors from
    /// any of the optional mmap paths.
    ///
    /// # Panics
    ///
    /// Panics if the two files disagree on `n` or `dim`.
    #[allow(clippy::cast_possible_truncation)] // row < n, and n was read as a u32
    pub fn open_with_tier_full(
        dir: &Path,
        tier: QuantKind,
        mmap_tier: bool,
        mmap_graph: bool,
    ) -> io::Result<DiskVamanaIndex> {
        let (base, dim, l_search) = open_segment(dir, tier, mmap_tier, mmap_graph)?;


        // Reopen the durable runs. A run directory with a `run.ok` marker was
        // fully written and fsynced before the WAL was compacted past it: it
        // reopens as a run. One without the marker predates its own
        // `flush_finish` (or predates markers entirely) - the WAL still
        // covers its rows, so it is deleted and its contents come back
        // through the replay below. This replaces the old behaviour of
        // deleting EVERY run and replaying everything-since-the-last-fold
        // into the delta: measured on the demo, a restart put 218k rows
        // (900 MB) back into RAM as a flat-scanned delta.
        let mut run_seqs: Vec<u64> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(seq) = name.strip_prefix("run-").and_then(|s| s.parse::<u64>().ok()) {
                    if entry.path().join(RUN_OK_FILE).exists() {
                        run_seqs.push(seq);
                    } else {
                        let _ = std::fs::remove_dir_all(entry.path()); // WAL-covered
                    }
                }
            }
        }
        run_seqs.sort_unstable();
        let mut runs: Vec<Segment> = Vec::with_capacity(run_seqs.len());
        for &seq in &run_seqs {
            let (seg, rdim, _) =
                open_segment(&dir.join(format!("run-{seq}")), tier, false, false)?;
            if rdim != dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("run-{seq} dim {rdim} != base dim {dim}"),
                ));
            }
            runs.push(seg);
        }
        let run_seq_next = run_seqs.last().map_or(0, |&s| s + 1);
        // Live ids = the union across base and runs (a run row can shadow a
        // base row; it is still one live id).
        let mut live_ids: AHashSet<u64> = base.id_to_main_row.keys().copied().collect();
        for run in &runs {
            live_ids.extend(run.id_to_main_row.keys().copied());
        }
        let live_count = live_ids.len();
        drop(live_ids);

        // Replay the delta WAL: streaming inserts/deletes since the last
        // consolidation that have not yet been folded into the graph.
        let log_path = dir.join(DELTA_LOG_FILE);
        let wal = std::fs::read(&log_path).unwrap_or_default();
        let (wal_format, wal_ops) = decode_wal(&wal, dim)?;
        let delta_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let mut index = DiskVamanaIndex {
            dim,
            l_search,
            base,
            runs,
            run_dirs: run_seqs,
            tier,
            run_seq: run_seq_next,
            dir: dir.to_path_buf(),
            delta: AHashMap::new(),
            flushing: AHashMap::new(),
            auto_flush: true,
            tombstones: AHashSet::new(),
            live_count,
            delta_log,
            wal_format,
            tq1: Box::default(),
            ivf: None,
            attr: None,
        };
        index.replay_wal_ops(wal_ops);
        index.load_attr();
        index.load_ivf(); // rebuilds the zone-map if attr is present
        Ok(index)
    }

    /// Apply decoded WAL operations to the in-memory delta state.
    fn replay_wal_ops(&mut self, ops: Vec<DeltaWalOp>) {
        for op in ops {
            match op {
                DeltaWalOp::Insert { id, vector } => self.apply_insert(id, vector),
                DeltaWalOp::Delete { id } => {
                    self.apply_delete(id);
                }
            }
        }
    }

    /// Apply an insert to the in-RAM delta (no WAL write).
    fn apply_insert(&mut self, id: u64, vector: Vec<f32>) {
        let was_live = self.is_live(id);
        self.tombstones.remove(&id);
        self.delta.insert(id, vector);
        if !was_live {
            self.live_count += 1;
        }
    }

    /// Apply a delete to the in-RAM state (no WAL write). Returns prior liveness.
    fn apply_delete(&mut self, id: u64) -> bool {
        let was_live = self.is_live(id);
        self.delta.remove(&id);
        // Drop from the flush staging too, so a deleted id in an in-flight flush
        // is not flat-scanned as live (the run copy is tombstone-masked).
        self.flushing.remove(&id);
        self.tombstones.insert(id);
        if was_live {
            self.live_count -= 1;
        }
        was_live
    }

    /// Create an empty on-disk index: an empty graph plus an empty vectors
    /// file. Every insert lands in the delta until the first
    /// [`consolidate`](Self::consolidate) folds it into a real graph.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory or files cannot be written.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`.
    pub fn create_empty(dir: &Path, dim: usize, l_search: usize) -> io::Result<DiskVamanaIndex> {
        Self::create_empty_with_tier(dir, dim, l_search, QuantKind::Int8)
    }

    /// Like [`create_empty`](Self::create_empty) but pins the RW tier-1 quantiser
    /// (persisted in `tier.kind`, so every later `open`/`consolidate` rebuilds it).
    /// `Int8` (default) or `TurboQuant { bits }` for sub-int8 RAM on live writes;
    /// `Pq` is rejected here (it needs a trained codebook, so it stays serve-only).
    ///
    /// # Errors
    ///
    /// I/O errors writing the initial files.
    pub fn create_empty_with_tier(
        dir: &Path,
        dim: usize,
        l_search: usize,
        tier: QuantKind,
    ) -> io::Result<DiskVamanaIndex> {
        assert!(dim > 0, "dim must be positive");
        assert!(
            matches!(tier, QuantKind::Int8 | QuantKind::TurboQuant { .. }),
            "RW disk tier must be int8 or turboquant; pq/f32/binary are not incrementally rebuildable here"
        );
        std::fs::create_dir_all(dir)?;
        write_tier(dir, tier)?;
        // The base starts in generation slot g0; CURRENT points at it. A
        // consolidate later builds g1 and flips the pointer atomically.
        let g0 = dir.join("g0");
        std::fs::create_dir_all(&g0)?;
        write_graph_vmn(&g0.join(GRAPH_FILE), 0, dim, 0, MAX_R, l_search, &[], &[])?;
        write_vectors_bin(
            &g0.join(VECTORS_FILE),
            &InMemoryVectorSource::new(Vec::new(), dim),
        )?;
        set_current_slot(dir, 0)?;
        write_framed_wal(&dir.join(DELTA_LOG_FILE), &[])?;
        DiskVamanaIndex::open(dir)
    }

    /// Vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of live vectors (main + delta, minus tombstones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// True if the index holds no live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Number of vectors in the in-RAM delta buffer (consolidation trigger).
    #[must_use]
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Number of vectors in the consolidated main graph.
    #[must_use]
    pub fn main_len(&self) -> usize {
        self.base.main_n as usize
    }

    /// Number of flushed LSM runs currently searched alongside the base.
    /// Each run adds a (shallow) graph walk per query, so a growing count is
    /// the signal that a consolidate is due.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Total rows across all flushed runs. With the off-thread flush the delta
    /// stays small (flushed at `FLUSH`), so this - not `delta_len` - is what a
    /// server watches to decide when a full consolidate (fold runs into base +
    /// truncate the WAL) is due.
    #[must_use]
    pub fn run_rows(&self) -> usize {
        self.runs.iter().map(|r| r.main_n as usize).sum()
    }

    /// Live tombstones (deleted ids not yet reclaimed). A cheap gate for the
    /// delete-patch trigger: `tombstone_count() / main_len()` approximates the
    /// dead fraction of the base without the O(base) scan `delete_patch_begin`
    /// does. Over-counts slightly (it also covers tombstoned run/delta ids), so
    /// it is a trigger heuristic, not an exact base-dead count.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Graph entry point (the approximate medoid). Used by an external walk
    /// that drives the graph with its own proxy distance (the PQ-tier gate).
    #[must_use]
    pub fn medoid(&self) -> VecId {
        self.base.medoid
    }

    /// Out-edges of node `id`. Used by an external walk that drives the graph
    /// with its own proxy distance (the PQ-tier gate).
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range.
    #[must_use]
    pub fn neighbors(&self, id: VecId) -> &[VecId] {
        self.base.nodes[id as usize].slice()
    }

    /// Bytes held in RAM: graph + ids + int8 tier + the (small) f32 delta.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let delta_bytes: usize = self.delta.values().map(|v| v.len() * 4 + 24).sum();
        let seg_bytes = |s: &Segment| {
            s.nodes.len() * std::mem::size_of::<Node>()
                + s.ids.len() * std::mem::size_of::<u64>()
                + s.id_to_main_row.len() * 16
                + s.quant.memory_bytes()
        };
        seg_bytes(&self.base)
            + self.runs.iter().map(seg_bytes).sum::<usize>()
            + delta_bytes
            + self.tombstones.len() * 8
    }

    /// True if `id` currently resolves to a live vector.
    fn is_live(&self, id: u64) -> bool {
        !self.tombstones.contains(&id)
            && (self.delta.contains_key(&id)
                || self.flushing.contains_key(&id)
                || self.base.id_to_main_row.contains_key(&id)
                || self.runs.iter().any(|r| r.id_to_main_row.contains_key(&id)))
    }

    /// True if `id` is a live (non-tombstoned) vector in this index. Cheap,
    /// in-memory; used by the server's per-tenant vector quota to tell an
    /// insert from an overwrite without touching disk.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.is_live(id)
    }

    /// Every live (non-tombstoned) vector id: main ids plus streaming-delta
    /// ids, minus tombstones. Used to reclaim per-id sidecar state (e.g.
    /// payload blobs) when the whole index is dropped. In-memory, no disk read.
    #[must_use]
    pub fn live_ids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .base
            .ids
            .iter()
            .copied()
            .chain(self.runs.iter().flat_map(|r| r.ids.iter().copied()))
            .chain(self.delta.keys().copied())
            .chain(self.flushing.keys().copied())
            .filter(|id| !self.tombstones.contains(id))
            .collect();
        // A delta overwrite of a main id appears in both sources; dedup.
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Exact top-`k` `(id, cosine)` over just the candidate `ids`, the
    /// brute-force path a filtered search takes once a predicate has narrowed
    /// the corpus. Full-precision f32 cosine (one disk read per main-resident
    /// id), so the result is exact. Non-live or unknown ids are skipped.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading a stored vector fails.
    pub fn score_ids(&self, query: &[f32], ids: &[u64], k: usize) -> io::Result<Vec<(u64, f32)>> {
        let mut scored: Vec<(u64, f32)> = Vec::new();
        for &id in ids {
            if let Some(v) = self.get(id)? {
                scored.push((id, cosine_f32(query, &v)));
            }
        }
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        Ok(scored)
    }

    /// Score an explicit id set with the in-RAM quantized proxy, then f32-rerank
    /// the top `rerank` survivors. Unlike [`score_ids`](Self::score_ids) (exact
    /// f32 scan = one disk read per id) this reads only `rerank` vectors from
    /// disk regardless of `|ids|`, so it scales to large matching sets (broad
    /// filters): the proxy scan is in-RAM and NEON-fast, the disk cost is bounded.
    /// Recall is the proxy's ranking into the rerank window - the same model as
    /// the ANN walk, without the navigation. `rerank` is the disk-read budget
    /// (e.g. `k*8`).
    ///
    /// # Errors
    ///
    /// I/O error if a re-rank read from `vectors.bin` fails.
    pub fn score_ids_quantized(
        &self,
        query: &[f32],
        ids: &[u64],
        k: usize,
        rerank: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::VsearchHybrid);
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchHybridScored,
            ids.len() as u64,
        );
        let phase_t0 = Instant::now();
        // Per-segment query code; proxy_rescore gives the asym-quality ordering.
        let base_code = self.base.quant.quantize_query(query);
        let run_codes: Vec<_> = self
            .runs
            .iter()
            .map(|r| r.quant.quantize_query(query))
            .collect();
        // (proxy score, seg_idx: 0=base / 1.. = run, row). Delta ids are f32 in
        // RAM already, so they go straight to the finalist list.
        let mut cand: Vec<(i32, usize, VecId)> = Vec::new();
        let mut scored: Vec<(f32, u64)> = Vec::new();
        // Fast path for the folded steady state (no tombstones, no delta, no
        // runs - exactly where a big filtered scan lands after consolidate):
        // one hash lookup per id instead of four. The id set comes from
        // Filter::evaluate sorted and unique, so the dedup set is only needed
        // when newer locations could shadow (measured 212ns/id on the demo
        // storm against a 40ns scoring kernel - the loop, not the math).
        if self.tombstones.is_empty() && self.delta.is_empty() && self.runs.is_empty() {
            cand.reserve(ids.len());
            for &id in ids {
                if let Some(&row) = self.base.id_to_main_row.get(&id) {
                    let p = self.base.quant.proxy_rescore(row as usize, &base_code);
                    cand.push((p, 0, row));
                }
            }
        } else {
            let mut seen: AHashSet<u64> = AHashSet::new();
            for &id in ids {
                if self.tombstones.contains(&id) || !seen.insert(id) {
                    continue;
                }
                if let Some(v) = self.delta.get(&id) {
                    scored.push((cosine_f32(query, v), id));
                    continue;
                }
                // Newest run wins, then base (consolidate precedence).
                if let Some((ri, &row)) = self
                    .runs
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(ri, r)| r.id_to_main_row.get(&id).map(|row| (ri, row)))
                {
                    let p = self.runs[ri]
                        .quant
                        .proxy_rescore(row as usize, &run_codes[ri]);
                    cand.push((p, ri + 1, row));
                } else if let Some(&row) = self.base.id_to_main_row.get(&id) {
                    let p = self.base.quant.proxy_rescore(row as usize, &base_code);
                    cand.push((p, 0, row));
                }
            }
        }
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchHybridScoreNanos,
            phase_t0.elapsed().as_nanos() as u64,
        );
        let phase_t0 = Instant::now();
        // Keep the top `rerank` by proxy (higher = closer), then f32-rerank them.
        let take = rerank.max(k);
        if cand.len() > take {
            cand.select_nth_unstable_by(take, |a, b| b.0.cmp(&a.0));
            cand.truncate(take);
        }
        let rerank_rows = cand.len() as u64;
        for (_p, seg_idx, row) in cand {
            let seg = if seg_idx == 0 {
                &self.base
            } else {
                &self.runs[seg_idx - 1]
            };
            let v = self.read_vector(seg, row)?;
            scored.push((cosine_f32(query, &v), seg.ids[row as usize]));
        }
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchHybridRerankNanos,
            phase_t0.elapsed().as_nanos() as u64,
        );
        skeg_telemetry::add_counter(skeg_telemetry::Counter::VsearchHybridReads, rerank_rows);
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        Ok(scored.into_iter().map(|(s, id)| (id, s)).collect())
    }

    /// Insert or overwrite the vector for `id`. The vector lands in the in-RAM
    /// delta and is appended to the WAL; [`consolidate`](Self::consolidate)
    /// folds it into the graph.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the WAL append fails, or `InvalidInput` if
    /// `vector.len()` does not equal the index dimension. It returns rather
    /// than panics because the caller is a server thread holding other
    /// vindexes: a bad dimension from one client must not take them down.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> io::Result<()> {
        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector has {} dims, index has {}", vector.len(), self.dim),
            ));
        }
        let op = DeltaWalOp::Insert {
            id,
            vector: vector.to_vec(),
        };
        let rec = encode_wal_record(self.wal_format, &op);
        self.delta_log.write_all(&rec)?;
        self.replay_wal_ops(vec![op]);
        // Keep the brute-forced L0 small: once it fills, fold it into a navigable
        // run so search stays sub-linear. INLINE (synchronous) - fine for direct
        // use (benches, bulk load). A server sets `auto_flush(false)` and drives
        // the flush OFF-THREAD from its maintenance loop, so ingest never blocks
        // the request thread on a run build.
        if self.auto_flush && self.delta.len() >= Self::FLUSH {
            self.flush()?;
        }
        Ok(())
    }

    /// Turn the inline auto-flush on (default) or off. With it off, the delta
    /// grows unbounded until the caller drives [`flush_begin`](Self::flush_begin)
    /// / [`flush_finish`](Self::flush_finish) off-thread. Meant for the server.
    pub fn set_auto_flush(&mut self, on: bool) {
        self.auto_flush = on;
    }

    /// Tombstone `id`. Returns `true` if it was live.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the WAL append fails.
    pub fn delete(&mut self, id: u64) -> io::Result<bool> {
        let op = DeltaWalOp::Delete { id };
        let rec = encode_wal_record(self.wal_format, &op);
        self.delta_log.write_all(&rec)?;
        Ok(self.apply_delete(id))
    }

    /// The current f32 vector for `id`, or `None` if absent/tombstoned.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a main-vector read from disk fails.
    pub fn get(&self, id: u64) -> io::Result<Option<Vec<f32>>> {
        if self.tombstones.contains(&id) {
            return Ok(None);
        }
        if let Some(v) = self.delta.get(&id) {
            return Ok(Some(v.clone()));
        }
        // Staged for an in-flight flush (delta > flushing).
        if let Some(v) = self.flushing.get(&id) {
            return Ok(Some(v.clone()));
        }
        if let Some(&row) = self.base.id_to_main_row.get(&id) {
            return Ok(Some(self.read_vector(&self.base, row)?));
        }
        // Newest run wins on a shadowed id; runs are searched after the base.
        for run in &self.runs {
            if let Some(&row) = run.id_to_main_row.get(&id) {
                return Ok(Some(self.read_vector(run, row)?));
            }
        }
        Ok(None)
    }

    /// Read one f32 vector from `vectors.bin` by positioned read, through the
    /// segment's row cache when one is configured.
    fn read_vector(&self, seg: &Segment, id: VecId) -> io::Result<Vec<f32>> {
        if let Some(cache) = &seg.row_cache
            && let Some(v) = cache.lock().expect("row cache poisoned").get(id)
        {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::RerankCacheHits);
            return Ok(v);
        }
        let offset = HEADER_LEN as u64 + u64::from(id) * self.dim as u64 * 4;
        let mut buf = vec![0u8; self.dim * 4];
        seg.vectors_file.read_exact_at(&mut buf, offset)?;
        let v: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Some(cache) = &seg.row_cache {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::RerankCacheMisses);
            cache.lock().expect("row cache poisoned").put(id, v.clone());
        }
        Ok(v)
    }

    /// Approximate top-`k` `(id, cosine)` for `query`. A graph walk over the
    /// main index (int8 tier in RAM) plus a flat scan of the delta; survivors
    /// are re-ranked with exact f32 cosine - main vectors from disk, delta
    /// vectors from RAM. Tombstoned and delta-shadowed ids are filtered out.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a re-rank read from `vectors.bin` fails.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    pub fn search(&self, query: &[f32], k: usize) -> io::Result<Vec<(u64, f32)>> {
        self.search_with_l(query, k, 0)
    }

    /// Like [`search`](Self::search) but with an explicit search-list size.
    /// `l_search == 0` uses the index default; a non-zero value overrides it -
    /// the query-time effort knob (bigger = higher recall, slower walk).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a re-rank read from `vectors.bin` fails.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    pub fn search_with_l(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        self.search_inner(query, k, l_search, None, &[], 1.0, None, None)
    }

    /// Like [`search_with_l`](Self::search_with_l) but also overrides the re-rank
    /// budget - the number of candidates read from disk and scored with exact
    /// f32 (the recall/disk-read knob). `0` uses the default (`k*4`). Higher =
    /// more disk reads, higher recall; with the tq1 hybrid the candidates are
    /// asym-ordered so each extra read is well spent. Query-time only: does not
    /// touch the write path.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a re-rank read from `vectors.bin` fails.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    pub fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
        rerank: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        let rr = (rerank != 0).then_some(rerank);
        self.search_inner(query, k, l_search, None, &[], 1.0, rr, None)
    }

    /// Enable the online tq1 proxy controller (no-op unless the tier is tq1).
    /// Seeds it from the dim prior; [`search_adaptive`](Self::search_adaptive)
    /// then picks the proxy per query and learns from shadow A/B samples.
    pub fn enable_tq1_controller(&self) {
        if matches!(self.tier, QuantKind::TurboQuant { bits: 1 }) {
            let prior = crate::quant::tq1_proxy_mode_for(self.dim, 1);
            // Agreement (hybrid-vs-asym top-k overlap) is a coarse k-step signal,
            // so a looser tolerance than the pure-recall controller default.
            *self.tq1.ctl.lock().expect("tq1 ctl") =
                Some(Tq1ProxyController::new(prior).with_policy(0.1, 10, 3));
        }
    }

    /// The controller's current proxy mode, if enabled. Observability.
    #[must_use]
    pub fn tq1_controller_mode(&self) -> Option<Tq1ProxyMode> {
        self.tq1
            .ctl
            .lock()
            .expect("tq1 ctl")
            .as_ref()
            .map(Tq1ProxyController::mode)
    }

    /// Adaptive search. With the controller enabled it serves the query with the
    /// learned proxy mode (one full walk). On ~1/`SHADOW_EVERY` queries it ALSO
    /// runs a cheap shadow A/B - two SHORT walks (`l_search = SHADOW_L`) of
    /// hybrid vs asym - to measure their top-k agreement and feed the controller.
    /// The served result is always the full-quality current-mode walk, so a
    /// shadow query costs ~1 full walk + 2 short walks (~1.3x), not 2x: the
    /// user's "keep the 2x cheap" constraint. The controller's EMA averages the
    /// samples (the evaluation window), so it never reacts to one noisy query.
    /// Falls back to plain `search` when the controller is disabled / not tq1.
    ///
    /// # Errors
    ///
    /// I/O error if a re-rank read fails.
    pub fn search_adaptive(&self, query: &[f32], k: usize) -> io::Result<Vec<(u64, f32)>> {
        /// Short walk size for the shadow measurement - cheap so the A/B does not
        /// double query latency; still discriminative for the proxy comparison.
        const SHADOW_L: usize = 64;
        let decision = {
            let guard = self.tq1.ctl.lock().expect("tq1 ctl");
            guard.as_ref().map(|c| {
                let ctr = self
                    .tq1
                    .ctr
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                (c.mode(), c.should_shadow(ctr))
            })
        };
        let Some((mode, shadow)) = decision else {
            return self.search(query, k);
        };
        // Serve the query at full quality in the current mode (one walk).
        let served = self.search_inner(query, k, 0, None, &[], 1.0, None, Some(mode))?;
        if shadow && k > 0 {
            // Cheap measurement: two short walks, compared, fed to the controller.
            let sh = self.search_inner(
                query,
                k,
                SHADOW_L,
                None,
                &[],
                1.0,
                None,
                Some(Tq1ProxyMode::Hybrid),
            )?;
            let sa = self.search_inner(
                query,
                k,
                SHADOW_L,
                None,
                &[],
                1.0,
                None,
                Some(Tq1ProxyMode::Asymmetric),
            )?;
            let set_a: AHashSet<u64> = sa.iter().map(|(id, _)| *id).collect();
            let agree = sh.iter().filter(|(id, _)| set_a.contains(id)).count() as f32 / k as f32;
            if let Some(c) = self.tq1.ctl.lock().expect("tq1 ctl").as_mut() {
                c.record_shadow(agree, 1.0);
            }
        }
        Ok(served)
    }

    /// Filtered search: only ids for which `matches` returns true enter the
    /// result. Oversamples the frontier and reranks all of it so enough matching
    /// candidates survive the post-filter. `seeds` are external ids drawn from
    /// the matching set; the walk also starts from them (mapped to graph rows)
    /// so it begins inside the matching region rather than only at the medoid -
    /// the fix for filters whose matches cluster away from the query.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a re-rank read from `vectors.bin` fails.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
        matches: &dyn Fn(u64) -> bool,
        seeds: &[u64],
        selectivity: f32,
    ) -> io::Result<Vec<(u64, f32)>> {
        self.search_inner(
            query,
            k,
            l_search,
            Some(matches),
            seeds,
            selectivity,
            None,
            None,
        )
    }

    /// Shared search core. `matches == None` is the plain ANN search; `Some`
    /// keeps only matching ids. `selectivity` = |matching| / live is the walk
    /// planner's input: a DENSE filter (matches everywhere) needs only a single
    /// navigate-all walk + filter-at-rerank (~plain-search cost), while a SPARSE
    /// one needs the oversampled admit-gated + navigate-all two-walk.
    #[allow(clippy::cast_precision_loss)] // proxy is an ordering key, exact value irrelevant
    fn search_inner(
        &self,
        query: &[f32],
        k: usize,
        l_search: usize,
        matches: Option<&dyn Fn(u64) -> bool>,
        seeds: &[u64],
        selectivity: f32,
        rerank_override: Option<usize>,
        tq1_mode: Option<Tq1ProxyMode>,
    ) -> io::Result<Vec<(u64, f32)>> {
        /// Frontier blow-up for a SPARSE filtered walk, so the post-filter still
        /// leaves enough matching candidates. Capped at the main graph size.
        const FILTER_OVERSAMPLE: usize = 4;
        /// At or above this matching fraction, the filter is "dense": matches sit
        /// near the query, so one navigate-all walk + filter-at-rerank suffices.
        const DENSE_SELECTIVITY: f32 = 0.10;
        let dense = matches.is_some() && selectivity >= DENSE_SELECTIVITY;
        if query.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("query has {} dims, index has {}", query.len(), self.dim),
            ));
        }
        let filtered = matches.is_some();
        if self.live_count == 0 || k == 0 {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(OrderedFloat<f32>, u64)> = Vec::new();

        // Graph walk over every segment (the base plus any LSM runs); each
        // contributes candidates that merge into `scored`. An empty segment is
        // skipped. The delta is brute-forced once below.
        // Per-segment WALK (in-RAM, cheap) collects proxy-ranked candidate rows
        // tagged with their segment; one GLOBAL re-rank below bounds the disk
        // reads regardless of how many segments (base + runs) there are. This is
        // what keeps query latency flat as the LSM accumulates runs.
        let segs: Vec<&Segment> = std::iter::once(&self.base)
            .chain(self.runs.iter())
            .collect();
        // Global re-rank budget (disk reads). Bounded so latency tracks `k`, not
        // the corpus size or the segment count.
        let rerank = if let Some(r) = rerank_override {
            r.max(k)
        } else if filtered {
            if dense {
                ((k as f32 / selectivity).ceil() as usize * 2).clamp(64, 1024)
            } else {
                (k * 8).max(64)
            }
        } else {
            // Default disk-rerank budget. k*8 (not k*4): the rerank budget, not
            // L_search or l_build, was the recall ceiling - across every
            // embedding set, k*4=40 capped recall at ~0.94-0.98 while k*8=80
            // lifts it to 0.99+ for +10-30% latency (it saturates by ~k*16), and
            // it is query-time only so writes and RAM are untouched. Override
            // per query with `search_with_params`.
            (k * 8).max(64)
        };
        if filtered {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::VsearchFiltered);
        }
        let sketch = query_sketch(query);
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::VsearchInner);
        let phase_t0 = Instant::now();
        let mut all_cand: Vec<(f32, usize, VecId)> = Vec::new();
        // One normalize per search, not one per segment plus one for the IVF
        // seeds (dim-sized alloc + sqrt each).
        let qn = normalized(query);
        for (seg_idx, seg) in segs.iter().enumerate() {
            if seg.main_n == 0 {
                continue;
            }
            let code = seg.quant.quantize_query_with_mode(&qn, tq1_mode);
            // The base (segment 0) carries the deep beam. Runs are small and only
            // feed the global re-rank, so they walk a shallow list - this keeps
            // query latency flat as runs accumulate, instead of paying a full
            // l_search beam per run.
            let walk_base = if seg_idx == 0 {
                if l_search == 0 {
                    self.l_search
                } else {
                    l_search
                }
            } else {
                (k * 4).max(16)
            };
            // A sparse filtered walk oversamples the frontier so the post-filter
            // still leaves enough matches; a dense or plain walk does not.
            let list_size = if filtered && !dense {
                (walk_base.max(k) * FILTER_OVERSAMPLE).min(seg.main_n as usize)
            } else {
                walk_base.max(k)
            };
            let early = (!filtered && speed_enabled()).then_some(EarlyTerm {
                k: rerank,
                window: speed_window(),
            });
            let mut visited = VisitedBitset::new(seg.main_n as usize);
            let mut seen = VisitedBitset::new(seg.main_n as usize);
            // Medoid plus, for a filtered walk, matching seed rows so the walk can
            // start inside a matching cluster that sits away from the query.
            let mut seed_rows: Vec<VecId> = vec![seg.medoid];
            for &id in seeds {
                if let Some(&r) = seg.id_to_main_row.get(&id)
                    && !self.tombstones.contains(&id)
                {
                    seed_rows.push(r);
                }
            }
            // IVF cell seeds (base segment only, opt-in): a NOVEL query
            // starts inside its own semantic cell instead of at the medoid -
            // the complement of the entry cache, which only serves repeats.
            // Extra seeds only add candidates.
            if seg_idx == 0
                && ivf_seeds_enabled()
                && let Some(router) = &self.ivf
            {
                seed_rows.extend(
                    router
                        .query_cell_seeds(&qn, 4)
                        .into_iter()
                        .filter(|&r| r < seg.main_n),
                );
            }
            // Semantic entry seeds (base segment only): where similar queries
            // landed before. Extra seeds only add candidates.
            if seg_idx == 0
                && let Some(cache) = &seg.entry_cache
            {
                match cache.lock().expect("entry cache poisoned").get(sketch) {
                    Some(rows) => {
                        skeg_telemetry::tick_counter(skeg_telemetry::Counter::EntryCacheHits);
                        seed_rows.extend(rows.iter().copied().filter(|&r| r < seg.main_n));
                    }
                    None => skeg_telemetry::tick_counter(
                        skeg_telemetry::Counter::EntryCacheMisses,
                    ),
                }
            }
            // Prototype nav: exact f32 (read from disk) steers the walk when
            // enabled; otherwise the cheap tq1 proxy. Storage is tq1 either way.
            let nav_f32 = nav_f32_enabled();
            let dist = |id: VecId| -> f32 {
                if nav_f32 {
                    match self.read_vector(seg, id) {
                        Ok(v) => -cosine_f32(query, &v),
                        Err(_) => -(seg.quant.proxy(id as usize, &code) as f32),
                    }
                } else {
                    -(seg.quant.proxy(id as usize, &code) as f32)
                }
            };
            let nbrs = |id: VecId| -> SmallVec<[VecId; MAX_R]> {
                seg.nodes[id as usize].slice().iter().copied().collect()
            };
            let mut cand: Vec<(f32, VecId)> = Vec::new();
            let mut walk = |seeds: &[VecId],
                            admit: Option<&dyn Fn(VecId) -> bool>,
                            early: Option<EarlyTerm>| {
                greedy_search(
                    seeds,
                    list_size,
                    early,
                    dist,
                    nbrs,
                    admit,
                    &mut visited,
                    &mut seen,
                    None,
                )
            };
            if filtered && !dense {
                // Two walks unioned (see history): an admit-gated walk recovers
                // clustered matches, a navigate-all walk recovers scattered ones.
                let admit = |row: VecId| -> bool {
                    let id = seg.ids[row as usize];
                    !self.tombstones.contains(&id)
                        && !self.delta.contains_key(&id)
                        && !self.flushing.contains_key(&id)
                        && matches.is_none_or(|m| m(id))
                };
                cand.extend(walk(&seed_rows, Some(&admit), None).iter());
                cand.extend(walk(&[seg.medoid], None, None).iter());
            } else {
                let walk_early = if dense { None } else { early };
                cand.extend(walk(&seed_rows, None, walk_early).iter());
            }
            // tq1 hybrid: the walk navigated with the cheap popcount proxy, but
            // the candidate ordering here gates the bounded disk-read rerank
            // budget below. Re-score the survivors with the asymmetric proxy
            // (in-RAM, no disk) so the reads land on the best candidates. Other
            // modes keep the walk's proxy value at zero extra cost.
            skeg_telemetry::add_counter(
                skeg_telemetry::Counter::VsearchWalkHops,
                visited.count_set() as u64,
            );
            let hybrid = code.is_tq1_hybrid();
            for (proxy, row) in cand {
                let score = if hybrid {
                    -(seg.quant.proxy_rescore(row as usize, &code) as f32)
                } else {
                    proxy
                };
                all_cand.push((score, seg_idx, row));
            }
        }
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchWalkNanos,
            phase_t0.elapsed().as_nanos() as u64,
        );
        let phase_t0 = Instant::now();
        // Global re-rank: best-by-proxy first across every segment, bounded disk
        // reads, dedup by id (a later segment can re-surface the same id).
        all_cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let rerank_span = tracing::info_span!(
            "vsearch.rerank",
            candidates = all_cand.len(),
            disk_reads = tracing::field::Empty,
        );
        let _rg = rerank_span.enter();
        let mut disk_reads: usize = 0;
        let mut reranked_ids: AHashSet<u64> = AHashSet::new();
        // Adaptive rerank (prototype): min-heap of the k best exact cosines gives
        // the current k-th threshold; a candidate whose proxy-estimated cosine +
        // margin can't reach it (and, in best-first order, neither can the rest)
        // stops the disk reads early. Off => read the full `rerank` budget.
        // Adaptive only when the pscore is cosine-scaled: tq1 asym-family
        // (bit-plane / asym / hybrid-rescore). Popcount is Hamming, and int8/PQ/
        // tq2/tq4 report no tq1 mode - for those the bound would compare a
        // wrong-scale estimate to the k-th cosine and skip every candidate.
        // tq1 asym family: per-vector reconstruction-quality bound (opt-in).
        // tq2/tq4: cosine-scaled proxy with a flat conservative margin
        // (default on). Other tiers report neither and never skip.
        let adaptive = if matches!(self.base.quant.tq1_proxy_mode(), Some(m) if m != Tq1ProxyMode::Popcount)
        {
            adaptive_rr()
        } else if self.base.quant.turboquant_cos_scaled_bits().is_some() {
            adaptive_rr_tq24()
        } else {
            None
        };
        let mut topk: std::collections::BinaryHeap<std::cmp::Reverse<OrderedFloat<f32>>> =
            std::collections::BinaryHeap::new();
        for (pscore, seg_idx, row) in all_cand {
            if disk_reads >= rerank {
                break;
            }
            let seg = segs[seg_idx];
            let id = seg.ids[row as usize];
            if self.tombstones.contains(&id)
                || self.delta.contains_key(&id)
                || self.flushing.contains_key(&id)
            {
                continue;
            }
            if let Some(m) = matches
                && !m(id)
            {
                continue;
            }
            if !reranked_ids.insert(id) {
                continue;
            }
            // No-rerank mode = rank by the proxy estimate alone, zero
            // disk reads. For apples-to-apples vs the TurboQuant blog (pure
            // quantized recall, no f32 rerank). pscore = -(proxy i32).
            if no_rerank() {
                disk_reads += 1; // counts as "candidate considered", bounds by rerank
                scored.push((OrderedFloat(-pscore / 1.0e7), id));
                continue;
            }
            if let Some(c) = adaptive
                && topk.len() >= k
            {
                // Per-vector RaBitQ-style bound: err = C*sqrt(1-g^2), g = code
                // reconstruction quality (1/scale). Good code => tight => stop
                // sooner; poor code => loose => keep reading. pscore = -(proxy i32).
                let cos_est = -pscore / 1.0e7;
                // tq1 asym: per-vector bound from the code's reconstruction
                // quality. tq2/tq4 (no g): flat margin normalised by
                // sqrt(1024/dim) - the ADC estimate's noise scales like
                // 1/sqrt(dim), so one margin value holds across dimensions
                // (calibrated at dim 1024; at dim 16 it widens 8x and the
                // skip goes quiet instead of eating recall).
                let tau = topk.peek().map(|r| r.0.0).unwrap_or(f32::MIN);
                match seg.quant.tq1_recon_g(row as usize) {
                    // Per-vector bound: a poor code further down may still
                    // pass, so only this candidate is skipped.
                    Some(g) => {
                        let margin = c * (1.0 - g * g).max(0.0).sqrt();
                        if cos_est + margin < tau {
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::RerankAdaptiveSkips,
                            );
                            continue;
                        }
                    }
                    // Flat margin (tq2/tq4), normalised by sqrt(1024/dim):
                    // candidates arrive best-proxy-first, so once one fails
                    // the bound every later one fails it too - stop, don't
                    // wander the rest of the pool.
                    None => {
                        let margin = c * (1024.0 / self.dim as f32).sqrt();
                        if cos_est + margin < tau {
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::RerankAdaptiveSkips,
                            );
                            break;
                        }
                    }
                }
            }
            let v = self.read_vector(seg, row)?;
            disk_reads += 1;
            let c = cosine_f32(query, &v);
            if adaptive.is_some() {
                topk.push(std::cmp::Reverse(OrderedFloat(c)));
                if topk.len() > k {
                    topk.pop();
                }
            }
            scored.push((OrderedFloat(c), id));
        }
        rerank_span.record("disk_reads", disk_reads);
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchRerankNanos,
            phase_t0.elapsed().as_nanos() as u64,
        );
        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchRerankReads,
            disk_reads as u64,
        );
        let phase_t0 = Instant::now();

        // Flat scan of the delta (small, in RAM). Delta entries are always live.
        for (&id, v) in &self.delta {
            if matches.is_none_or(|m| m(id)) {
                scored.push((OrderedFloat(cosine_f32(query, v)), id));
            }
        }
        // Flat scan the in-flight flush staging too, skipping ids the delta
        // already covers (delta wins). Empty except during an off-thread flush.
        for (&id, v) in &self.flushing {
            if !self.delta.contains_key(&id) && matches.is_none_or(|m| m(id)) {
                scored.push((OrderedFloat(cosine_f32(query, v)), id));
            }
        }

        skeg_telemetry::add_counter(
            skeg_telemetry::Counter::VsearchDeltaNanos,
            phase_t0.elapsed().as_nanos() as u64,
        );
        scored.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
        scored.truncate(k);
        // Remember where this query landed: its base-row winners become the
        // walk seeds of the next query with the same sketch.
        if let Some(cache) = &self.base.entry_cache {
            let rows: SmallVec<[VecId; 8]> = scored
                .iter()
                .filter_map(|&(_, id)| self.base.id_to_main_row.get(&id).copied())
                .take(8)
                .collect();
            if !rows.is_empty() {
                cache.lock().expect("entry cache poisoned").put(sketch, rows);
            }
        }
        Ok(scored
            .into_iter()
            .map(|(s, id)| (id, s.into_inner()))
            .collect())
    }

    /// A uniform sample of the base graph for visual exploration: up to
    /// `count` seed rows (stride-picked) plus their out-neighbours, as
    /// `(id, degree)` nodes and `(from_id, to_id)` edges. Read-only, RAM
    /// only (adjacency + ids), no distances.
    #[must_use]
    pub fn graph_sample(&self, count: usize) -> (Vec<(u64, u32)>, Vec<(u64, u64)>) {
        let n = self.base.main_n as usize;
        if n == 0 || count == 0 {
            return (Vec::new(), Vec::new());
        }
        // CONNECTED patches instead of scattered stars: a stride sample with
        // one hop kept ~0.1% of edges (both endpoints rarely sampled) and drew
        // dust. A few seeds per call expand breadth-first until the budget is
        // spent, so the local edge structure comes out whole.
        let n_seeds = (count / 40).clamp(2, 16);
        let stride = (n / n_seeds).max(1);
        let mut rows: Vec<u32> = Vec::with_capacity(count);
        let mut in_sample: AHashSet<u32> = AHashSet::new();
        let mut queue: std::collections::VecDeque<u32> = (0..n)
            .step_by(stride)
            .take(n_seeds)
            .map(|r| r as u32)
            .collect();
        for &s in &queue {
            in_sample.insert(s);
        }
        while let Some(r) = queue.pop_front() {
            rows.push(r);
            if rows.len() + queue.len() >= count {
                continue; // drain what is queued, expand no further
            }
            for &nb in self.base.nodes[r as usize].slice() {
                if in_sample.insert(nb) {
                    queue.push_back(nb);
                }
            }
        }
        rows.extend(queue);
        let mut edges: Vec<(u64, u64)> = Vec::new();
        for &r in &rows {
            for &nb in self.base.nodes[r as usize].slice() {
                if in_sample.contains(&nb) {
                    edges.push((self.base.ids[r as usize], self.base.ids[nb as usize]));
                }
            }
        }
        let nodes = rows
            .into_iter()
            .map(|r| (self.base.ids[r as usize], self.base.nodes[r as usize].degree))
            .collect();
        (nodes, edges)
    }

    /// Run the main-graph greedy walk for `query` and return the ordered
    /// sequence of graph node rows it expands - the on-disk access pattern a
    /// paged graph store would see. For cache-locality analysis (the gate
    /// before paged storage); not part of a normal search.
    ///
    /// # Errors
    ///
    /// Returns an I/O error only for signature parity with [`search`](Self::search);
    /// the walk itself is in-RAM.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal the index dimension.
    #[allow(clippy::cast_precision_loss)] // proxy is an ordering key, exact value irrelevant
    pub fn search_node_trace(&self, query: &[f32]) -> io::Result<Vec<VecId>> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let mut trace = Vec::new();
        if self.base.main_n > 0 {
            let code = self.base.quant.quantize_query(&normalized(query));
            let mut visited = VisitedBitset::new(self.base.main_n as usize);
            let mut seen = VisitedBitset::new(self.base.main_n as usize);
            greedy_search(
                &[self.base.medoid],
                self.l_search,
                None, // trace: want the full walk, no early termination
                |id| -(self.base.quant.proxy(id as usize, &code) as f32),
                |id| {
                    self.base.nodes[id as usize]
                        .slice()
                        .iter()
                        .copied()
                        .collect()
                },
                None, // trace: no filter admission
                &mut visited,
                &mut seen,
                Some(&mut trace),
            );
        }
        Ok(trace)
    }

    /// BFS node ordering from the medoid: `result[k]` is the original node row
    /// a page-aware layout would place at position `k`. Graph-adjacent nodes
    /// land at adjacent positions, so a paged store co-locates them. For the
    /// cache-locality reorder experiment; not used by a normal search.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // node count is a u32
    pub fn bfs_order(&self) -> Vec<VecId> {
        let n = self.base.main_n as usize;
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        if n > 0 {
            seen[self.base.medoid as usize] = true;
            queue.push_back(self.base.medoid);
        }
        while let Some(cur) = queue.pop_front() {
            order.push(cur);
            for &nbr in self.base.nodes[cur as usize].slice() {
                if !seen[nbr as usize] {
                    seen[nbr as usize] = true;
                    queue.push_back(nbr);
                }
            }
        }
        // A node unreachable from the medoid (should not happen after
        // patch_connectivity) is appended so the permutation stays total.
        for (id, &s) in seen.iter().enumerate() {
            if !s {
                order.push(id as VecId);
            }
        }
        order
    }

    /// L0 (delta) size that triggers a flush into a navigable run. Small, so a
    /// flush is a cheap few-thousand-vector bulk build and the brute-forced
    /// delta search never grows large.
    const FLUSH: usize = 4096;

    /// Build the L0 delta into a fresh immutable run and clear L0. A pure
    /// in-process optimisation: the WAL is left intact, so a crash mid-flush
    /// loses nothing (the run is rebuildable from the WAL). See
    /// `docs/adr-incremental-flush.md`. No-op when L0 is empty.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if building, saving, or opening the run fails.
    fn flush(&mut self) -> io::Result<()> {
        if self.delta.is_empty() {
            return Ok(());
        }
        let mut vectors: Vec<f32> = Vec::with_capacity(self.delta.len() * self.dim);
        let mut ids: Vec<u64> = Vec::with_capacity(self.delta.len());
        for (&id, v) in &self.delta {
            vectors.extend_from_slice(v);
            ids.push(id);
        }
        let seq = self.run_seq;
        let run_dir = self.dir.join(format!("run-{seq}"));
        self.run_seq += 1;
        let rebuilt = build_disk_graph(self.tier, vectors, ids, self.dim, &disk_build_config());
        rebuilt.save(&run_dir)?;
        // Open the run with this index's tier and keep only its base segment; the
        // rest of the opened index (an empty delta/WAL over the run dir) is dropped.
        let run = DiskVamanaIndex::open_with_tier(&run_dir, self.tier)?;
        self.runs.push(run.base);
        self.run_dirs.push(seq);
        self.delta.clear();
        Ok(())
    }

    /// Delete every flushed run directory and drop the in-RAM runs. Called once
    /// `consolidate` has folded them into a fresh base.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a run directory cannot be removed.
    fn discard_runs(&mut self) -> io::Result<()> {
        for seq in 0..self.run_seq {
            let d = self.dir.join(format!("run-{seq}"));
            if d.exists() {
                std::fs::remove_dir_all(&d)?;
            }
        }
        self.runs.clear();
        self.run_dirs.clear();
        Ok(())
    }

    /// Survivor locations that are not a segment index: the in-RAM layers.
    const LOC_DELTA: usize = usize::MAX;
    /// The staging map an in-flight flush moved the delta into.
    const LOC_FLUSHING: usize = usize::MAX - 1;

    /// Fold the base, every run, the delta, and tombstones into one fresh
    /// on-disk graph, then re-open. The newest version of each id wins (delta,
    /// then runs newest-to-oldest, then base); tombstoned ids are dropped.
    /// Heavy - meant for a background task. A no-op if nothing is live.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if rebuilding, re-saving, or re-opening fails.
    pub fn consolidate(&mut self) -> io::Result<()> {
        let dim = self.dim;
        // Collect the surviving (id, location) refs only - ~24 B each, not the
        // full 2 GB of vectors. Precedence (newest wins): delta > runs (newest
        // first) > base; `seen` keeps each id once, tombstoned ids never enter.
        // `loc`: usize::MAX => delta, else index into `segs` (0 = base, 1.. = runs).
        let segs: Vec<&Segment> = std::iter::once(&self.base)
            .chain(self.runs.iter())
            .collect();
        let mut seen: AHashSet<u64> = AHashSet::new();
        let mut survivors: Vec<(u64, usize, u32)> = Vec::new();
        // `delta`, then `flushing`: the documented precedence is
        // `delta > flushing > runs > base`. Skipping the staging map here loses
        // every vector an in-flight `flush_begin` moved out of the delta, since
        // this function truncates the WAL and reopens from disk, which drops
        // the in-memory map too. The window is reachable: the engine releases
        // the write lock while a flush builds off-thread, and the client
        // command `SKEG.VINDEX.CONSOLIDATE` takes that lock and lands here.
        for &id in self.delta.keys() {
            if seen.insert(id) {
                survivors.push((id, Self::LOC_DELTA, 0));
            }
        }
        for &id in self.flushing.keys() {
            if seen.insert(id) {
                survivors.push((id, Self::LOC_FLUSHING, 0));
            }
        }
        for (ri, run) in self.runs.iter().enumerate().rev() {
            for row in 0..run.main_n {
                let id = run.ids[row as usize];
                if self.tombstones.contains(&id) || !seen.insert(id) {
                    continue;
                }
                survivors.push((id, ri + 1, row));
            }
        }
        for row in 0..self.base.main_n {
            let id = self.base.ids[row as usize];
            if self.tombstones.contains(&id) || !seen.insert(id) {
                continue;
            }
            survivors.push((id, 0, row));
        }
        if survivors.is_empty() {
            return Ok(());
        }
        // Rebuild in id order so a query's near-neighbours land at nearby
        // vectors.bin rows and the re-rank's f32 reads stay cache-local. Without
        // it the fold order scatters them and 500k+ search latency regresses ~1.5x
        // (root-caused: the re-rank is disk-read bound at scale).
        survivors.sort_unstable_by_key(|&(id, _, _)| id);
        let mut vectors: Vec<f32> = Vec::with_capacity(survivors.len() * dim);
        let mut ids: Vec<u64> = Vec::with_capacity(survivors.len());
        for (id, loc, row) in survivors {
            match loc {
                Self::LOC_DELTA => vectors.extend_from_slice(&self.delta[&id]),
                Self::LOC_FLUSHING => vectors.extend_from_slice(&self.flushing[&id]),
                seg => vectors.extend(self.read_vector(segs[seg], row)?),
            }
            ids.push(id);
        }
        let dir = self.dir.clone();
        let tier = self.tier;
        // Drop the persisted router FIRST: this consolidate reorders the base
        // rows, so the old sidecar's cell_of is about to be invalid. Removing it
        // before save() means a crash anywhere below leaves NO sidecar (open
        // rebuilds none, the idle task builds a fresh one) rather than a stale
        // one that a length-only load guard could accept against reordered rows.
        // We do NOT rebuild the IVF here - consolidate runs INLINE on the ingest
        // path (shard auto-consolidates when delta >= main); an inline IVF build
        // (re-read all vectors + k-means) stalls ingest. The background
        // idle-consolidate rebuilds it off the request path; until then filtered
        // search falls back to the exact scan (correct, just O(|s|)).
        let _ = std::fs::remove_file(dir.join(IVF_FILE));
        // Env-gated phase timing to find the real consolidate cost
        // before optimizing it. Off unless SKEG_CONSOLIDATE_TIMING is set.
        let timing = std::env::var("SKEG_CONSOLIDATE_TIMING").is_ok();
        let t = std::time::Instant::now();
        let rebuilt = build_disk_graph(tier, vectors, ids, dim, &disk_build_config());
        let build_ms = t.elapsed().as_millis();
        rebuilt.save(&dir)?;
        let save_ms = t.elapsed().as_millis();
        self.discard_runs()?;
        // The delta + runs are now folded into the graph: the WAL must start
        // empty so the reopen below does not replay stale records.
        write_framed_wal(&dir.join(DELTA_LOG_FILE), &[])?;
        *self = DiskVamanaIndex::open_with_tier(&dir, tier)?;
        if timing {
            let reopen_ms = t.elapsed().as_millis() - save_ms;
            eprintln!(
                "consolidate n={} build={build_ms}ms save={}ms reopen(reread+requant)={reopen_ms}ms",
                self.base.main_n,
                save_ms - build_ms,
            );
        }
        Ok(())
    }

    /// Begin a background consolidate: the short, exclusive phase.
    ///
    /// Flushes the delta into a run (so the snapshot is over immutable segments
    /// only), collects the surviving vectors (tombstone-masked, newest-wins, id
    /// order), and records the WAL high-water offset. Everything appended to the
    /// WAL after this point (inserts, deletes, and any runs they flush) is NOT
    /// in the snapshot and survives [`consolidate_finish`](Self::consolidate_finish)
    /// via WAL-suffix replay.
    ///
    /// Returns `None` when there is nothing to fold. At most one job should be
    /// outstanding per index; the caller enforces that (the engine does not).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the flush or a vector read fails.
    pub fn consolidate_begin(&mut self) -> io::Result<Option<ConsolidateJob>> {
        // Cheap snapshot: NO flush (which would build a graph from the delta) and
        // NO O(live) vector reads. The delta is the newest layer, captured in
        // RAM; base/run survivors are recorded as (seg, row) locations and read
        // off-thread in `build`. All the heavy work moves to the background.
        let mut seen: AHashSet<u64> = AHashSet::new();
        let cap = self.delta.len() + self.flushing.len();
        let mut delta_ids: Vec<u64> = Vec::with_capacity(cap);
        let mut delta_vectors: Vec<f32> = Vec::with_capacity(cap * self.dim);
        // delta first (newest), then any in-flight flush staging (delta shadows
        // it via `seen`). Tombstoned ids are already removed from both.
        for (&id, v) in self.delta.iter().chain(self.flushing.iter()) {
            if seen.insert(id) {
                delta_ids.push(id);
                delta_vectors.extend_from_slice(v);
            }
        }
        // Base/runs, newest run first; delta already shadows via `seen`.
        let mut survivors: Vec<(u64, usize, u32)> = Vec::new();
        for (ri, run) in self.runs.iter().enumerate().rev() {
            for row in 0..run.main_n {
                let id = run.ids[row as usize];
                if self.tombstones.contains(&id) || !seen.insert(id) {
                    continue;
                }
                survivors.push((id, ri + 1, row));
            }
        }
        for row in 0..self.base.main_n {
            let id = self.base.ids[row as usize];
            if self.tombstones.contains(&id) || !seen.insert(id) {
                continue;
            }
            survivors.push((id, 0, row));
        }
        if survivors.is_empty() && delta_ids.is_empty() {
            return Ok(None);
        }
        // Dup the vectors.bin fds [base, run0, ...] for off-thread reads (O(1)).
        let mut seg_files: Vec<File> = Vec::with_capacity(1 + self.runs.len());
        seg_files.push(self.base.vectors_file.try_clone()?);
        for run in &self.runs {
            seg_files.push(run.vectors_file.try_clone()?);
        }
        let wal_offset = self.delta_log.metadata()?.len();
        // Route: keep the base edges when most of the work would be redoing
        // them. `base_live` are rows whose edges survive verbatim or bridged;
        // everything else has to be inserted fresh either way, so it is the
        // honest measure of what reuse can save. Capturing the adjacency is a
        // memcpy of the Node array (~260 B/row) under this short lock; it is
        // taken only when the patched route will actually run.
        let base_live = survivors.iter().filter(|s| s.1 == 0).count();
        let new_rows = delta_ids.len() + (survivors.len() - base_live);
        let patch = if patch_fold_route(new_rows, base_live, self.base.main_n as usize) {
            let n = self.base.main_n as usize;
            Some(PatchBase {
                adj: (0..n).map(|r| self.base.nodes[r]).collect(),
                medoid: self.base.medoid,
            })
        } else {
            None
        };
        Ok(Some(ConsolidateJob {
            delta_vectors,
            delta_ids,
            survivors,
            seg_files,
            dim: self.dim,
            tier: self.tier,
            run_seq_high: self.run_seq,
            wal_offset,
            patch,
        }))
    }

    /// Finish a background consolidate: the short, exclusive swap phase.
    ///
    /// Replaces the base with the graph the job built, discards every run
    /// (post-begin runs are rebuilt from the WAL suffix), rewrites the WAL to
    /// hold only the records appended after `consolidate_begin`, and reopens.
    /// Post-begin inserts land back in the delta (re-flushing if large) and
    /// post-begin deletes land back in the tombstone set, so no write that
    /// raced the background build is lost.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a file move, the WAL rewrite, or the reopen fails.
    pub fn consolidate_finish(&mut self, built: ConsolidateBuilt) -> io::Result<()> {
        tracing::info!(
            "consolidate_finish: installing {} base ({} rows)",
            if built.patched { "patched" } else { "rebuilt" },
            built.base.main_n,
        );
        let dir = self.dir.clone();
        let tier = self.tier;
        // The rebuild reorders base rows: drop the router sidecar first, exactly
        // like the inline consolidate (crash below leaves no sidecar, not a
        // stale one).
        let _ = std::fs::remove_file(dir.join(IVF_FILE));
        // WAL suffix = everything appended after begin. Copy it out before the
        // reopen truncates.
        let wal_path = dir.join(DELTA_LOG_FILE);
        let wal = std::fs::read(&wal_path)?;
        let suffix_start = usize::try_from(built.wal_offset).unwrap_or(wal.len());
        let suffix = wal.get(suffix_start..).unwrap_or(&[]).to_vec();
        let suffix_ops = decode_wal_payload(self.wal_format, &suffix, self.dim)?;
        // Swap in the built base, drop every run dir (pre-begin runs are folded
        // into the new base; post-begin runs replay from the WAL suffix).
        // Atomic base swap: the whole new generation (graph + vectors + tier
        // cache) is installed under one CURRENT flip - never a torn mix of
        // old and new files (review P0). built.base's fds follow the inodes
        // across the rename.
        install_base_generation(&dir, &built.tmp)?;
        self.run_seq = built.run_seq_high.max(self.run_seq);
        self.discard_runs()?;
        // Re-encode the post-begin suffix as V2.
        write_framed_wal(&wal_path, &suffix_ops)?;
        // Surgical swap: install the prebuilt base (tier already built off-thread)
        // and reconstruct the in-RAM state the way `open` would, then replay the
        // WAL suffix - WITHOUT a full reopen, so the O(live) tier rebuild does not
        // run here on the shard thread. `tier` is unused now (the segment carries
        // its own quant); keep the arg-free reopen out of the hot path.
        let _ = tier;
        let delta_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;
        self.base = built.base;
        self.delta.clear();
        self.tombstones.clear();
        self.live_count = self.base.main_n as usize;
        self.delta_log = delta_log;
        self.wal_format = DeltaWalFormat::FramedV2;
        *self.tq1 = Default::default();
        self.ivf = None;
        self.replay_wal_ops(suffix_ops);
        Ok(())
    }

    /// Begin a background runs-only merge (Level 2 of the write-heavy path):
    /// snapshot the live vectors from the front runs (tombstone-masked,
    /// newer-run-wins) so an off-thread [`RunMergeJob::build`] can fold them into
    /// one replacement run, leaving the base untouched. O(runs), not O(live-set),
    /// so it can run often enough to keep the run count - and thus per-query walk
    /// cost - bounded under fast churn, deferring the expensive base rebuild to a
    /// rare event. Needs no WAL manipulation: runs are immutable, so racing
    /// inserts/deletes/flushes are handled by search precedence alone.
    ///
    /// Returns `None` with fewer than two runs (nothing to compact).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if a run vector read fails.
    pub fn merge_runs_begin(&mut self) -> io::Result<Option<RunMergeJob>> {
        let n_merged = self.runs.len();
        if n_merged < 2 {
            return Ok(None);
        }
        // Newer runs win on a re-inserted id: fold the front runs newest-first.
        let mut seen: AHashSet<u64> = AHashSet::new();
        let mut survivors: Vec<(usize, u32)> = Vec::new();
        for ri in (0..n_merged).rev() {
            let run = &self.runs[ri];
            for row in 0..run.main_n {
                let id = run.ids[row as usize];
                if self.tombstones.contains(&id) || !seen.insert(id) {
                    continue;
                }
                survivors.push((ri, row));
            }
        }
        if survivors.is_empty() {
            // Every folded run entry is tombstoned or shadowed: just drop them.
            let old_dirs = self.run_dirs[..n_merged].to_vec();
            let keep_runs = self.runs.split_off(n_merged);
            let keep_dirs = self.run_dirs.split_off(n_merged);
            self.runs = keep_runs;
            self.run_dirs = keep_dirs;
            for seq in old_dirs {
                let d = self.dir.join(format!("run-{seq}"));
                if d.exists() {
                    std::fs::remove_dir_all(&d)?;
                }
            }
            return Ok(None);
        }
        // id order for re-rank cache locality, same rule as consolidate.
        survivors.sort_unstable_by_key(|&(ri, row)| self.runs[ri].ids[row as usize]);
        let ids: Vec<u64> = survivors
            .iter()
            .map(|&(ri, row)| self.runs[ri].ids[row as usize])
            .collect();
        // Dup the folded runs' vectors.bin fds (O(1)); the reads happen off-thread
        // in build. seg_index in `survivors` is the run index 0..n_merged.
        let mut seg_files: Vec<File> = Vec::with_capacity(n_merged);
        for ri in 0..n_merged {
            seg_files.push(self.runs[ri].vectors_file.try_clone()?);
        }
        let old_dirs = self.run_dirs[..n_merged].to_vec();
        let merged_seq = self.run_seq;
        self.run_seq += 1;
        Ok(Some(RunMergeJob {
            survivors,
            ids,
            seg_files,
            dim: self.dim,
            tier: self.tier,
            merged_seq,
            n_merged,
            old_dirs,
        }))
    }

    /// Finish a background runs-only merge: slot the built run in place of the
    /// front `n_merged` runs it folded, keep any runs flushed after begin (they
    /// are newer, so they stay ahead of the merged run in search order), and
    /// delete the folded run dirs. The base, delta, tombstones, and WAL are
    /// untouched.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if opening the merged run or deleting a dir fails.
    pub fn merge_runs_finish(&mut self, built: RunMergeBuilt) -> io::Result<()> {
        // The merged run replaces marked runs; it must be marked itself or a
        // reopen would drop it AND find no WAL rows to rebuild it from.
        self.mark_run_durable(built.merged_seq)?;
        // The merged run was already opened off-thread in build; splice it in.
        let keep_runs = self.runs.split_off(built.n_merged);
        let keep_dirs = self.run_dirs.split_off(built.n_merged);
        self.runs = std::iter::once(built.merged).chain(keep_runs).collect();
        self.run_dirs = std::iter::once(built.merged_seq).chain(keep_dirs).collect();
        for seq in built.old_dirs {
            let d = self.dir.join(format!("run-{seq}"));
            if d.exists() {
                std::fs::remove_dir_all(&d)?;
            }
        }
        Ok(())
    }

    /// Begin an OFF-THREAD flush of the delta into a run (short, exclusive).
    /// Moves the delta into the `flushing` staging buffer (kept searchable) and
    /// hands its vectors to the returned job; the run build + open happens off
    /// the caller in [`FlushJob::build`]. `None` if the delta is empty or a flush
    /// is already in flight. No WAL touch: the delta's inserts stay in the WAL,
    /// so a crash before `flush_finish` replays them into the delta on reopen.
    ///
    /// # Errors
    ///
    /// Infallible today; returns `io::Result` for symmetry with the other
    /// begin/finish pairs.
    pub fn flush_begin(&mut self) -> io::Result<Option<FlushJob>> {
        if self.delta.is_empty() || !self.flushing.is_empty() {
            return Ok(None);
        }
        self.flushing = std::mem::take(&mut self.delta);
        let mut vectors: Vec<f32> = Vec::with_capacity(self.flushing.len() * self.dim);
        let mut ids: Vec<u64> = Vec::with_capacity(self.flushing.len());
        for (&id, v) in &self.flushing {
            vectors.extend_from_slice(v);
            ids.push(id);
        }
        let seq = self.run_seq;
        self.run_seq += 1;
        Ok(Some(FlushJob {
            vectors,
            ids,
            dim: self.dim,
            tier: self.tier,
            seq,
        }))
    }

    /// Splice the flushed run in (short, exclusive) and clear the staging. The
    /// run is the newest, so it goes at the end (newest-wins in search). Ids
    /// re-inserted during the build are in the fresh delta and shadow the run
    /// copy; deleted ones are tombstone-masked.
    ///
    /// # Errors
    ///
    /// Infallible today; `io::Result` for symmetry.
    pub fn flush_finish(&mut self, built: FlushBuilt) -> io::Result<()> {
        // Durability order matters. 1) fsync the run's files; 2) write and
        // fsync `run.ok` - from here a reopen loads this run instead of
        // replaying its rows; 3) install; 4) compact the WAL down to the
        // current delta plus the live tombstones. A crash between 2 and 4
        // leaves the run marked AND its rows still in the WAL: the replayed
        // delta copies shadow the identical run copies, which costs RAM until
        // the next flush and nothing else.
        let seq = built.seq;
        self.mark_run_durable(seq)?;
        self.runs.push(built.run);
        self.run_dirs.push(seq);
        self.flushing.clear();
        self.compact_wal()?;
        Ok(())
    }

    /// Fsync every file of `run-{seq}` and write its `run.ok` marker.
    fn mark_run_durable(&self, seq: u64) -> io::Result<()> {
        let d = self.dir.join(format!("run-{seq}"));
        // Durability order (a crash between any two steps must not leave the
        // marker durable while a data file is not): 1) fsync each data file's
        // contents; 2) fsync the directory so those files' entries are durable;
        // 3) only then write + fsync run.ok; 4) fsync the directory again so
        // the marker's entry lands last. Reversing 2 and 3 (the old order)
        // risked a marked-but-unreadable run whose rows the WAL had already
        // compacted away - real data loss (review finding).
        for entry in std::fs::read_dir(&d)? {
            let path = entry?.path();
            if path.is_file() {
                File::open(&path)?.sync_all()?;
            }
        }
        File::open(&d)?.sync_all()?;
        let marker = d.join(RUN_OK_FILE);
        std::fs::write(&marker, b"ok")?;
        File::open(&marker)?.sync_all()?;
        File::open(&d)?.sync_all()?;
        Ok(())
    }

    /// Rewrite the WAL to exactly the recoverable in-RAM state: one delete per
    /// live tombstone, one insert per delta row. Written to a temp file and
    /// renamed, then the append handle is reopened on the new inode.
    fn compact_wal(&mut self) -> io::Result<()> {
        let mut ops: Vec<DeltaWalOp> = Vec::with_capacity(self.tombstones.len() + self.delta.len());
        for &id in &self.tombstones {
            ops.push(DeltaWalOp::Delete { id });
        }
        for (&id, v) in &self.delta {
            ops.push(DeltaWalOp::Insert { id, vector: v.clone() });
        }
        let path = self.dir.join(DELTA_LOG_FILE);
        let tmp = self.dir.join("delta.log.compact");
        write_framed_wal(&tmp, &ops)?;
        // Open the append handle on the TEMP inode, fsync, then rename: the
        // handle follows the inode to its new name, so there is no window
        // where delta_log points at a renamed-over, unlinked file that would
        // swallow later WAL appends (review finding). The old handle is
        // dropped only once the new one is in hand.
        let new_log = std::fs::OpenOptions::new().append(true).open(&tmp)?;
        new_log.sync_all()?;
        std::fs::rename(&tmp, &path)?;
        File::open(&self.dir)?.sync_all()?; // the rename's directory entry
        self.delta_log = new_log;
        self.wal_format = DeltaWalFormat::FramedV2;
        Ok(())
    }

    /// Snapshot for a background delete-patch of the base graph (short,
    /// exclusive). Flushes the delta, then marks every base row that is dead -
    /// tombstoned, or shadowed by a newer copy in a run - and captures the base
    /// adjacency + vectors so [`DeletePatchJob::build`] can re-prune only the
    /// survivors that lost a neighbour (O(affected), no greedy rebuild).
    /// Returns `None` when there is nothing to reclaim, or when the patch would
    /// empty the base (a full `consolidate` is the right tool then).
    ///
    /// # Errors
    ///
    /// I/O error if the flush or a base-vector read fails.
    pub fn delete_patch_begin(&mut self) -> io::Result<Option<DeletePatchJob>> {
        // NO flush (that would build a run graph on the caller). A base row is
        // dead if tombstoned OR shadowed by a newer copy anywhere - run, delta,
        // or flush staging; the newer copy wins in search, so dropping the stale
        // base row is safe without moving the delta into a run first.
        let n = self.base.main_n as usize;
        if n == 0 {
            return Ok(None);
        }
        let mut shadowed: AHashSet<u64> = AHashSet::new();
        for run in &self.runs {
            for &id in &run.ids {
                shadowed.insert(id);
            }
        }
        shadowed.extend(self.delta.keys().copied());
        shadowed.extend(self.flushing.keys().copied());
        let mut dead = vec![false; n];
        let mut dead_count = 0usize;
        for row in 0..n {
            let id = self.base.ids[row];
            if self.tombstones.contains(&id) || shadowed.contains(&id) {
                dead[row] = true;
                dead_count += 1;
            }
        }
        if dead_count == 0 || n - dead_count == 0 {
            return Ok(None);
        }
        // Copy the adjacency (fast memcpy of the Node array) but DEFER the
        // O(live) vector reads to build via a duped fd.
        let adj: Vec<Node> = (0..n).map(|r| self.base.nodes[r]).collect();
        let seg_file = self.base.vectors_file.try_clone()?;
        Ok(Some(DeletePatchJob {
            seg_file,
            n_rows: n,
            ids: self.base.ids.clone(),
            adj,
            dead,
            medoid: self.base.medoid,
            dim: self.dim,
            l_search: self.l_search,
            tier: self.tier,
        }))
    }

    /// Swap the patched base in (short, exclusive). The patch touched only the
    /// base graph, so - unlike `consolidate_finish` - runs, delta, tombstones,
    /// and the WAL are all left as they are. That is safe across a restart: every
    /// row the patch removed was either tombstoned (the WAL still re-tombstones
    /// it) or shadowed by a run (the WAL still holds the newer copy), so a plain
    /// reopen reconstructs the same live set. The patched base's `vectors.bin`
    /// fd, opened via the sidecar, keeps pointing at the renamed inode.
    ///
    /// # Errors
    ///
    /// I/O error if the sidecar open or the file renames fail.
    pub fn delete_patch_finish(&mut self, built: DeletePatchBuilt) -> io::Result<()> {
        let dir = self.dir.clone();
        // The patched base was already opened off-thread in build; swap it in.
        let new_base = built.base;
        // The patch reordered base rows: drop the stale router sidecar.
        let _ = std::fs::remove_file(dir.join(IVF_FILE));
        install_base_generation(&dir, &built.tmp)?;
        self.base = new_base;
        self.ivf = None;
        // Drop tombstones that no longer cover anything (their only copy was a
        // base row we just removed); the WAL still carries them for a restart,
        // and the next consolidate clears them for good. Keeps the live filter
        // from growing without bound under sustained delete-patching.
        let mut present: AHashSet<u64> = AHashSet::with_capacity(self.base.ids.len());
        present.extend(self.base.ids.iter().copied());
        for run in &self.runs {
            present.extend(run.ids.iter().copied());
        }
        self.tombstones.retain(|id| present.contains(id));
        Ok(())
    }

    /// True if this index is large enough to benefit from a routed filtered
    /// search (below it, an exact scan of the match set is already cheap). Used
    /// by the background idle-consolidate to decide whether to build the router.
    #[must_use]
    pub fn wants_ivf(&self) -> bool {
        /// Base size at/above which the IVF router pays for itself.
        const IVF_MIN: u32 = 50_000;
        self.ivf.is_none() && self.base.main_n >= IVF_MIN
    }

    /// Build the coarse IVF router over the base segment (the "cells" branch of
    /// hybrid filtered search). In-memory; rebuild after a `consolidate`. Reads
    /// the base vectors once for k-means. `n_cells == 0` picks ~sqrtn.
    ///
    /// # Errors
    /// I/O error if a base vector read fails.
    pub fn build_ivf(&mut self, n_cells: usize, iters: usize) -> io::Result<()> {
        let n = self.base.main_n;
        if n == 0 {
            self.ivf = None;
            return Ok(());
        }
        let dim = self.dim;
        let mut all = vec![0.0f32; n as usize * dim];
        for r in 0..n {
            let v = self.read_vector(&self.base, r)?;
            all[r as usize * dim..r as usize * dim + dim].copy_from_slice(&v);
        }
        let n_cells = if n_cells == 0 {
            IvfRouter::cells_for(n as usize)
        } else {
            n_cells
        };
        let router = IvfRouter::build(&all, n, dim, n_cells, iters);
        let _ = std::fs::write(self.dir.join(IVF_FILE), router.to_bytes());
        self.ivf = Some(Box::new(router));
        self.refresh_zonemap();
        Ok(())
    }

    /// Begin an OFF-THREAD IVF rebuild: dup the base vectors fd (O(1)). `None`
    /// if the base is empty. The read + k-means happen in [`IvfJob::build`].
    ///
    /// # Errors
    ///
    /// I/O error if the fd cannot be duped.
    pub fn ivf_begin(&self, n_cells: usize, iters: usize) -> io::Result<Option<IvfJob>> {
        let n = self.base.main_n;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(IvfJob {
            seg_file: self.base.vectors_file.try_clone()?,
            n,
            dim: self.dim,
            n_cells,
            iters,
        }))
    }

    /// Install the router built off-thread (short, exclusive) and persist it.
    ///
    /// # Errors
    ///
    /// Infallible today; `io::Result` for symmetry.
    pub fn ivf_finish(&mut self, built: IvfBuilt) -> io::Result<()> {
        let _ = std::fs::write(self.dir.join(IVF_FILE), built.router.to_bytes());
        self.ivf = Some(Box::new(built.router));
        self.refresh_zonemap();
        Ok(())
    }

    /// Load the persisted IVF router, if present + consistent with the base.
    /// Best-effort: a missing / stale sidecar just leaves the router unbuilt.
    fn load_ivf(&mut self) {
        let Ok(bytes) = std::fs::read(self.dir.join(IVF_FILE)) else {
            return;
        };
        if let Some(r) = IvfRouter::from_bytes(&bytes)
            && r.len() == self.base.main_n as usize
        {
            self.ivf = Some(Box::new(r));
            self.refresh_zonemap();
        }
    }

    /// Load the persisted attribute column, if present + sized to the base.
    /// Best-effort: a missing / stale sidecar just leaves `attr` unset.
    fn load_attr(&mut self) {
        let Ok(bytes) = std::fs::read(self.dir.join(ATTR_FILE)) else {
            return;
        };
        let n = self.base.main_n as usize;
        if bytes.len() != n * 8 {
            return; // stale (base resized) - ignore
        }
        let attr: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.attr = Some(attr);
    }

    /// (Re)build the router's zone-map from the attribute column. Called after
    /// any router install and by [`set_attr`]; the zone-map is RAM-only (not in
    /// the IVF sidecar), so it must be rebuilt whenever either side changes.
    fn refresh_zonemap(&mut self) {
        if let (Some(r), Some(a)) = (self.ivf.as_mut(), self.attr.as_ref())
            && a.len() == r.len()
        {
            r.set_zonemap(a);
        }
    }

    /// Attach a u64 attribute column (one value per BASE row, in row order) for
    /// range-filtered search. Persists a sidecar and builds the router zone-map
    /// if the IVF router is present. The column is opaque - a time axis, an
    /// importance score, any monotone-comparable key.
    ///
    /// # Errors
    /// I/O error if the sidecar write fails.
    ///
    /// # Panics
    /// Panics if `attr.len()` does not equal the base row count.
    pub fn set_attr(&mut self, attr: &[u64]) -> io::Result<()> {
        assert_eq!(
            attr.len(),
            self.base.main_n as usize,
            "attr/base-rows mismatch"
        );
        let mut bytes = Vec::with_capacity(attr.len() * 8);
        for &a in attr {
            bytes.extend_from_slice(&a.to_le_bytes());
        }
        std::fs::write(self.dir.join(ATTR_FILE), &bytes)?;
        self.attr = Some(attr.to_vec());
        self.refresh_zonemap();
        Ok(())
    }

    /// Hybrid filtered search over the base. `s` = the filter's SORTED matching
    /// external ids. Planner by |s|: tiny -> exact quantized scan of `s`; larger
    /// -> IVF-routed shortlist (query-nearest S-cells) then quantized scan +
    /// f32 rerank. Falls back to a plain quantized scan of `s` if the router is
    /// not built. `rerank` = disk-read budget (e.g. k*8).
    ///
    /// # Errors
    /// I/O error if a re-rank read fails.
    pub fn search_filtered_hybrid(
        &self,
        query: &[f32],
        s: &[u64],
        k: usize,
        rerank: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        /// Below this many matches, an exact scan of `s` is cheaper than the
        /// IVF route. The compiled default was measured BEFORE the sdot
        /// kernels made scanning 3,6x cheaper, and the demo's default filter
        /// lands at ~12,3k ids per shard - exactly on this edge, so the
        /// router never engaged and every filtered query paid the full scan.
        /// `SKEG_HYBRID_SCAN_MAX` overrides while the crossover is
        /// re-measured on current kernels.
        fn scan_max() -> usize {
            static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *V.get_or_init(|| {
                std::env::var("SKEG_HYBRID_SCAN_MAX")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(12_288)
            })
        }
        /// Candidate budget the router narrows `s` down to before scoring.
        const SHORTLIST: usize = 4_096;
        if k == 0 || s.is_empty() {
            return Ok(Vec::new());
        }
        match &self.ivf {
            Some(router) if s.len() > scan_max() => {
                // Map external ids -> base rows (skip ids not in the base: delta
                // ids fall through to a direct scan of the whole `s`).
                let s_rows: Vec<u64> = s
                    .iter()
                    .filter_map(|id| self.base.id_to_main_row.get(id).map(|&r| u64::from(r)))
                    .collect();
                if s_rows.len() < s.len() {
                    // Some matches are outside the base (delta): can't route them
                    // reliably, so scan all of `s` exactly.
                    return self.score_ids_quantized(query, s, k, rerank);
                }
                // NOTE (bounded, transient): a match that IS in the base but was
                // re-inserted into a run/delta is routed here by its stale base
                // vector, so an updated id whose new vector is a true top-k hit
                // can be dropped from the shortlist (recall only - surviving ids
                // are still scored against their newest vector by
                // score_ids_quantized). Self-heals at the next consolidate, which
                // folds runs/delta and rebuilds the router.
                let q = normalized(query);
                let short_rows = router.probe(&q, &s_rows, SHORTLIST.max(rerank));
                let shortlist: Vec<u64> = short_rows
                    .iter()
                    .map(|&r| self.base.ids[r as usize])
                    .collect();
                self.score_ids_quantized(query, &shortlist, k, rerank)
            }
            _ => self.score_ids_quantized(query, s, k, rerank),
        }
    }

    /// Range-filtered search: top-k nearest to `query` among base rows whose
    /// attribute (set via [`set_attr`](Self::set_attr)) is in `[lo, hi]`.
    ///
    /// Self-routing on the estimated match count (O(cells), from the zone-map,
    /// no attribute scan): a WIDE range takes the zone-map path
    /// ([`probe_range`](IvfRouter::probe_range)) that never materialises the id
    /// set; a NARROW range materialises the matching ids and reuses
    /// [`search_filtered_hybrid`](Self::search_filtered_hybrid), whose own small-
    /// set path is already cheap. Both rerank through `score_ids_quantized`, so
    /// recall matches the filtered path. Requires a zone-map (attr + IVF router);
    /// without one it falls back to a full attribute scan + the filtered path.
    ///
    /// # Errors
    /// I/O error if a re-rank read fails.
    pub fn search_range(
        &self,
        query: &[f32],
        lo: u64,
        hi: u64,
        k: usize,
        rerank: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        /// Match-count crossover: above it, avoiding the id-set materialisation
        /// wins; below it, the filtered path's small-set scan is cheaper
        /// (measured ~= SHORTLIST, the router's candidate budget).
        const RANGE_MIN: usize = 4_096;
        const SHORTLIST: usize = 4_096;
        if k == 0 {
            return Ok(Vec::new());
        }
        let (Some(router), Some(attr)) = (&self.ivf, &self.attr) else {
            // No zone-map: fall back to a full scan into the filtered path.
            let s = self.ids_in_range(lo, hi);
            return self.search_filtered_hybrid(query, &s, k, rerank);
        };
        if router.estimate_range_count(lo, hi) >= RANGE_MIN {
            // Wide: zone-map path, no id set built.
            let q = normalized(query);
            let short_rows = router.probe_range(&q, lo, hi, attr, SHORTLIST.max(rerank));
            let ids: Vec<u64> = short_rows
                .iter()
                .map(|&r| self.base.ids[r as usize])
                .collect();
            self.score_ids_quantized(query, &ids, k, rerank)
        } else {
            // Narrow: materialise the matching ids, reuse the filtered path.
            let s = self.ids_in_range(lo, hi);
            self.search_filtered_hybrid(query, &s, k, rerank)
        }
    }

    /// TEST/BENCH: the range planner's estimate and whether `search_range` would
    /// take the zone-map (wide) path. Not part of the stable API.
    #[doc(hidden)]
    #[must_use]
    pub fn debug_range_plan(&self, lo: u64, hi: u64) -> (usize, bool) {
        match &self.ivf {
            Some(r) => {
                let est = r.estimate_range_count(lo, hi);
                (est, est >= 4_096)
            }
            None => (0, false),
        }
    }

    /// Sorted external ids of base rows whose attribute is in `[lo, hi]`. O(base).
    /// Empty if no attribute column is set.
    fn ids_in_range(&self, lo: u64, hi: u64) -> Vec<u64> {
        let Some(attr) = &self.attr else {
            return Vec::new();
        };
        let mut s: Vec<u64> = attr
            .iter()
            .enumerate()
            .filter(|&(_, &a)| (lo..=hi).contains(&a))
            .map(|(row, _)| self.base.ids[row])
            .collect();
        s.sort_unstable();
        s
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // test sizes are tiny
mod tests {

    /// A wrong dimension is a client mistake, not a reason to abort. The
    /// server used to pre-check it purely because these two entry points
    /// asserted, and a panic here takes down a shard thread that is serving
    /// every other vindex on it.
    #[test]
    fn wrong_dimension_is_an_error_not_a_panic() {
        let dir = tempfile::TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-dim");
        let dim = 16;
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.insert(1, &vec![0.5f32; dim]).unwrap();

        let short = idx.insert(2, &vec![0.5f32; dim - 1]).unwrap_err();
        assert_eq!(short.kind(), io::ErrorKind::InvalidInput);
        let long = idx.insert(3, &vec![0.5f32; dim + 1]).unwrap_err();
        assert_eq!(long.kind(), io::ErrorKind::InvalidInput);

        let bad_query = idx.search(&vec![0.5f32; dim + 4], 5).unwrap_err();
        assert_eq!(bad_query.kind(), io::ErrorKind::InvalidInput);

        // The index is still usable: the rejected calls changed nothing.
        assert!(idx.get(1).unwrap().is_some());
        assert_eq!(idx.search(&vec![0.5f32; dim], 5).unwrap().len(), 1);
    }

    /// The documented precedence is `delta > flushing > runs > base`. The
    /// synchronous `consolidate` collected survivors from `delta`, `runs` and
    /// `base` only, so anything staged in `flushing` by an in-flight
    /// `flush_begin` was outside the fold. `consolidate` then truncates the WAL
    /// and reopens from disk, which drops the in-memory staging map.
    ///
    /// The window is reachable: `off_thread_maintenance` releases the write
    /// lock while the flush builds off-thread, and the client command
    /// `SKEG.VINDEX.CONSOLIDATE` takes that lock and calls this exact path.
    #[test]
    fn consolidate_keeps_vectors_staged_by_an_in_flight_flush() {
        let dir = tempfile::TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-flush-consolidate");
        let dim = 16;
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        let vec_for =
            |id: u64| -> Vec<f32> { (0..dim).map(|d| (id as f32 + d as f32) / 100.0).collect() };
        // A non-empty base first: otherwise the fold finds no survivors at all
        // and `consolidate` returns early without truncating the WAL, which
        // walks past the window this test is about.
        for id in 0u64..200 {
            idx.insert(id, &vec_for(id)).unwrap();
        }
        idx.consolidate().unwrap();
        assert!(
            idx.get(0).unwrap().is_some(),
            "base did not survive its fold"
        );

        // Now stage a second batch for a flush, and consolidate before the
        // flush lands. The staged ids live only in `flushing` and the WAL.
        for id in 200u64..400 {
            idx.insert(id, &vec_for(id)).unwrap();
        }
        let job = idx.flush_begin().unwrap().expect("delta is non-empty");
        idx.consolidate().unwrap();
        // Every id must still be retrievable. Before the fix the fold missed
        // them, the WAL was truncated, and the reopen dropped the staging map.
        for id in 0u64..400 {
            assert!(
                idx.get(id).unwrap().is_some(),
                "id {id} was staged for flush and lost by consolidate",
            );
        }
        drop(job);
    }
    use super::*;
    use ordered_float::OrderedFloat;

    /// The route decision, pinned at its measured boundaries. The dead-fraction
    /// guard exists because its absence cost 407-475s per shard in production:
    /// a 39%-dead base took the patched route and lost to the rebuild, exactly
    /// where the delete-patch verdict said the bridge+re-prune regime loses
    /// (crossover 20-25%). 20% is inside the winning region, not on the edge.
    #[test]
    fn fold_route_honours_the_measured_boundaries() {
        // Empty base: nothing to reuse.
        assert!(!route_from_shape(100, 0, 0));
        // Growth-only, healthy base: patched.
        assert!(route_from_shape(50_000, 100_000, 100_000));
        // New mass above the live base: full.
        assert!(!route_from_shape(150_000, 100_000, 100_000));
        // Dead fraction at the guard (20%): still patched.
        assert!(route_from_shape(10_000, 80_000, 100_000));
        // Just past it: full. This is the case that shipped slow.
        assert!(!route_from_shape(10_000, 79_000, 100_000));
        // The production incident's shape: 458k base rows, ~180k of them
        // rewritten and therefore dead (~39%). Must be full.
        assert!(!route_from_shape(180_000, 278_000, 458_000));
        // The same shape as one shard saw it.
        assert!(!route_from_shape(22_000, 34_000, 56_000));
    }

    /// The invariant a fold must never break, whichever route builds it: every
    /// live id stays findable by its own vector, and a deleted id never comes
    /// back. This is the harness that has to stay green when the fold flips
    /// from "rebuild everything" to "keep the base edges, insert what is new".
    ///
    /// A wrong graph fails this loudly: a lost node stops answering for
    /// itself, and a resurrected tombstone shows up in someone's results.
    #[test]
    fn incremental_fold_keeps_every_id_findable_and_no_resurrection() {
        let dir = tempfile::TempDir::new().unwrap();
        let vdir = dir.path().join("vindex-patched-fold");
        let dim = 16;
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &vdir,
            dim,
            200,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        // Deterministic, well-spread vectors: self-search must return self.
        let vec_for = |id: u64| -> Vec<f32> {
            use rand::Rng;
            let mut rng = StdRng::seed_from_u64(0xF01D ^ id);
            (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
        };

        // First fold: empty base, so this exercises the full-build route and
        // gives the second fold a base worth reusing.
        for id in 0u64..2000 {
            idx.insert(id, &vec_for(id)).unwrap();
        }
        let job = idx.consolidate_begin().unwrap().expect("work to fold");
        let built = job.build(&vdir).unwrap();
        assert!(!built.patched, "an empty base must take the full route");
        idx.consolidate_finish(built).unwrap();

        // Churn: delete some base ids, add new ones. The new mass is well under
        // the live base, which is the regime the cheap route is for.
        for id in (0u64..2000).step_by(40) {
            assert!(idx.delete(id).unwrap(), "delete of {id} did not land");
        }
        for id in 2000u64..2600 {
            idx.insert(id, &vec_for(id)).unwrap();
        }
        let job = idx.consolidate_begin().unwrap().expect("work to fold");
        let built = job.build(&vdir).unwrap();
        assert!(
            built.patched,
            "600 new rows against 1950 live base rows must take the patched route"
        );
        idx.consolidate_finish(built).unwrap();

        // Every live id answers for itself...
        let mut misses = Vec::new();
        for id in 0u64..2600 {
            let deleted = id < 2000 && id % 40 == 0;
            if deleted {
                continue;
            }
            let hits = idx.search(&vec_for(id), 1).unwrap();
            if hits.first().map(|&(hid, _)| hid) != Some(id) {
                misses.push(id);
            }
        }
        assert!(
            misses.is_empty(),
            "{} ids no longer answer for themselves after the fold; first: {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
        // ...and no tombstone rises.
        for id in (0u64..2000).step_by(40) {
            let hits = idx.search(&vec_for(id), 5).unwrap();
            assert!(
                hits.iter().all(|&(hid, _)| hid != id),
                "deleted id {id} came back through the fold"
            );
        }

        // Recall@10 against brute force on perturbed queries. Honest scope
        // note: at dim 16 this does NOT falsify a degenerate build. A mutation
        // that skipped inserting the new points entirely still passed, because
        // `patch_connectivity` attaches each stranded node to its exact
        // nearest neighbour and in low dimension that single edge is enough
        // for full recall. Structural quality is guarded by the recall gate on
        // real 100-dim embeddings (bench/inplace_gate.py), not here; this
        // check only catches gross breakage.
        let live: Vec<u64> = (0u64..2600)
            .filter(|id| !(*id < 2000 && id % 40 == 0))
            .collect();
        let mut total_hits = 0usize;
        let mut queries = 0usize;
        for probe in (0..2600u64).step_by(26) {
            let mut q = vec_for(probe);
            for (j, v) in q.iter_mut().enumerate() {
                *v += ((probe as f32 + j as f32).sin()) * 0.05;
            }
            let mut truth: Vec<(OrderedFloat<f32>, u64)> = live
                .iter()
                .map(|&id| (OrderedFloat(dist(&q, &vec_for(id))), id))
                .collect();
            truth.sort_unstable();
            let want: AHashSet<u64> = truth[..10].iter().map(|&(_, id)| id).collect();
            let got = idx.search(&q, 10).unwrap();
            total_hits += got.iter().filter(|&&(id, _)| want.contains(&id)).count();
            queries += 1;
        }
        let recall = total_hits as f64 / (queries * 10) as f64;
        assert!(
            recall >= 0.95,
            "recall@10 vs brute force is {recall:.3} after the patched fold; \
             the graph is navigable but structurally degraded"
        );
    }

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    /// Row `id` of a row-major store. A test helper: production code reads
    /// rows through [`VectorSource`].
    fn row(vectors: &[f32], id: u32, dim: usize) -> &[f32] {
        let start = id as usize * dim;
        &vectors[start..start + dim]
    }

    fn brute_force(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u64> {
        let n = vectors.len() / dim;
        let mut scored: Vec<(OrderedFloat<f32>, u64)> = (0..n)
            .map(|i| {
                (
                    OrderedFloat(dist(query, row(vectors, i as u32, dim))),
                    i as u64,
                )
            })
            .collect();
        scored.sort_unstable();
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[test]
    fn vector_writer_preserves_header_and_row_major_payload() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let values = vec![1.25f32, -2.5, 0.0, 3.75, 4.5, -6.25];
        let source = InMemoryVectorSource::new(values.clone(), 3);
        write_vectors_bin(&path, &source).unwrap();

        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            bytes.len(),
            HEADER_LEN + values.len() * std::mem::size_of::<f32>()
        );
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            VEC_MAGIC
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            FORMAT_VERSION
        );
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        let payload: Vec<f32> = bytes[HEADER_LEN..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(payload, values);
    }

    #[test]
    fn sequential_row_reader_preserves_a_multi_block_range() {
        let dim = 7;
        let n = 80_000;
        let values: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.25).collect();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let source = InMemoryVectorSource::new(values.clone(), dim);
        write_vectors_bin(&path, &source).unwrap();
        let file = File::open(path).unwrap();

        let start = 123usize;
        let rows = 60_000usize;
        let got = read_f32_rows_sequential(&file, start, rows, dim).unwrap();
        assert_eq!(got, values[start * dim..(start + rows) * dim]);
    }

    #[test]
    fn sequential_row_reader_rejects_a_zero_dimension() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("vectors.bin");
        std::fs::write(&path, []).unwrap();
        let file = File::open(path).unwrap();

        let error = read_f32_rows_sequential(&file, 0, 1, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn build_100_vectors_fully_connected() {
        let dim = 48;
        let n = 100;
        let vectors = random_vectors(n, dim, 1);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());

        // BFS from the medoid must reach every node (patch_connectivity guarantees it).
        let (_, count) = reachable_from_medoid(&index.nodes, index.n, index.medoid);
        assert_eq!(
            count, index.n,
            "graph must be fully reachable from the medoid"
        );
    }

    #[test]
    fn search_recall_at_10() {
        let dim = 64;
        let n = 1000;
        let vectors = random_vectors(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());

        let mut rng = StdRng::seed_from_u64(777);
        let mut hits = 0;
        let mut total = 0;
        for _ in 0..50 {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let want = brute_force(&vectors, dim, &query, 10);
            let got: Vec<u64> = index
                .search(&query, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.98,
            "Vamana recall@10 = {recall:.4} (target >= 0.98)"
        );
    }

    #[test]
    fn robust_prune_excludes_self() {
        let dim = 16;
        let vectors = random_vectors(20, dim, 3);
        // candidates deliberately include p itself.
        let mut cand: Vec<(f32, VecId)> = (0..20u32)
            .map(|id| (dist(row(&vectors, 5, dim), row(&vectors, id, dim)), id))
            .collect();
        let src = InMemoryVectorSource::new(vectors, dim);
        let result = robust_prune(5, &mut cand, 1.2, 8, &src);
        assert!(
            !result.contains(&5),
            "a node must never be its own neighbour"
        );
    }

    #[test]
    fn robust_prune_satisfies_alpha_condition() {
        // With a non-binding `r` (>= candidate count) every candidate is either
        // selected or pruned by the alpha rule - never just cut off by the
        // degree limit. So a pruned candidate must be dominated by some
        // selected neighbour: alpha * d(p*, v) <= d(p, v).
        let dim = 24;
        let n = 60u32;
        let vectors = random_vectors(n as usize, dim, 9);
        let p = 0u32;
        let alpha = 1.2;
        let mut cand: Vec<(f32, VecId)> = (1..n)
            .map(|id| (dist(row(&vectors, p, dim), row(&vectors, id, dim)), id))
            .collect();
        let original: Vec<(f32, VecId)> = cand.clone();
        let src = InMemoryVectorSource::new(vectors.clone(), dim);
        let result = robust_prune(p, &mut cand, alpha, MAX_R, &src);

        for &(d_pv, v) in &original {
            if result.contains(&v) {
                continue;
            }
            let dominated = result
                .iter()
                .any(|&star| alpha * dist(row(&vectors, star, dim), row(&vectors, v, dim)) <= d_pv);
            assert!(
                dominated,
                "pruned candidate {v} not dominated by any neighbour"
            );
        }
    }

    #[test]
    fn greedy_search_terminates_on_any_graph() {
        // A random graph that never went through the build still terminates.
        let dim = 16;
        let n = 80;
        let vectors = random_vectors(n, dim, 5);
        let mut nodes = vec![Node::new(); n];
        init_random_graph(&mut nodes, n as u32, 12, 99);
        let query = vec![0.3f32; dim];
        let mut visited = VisitedBitset::new(n);
        let mut seen = VisitedBitset::new(n);
        let list = greedy_search(
            &[0],
            50,
            None,
            |id| dist(&query, row(&vectors, id, dim)),
            |id| nodes[id as usize].slice().iter().copied().collect(),
            None,
            &mut visited,
            &mut seen,
            None,
        );
        assert!(list.iter().count() > 0, "search must return candidates");
        assert!(
            visited.iter().count() <= n,
            "visited set bounded by node count"
        );
    }

    #[test]
    fn graph_file_roundtrip() {
        let dim = 32;
        let n = 300;
        let vectors = random_vectors(n, dim, 11);
        let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 1).collect(); // non-trivial ids
        let index = VamanaIndex::build(vectors, ids.clone(), dim, &VamanaConfig::default());

        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        assert_eq!(disk.len(), index.len());
        assert_eq!(disk.dim(), dim);
        assert_eq!(disk.base.medoid, index.medoid);
        assert_eq!(disk.base.ids, ids);
        for (a, b) in disk.base.nodes.iter().zip(index.nodes.iter()) {
            assert_eq!(a.degree, b.degree, "node degree must survive the roundtrip");
            assert_eq!(
                a.slice(),
                b.slice(),
                "node edges must survive the roundtrip"
            );
        }
    }

    #[test]
    fn disk_search_recall_at_10() {
        let dim = 64;
        let n = 1000;
        let vectors = random_vectors(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());

        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        let mut rng = StdRng::seed_from_u64(777);
        let mut hits = 0;
        let mut total = 0;
        for _ in 0..50 {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let want = brute_force(&vectors, dim, &query, 10);
            let got: Vec<u64> = disk
                .search(&query, 10)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.95,
            "on-disk Vamana recall@10 = {recall:.4} (target >= 0.95)"
        );
    }

    // Range-filtered search over the disk index: every result honours [lo,hi],
    // recall matches a brute filtered top-k, and the attr column + zone-map
    // survive a save/reopen. ids == row so the range predicate is exact.
    #[test]
    fn disk_search_range() {
        let dim = 48;
        let n = 4000; // > RANGE_MIN so the zone-map path engages for a wide range
        let vectors = random_vectors(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();

        let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();
        let attr: Vec<u64> = (0..n as u64).collect(); // attr[row] = row = id
        disk.set_attr(&attr).unwrap();
        disk.build_ivf(0, 6).unwrap();

        let check = |d: &DiskVamanaIndex, lo: u64, hi: u64| {
            let mut rng = StdRng::seed_from_u64(9);
            let (mut hits, mut total) = (0usize, 0usize);
            for _ in 0..30 {
                let q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let got: Vec<u64> = d
                    .search_range(&q, lo, hi, 10, 80)
                    .unwrap()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                assert!(
                    got.iter().all(|&id| (lo..=hi).contains(&id)),
                    "result out of [{lo},{hi}]"
                );
                // brute filtered truth
                let mut truth: Vec<(f32, u64)> = (lo..=hi)
                    .map(|id| {
                        (
                            cosine_f32(&q, &vectors[id as usize * dim..id as usize * dim + dim]),
                            id,
                        )
                    })
                    .collect();
                truth.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                let want: std::collections::HashSet<u64> =
                    truth.iter().take(10).map(|&(_, id)| id).collect();
                hits += got.iter().filter(|id| want.contains(id)).count();
                total += want.len().min(10);
            }
            hits as f64 / total as f64
        };

        // Wide range (zone-map path) and narrow range (materialise path).
        assert!(check(&disk, 1000, 3500) >= 0.9, "wide-range recall");
        assert!(check(&disk, 100, 260) >= 0.9, "narrow-range recall");

        // Persistence: reopen must reload attr + rebuild the zone-map.
        drop(disk);
        let disk2 = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert!(disk2.attr.is_some(), "attr column must survive reopen");
        assert!(
            check(&disk2, 1000, 3500) >= 0.9,
            "wide-range recall after reopen"
        );
    }

    // RW disk index with a TurboQuant tier: the live write path (create -> insert
    // via delta -> consolidate) rebuilds a tq2 tier, the kind survives a reopen
    // (persisted in tier.kind), search recall holds, and the tier is leaner in RAM
    // than int8. This is "lean live writes": sub-int8 RAM WITHOUT a trained codebook.
    #[test]
    fn disk_rw_turboquant_tier() {
        let dim = 64;
        let n = 2000;
        let vectors = random_vectors(n, dim, 42);
        let tmp = tempfile::TempDir::new().unwrap();

        let mut tq = DiskVamanaIndex::create_empty_with_tier(
            tmp.path(),
            dim,
            100,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        for id in 0..n {
            tq.insert(id as u64, row(&vectors, id as u32, dim)).unwrap();
        }
        tq.consolidate().unwrap();
        // tier.kind persisted: a fresh `open` (no explicit tier) rebuilds tq2.
        drop(tq);
        let tq = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(tq.len(), n, "all vectors live after RW tq build");

        let mut rng = StdRng::seed_from_u64(7);
        let mut hits = 0;
        let mut total = 0;
        for _ in 0..50 {
            let q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let want = brute_force(&vectors, dim, &q, 10);
            let got: Vec<u64> = tq
                .search(&q, 10)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.90,
            "tq2 RW recall@10 = {recall:.4} (target >= 0.90)"
        );

        // The tq2 tier (dim/4 bytes/vec) is leaner in RAM than int8 (dim bytes/vec).
        let i8dir = tmp.path().join("i8");
        let mut i8 =
            DiskVamanaIndex::create_empty_with_tier(&i8dir, dim, 100, QuantKind::Int8).unwrap();
        for id in 0..n {
            i8.insert(id as u64, row(&vectors, id as u32, dim)).unwrap();
        }
        i8.consolidate().unwrap();
        assert!(
            tq.resident_bytes() < i8.resident_bytes(),
            "tq2 RAM {} should be < int8 RAM {}",
            tq.resident_bytes(),
            i8.resident_bytes()
        );
    }

    // Filtered walk recall: against the exact top-10 over the matching subset
    // (the ground truth), the oversampled filtered walk recovers >= 0.90 at a
    // low-selectivity (~50%) filter, and never returns a non-matching id.
    #[test]
    fn disk_filtered_search_recall() {
        let dim = 64;
        let n = 1000;
        let vectors = random_vectors(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        let matches = |id: u64| id % 2 == 0; // even ids, ~50% selectivity
        // A handful of matching ids spread across the set, used as walk seeds.
        let seeds: Vec<u64> = (0..n as u64)
            .filter(|&id| matches(id))
            .step_by(64)
            .collect();
        let mut rng = StdRng::seed_from_u64(777);
        let mut hits = 0;
        let mut total = 0;
        for _ in 0..50 {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            // Ground truth: exact top-10 cosine over just the matching ids.
            let mut scored: Vec<(f32, u64)> = (0..n as u64)
                .filter(|&id| matches(id))
                .map(|id| (cosine_f32(&query, row(&vectors, id as u32, dim)), id))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let want: Vec<u64> = scored.iter().take(10).map(|&(_, id)| id).collect();

            let got: Vec<u64> = disk
                .search_filtered(&query, 10, 0, &matches, &seeds, 0.5)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert!(
                got.iter().all(|&id| matches(id)),
                "filtered walk returned a non-matching id"
            );
            hits += got.iter().filter(|id| want.contains(id)).count();
            total += want.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.90,
            "filtered walk recall@10 = {recall:.4} (target >= 0.90)"
        );
    }

    #[test]
    fn disk_search_finds_exact_match() {
        let dim = 48;
        let n = 500;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        let query = row(&vectors, 222, dim).to_vec();
        let hits = disk.search(&query, 5).unwrap();
        assert_eq!(hits[0].0, 222, "exact match must rank first after re-rank");
        assert!((hits[0].1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn search_finds_exact_match() {
        let dim = 32;
        let n = 400;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        // Query equal to stored vector #137 must return it at cosine ~1.
        let query = row(&vectors, 137, dim).to_vec();
        let hits = index.search(&query, 5);
        assert_eq!(hits[0].0, 137, "exact match must rank first");
        assert!(
            (hits[0].1 - 1.0).abs() < 1e-4,
            "self-cosine ~1, got {}",
            hits[0].1
        );
    }

    #[test]
    fn disk_streaming_insert_searchable() {
        let dim = 48;
        let n = 400;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        // A vector inserted after open lands in the delta and must be found.
        let newv: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 0.9 } else { -0.9 })
            .collect();
        disk.insert(9999, &newv).unwrap();
        assert_eq!(disk.len(), n + 1);
        assert_eq!(disk.delta_len(), 1);
        let hits = disk.search(&newv, 1).unwrap();
        assert_eq!(hits[0].0, 9999, "the freshly inserted vector must be found");
        assert!((hits[0].1 - 1.0).abs() < 1e-4);
        assert_eq!(disk.get(9999).unwrap().as_deref(), Some(newv.as_slice()));
    }

    #[test]
    fn disk_delete_tombstone_filtered() {
        let dim = 32;
        let n = 300;
        let vectors = random_vectors(n, dim, 2);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        assert!(disk.delete(42).unwrap(), "delete of a live id returns true");
        assert!(!disk.delete(42).unwrap(), "second delete returns false");
        assert_eq!(disk.len(), n - 1);
        let query = row(&vectors, 42, dim).to_vec();
        let hits = disk.search(&query, 10).unwrap();
        assert!(
            hits.iter().all(|&(id, _)| id != 42),
            "tombstoned id must not appear"
        );
        assert!(
            disk.get(42).unwrap().is_none(),
            "get of a deleted id -> None"
        );
    }

    #[test]
    fn disk_consolidate_preserves_live_set() {
        let dim = 40;
        let n = 300;
        let vectors = random_vectors(n, dim, 5);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();
        let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();

        // Insert 50 fresh vectors, delete 30 old ones, then consolidate.
        let extra = random_vectors(50, dim, 99);
        for j in 0..50usize {
            disk.insert(10_000 + j as u64, &extra[j * dim..(j + 1) * dim])
                .unwrap();
        }
        for id in 0..30u64 {
            disk.delete(id).unwrap();
        }
        let before = disk.len();
        assert_eq!(before, n + 50 - 30);

        disk.consolidate().unwrap();
        assert_eq!(disk.len(), before, "consolidation preserves the live count");
        assert_eq!(disk.delta_len(), 0, "delta is empty after consolidation");

        // An inserted vector survives consolidation; a deleted id stays gone.
        let hits = disk.search(&extra[0..dim], 1).unwrap();
        assert_eq!(hits[0].0, 10_000, "inserted vector survives consolidation");
        let deleted = disk.search(row(&vectors, 5, dim), 10).unwrap();
        assert!(
            deleted.iter().all(|&(id, _)| id != 5),
            "deleted id stays gone"
        );
    }

    #[test]
    fn disk_open_with_pq_tier_searches() {
        // dim divisible by m. The PQ tier drives the walk; the f32 re-rank
        // then puts an exact-match query (a corpus vector, cosine 1.0) top-1.
        let dim = 64;
        let n = 1500;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();

        let disk =
            DiskVamanaIndex::open_with_tier(tmp.path(), QuantKind::Pq { m: 16, k: 64 }).unwrap();
        assert_eq!(disk.len(), n);

        let probes: Vec<usize> = (0..n).step_by(50).collect();
        let hits = probes
            .iter()
            .filter(|&&q| {
                let top = disk.search(row(&vectors, q as u32, dim), 1).unwrap();
                top.first().map(|&(id, _)| id) == Some(q as u64)
            })
            .count();
        assert!(
            hits + 2 >= probes.len(),
            "PQ-tier walk found {hits}/{} exact matches",
            probes.len()
        );
    }

    #[test]
    fn disk_open_with_turboquant_tier_searches() {
        // Same shape as the PQ tier integration test, but the walk is driven
        // by the TurboQuant proxy (4-bit). Top-1 still recovers exact matches
        // for corpus vectors after the f32 re-rank.
        let dim = 64;
        let n = 1500;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors.clone(), ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();

        for bits in [1u8, 2, 4] {
            let disk = DiskVamanaIndex::open_with_tier(tmp.path(), QuantKind::TurboQuant { bits })
                .unwrap();
            assert_eq!(disk.len(), n);
            let probes: Vec<usize> = (0..n).step_by(50).collect();
            let hits = probes
                .iter()
                .filter(|&&q| {
                    let top = disk.search(row(&vectors, q as u32, dim), 1).unwrap();
                    top.first().map(|&(id, _)| id) == Some(q as u64)
                })
                .count();
            // Allow 3 misses across 30 probes for the 1-bit tier (the
            // coarsest); 4-bit lands near-perfect.
            let allow = if bits == 1 { 3 } else { 2 };
            assert!(
                hits + allow >= probes.len(),
                "TurboQuant {bits}-bit walk found {hits}/{} exact matches",
                probes.len()
            );
        }
    }

    #[test]
    fn disk_reopen_recovers_delta_via_wal() {
        let dim = 48;
        let n = 200;
        let vectors = random_vectors(n, dim, 3);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();

        // Open, stream-insert a vector and delete a main id, then drop the
        // index - the in-RAM delta is gone, only the WAL on disk remains.
        let v_a: Vec<f32> = (0..dim)
            .map(|i| if i % 3 == 0 { 0.8 } else { -0.2 })
            .collect();
        {
            let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();
            disk.insert(5000, &v_a).unwrap();
            assert!(disk.delete(7).unwrap());
            assert_eq!(disk.len(), n); // +1 insert, -1 delete
        }

        // Reopen: the WAL replay must restore the streamed insert and delete.
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(disk.len(), n, "live count restored after WAL replay");
        assert_eq!(
            disk.get(5000).unwrap().as_deref(),
            Some(v_a.as_slice()),
            "streamed insert recovered from the WAL",
        );
        assert!(
            disk.get(7).unwrap().is_none(),
            "delete recovered from the WAL"
        );
        let hits = disk.search(&v_a, 1).unwrap();
        assert_eq!(hits[0].0, 5000, "the WAL-recovered vector is searchable");
    }

    #[test]
    fn disk_reopen_rejects_corrupt_framed_wal_record() {
        let tmp = tempfile::TempDir::new().unwrap();
        let wal_path = tmp.path().join(DELTA_LOG_FILE);
        {
            let mut disk = DiskVamanaIndex::create_empty(tmp.path(), 4, 64).unwrap();
            disk.insert(7, &[1.0; 4]).unwrap();
            disk.insert(8, &[2.0; 4]).unwrap();
        }
        let mut wal = std::fs::read(&wal_path).unwrap();
        assert!(
            wal.starts_with(b"SKWL\x02"),
            "new vector WALs are framed and versioned"
        );
        // Corrupt the first record; id 8 must not be silently skipped.
        wal[DELTA_WAL_V2_MAGIC.len() + 9] ^= 0x01;
        std::fs::write(wal_path, wal).unwrap();

        let Err(err) = DiskVamanaIndex::open(tmp.path()) else {
            panic!("corrupt framed WAL must refuse recovery");
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn disk_reopen_ignores_only_torn_framed_wal_tail() {
        let tmp = tempfile::TempDir::new().unwrap();
        let wal_path = tmp.path().join(DELTA_LOG_FILE);
        {
            let mut disk = DiskVamanaIndex::create_empty(tmp.path(), 4, 64).unwrap();
            disk.insert(1, &[1.0; 4]).unwrap();
            disk.insert(2, &[2.0; 4]).unwrap();
        }
        let mut wal = std::fs::read(&wal_path).unwrap();
        assert!(wal.starts_with(b"SKWL\x02"));
        wal.truncate(wal.len() - 3);
        std::fs::write(wal_path, wal).unwrap();

        let reopened = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert!(
            reopened.get(1).unwrap().is_some(),
            "completed record survives"
        );
        assert!(reopened.get(2).unwrap().is_none(), "torn tail is ignored");
    }

    #[test]
    fn disk_reopen_accepts_legacy_wal_and_upgrades_on_consolidate() {
        let tmp = tempfile::TempDir::new().unwrap();
        let wal_path = tmp.path().join(DELTA_LOG_FILE);
        {
            let _disk = DiskVamanaIndex::create_empty(tmp.path(), 4, 64).unwrap();
        }
        let mut legacy = vec![0];
        legacy.extend_from_slice(&7u64.to_le_bytes());
        for x in [1.0f32; 4] {
            legacy.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(&wal_path, legacy).unwrap();

        let mut reopened = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(reopened.get(7).unwrap(), Some(vec![1.0; 4]));
        reopened.consolidate().unwrap();
        assert!(
            std::fs::read(wal_path)
                .unwrap()
                .starts_with(DELTA_WAL_V2_MAGIC),
            "a successful consolidate migrates the legacy WAL"
        );
    }

    // Insert past the L0 flush threshold so at least one run is built; returns
    // the corpus so callers can probe specific vectors.
    fn fill_past_flush(disk: &mut DiskVamanaIndex, n: usize, dim: usize) -> Vec<f32> {
        let vectors = random_vectors(n, dim, 7);
        for id in 0..n {
            disk.insert(id as u64, &vectors[id * dim..(id + 1) * dim])
                .unwrap();
        }
        vectors
    }

    #[test]
    fn disk_flush_builds_a_run_and_stays_searchable() {
        let (dim, n) = (16, 5000); // > FLUSH (4096): one flush fires
        let tmp = tempfile::TempDir::new().unwrap();
        let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        let vectors = fill_past_flush(&mut disk, n, dim);

        assert!(!disk.runs.is_empty(), "a flush must have built a run");
        assert!(disk.delta.len() < 4096, "L0 stayed bounded after flushing");
        assert_eq!(disk.len(), n, "every inserted vector is live");
        let q = &vectors[100 * dim..101 * dim];
        assert_eq!(
            disk.search(q, 1).unwrap()[0].0,
            100,
            "a flushed vector is its own NN"
        );
    }

    #[test]
    fn disk_consolidate_folds_runs_into_base() {
        let (dim, n) = (16, 5000);
        let tmp = tempfile::TempDir::new().unwrap();
        let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        let vectors = fill_past_flush(&mut disk, n, dim);
        disk.consolidate().unwrap();

        assert!(disk.runs.is_empty(), "consolidate clears the runs");
        assert_eq!(
            disk.base.main_n as usize, n,
            "all vectors folded into the base"
        );
        assert_eq!(disk.len(), n);
        let run_dirs = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
            .count();
        assert_eq!(run_dirs, 0, "run directories are removed once folded in");
        let q = &vectors[4500 * dim..4501 * dim];
        assert_eq!(
            disk.search(q, 1).unwrap()[0].0,
            4500,
            "search exact after consolidate"
        );
    }

    #[test]
    fn disk_reopen_after_flush_recovers_via_wal() {
        let (dim, n) = (16, 5000);
        let tmp = tempfile::TempDir::new().unwrap();
        let vectors = {
            let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
            let v = fill_past_flush(&mut disk, n, dim);
            assert!(!disk.runs.is_empty(), "flushed before drop");
            v
        }; // dropped: in-RAM runs gone, only the WAL + stale run dirs on disk

        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(disk.len(), n, "WAL replay restores every flushed vector");
        let q = &vectors[100 * dim..101 * dim];
        assert_eq!(
            disk.search(q, 1).unwrap()[0].0,
            100,
            "recovered vector is searchable"
        );
        let stale = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
            .count();
        assert_eq!(stale, 0, "stale run dirs are cleaned on open");
    }

    #[test]
    fn disk_wal_cleared_after_consolidation() {
        let dim = 32;
        let n = 150;
        let vectors = random_vectors(n, dim, 8);
        let ids: Vec<u64> = (0..n as u64).collect();
        let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());
        let tmp = tempfile::TempDir::new().unwrap();
        index.save(tmp.path()).unwrap();

        let mut disk = DiskVamanaIndex::open(tmp.path()).unwrap();
        for j in 0..20u64 {
            disk.insert(9000 + j, &random_vectors(1, dim, 100 + j))
                .unwrap();
        }
        disk.consolidate().unwrap();
        // Consolidation leaves only the V2 header.
        let wal = std::fs::read(tmp.path().join("delta.log")).unwrap();
        assert_eq!(
            wal, DELTA_WAL_V2_MAGIC,
            "consolidation leaves a V2 WAL header with no records"
        );
        let reopened = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(reopened.len(), n + 20);
        assert_eq!(reopened.delta_len(), 0, "reopen replays an empty WAL");
    }

    #[test]
    fn disk_empty_index_searches_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let disk = DiskVamanaIndex::create_empty(tmp.path(), 8, 64).unwrap();
        assert!(disk.is_empty());
        assert_eq!(disk.len(), 0);
        assert!(disk.search(&[0.0; 8], 5).unwrap().is_empty());
        assert!(disk.get(123).unwrap().is_none());
    }

    #[test]
    fn disk_insert_overwrite_returns_latest() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        let v1 = vec![1.0f32; dim];
        let mut v2 = vec![-1.0f32; dim];
        v2[0] = 1.0;
        disk.insert(1, &v1).unwrap();
        disk.insert(1, &v2).unwrap(); // overwrite same id
        assert_eq!(disk.len(), 1, "an overwrite does not change the live count");
        assert_eq!(disk.get(1).unwrap(), Some(v2));
    }

    #[test]
    fn disk_delete_then_reinsert_resurrects() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        let v = vec![0.5f32; dim];
        disk.insert(7, &v).unwrap();
        assert!(disk.delete(7).unwrap());
        assert_eq!(disk.len(), 0);
        disk.insert(7, &v).unwrap(); // resurrect a tombstoned id
        assert_eq!(disk.len(), 1);
        assert!(disk.get(7).unwrap().is_some());
    }

    #[test]
    fn disk_wal_tolerates_truncated_tail() {
        let dim = 12;
        let tmp = tempfile::TempDir::new().unwrap();
        {
            let mut disk = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
            disk.insert(1, &vec![0.1f32; dim]).unwrap();
            disk.insert(2, &vec![0.2f32; dim]).unwrap();
        }
        // Simulate a crash mid-append: drop the last 5 bytes of the WAL.
        let log = tmp.path().join("delta.log");
        let bytes = std::fs::read(&log).unwrap();
        std::fs::write(&log, &bytes[..bytes.len() - 5]).unwrap();
        // Reopen: the intact record replays, the truncated one is dropped,
        // and crucially there is no panic.
        let disk = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert!(
            disk.get(1).unwrap().is_some(),
            "the intact WAL record replays"
        );
        assert!(
            disk.len() <= 2,
            "a truncated trailing record is skipped, not fatal"
        );
    }

    #[test]
    fn search_k_larger_than_n_is_clamped() {
        let dim = 24;
        let vectors = random_vectors(10, dim, 4);
        let ids: Vec<u64> = (0..10u64).collect();
        let index = VamanaIndex::build(vectors, ids, dim, &VamanaConfig::default());
        let hits = index.search(&vec![0.2f32; dim], 100);
        assert!(hits.len() <= 10, "k > n returns at most n results");
    }

    /// The background split (begin/build/finish) must land on the same state as
    /// the inline consolidate for the same operation history.
    #[test]
    fn background_consolidate_matches_inline() {
        let dim = 16;
        let n = 300;
        let vecs = random_vectors(n, dim, 11);
        let mk = |tmp: &std::path::Path| {
            let mut idx = DiskVamanaIndex::create_empty(tmp, dim, 64).unwrap();
            for (i, v) in vecs.chunks_exact(dim).enumerate() {
                idx.insert(i as u64, v).unwrap();
            }
            for id in 0..40u64 {
                idx.delete(id * 3).unwrap(); // scattered deletes
            }
            idx
        };
        let t_inline = tempfile::TempDir::new().unwrap();
        let mut inline = mk(t_inline.path());
        inline.consolidate().unwrap();

        let t_bg = tempfile::TempDir::new().unwrap();
        let mut bg = mk(t_bg.path());
        let job = bg.consolidate_begin().unwrap().expect("non-empty");
        let built = job.build(t_bg.path()).unwrap();
        bg.consolidate_finish(built).unwrap();

        assert_eq!(bg.len(), inline.len(), "same live count");
        assert_eq!(bg.run_count(), 0, "runs folded");
        assert_eq!(bg.delta_len(), 0, "no post-begin writes -> empty delta");
        let q = &vecs[..dim];
        let a = inline.search(q, 10).unwrap();
        let b = bg.search(q, 10).unwrap();
        assert_eq!(
            a.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            b.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "same top-k"
        );
    }

    /// Writes that race the background build (inserts, deletes of snapshot ids,
    /// deletes of post-begin ids) must all survive the finish swap, in RAM and
    /// across a restart.
    #[test]
    fn writes_during_background_build_survive() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        let vecs = random_vectors(200, dim, 7);
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        let job = idx.consolidate_begin().unwrap().expect("non-empty");

        // Race the build: new inserts, a delete of a folded id, and an
        // insert-then-delete entirely inside the window.
        idx.insert(1000, &vec![0.25f32; dim]).unwrap();
        idx.insert(1001, &vec![0.35f32; dim]).unwrap();
        assert!(idx.delete(5).unwrap(), "folded id was live");
        idx.insert(1002, &vec![0.45f32; dim]).unwrap();
        assert!(idx.delete(1002).unwrap(), "post-begin id was live");

        let built = job.build(tmp.path()).unwrap();
        idx.consolidate_finish(built).unwrap();

        assert_eq!(idx.len(), 200 + 2 - 1, "200 folded + 2 new - 1 deleted");
        assert!(idx.get(1000).unwrap().is_some(), "post-begin insert kept");
        assert!(idx.get(1001).unwrap().is_some(), "post-begin insert kept");
        assert!(
            idx.get(5).unwrap().is_none(),
            "post-begin delete of folded id holds"
        );
        assert!(
            idx.get(1002).unwrap().is_none(),
            "insert+delete in window stays dead"
        );

        // The same must hold from disk alone (WAL suffix replay).
        drop(idx);
        let reopened = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(reopened.len(), 201);
        assert!(reopened.get(1000).unwrap().is_some());
        assert!(reopened.get(5).unwrap().is_none());
        assert!(reopened.get(1002).unwrap().is_none());
    }

    /// A flush that fires during the background window (delta reaching FLUSH)
    /// creates post-begin runs; finish discards their dirs and the WAL suffix
    /// replays them, so nothing is lost and no stale run dir survives.
    #[test]
    fn post_begin_flush_survives_finish() {
        let dim = 8;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        for i in 0..100u64 {
            idx.insert(i, &vec![0.1f32 + i as f32 * 1e-3; dim]).unwrap();
        }
        let job = idx.consolidate_begin().unwrap().expect("non-empty");
        // Enough post-begin inserts to cross FLUSH and mint a run mid-window.
        for i in 1000..(1000 + DiskVamanaIndex::FLUSH as u64 + 10) {
            idx.insert(i, &vec![0.2f32 + (i as f32) * 1e-6; dim])
                .unwrap();
        }
        assert!(idx.run_count() > 0, "a post-begin run exists");
        let built = job.build(tmp.path()).unwrap();
        idx.consolidate_finish(built).unwrap();
        assert_eq!(
            idx.len(),
            100 + DiskVamanaIndex::FLUSH + 10,
            "folded + post-begin (replayed through the WAL suffix)"
        );
        assert!(
            idx.get(1000).unwrap().is_some(),
            "post-begin run content kept"
        );
    }

    /// A runs-only merge folds the runs into one, preserves the live set and the
    /// search result, and leaves the base untouched.
    #[test]
    fn runs_merge_folds_and_preserves_search() {
        let dim = 16;
        let n = 3 * DiskVamanaIndex::FLUSH; // several runs
        let vecs = random_vectors(n, dim, 21);
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        // Scattered deletes across the runs.
        for id in (0..n as u64).step_by(7) {
            idx.delete(id).unwrap();
        }
        let live_before = idx.len();
        let runs_before = idx.run_count();
        assert!(
            runs_before >= 2,
            "need multiple runs to merge, got {runs_before}"
        );

        let job = idx.merge_runs_begin().unwrap().expect("runs to merge");
        let built = job.build(tmp.path()).unwrap();
        idx.merge_runs_finish(built).unwrap();

        assert_eq!(idx.run_count(), 1, "runs folded into one");
        assert_eq!(idx.len(), live_before, "live set unchanged by the merge");
        // Deleted ids stay gone; a surviving id self-matches.
        assert!(idx.get(0).unwrap().is_none(), "deleted id 0 stays gone");
        let survivor = 1u64; // 1 % 7 != 0
        assert!(idx.get(survivor).unwrap().is_some());
        let q = &vecs[survivor as usize * dim..(survivor as usize + 1) * dim];
        let hit = idx.search(q, 1).unwrap();
        assert_eq!(hit[0].0, survivor, "query returns its own vector as top-1");
    }

    /// Inserts, deletes, and flushes that race a background runs-merge all
    /// survive: the base/delta/WAL are untouched, so search precedence resolves
    /// everything.
    #[test]
    fn writes_during_runs_merge_survive() {
        let dim = 16;
        let n = 3 * DiskVamanaIndex::FLUSH;
        let vecs = random_vectors(n, dim, 22);
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        let live_before = idx.len();
        let job = idx.merge_runs_begin().unwrap().expect("runs to merge");

        // Race the build.
        idx.insert(90_000, &vec![0.11f32; dim]).unwrap();
        assert!(idx.delete(1).unwrap(), "delete a folded-in id"); // 1 % 7 != 0, was live
        idx.insert(90_001, &vec![0.22f32; dim]).unwrap();
        assert!(idx.delete(90_001).unwrap(), "insert+delete in the window");

        let built = job.build(tmp.path()).unwrap();
        idx.merge_runs_finish(built).unwrap();

        assert_eq!(
            idx.len(),
            live_before + 1 - 1,
            "one new live (90000), one folded id deleted"
        );
        assert!(idx.get(90_000).unwrap().is_some(), "post-begin insert kept");
        assert!(
            idx.get(1).unwrap().is_none(),
            "post-begin delete of a folded id holds"
        );
        assert!(
            idx.get(90_001).unwrap().is_none(),
            "insert+delete in window stays dead"
        );

        // Reopen: the merge is transient (no WAL touch); the WAL replays the full
        // history, so the live set is identical from disk alone.
        drop(idx);
        let re = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(re.len(), live_before);
        assert!(re.get(90_000).unwrap().is_some());
        assert!(re.get(1).unwrap().is_none());
    }

    /// Delete-patch removes the tombstoned rows from the base graph in place
    /// (O(deleted), no full rebuild) while keeping the survivors searchable.
    #[test]
    fn delete_patch_removes_deleted_and_preserves_search() {
        let dim = 16;
        let n = 400usize;
        let vecs = random_vectors(n, dim, 31);
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap(); // land every vector in the base
        let base_before = idx.main_len();
        assert_eq!(base_before, n, "all rows in base");

        let deleted: Vec<u64> = (0..n as u64).step_by(5).collect();
        for &id in &deleted {
            idx.delete(id).unwrap();
        }
        let live_before = idx.len();

        let job = idx
            .delete_patch_begin()
            .unwrap()
            .expect("dead rows to reclaim");
        let built = job.build(tmp.path()).unwrap();
        idx.delete_patch_finish(built).unwrap();

        assert_eq!(
            idx.main_len(),
            n - deleted.len(),
            "base shrank by the deletes"
        );
        assert_eq!(idx.len(), live_before, "live set unchanged by the patch");
        for &id in &deleted {
            assert!(idx.get(id).unwrap().is_none(), "deleted id {id} stays gone");
        }
        // The rewired graph still reaches its survivors: top-1 self-match on a
        // broad sample.
        let mut self_hits = 0usize;
        let survivors: Vec<u64> = (0..n as u64).filter(|id| id % 5 != 0).collect();
        for &id in &survivors {
            let q = &vecs[id as usize * dim..(id as usize + 1) * dim];
            if idx.search(q, 1).unwrap().first().is_some_and(|h| h.0 == id) {
                self_hits += 1;
            }
        }
        let frac = self_hits as f64 / survivors.len() as f64;
        assert!(
            frac >= 0.90,
            "patched graph stays connected: {frac:.3} self-match"
        );
    }

    /// Writes that race a background delete-patch survive, and the result is
    /// correct from disk alone after a reopen - the patch leaves runs/delta/WAL
    /// untouched, so the removed rows (tombstoned or run-shadowed) still replay.
    #[test]
    fn delete_patch_survives_race_and_reopen() {
        let dim = 16;
        let n = 300usize;
        let vecs = random_vectors(n, dim, 32);
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty(tmp.path(), dim, 64).unwrap();
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        for id in (0..n as u64).step_by(5) {
            idx.delete(id).unwrap(); // dead-at-begin base rows
        }
        let live_before = idx.len();

        let job = idx
            .delete_patch_begin()
            .unwrap()
            .expect("dead rows to reclaim");
        // Race the off-thread build.
        idx.insert(90_000, &vec![0.11f32; dim]).unwrap(); // new live
        assert!(idx.delete(1).unwrap(), "delete a survivor mid-patch"); // 1 % 5 != 0
        idx.insert(90_001, &vec![0.22f32; dim]).unwrap();
        assert!(idx.delete(90_001).unwrap(), "insert+delete in the window");
        let built = job.build(tmp.path()).unwrap();
        idx.delete_patch_finish(built).unwrap();

        // +1 (90000 live), -1 (id 1 deleted). Deletes of already-dead rows n/a.
        assert_eq!(idx.len(), live_before + 1 - 1);
        assert!(idx.get(90_000).unwrap().is_some(), "post-begin insert kept");
        assert!(
            idx.get(1).unwrap().is_none(),
            "post-begin delete of a survivor holds"
        );
        assert!(
            idx.get(90_001).unwrap().is_none(),
            "insert+delete in window stays dead"
        );
        assert!(idx.get(0).unwrap().is_none(), "pre-begin delete stays gone");

        // Reopen from disk: base is patched, WAL replays the full history.
        drop(idx);
        let re = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(re.len(), live_before, "live count identical from disk");
        assert!(re.get(90_000).unwrap().is_some());
        assert!(re.get(1).unwrap().is_none());
        assert!(re.get(0).unwrap().is_none());
        let survivor = 2u64;
        assert!(
            re.get(survivor).unwrap().is_some(),
            "a survivor is still present"
        );
    }

    // ── Off-thread flush (feat/off-thread-flush) ────────────────────────────

    // The staged delta stays searchable during the off-thread build, then lands
    // in a run; the live set is unchanged and everything remains retrievable.
    #[test]
    fn off_thread_flush_stages_and_splices() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            tmp.path(),
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        let base = random_vectors(500, dim, 7);
        for (i, v) in base.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        let more = random_vectors(300, dim, 8);
        for (i, v) in more.chunks_exact(dim).enumerate() {
            idx.insert(500 + i as u64, v).unwrap();
        }
        let live = idx.len();
        assert_eq!(idx.delta_len(), 300);

        let job = idx.flush_begin().unwrap().expect("delta to flush");
        assert_eq!(idx.delta_len(), 0, "delta moved to staging");
        // A staged entry is still found DURING the flush (search + get).
        let x = 600u64;
        let qv = &more[(x - 500) as usize * dim..(x - 500 + 1) as usize * dim];
        assert_eq!(
            idx.search(qv, 1).unwrap()[0].0,
            x,
            "staged searchable mid-flush"
        );
        assert!(idx.get(x).unwrap().is_some(), "staged get() mid-flush");

        let built = job.build(tmp.path()).unwrap();
        idx.flush_finish(built).unwrap();
        assert_eq!(idx.run_count(), 1, "flushed into one run");
        assert_eq!(idx.len(), live, "live set unchanged");
        assert_eq!(
            idx.search(qv, 1).unwrap()[0].0,
            x,
            "searchable after finish"
        );
    }

    /// A crash mid base-swap (the new slot written but CURRENT not yet
    /// flipped) leaves the OLD generation live and intact - the torn-swap
    /// window the generation pointer closes.
    #[test]
    fn a_crash_before_the_pointer_flip_keeps_the_old_base() {
        let dim = 16;
        let tier = QuantKind::TurboQuant { bits: 2 };
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx =
            DiskVamanaIndex::create_empty_with_tier(tmp.path(), dim, 64, tier).unwrap();
        let vecs = random_vectors(300, dim, 55);
        for (id, v) in vecs.chunks_exact(dim).enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        let base_before = idx.main_len();
        assert!(base_before >= 300);
        drop(idx);

        // Simulate a crash mid-swap: a half-built inactive slot exists, but
        // CURRENT still names the live one. `install`'s "remove a torn prior
        // attempt" clause and the untouched pointer must ignore it.
        let live = current_slot(tmp.path()).expect("CURRENT written");
        let dead = if live == 0 { 1u8 } else { 0u8 };
        let dead_dir = tmp.path().join(format!("g{dead}"));
        std::fs::create_dir_all(&dead_dir).unwrap();
        std::fs::write(dead_dir.join(GRAPH_FILE), b"garbage").unwrap();

        let re = DiskVamanaIndex::open_with_tier(tmp.path(), tier).unwrap();
        assert_eq!(re.main_len(), base_before, "reopen must serve the old base");
        for id in [0u64, 150, 299] {
            assert!(re.get(id).unwrap().is_some(), "id {id} lost across the torn swap");
        }
    }

    /// A repeated query must hit the semantic entry cache and return the
    /// same results it returned cold.
    #[test]
    fn entry_cache_seeds_repeat_queries_without_changing_results() {
        let dim = 32;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            tmp.path(),
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        for (i, v) in random_vectors(3000, dim, 77).chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        if !entry_cache_enabled() {
            return; // the suite also runs with SKEG_ENTRY_CACHE=0
        }
        let q: Vec<f32> = random_vectors(1, dim, 78);
        let hits0 =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::EntryCacheHits);
        let cold = idx.search(&q, 10).unwrap();
        let warm = idx.search(&q, 10).unwrap();
        assert_eq!(
            cold.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            warm.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            "seeded results differ from cold results"
        );
        assert!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::EntryCacheHits) > hits0,
            "the repeat query never hit the entry cache"
        );
    }

    /// A reopen must keep the flushed runs as runs, not replay them into the
    /// delta.
    ///
    /// `clean_stale_runs` used to delete every run directory at open and
    /// recover the whole WAL into RAM: correct, and O(everything-since-last-
    /// fold). The demo restarted with 218k rows in the delta - 900 MB of RAM
    /// and a flat scan on every search - because a growth's worth of runs
    /// was thrown away and re-read. Runs are durable graphs on disk; a
    /// reopen re-opens them and replays only the WAL suffix (the current
    /// delta).
    #[test]
    fn reopen_keeps_flushed_runs_out_of_the_delta() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let tier = QuantKind::TurboQuant { bits: 2 };
        let mut idx =
            DiskVamanaIndex::create_empty_with_tier(tmp.path(), dim, 64, tier).unwrap();
        idx.set_auto_flush(false);
        let base = random_vectors(400, dim, 21);
        for (i, v) in base.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        // Two flushed runs plus a small live delta.
        for batch in 0..2u64 {
            let more = random_vectors(200, dim, 22 + batch);
            for (i, v) in more.chunks_exact(dim).enumerate() {
                idx.insert(1000 + batch * 1000 + i as u64, v).unwrap();
            }
            let job = idx.flush_begin().unwrap().expect("delta to flush");
            let built = job.build(tmp.path()).unwrap();
            idx.flush_finish(built).unwrap();
        }
        let tail = random_vectors(50, dim, 30);
        for (i, v) in tail.chunks_exact(dim).enumerate() {
            idx.insert(5000 + i as u64, v).unwrap();
        }
        let live = idx.len();
        assert_eq!(idx.run_count(), 2);
        drop(idx);

        let re = DiskVamanaIndex::open_with_tier(tmp.path(), tier).unwrap();
        assert_eq!(re.len(), live, "live set survives the reopen");
        assert_eq!(re.run_count(), 2, "flushed runs reopened as runs");
        assert_eq!(re.delta_len(), 50, "only the WAL suffix lands in the delta");
        // Every id from every location is still findable.
        for id in [0u64, 399, 1000, 1199, 2000, 2199, 5000, 5049] {
            assert!(re.get(id).unwrap().is_some(), "id {id} lost across reopen");
        }
        // And a fresh insert keeps working (run_seq must not collide).
        let mut re = re;
        re.insert(9000, &vec![0.25f32; dim]).unwrap();
        let job = re.flush_begin().unwrap().expect("delta to flush");
        let built = job.build(tmp.path()).unwrap();
        re.flush_finish(built).unwrap();
        assert_eq!(re.run_count(), 3, "post-reopen flush adds a new run");
    }

    // Inserts, deletes, and re-inserts that race the off-thread flush all resolve
    // correctly (delta > flushing > run precedence), and a reopen from disk gives
    // the same live set (the WAL was never truncated by the flush).
    #[test]
    fn writes_during_off_thread_flush_survive() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            tmp.path(),
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        let base = random_vectors(500, dim, 9);
        for (i, v) in base.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        let more = random_vectors(300, dim, 10);
        for (i, v) in more.chunks_exact(dim).enumerate() {
            idx.insert(500 + i as u64, v).unwrap();
        }
        let live = idx.len();

        let job = idx.flush_begin().unwrap().unwrap();
        idx.insert(9000, &vec![0.5f32; dim]).unwrap(); // new
        assert!(idx.delete(500).unwrap(), "delete a staged id"); // 500 is staged
        idx.insert(501, &vec![0.7f32; dim]).unwrap(); // overwrite a staged id
        let built = job.build(tmp.path()).unwrap();
        idx.flush_finish(built).unwrap();

        assert!(idx.get(9000).unwrap().is_some(), "post-begin insert kept");
        assert!(idx.get(500).unwrap().is_none(), "deleted staged id gone");
        assert_eq!(
            idx.get(501).unwrap().unwrap(),
            vec![0.7f32; dim],
            "re-insert wins over run"
        );
        assert_eq!(idx.len(), live + 1 - 1, "one new, one deleted");

        drop(idx);
        let re = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(re.len(), live, "same live set from disk");
        assert!(re.get(9000).unwrap().is_some());
        assert!(re.get(500).unwrap().is_none());
        assert_eq!(re.get(501).unwrap().unwrap(), vec![0.7f32; dim]);
    }

    // A crash between flush_begin and flush_finish loses nothing: the staged
    // batch's inserts are still in the WAL, so a reopen replays them.
    #[test]
    fn off_thread_flush_crash_before_finish_recovers() {
        let dim = 16;
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            tmp.path(),
            dim,
            64,
            QuantKind::TurboQuant { bits: 2 },
        )
        .unwrap();
        idx.set_auto_flush(false);
        let base = random_vectors(400, dim, 11);
        for (i, v) in base.chunks_exact(dim).enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        let more = random_vectors(300, dim, 12);
        for (i, v) in more.chunks_exact(dim).enumerate() {
            idx.insert(400 + i as u64, v).unwrap();
        }
        let live = idx.len();
        let job = idx.flush_begin().unwrap().unwrap();
        let _built = job.build(tmp.path()).unwrap(); // built but NEVER finished
        drop(idx); // crash

        let re = DiskVamanaIndex::open(tmp.path()).unwrap();
        assert_eq!(re.len(), live, "live set recovered from the WAL");
        assert!(re.get(500).unwrap().is_some(), "a staged id recovered");
    }

    // set_auto_flush(false) stops the inline flush; the default keeps it.
    #[test]
    fn set_auto_flush_off_disables_inline_flush() {
        let dim = 16;
        let n = DiskVamanaIndex::FLUSH + 500;
        let vecs = random_vectors(n, dim, 13);
        let tmp0 = tempfile::TempDir::new().unwrap();
        let mut off = DiskVamanaIndex::create_empty(tmp0.path(), dim, 64).unwrap();
        off.set_auto_flush(false);
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            off.insert(i as u64, v).unwrap();
        }
        assert_eq!(off.run_count(), 0, "auto_flush off => no inline flush");
        assert!(
            off.delta_len() >= DiskVamanaIndex::FLUSH,
            "delta grew past FLUSH"
        );

        let tmp1 = tempfile::TempDir::new().unwrap();
        let mut on = DiskVamanaIndex::create_empty(tmp1.path(), dim, 64).unwrap();
        for (i, v) in vecs.chunks_exact(dim).enumerate() {
            on.insert(i as u64, v).unwrap();
        }
        assert!(on.run_count() >= 1, "default auto_flush flushes inline");
    }
}
