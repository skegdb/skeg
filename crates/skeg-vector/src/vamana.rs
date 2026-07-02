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
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use skeg_simd::cosine_f32;
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

// ── graph node ────────────────────────────────────────────────────────────────

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

// ── search list ───────────────────────────────────────────────────────────────

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

// ── greedy search ─────────────────────────────────────────────────────────────

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

// ── robust prune ──────────────────────────────────────────────────────────────

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

// ── build ─────────────────────────────────────────────────────────────────────

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
/// `SKEG_L_BUILD` env var overrides `l_build` for benchmarking/tuning; otherwise
/// the validated default (64) is used.
fn disk_build_config() -> VamanaConfig {
    let mut cfg = VamanaConfig::default();
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

// ── build profiling ───────────────────────────────────────────────────────────
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
    medoid: VecId,
    p: VecId,
    alpha: f32,
    r: usize,
    l_build: usize,
    scratch: &mut BuildScratch,
) {
    let p_vec = source.row(p);
    let t_walk = Instant::now();
    greedy_search(
        &[medoid],
        l_build,
        None, // build: never early-terminate (full candidate pool for prune)
        |id| dist(p_vec, source.row(id)),
        |id| locked_neighbors(graph, id),
        None, // build: no filter admission
        &mut scratch.visited,
        &mut scratch.seen,
        None,
    );
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
            insert_point_concurrent(graph, source, medoid, p, alpha, r, l_build, scratch);
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
    for u in 0..n {
        if reachable[u as usize] {
            continue;
        }
        let u_vec = source.row(u);
        let mut best = medoid;
        let mut best_d = f32::INFINITY;
        for v in 0..n {
            if reachable[v as usize] {
                let d = dist(u_vec, source.row(v));
                if d < best_d {
                    best_d = d;
                    best = v;
                }
            }
        }
        nodes[best as usize].try_push(u, r);
    }
}

// ── public index ──────────────────────────────────────────────────────────────

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
        assert!(!ids.is_empty(), "Vamana needs at least one vector");
        assert_eq!(vectors.len(), ids.len(), "source/ids length mismatch");
        let dim = vectors.dim();
        let n = ids.len() as u32;

        let mut plain = vec![Node::new(); n as usize];
        init_random_graph(&mut plain, n, config.r, config.seed);
        let medoid = approximate_medoid(&*vectors, n, config.medoid_sample, config.seed);

        // Both passes run in parallel across the rayon pool; the graph is a
        // Vec<Mutex<Node>> for the duration of the build, then unwrapped.
        let graph: Vec<Mutex<Node>> = plain.into_iter().map(Mutex::new).collect();
        run_pass_parallel(
            &graph,
            &*vectors,
            n,
            medoid,
            config.alpha1,
            config.r,
            config.l_build,
            config.seed,
        );
        run_pass_parallel(
            &graph,
            &*vectors,
            n,
            medoid,
            config.alpha2,
            config.r,
            config.l_build,
            config.seed.wrapping_add(1),
        );
        let mut nodes: Vec<Node> = graph.into_iter().map(Mutex::into_inner).collect();
        patch_connectivity(&mut nodes, &*vectors, n, medoid, config.r);

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
            window: 5,
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

// ── on-disk format ────────────────────────────────────────────────────────────
//
// graph.vmn   : 64-byte header, then n u64 ids, then n nodes
//               (each: degree u32 + MAX_R neighbour u32s).
// vectors.bin : 64-byte header, then n*dim f32 row-major.
//
// The graph and an int8 tier-1 quantisation live in RAM; the f32 vectors stay
// on disk and are read (one positioned `read_exact_at` per candidate) only to
// re-rank the survivors of the graph walk.

const GRAPH_FILE: &str = "graph.vmn";
const VECTORS_FILE: &str = "vectors.bin";
/// Append-only WAL of delta inserts/deletes (replayed on open).
const DELTA_LOG_FILE: &str = "delta.log";
/// Persisted IVF router sidecar (centroids + cell assignment).
const IVF_FILE: &str = "ivf.bin";
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
    let mut f = BufWriter::new(File::create(path)?);
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
}

pub struct DiskVamanaIndex {
    dim: usize,
    l_search: usize,
    /// The immutable base segment. Future LSM runs join it as more segments.
    base: Segment,
    /// Additional immutable LSM runs, searched alongside `base`. Streaming
    /// writes flush from the delta into runs; `consolidate` folds them back.
    runs: Vec<Segment>,
    /// Tier-1 quantiser, kept so a `flush` builds runs with the same tier as
    /// the base (and so `consolidate` reopens with it).
    tier: QuantKind,
    /// Monotonic run-directory counter, so flushed run dirs never collide.
    run_seq: u64,
    dir: PathBuf,
    /// Streaming inserts since open / last consolidation: external id -> f32.
    delta: AHashMap<u64, Vec<f32>>,
    /// Tombstoned external ids (covers both main and delta).
    tombstones: AHashSet<u64>,
    live_count: usize,
    /// Append-only log of delta mutations, replayed on `open` so streaming
    /// inserts/deletes survive a restart. `consolidate` truncates it.
    delta_log: File,
    /// Online tq1 proxy controller, `None` unless enabled via
    /// [`enable_tq1_controller`](Self::enable_tq1_controller). Behind a mutex
    /// because `search` is `&self`; only touched on shadow queries.
    // ponytail: Box the whole tq1 runtime (mutex + counter) so this cold state
    // stays off DiskVamanaIndex as one pointer - else the pthread mutex inline
    // bloats skeg-server's VectorBackend enum past large_enum_variant.
    tq1: Box<Tq1Runtime>,
    /// Coarse IVF router: the "cells" branch of hybrid filtered search (sparse /
    /// medium filters). `None` until built via [`build_ivf`](Self::build_ivf).
    /// Boxed to keep DiskVamanaIndex small (VectorBackend large_enum_variant).
    ivf: Option<Box<IvfRouter>>,
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
        // graph.vmn
        let graph_bytes = std::fs::read(dir.join(GRAPH_FILE))?;
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
        let node_len = std::mem::size_of::<Node>();
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
            let file = skeg_platform::MappedFile::open(&dir.join(GRAPH_FILE))?;
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
                let mut neighbors = [0u32; MAX_R];
                for (k, slot) in neighbors.iter_mut().enumerate() {
                    *slot = read_u32(&graph_bytes, pos + 4 + k * 4);
                }
                v.push(Node { degree, neighbors });
                pos += node_len;
            }
            NodeBacking::Owned(v)
        };
        // `pos` is consumed by the in-RAM path; the mmap path skips ahead.
        let _ = pos;

        // vectors.bin: verify header, stream f32 to build the int8 tier.
        let vectors_file = File::open(dir.join(VECTORS_FILE))?;
        let mut vhdr = [0u8; HEADER_LEN];
        vectors_file.read_exact_at(&mut vhdr, 0)?;
        if read_u32(&vhdr, 0) != VEC_MAGIC || read_u32(&vhdr, 4) != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad vectors.bin header",
            ));
        }
        assert_eq!(read_u32(&vhdr, 8), n, "graph/vectors disagree on n");
        assert_eq!(
            read_u32(&vhdr, 12) as usize,
            dim,
            "graph/vectors disagree on dim"
        );

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
        let mut quant = match tier {
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
        };
        // TurboQuant tier only, opt-in. Persist the codes buffer to
        // `tier.cache.bin` and swap
        // the in-RAM `Vec<u8>` for a `MappedFile`; the OS page cache then
        // decides which pages stay resident. Other tiers (int8, pq) keep
        // their `Vec<u8>` representation - the experiment runs on
        // TurboQuant only.
        if mmap_tier && matches!(tier, QuantKind::TurboQuant { .. }) {
            quant.swap_turboquant_codes_to_mmap(&dir.join("tier.cache.bin"))?;
        }

        let id_to_main_row: AHashMap<u64, VecId> = ids
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row as VecId))
            .collect();

        // Replay the delta WAL: streaming inserts/deletes since the last
        // consolidation that have not yet been folded into the graph.
        let log_path = dir.join(DELTA_LOG_FILE);
        let wal = std::fs::read(&log_path).unwrap_or_default();
        let delta_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let mut index = DiskVamanaIndex {
            dim,
            l_search,
            base: Segment {
                main_n: n,
                nodes,
                ids,
                id_to_main_row,
                medoid,
                quant,
                vectors_file,
            },
            runs: Vec::new(),
            tier,
            run_seq: 0,
            dir: dir.to_path_buf(),
            delta: AHashMap::new(),
            tombstones: AHashSet::new(),
            live_count: n as usize,
            delta_log,
            tq1: Box::default(),
            ivf: None,
        };
        index.replay_wal(&wal);
        // Runs flushed before a restart are not reloaded; the WAL replay above
        // already put their vectors back in L0, so the stale dirs are redundant
        // (and would collide with `run-0` of this session). See the flush ADR.
        index.clean_stale_runs();
        index.load_ivf();
        Ok(index)
    }

    /// Best-effort removal of leftover `run-*` directories from a prior session.
    /// They are rebuildable from the WAL, so a failure to remove one is not
    /// fatal to opening the index.
    fn clean_stale_runs(&self) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with("run-") {
                let _ = std::fs::remove_dir_all(entry.path()); // stale, rebuildable
            }
        }
    }

    /// Replay the delta WAL into the in-RAM delta. Tolerant of a truncated
    /// trailing record (a crash mid-append): parsing stops at the first
    /// incomplete record.
    fn replay_wal(&mut self, wal: &[u8]) {
        let insert_len = 1 + 8 + self.dim * 4;
        let mut pos = 0;
        while pos < wal.len() {
            match wal[pos] {
                0 if pos + insert_len <= wal.len() => {
                    let id = u64::from_le_bytes(
                        wal[pos + 1..pos + 9]
                            .try_into()
                            .expect("8-byte window guarded by insert_len check"),
                    );
                    let v: Vec<f32> = wal[pos + 9..pos + insert_len]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    self.apply_insert(id, v);
                    pos += insert_len;
                }
                1 if pos + 9 <= wal.len() => {
                    let id = u64::from_le_bytes(
                        wal[pos + 1..pos + 9]
                            .try_into()
                            .expect("8-byte window guarded by len check"),
                    );
                    self.apply_delete(id);
                    pos += 9;
                }
                _ => break, // truncated or corrupt trailing record
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
        write_graph_vmn(&dir.join(GRAPH_FILE), 0, dim, 0, MAX_R, l_search, &[], &[])?;
        write_vectors_bin(
            &dir.join(VECTORS_FILE),
            &InMemoryVectorSource::new(Vec::new(), dim),
        )?;
        std::fs::write(dir.join(DELTA_LOG_FILE), [])?;
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
        let mut seen: AHashSet<u64> = AHashSet::new();
        for &id in ids {
            if self.tombstones.contains(&id) || !seen.insert(id) {
                continue;
            }
            if let Some(v) = self.delta.get(&id) {
                scored.push((cosine_f32(query, v), id));
                continue;
            }
            // Newest run wins, then base (matches consolidate precedence).
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
        // Keep the top `rerank` by proxy (higher = closer), then f32-rerank them.
        let take = rerank.max(k);
        if cand.len() > take {
            cand.select_nth_unstable_by(take, |a, b| b.0.cmp(&a.0));
            cand.truncate(take);
        }
        for (_p, seg_idx, row) in cand {
            let seg = if seg_idx == 0 {
                &self.base
            } else {
                &self.runs[seg_idx - 1]
            };
            let v = self.read_vector(seg, row)?;
            scored.push((cosine_f32(query, &v), seg.ids[row as usize]));
        }
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
    /// Returns an I/O error if the WAL append fails.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len()` does not equal the index dimension.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> io::Result<()> {
        assert_eq!(vector.len(), self.dim, "vector dim mismatch");
        // WAL record: [0x00][id u64 LE][f32 x dim LE].
        let mut rec = Vec::with_capacity(1 + 8 + self.dim * 4);
        rec.push(0u8);
        rec.extend_from_slice(&id.to_le_bytes());
        for &x in vector {
            rec.extend_from_slice(&x.to_le_bytes());
        }
        self.delta_log.write_all(&rec)?;
        self.apply_insert(id, vector.to_vec());
        // Keep the brute-forced L0 small: once it fills, fold it into a
        // navigable run so search stays sub-linear between consolidations.
        if self.delta.len() >= Self::FLUSH {
            self.flush()?;
        }
        Ok(())
    }

    /// Tombstone `id`. Returns `true` if it was live.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the WAL append fails.
    pub fn delete(&mut self, id: u64) -> io::Result<bool> {
        // WAL record: [0x01][id u64 LE].
        let mut rec = [0u8; 9];
        rec[0] = 1;
        rec[1..9].copy_from_slice(&id.to_le_bytes());
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

    /// Read one f32 vector from `vectors.bin` by positioned read.
    fn read_vector(&self, seg: &Segment, id: VecId) -> io::Result<Vec<f32>> {
        let offset = HEADER_LEN as u64 + u64::from(id) * self.dim as u64 * 4;
        let mut buf = vec![0u8; self.dim * 4];
        seg.vectors_file.read_exact_at(&mut buf, offset)?;
        Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
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
        assert_eq!(query.len(), self.dim, "query dim mismatch");
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
        let mut all_cand: Vec<(f32, usize, VecId)> = Vec::new();
        for (seg_idx, seg) in segs.iter().enumerate() {
            if seg.main_n == 0 {
                continue;
            }
            let code = seg
                .quant
                .quantize_query_with_mode(&normalized(query), tq1_mode);
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
                window: 5,
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
            let dist = |id: VecId| -(seg.quant.proxy(id as usize, &code) as f32);
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
        for (_, seg_idx, row) in all_cand {
            if disk_reads >= rerank {
                break;
            }
            let seg = segs[seg_idx];
            let id = seg.ids[row as usize];
            if self.tombstones.contains(&id) || self.delta.contains_key(&id) {
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
            let v = self.read_vector(seg, row)?;
            disk_reads += 1;
            scored.push((OrderedFloat(cosine_f32(query, &v)), id));
        }
        rerank_span.record("disk_reads", disk_reads);

        // Flat scan of the delta (small, in RAM). Delta entries are always live.
        for (&id, v) in &self.delta {
            if matches.is_none_or(|m| m(id)) {
                scored.push((OrderedFloat(cosine_f32(query, v)), id));
            }
        }

        scored.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .map(|(s, id)| (id, s.into_inner()))
            .collect())
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
        let run_dir = self.dir.join(format!("run-{}", self.run_seq));
        self.run_seq += 1;
        let rebuilt = VamanaIndex::build(vectors, ids, self.dim, &disk_build_config());
        rebuilt.save(&run_dir)?;
        // Open the run with this index's tier and keep only its base segment; the
        // rest of the opened index (an empty delta/WAL over the run dir) is dropped.
        let run = DiskVamanaIndex::open_with_tier(&run_dir, self.tier)?;
        self.runs.push(run.base);
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
        Ok(())
    }

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
        for &id in self.delta.keys() {
            if seen.insert(id) {
                survivors.push((id, usize::MAX, 0));
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
            if loc == usize::MAX {
                vectors.extend_from_slice(&self.delta[&id]);
            } else {
                vectors.extend(self.read_vector(segs[loc], row)?);
            }
            ids.push(id);
        }
        let dir = self.dir.clone();
        let tier = self.tier;
        let rebuilt = VamanaIndex::build(vectors, ids, dim, &disk_build_config());
        rebuilt.save(&dir)?;
        self.discard_runs()?;
        // The delta + runs are now folded into the graph: the WAL must start
        // empty so the reopen below does not replay stale records.
        std::fs::write(dir.join(DELTA_LOG_FILE), [])?;
        // The row order changed, so the persisted router is now stale: drop it.
        // We do NOT rebuild the IVF here - consolidate runs INLINE on the ingest
        // path (shard auto-consolidates when delta >= main), and an inline IVF
        // build (re-read all vectors + k-means) stalls ingest. The background
        // idle-consolidate rebuilds it off the request path; until then filtered
        // search falls back to the exact scan (correct, just O(|s|)).
        let _ = std::fs::remove_file(dir.join(IVF_FILE));
        *self = DiskVamanaIndex::open_with_tier(&dir, tier)?;
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
    /// the base vectors once for k-means. `n_cells == 0` picks ~√n.
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
        }
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
        /// Below this many matches, an exact scan of `s` is cheaper than the IVF
        /// route (measured: at 5k, qscan 0.9ms vs routed 1.9ms; the route wins
        /// from ~10k up, where qscan goes O(|S|)).
        const SCAN_MAX: usize = 12_288;
        /// Candidate budget the router narrows `s` down to before scoring.
        const SHORTLIST: usize = 4_096;
        if k == 0 || s.is_empty() {
            return Ok(Vec::new());
        }
        match &self.ivf {
            Some(router) if s.len() > SCAN_MAX => {
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
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // test sizes are tiny
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;
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
        // After consolidation the WAL is empty: a reopen replays nothing and
        // the (now graph-resident) vectors are still all there.
        let wal = std::fs::read(tmp.path().join("delta.log")).unwrap();
        assert!(wal.is_empty(), "WAL must be truncated by consolidation");
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
}
