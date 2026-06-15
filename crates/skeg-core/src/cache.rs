#![deny(unsafe_code)]

//! S3-FIFO cache - byte-budgeted RAM cache for hot keys.
//!
//! `F_NOCACHE` disables the OS page cache, so every uncached read hits the SSD.
//! This cache is the explicit replacement. S3-FIFO (Yang et al., SOSP 2023)
//! matches W-TinyLFU hit rates with three FIFO queues and no per-entry locks.
//!
//! The budget is **in bytes, strict**: an insert that would exceed the budget
//! evicts entries first, so `current_bytes` never crosses `budget_bytes`. A
//! count-based budget cannot bound RAM when value sizes vary (an embedding is
//! ~4 KB, a scalar a few bytes), which would break the project's RAM-frugal
//! goal.
//!
//! Three structures:
//!   - **Small** (`small`, ~10% of the budget): newcomers; one-hit wonders die.
//!   - **Main** (`main`, ~90%): survivors - entries touched >= 2x while in Small.
//!   - **Ghost** (`ghost`): fingerprints of recent evictions - a readmission hint.
//!
//! Each entry carries a 2-bit frequency counter (0..=3, saturating).
//!
//! The cache is **single-threaded per shard** - it lives behind the shard's
//! `RefCell`, never shared across threads - so no internal locking is needed.

use std::collections::VecDeque;

use ahash::{AHashMap, AHashSet};
use xxhash_rust::xxh3::xxh3_64;

/// Fixed per-entry bookkeeping charged to the budget, on top of key and value
/// bytes: the hashmap node, the key clone held in a queue, and the entry struct.
const ENTRY_OVERHEAD: usize = 64;

/// Ghost queue size, in fingerprints (8 bytes each -> ~512 KiB max).
const GHOST_CAPACITY: usize = 1 << 16;

/// Max consecutive under-share entries Main eviction will skip while looking for
/// an over-share victim, before falling back to plain FIFO. Bounds the fairness
/// cost per eviction and guarantees termination.
const MAX_FAIR_SKIPS: u32 = 16;

struct CacheEntry<V> {
    value: V,
    /// Bytes this entry charges against the budget.
    size: usize,
    freq: u8,
    in_small: bool,
    /// Owning tenant, for per-tenant byte accounting. `0` is the single-tenant
    /// (unscoped) default. A `u64` (low bits of the `u128` tenant id) so the
    /// entry stays 8-aligned: a `u128` field would force 16-byte struct
    /// alignment and roughly double `CacheEntry`, hurting eviction-walk locality.
    tenant: u64,
}

/// S3-FIFO cache mapping key bytes -> value `V`, bounded by a byte budget.
pub struct S3Fifo<V> {
    map: AHashMap<Vec<u8>, CacheEntry<V>>,
    small: VecDeque<Vec<u8>>,
    main: VecDeque<Vec<u8>>,
    ghost: VecDeque<u64>,
    ghost_set: AHashSet<u64>,
    /// Bytes charged to the unscoped default tenant `0`, kept inline so the
    /// single-tenant path pays an integer add, not a `u128` hash + probe.
    tenant0_bytes: usize,
    /// Bytes charged per non-zero tenant. With `tenant0_bytes` it sums to
    /// `total_bytes`; entries drop to absent (not zero) so the map tracks only
    /// tenants with live cache residency.
    per_tenant: AHashMap<u64, usize>,
    budget_bytes: usize,
    small_budget: usize,
    total_bytes: usize,
    small_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<V: Clone> S3Fifo<V> {
    /// Create a cache with a `budget_bytes` byte budget.
    ///
    /// # Panics
    ///
    /// Panics if `budget_bytes` is zero.
    #[must_use]
    pub fn new(budget_bytes: usize) -> Self {
        assert!(budget_bytes >= 1, "budget must be >= 1 byte");
        Self {
            map: AHashMap::new(),
            small: VecDeque::new(),
            main: VecDeque::new(),
            ghost: VecDeque::new(),
            ghost_set: AHashSet::new(),
            tenant0_bytes: 0,
            per_tenant: AHashMap::new(),
            budget_bytes,
            small_budget: (budget_bytes / 10).max(1),
            total_bytes: 0,
            small_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up `key`. On a hit, bumps the entry's frequency counter.
    pub fn get(&mut self, key: &[u8]) -> Option<V> {
        if let Some(e) = self.map.get_mut(key) {
            e.freq = (e.freq + 1).min(3);
            self.hits += 1;
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::CacheHits);
            Some(e.value.clone())
        } else {
            self.misses += 1;
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::CacheMisses);
            None
        }
    }

    /// Insert or overwrite `key` -> `value` for the unscoped (single-tenant)
    /// default tenant `0`. `value_bytes` is the value's size, charged (with the
    /// key and a fixed overhead) against the budget.
    pub fn insert(&mut self, key: &[u8], value: V, value_bytes: usize) {
        self.insert_for(key, value, value_bytes, 0);
    }

    /// Insert or overwrite `key` -> `value`, charging the bytes to `tenant` for
    /// per-tenant accounting. Rewriting a key under a different tenant moves its
    /// charged bytes from the old tenant to the new one.
    pub fn insert_for(&mut self, key: &[u8], value: V, value_bytes: usize, tenant: u128) {
        let tenant = tenant as u64;
        let entry_size = key.len() + value_bytes + ENTRY_OVERHEAD;

        if let Some(e) = self.map.get_mut(key) {
            // Overwrite in place; adjust the byte counters by the size delta.
            let old = e.size;
            let old_tenant = e.tenant;
            let was_small = e.in_small;
            e.value = value;
            e.size = entry_size;
            e.tenant = tenant;
            self.total_bytes = self.total_bytes + entry_size - old;
            if was_small {
                self.small_bytes = self.small_bytes + entry_size - old;
            }
            self.sub_tenant_bytes(old_tenant, old);
            self.add_tenant_bytes(tenant, entry_size);
            return;
        }

        // Strict: evict until the new entry fits within the budget.
        while self.total_bytes + entry_size > self.budget_bytes && !self.map.is_empty() {
            self.evict_one();
        }

        let kh = xxh3_64(key);
        let in_small = !self.ghost_set.remove(&kh);
        if in_small {
            self.small.push_back(key.to_vec());
            self.small_bytes += entry_size;
        } else {
            self.main.push_back(key.to_vec());
        }
        self.map.insert(
            key.to_vec(),
            CacheEntry {
                value,
                size: entry_size,
                freq: 0,
                in_small,
                tenant,
            },
        );
        self.total_bytes += entry_size;
        self.add_tenant_bytes(tenant, entry_size);
    }

    /// Add `n` bytes to a tenant's charged total. Tenant `0` takes the inline
    /// fast path; non-zero tenants hit the map.
    fn add_tenant_bytes(&mut self, tenant: u64, n: usize) {
        if tenant == 0 {
            self.tenant0_bytes += n;
        } else {
            *self.per_tenant.entry(tenant).or_insert(0) += n;
        }
    }

    /// Subtract `n` bytes from a tenant's charged total, dropping the map entry
    /// when it reaches zero so the map tracks only tenants with live residency.
    fn sub_tenant_bytes(&mut self, tenant: u64, n: usize) {
        if tenant == 0 {
            self.tenant0_bytes -= n;
        } else if let Some(v) = self.per_tenant.get_mut(&tenant) {
            *v -= n;
            if *v == 0 {
                self.per_tenant.remove(&tenant);
            }
        }
    }

    /// Bytes currently charged to `tenant` (0 if the tenant holds nothing).
    #[must_use]
    pub fn charged_bytes(&self, tenant: u128) -> usize {
        self.tenant_charged(tenant as u64)
    }

    /// Internal `u64`-keyed read used on the eviction hot path.
    fn tenant_charged(&self, tenant: u64) -> usize {
        if tenant == 0 {
            self.tenant0_bytes
        } else {
            self.per_tenant.get(&tenant).copied().unwrap_or(0)
        }
    }

    /// Equal fair-share target per active tenant: `budget / active_tenants`.
    fn share_target(&self) -> usize {
        let active = self.per_tenant.len() + usize::from(self.tenant0_bytes > 0);
        self.budget_bytes / active.max(1)
    }

    /// True if any tenant holds more than `target` (its fair share).
    fn has_over_share(&self, target: usize) -> bool {
        self.tenant0_bytes > target || self.per_tenant.values().any(|&b| b > target)
    }

    /// Remove `key`. Returns `true` if it was cached.
    ///
    /// The queue entry is left behind and skipped lazily on eviction.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if let Some(e) = self.map.remove(key) {
            self.total_bytes -= e.size;
            if e.in_small {
                self.small_bytes -= e.size;
            }
            self.sub_tenant_bytes(e.tenant, e.size);
            true
        } else {
            false
        }
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Byte budget.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget_bytes
    }

    /// Bytes currently charged to the budget (never exceeds `budget`).
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of entries evicted from the cache over its lifetime.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Fraction of lookups that hit, over the cache's lifetime.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let r = self.hits as f64 / total as f64;
            r
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /// Evict exactly one entry from the cache.
    fn evict_one(&mut self) {
        if self.small_bytes >= self.small_budget && self.evict_from_small() {
            return;
        }
        self.evict_from_main();
    }

    /// Drain Small: promote touched entries to Main, evict the first cold one.
    /// Returns `true` if an entry was actually removed from the cache.
    fn evict_from_small(&mut self) -> bool {
        while let Some(key) = self.small.pop_front() {
            let (freq, sz, tnt) = match self.map.get(&key) {
                Some(e) => (e.freq, e.size, e.tenant),
                None => continue, // stale (removed) - skip
            };
            if freq > 1 {
                if let Some(e) = self.map.get_mut(&key) {
                    e.freq = 0;
                    e.in_small = false;
                }
                self.small_bytes -= sz;
                self.main.push_back(key);
            } else {
                let kh = xxh3_64(&key);
                self.map.remove(&key);
                self.total_bytes -= sz;
                self.small_bytes -= sz;
                self.sub_tenant_bytes(tnt, sz);
                self.ghost_push(kh);
                self.evictions += 1;
                skeg_telemetry::tick_counter(skeg_telemetry::Counter::CacheEvictions);
                return true;
            }
        }
        false
    }

    /// Evict from Main, giving touched entries a second chance, and (under real
    /// multi-tenancy) biasing the victim toward tenants over their fair share so
    /// a greedy tenant cannot evict a small tenant's hot set.
    fn evict_from_main(&mut self) {
        // Fairness engages only with >1 tenant AND when someone is over share;
        // single-tenant short-circuits (`over_exists == false`) to plain S3-FIFO.
        let (target, over_exists) = if self.per_tenant.is_empty() {
            (0, false)
        } else {
            let t = self.share_target();
            (t, self.has_over_share(t))
        };
        let mut skips = 0u32;
        while let Some(key) = self.main.pop_front() {
            let (freq, sz, tnt) = match self.map.get(&key) {
                Some(e) => (e.freq, e.size, e.tenant),
                None => continue, // stale - skip
            };
            if freq > 0 {
                if let Some(e) = self.map.get_mut(&key) {
                    e.freq -= 1;
                }
                self.main.push_back(key);
                continue;
            }
            // freq == 0: eviction candidate. Spare an under-share tenant while an
            // over-share one exists, up to MAX_FAIR_SKIPS, to hit the greedy one.
            if over_exists && skips < MAX_FAIR_SKIPS && self.tenant_charged(tnt) <= target {
                self.main.push_back(key);
                skips += 1;
                continue;
            }
            self.map.remove(&key);
            self.total_bytes -= sz;
            self.sub_tenant_bytes(tnt, sz);
            self.evictions += 1;
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::CacheEvictions);
            return;
        }
    }

    fn ghost_push(&mut self, kh: u64) {
        if self.ghost_set.insert(kh) {
            self.ghost.push_back(kh);
        }
        while self.ghost.len() > GHOST_CAPACITY {
            if let Some(old) = self.ghost.pop_front() {
                self.ghost_set.remove(&old);
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A u32-valued entry costs key.len() + 4 + ENTRY_OVERHEAD bytes.
    // Most test keys are ~6-9 bytes, so an entry is ~74-77 bytes; budgets below
    // are sized in those units.
    const KIB: usize = 1024;

    #[test]
    fn test_cache_hit_after_set() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        c.insert(b"k", 42, 4);
        assert_eq!(c.get(b"k"), Some(42));
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        assert_eq!(c.get(b"ghost"), None);
    }

    #[test]
    fn test_cache_overwrite() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        c.insert(b"k", 1, 4);
        let after_first = c.current_bytes();
        c.insert(b"k", 2, 4);
        assert_eq!(c.get(b"k"), Some(2));
        assert_eq!(c.len(), 1);
        assert_eq!(
            c.current_bytes(),
            after_first,
            "same-size overwrite must not grow the budget"
        );
    }

    #[test]
    fn test_cache_remove() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        c.insert(b"k", 1, 4);
        assert!(c.remove(b"k"));
        assert_eq!(c.current_bytes(), 0, "remove must release bytes");
        assert!(!c.remove(b"k"));
        assert_eq!(c.get(b"k"), None);
    }

    #[test]
    fn test_strict_budget_never_exceeded() {
        // Strict eviction: current_bytes must never cross the budget.
        let budget = 8 * KIB;
        let mut c: S3Fifo<u32> = S3Fifo::new(budget);
        for i in 0u32..2000 {
            c.insert(format!("key{i}").as_bytes(), i, 4);
            assert!(
                c.current_bytes() <= budget,
                "budget exceeded at insert {i}: {} > {budget}",
                c.current_bytes()
            );
        }
    }

    #[test]
    fn test_eviction_counter() {
        let mut c: S3Fifo<u32> = S3Fifo::new(8 * KIB);
        assert_eq!(c.evictions(), 0);
        for i in 0u32..3000 {
            c.insert(format!("key{i}").as_bytes(), i, 4);
        }
        assert!(c.evictions() > 0, "inserting far past budget must evict");
    }

    #[test]
    fn test_value_size_affects_budget() {
        // A cache budgeted for a few large entries holds many more small ones.
        let budget = 64 * KIB;
        let mut big: S3Fifo<u8> = S3Fifo::new(budget);
        for i in 0u32..1000 {
            big.insert(format!("k{i}").as_bytes(), 0, 4096); // 4 KiB values
        }
        let big_count = big.len();

        let mut small: S3Fifo<u8> = S3Fifo::new(budget);
        for i in 0u32..1000 {
            small.insert(format!("k{i}").as_bytes(), 0, 8); // tiny values
        }
        assert!(
            small.len() > big_count * 4,
            "small values must pack denser: small={} big={big_count}",
            small.len()
        );
    }

    #[test]
    fn test_s3fifo_small_to_main_promotion() {
        // ~75-byte entries; an 8 KiB budget holds ~100, small ~10.
        let mut c: S3Fifo<u32> = S3Fifo::new(8 * KIB);
        c.insert(b"hot", 999, 4);
        for _ in 0..3 {
            assert_eq!(c.get(b"hot"), Some(999));
        }
        for i in 0u32..400 {
            c.insert(format!("cold{i}").as_bytes(), i, 4);
        }
        assert_eq!(
            c.get(b"hot"),
            Some(999),
            "a touched key must survive in Main"
        );
    }

    #[test]
    fn test_s3fifo_one_hit_wonder_evicted() {
        let mut c: S3Fifo<u32> = S3Fifo::new(8 * KIB);
        c.insert(b"cold_once", 1, 4);
        for i in 0u32..600 {
            c.insert(format!("flood{i}").as_bytes(), i, 4);
        }
        assert_eq!(
            c.get(b"cold_once"),
            None,
            "an untouched newcomer must be evicted"
        );
    }

    #[test]
    fn test_s3fifo_ghost_readmission() {
        let mut c: S3Fifo<u32> = S3Fifo::new(8 * KIB);
        c.insert(b"x", 1, 4);
        for i in 0u32..600 {
            c.insert(format!("f{i}").as_bytes(), i, 4);
        }
        assert_eq!(c.get(b"x"), None, "x should have been evicted");

        c.insert(b"x", 2, 4); // ghost hit routes x straight to Main
        for i in 0u32..40 {
            c.insert(format!("g{i}").as_bytes(), i, 4);
        }
        assert_eq!(
            c.get(b"x"),
            Some(2),
            "a ghost-readmitted key belongs in Main"
        );
    }

    #[test]
    fn test_hit_rate_tracking() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        c.insert(b"k", 1, 4);
        let _ = c.get(b"k"); // hit
        let _ = c.get(b"miss"); // miss
        assert!((c.hit_rate() - 0.5).abs() < 1e-9);
    }

    // ── Hit-rate quality: skewed vs uniform ──────────────────────────────────

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn unit(&mut self) -> f64 {
            #[allow(clippy::cast_precision_loss)]
            let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            v
        }
    }

    /// Run `accesses` lookups against a cache budgeted for ~10% of the keyspace.
    /// A fraction `hot_prob` of accesses target a hot set of 5% of `keyspace`.
    fn run_workload(keyspace: u32, accesses: usize, hot_prob: f64, seed: u64) -> f64 {
        // 4-byte key + 4-byte value + overhead.
        let entry = 4 + 4 + ENTRY_OVERHEAD;
        let budget = (keyspace as usize / 10) * entry;
        let hot_set = (keyspace / 20).max(1);
        let mut c: S3Fifo<u32> = S3Fifo::new(budget);
        let mut rng = Rng(seed);
        for _ in 0..accesses {
            #[allow(clippy::cast_possible_truncation)]
            let id = if rng.unit() < hot_prob {
                (rng.next_u64() % u64::from(hot_set)) as u32
            } else {
                (rng.next_u64() % u64::from(keyspace)) as u32
            };
            let key = id.to_le_bytes();
            if c.get(&key).is_none() {
                c.insert(&key, id, 4);
            }
        }
        c.hit_rate()
    }

    #[test]
    fn test_hit_rate_zipf_like() {
        let hr = run_workload(10_000, 200_000, 0.90, 0x9E37_79B9);
        assert!(hr > 0.80, "skewed hit rate too low: {hr:.3}");
    }

    #[test]
    fn test_hit_rate_uniform() {
        let hr = run_workload(10_000, 200_000, 0.0, 0x1234_5678);
        assert!(hr < 0.20, "uniform hit rate implausibly high: {hr:.3}");
        let skewed = run_workload(10_000, 200_000, 0.90, 0x1234_5678);
        assert!(
            skewed > hr * 3.0,
            "skew {skewed:.3} should dominate uniform {hr:.3}"
        );
    }

    // ── Per-tenant byte accounting ────────────────────────────────────────────

    // Charged size of one u32-valued entry with the given key.
    fn entry_size(key: &[u8]) -> usize {
        key.len() + 4 + ENTRY_OVERHEAD
    }

    #[test]
    fn test_cache_entry_stays_compact() {
        // The tenant tag is a u64, not u128, so the entry stays 8-aligned. A
        // u128 would force 16-byte struct alignment and roughly double the
        // entry, hurting eviction-walk locality (see the cache accounting bench).
        assert!(
            std::mem::size_of::<CacheEntry<u64>>() <= 32,
            "CacheEntry grew to {} bytes; keep the tenant tag 8-aligned",
            std::mem::size_of::<CacheEntry<u64>>()
        );
    }

    #[test]
    fn test_per_tenant_charged_bytes() {
        let mut c: S3Fifo<u32> = S3Fifo::new(64 * KIB);
        c.insert_for(b"a", 1, 4, 100);
        c.insert_for(b"b", 2, 4, 100);
        c.insert_for(b"c", 3, 4, 200);
        assert_eq!(c.charged_bytes(100), entry_size(b"a") + entry_size(b"b"));
        assert_eq!(c.charged_bytes(200), entry_size(b"c"));
        assert_eq!(c.charged_bytes(999), 0, "unknown tenant charges nothing");
        assert_eq!(
            c.charged_bytes(100) + c.charged_bytes(200),
            c.current_bytes()
        );
    }

    #[test]
    fn test_insert_defaults_tenant_zero() {
        let mut c: S3Fifo<u32> = S3Fifo::new(4 * KIB);
        c.insert(b"k", 1, 4);
        assert_eq!(
            c.charged_bytes(0),
            c.current_bytes(),
            "bare insert must charge tenant 0"
        );
    }

    #[test]
    fn test_per_tenant_invariant_under_eviction() {
        // Invariant: sum over tenants == current_bytes, held through eviction.
        let mut c: S3Fifo<u32> = S3Fifo::new(8 * KIB);
        for i in 0u32..3000 {
            let tenant = u128::from(i % 4);
            c.insert_for(format!("key{i}").as_bytes(), i, 4, tenant);
            let sum: usize = (0..4u128).map(|t| c.charged_bytes(t)).sum();
            assert_eq!(
                sum,
                c.current_bytes(),
                "per-tenant sum must equal total at insert {i}"
            );
        }
        assert!(c.evictions() > 0, "inserting past budget must evict");
    }

    #[test]
    fn test_per_tenant_remove_and_overwrite() {
        let mut c: S3Fifo<u32> = S3Fifo::new(64 * KIB);
        c.insert_for(b"k", 1, 4, 7);
        let after = c.charged_bytes(7);
        c.insert_for(b"k", 2, 4, 7); // same-size overwrite, same tenant
        assert_eq!(c.charged_bytes(7), after, "same-size overwrite keeps bytes");
        assert!(c.remove(b"k"));
        assert_eq!(c.charged_bytes(7), 0, "remove releases per-tenant bytes");
        assert_eq!(c.current_bytes(), 0);
    }

    #[test]
    fn test_per_tenant_overwrite_reassigns_tenant() {
        // A key rewritten under a different tenant moves its bytes across.
        let mut c: S3Fifo<u32> = S3Fifo::new(64 * KIB);
        c.insert_for(b"k", 1, 4, 7);
        c.insert_for(b"k", 2, 4, 9);
        assert_eq!(c.charged_bytes(7), 0, "old tenant released");
        assert_eq!(c.charged_bytes(9), entry_size(b"k"), "new tenant charged");
        assert_eq!(c.charged_bytes(9), c.current_bytes());
    }

    #[test]
    fn test_eviction_fairness_protects_small_tenant() {
        // Noisy neighbor: tenant 7 keeps a tiny hot set; tenant 9 floods a large
        // one. Both get promoted to Main. Fair eviction must keep tenant 7's set
        // resident (it is far under its fair share) instead of letting tenant 9's
        // flood evict it as plain FIFO would.
        let entry = 8 + 8 + ENTRY_OVERHEAD; // 8-byte key + u64 value + overhead
        let budget = 100 * entry;
        let mut c: S3Fifo<u64> = S3Fifo::new(budget);

        // Tenant 7: 5 keys, accessed repeatedly so they promote to Main.
        let hot: Vec<[u8; 8]> = (0..5u64).map(|i| (1_000_000 + i).to_le_bytes()).collect();
        for _ in 0..4 {
            for k in &hot {
                c.insert_for(k, 0, 8, 7);
                let _ = c.get(k);
                let _ = c.get(k);
            }
        }
        // Tenant 9: flood 500 keys, each touched twice -> also promoted, competing
        // in Main and forcing many evictions.
        for i in 0..500u64 {
            let k = i.to_le_bytes();
            c.insert_for(&k, 0, 8, 9);
            let _ = c.get(&k);
            let _ = c.get(&k);
        }

        let survived = hot.iter().filter(|k| c.get(k.as_slice()).is_some()).count();
        assert!(
            survived >= 4,
            "fair eviction must protect the small tenant's hot set: {survived}/5 survived"
        );
    }
}
