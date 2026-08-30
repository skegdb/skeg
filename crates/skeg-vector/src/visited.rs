//! Compact bitset for tracking visited nodes during the Vamana greedy walk.
//!
//! Drop-in replacement for `AHashSet<VecId>` in the access pattern of
//! [`crate::vamana::greedy_search`]: ~100 expansions x ~64 neighbours per
//! query = ~6400 insert/lookup operations. AHashSet hashes (xxh3/wyhash
//! depending on the backend) and probes; VisitedBitset does one `idx/64`
//! index plus a bit shift. Memory: `N/8` bytes vs ~16 bytes per entry for
//! AHashSet (at 1M nodes: 128 KiB vs ~16 MiB).
//!
//! The primitive gate (6.20x faster than AHashSet on the walk access
//! pattern) and the integration gate (+6-8% QPS, -7-9% latency, recall
//! identical on mxbai/MiniLM) are both green.

use crate::vamana::VecId;

const WORD_BITS: usize = 64;

/// Bitset with `n` slots, `n/64` `u64` words. Explicit reset between queries
/// via [`Self::clear`] (fills `n/64` words with zero, ~125 KiB writes for
/// N=1M - fast at memory bandwidth on contiguous storage).
#[derive(Debug, Clone)]
pub struct VisitedBitset {
    bits: Vec<u64>,
    capacity: u32,
    /// Word indexes that became non-zero since the last clear. A walk touches
    /// a few hundred nodes out of up to 1M slots; clear() and iter() over the
    /// dirty words alone turn two full-bitset sweeps per insert (~250 KiB of
    /// scratch traffic each) into O(visited).
    dirty: Vec<u32>,
}

impl VisitedBitset {
    /// Number of set bits: after a walk, the nodes it expanded. One popcount
    /// per word, outside any hot loop.
    #[must_use]
    pub fn count_set(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }


    /// Create a bitset with `n` slots. Allocation: `n/64` `u64` words, all
    /// zeroed.
    ///
    /// # Panics
    ///
    /// Panics if `n > u32::MAX as usize`. `VecId` is `u32`, no larger
    /// graphs are supported.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let capacity = u32::try_from(n).expect("VisitedBitset capacity must fit u32");
        let words = n.div_ceil(WORD_BITS);
        Self {
            bits: vec![0u64; words],
            capacity,
            dirty: Vec::new(),
        }
    }

    /// Set bit `idx` and return the previous state. Mirrors
    /// `HashSet::insert` semantics with the bool flipped: `true` means
    /// "already present" (insert is a no-op), `false` means "newly set".
    ///
    /// # Panics
    ///
    /// Panics in debug if `idx >= capacity`. In release the bounds check
    /// is delegated to the `bits[word]` indexing, which panics in the
    /// idiomatic Rust way.
    #[inline]
    pub fn test_and_set(&mut self, idx: VecId) -> bool {
        debug_assert!(
            idx < self.capacity,
            "idx {idx} >= capacity {}",
            self.capacity
        );
        let word_idx = idx as usize / WORD_BITS;
        let bit = idx as usize % WORD_BITS;
        let mask = 1u64 << bit;
        let word = self.bits[word_idx];
        if word == 0 {
            self.dirty.push(word_idx as u32);
        }
        let was_set = (word & mask) != 0;
        self.bits[word_idx] = word | mask;
        was_set
    }

    /// Return `true` if `idx` is set (read-only).
    #[inline]
    #[must_use]
    pub fn is_set(&self, idx: VecId) -> bool {
        debug_assert!(idx < self.capacity);
        let word_idx = idx as usize / WORD_BITS;
        let bit = idx as usize % WORD_BITS;
        (self.bits[word_idx] & (1u64 << bit)) != 0
    }

    /// Reset every bit to zero. Call between consecutive queries that reuse
    /// the same bitset (the `BuildScratch` pattern in vamana).
    #[inline]
    pub fn clear(&mut self) {
        for &w in &self.dirty {
            self.bits[w as usize] = 0;
        }
        self.dirty.clear();
    }

    /// Number of slots.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Iterate over the set indices in ascending order. For each non-zero
    /// word, extract bits via repeated `trailing_zeros`. Cache-friendly
    /// (sequential access across the `n/64` words, ~125 KiB at N=1M).
    ///
    /// Equivalent to `AHashSet::iter()` but deterministic-ordered (indices
    /// ascending). Used by `vamana::insert_point_concurrent` to build the
    /// candidate pool for `robust_prune` after the greedy walk.
    #[must_use]
    pub fn iter(&self) -> SetBitsIter<'_> {
        // Ascending order (callers depend on determinism): sort the dirty
        // word list - a few hundred u32s - instead of sweeping every word.
        let mut dirty = self.dirty.clone();
        dirty.sort_unstable();
        SetBitsIter {
            bits: &self.bits,
            dirty,
            pos: 0,
            current_word: 0,
            word_idx: 0,
        }
    }
}

/// Iterator over the set indices of [`VisitedBitset`], ascending order.
/// Walks only the dirty (non-zero) words, sorted ascending.
pub struct SetBitsIter<'a> {
    bits: &'a [u64],
    dirty: Vec<u32>,
    pos: usize,
    current_word: u64,
    word_idx: u32,
}

impl Iterator for SetBitsIter<'_> {
    type Item = VecId;

    fn next(&mut self) -> Option<VecId> {
        while self.current_word == 0 {
            if self.pos >= self.dirty.len() {
                return None;
            }
            self.word_idx = self.dirty[self.pos];
            self.pos += 1;
            self.current_word = self.bits[self.word_idx as usize];
        }
        let bit = self.current_word.trailing_zeros();
        self.current_word &= self.current_word - 1;
        Some(self.word_idx * WORD_BITS as u32 + bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_all_unset() {
        let b = VisitedBitset::new(100);
        for i in 0..100 {
            assert!(!b.is_set(i));
        }
    }

    #[test]
    fn test_and_set_returns_previous_state() {
        let mut b = VisitedBitset::new(10);
        assert!(!b.test_and_set(3)); // first set: was unset
        assert!(b.test_and_set(3)); // second set: was set
        assert!(b.test_and_set(3)); // still set
    }

    #[test]
    fn independent_bits() {
        let mut b = VisitedBitset::new(200);
        b.test_and_set(7);
        b.test_and_set(63);
        b.test_and_set(64); // boundary across words
        b.test_and_set(128);
        b.test_and_set(199);

        assert!(b.is_set(7));
        assert!(b.is_set(63));
        assert!(b.is_set(64));
        assert!(b.is_set(128));
        assert!(b.is_set(199));

        // Unrelated bits stay unset.
        for i in [0, 1, 6, 8, 62, 65, 127, 129, 198] {
            assert!(!b.is_set(i), "bit {i} should be unset");
        }
    }

    #[test]
    fn clear_resets_everything() {
        let mut b = VisitedBitset::new(1000);
        for i in 0..1000 {
            b.test_and_set(i);
        }
        b.clear();
        for i in 0..1000 {
            assert!(!b.is_set(i), "bit {i} should be cleared");
        }
    }

    #[test]
    fn handles_non_multiple_of_64() {
        // 100 slots: needs 2 words (128 bits), but only 0..100 are valid.
        let mut b = VisitedBitset::new(100);
        b.test_and_set(99);
        assert!(b.is_set(99));
        assert!(!b.is_set(0));
        assert!(!b.is_set(50));
        b.clear();
        assert!(!b.is_set(99));
    }

    #[test]
    fn capacity_returned_correctly() {
        let b = VisitedBitset::new(0);
        assert_eq!(b.capacity(), 0);
        let b = VisitedBitset::new(1);
        assert_eq!(b.capacity(), 1);
        let b = VisitedBitset::new(64);
        assert_eq!(b.capacity(), 64);
        let b = VisitedBitset::new(65);
        assert_eq!(b.capacity(), 65);
        let b = VisitedBitset::new(1_000_000);
        assert_eq!(b.capacity(), 1_000_000);
    }

    #[test]
    fn iter_returns_set_bits_in_order() {
        let mut b = VisitedBitset::new(200);
        let ids = [3, 7, 63, 64, 65, 128, 199];
        for &id in &ids {
            b.test_and_set(id);
        }
        let collected: Vec<u32> = b.iter().collect();
        assert_eq!(collected, ids);
    }

    #[test]
    fn iter_empty_bitset_yields_nothing() {
        let b = VisitedBitset::new(1000);
        assert_eq!(b.iter().count(), 0);
    }

    #[test]
    fn iter_after_clear_yields_nothing() {
        let mut b = VisitedBitset::new(200);
        b.test_and_set(50);
        b.test_and_set(150);
        b.clear();
        assert_eq!(b.iter().count(), 0);
    }

    #[test]
    fn reuse_across_many_clear_cycles_matches_hashset() {
        use ahash::AHashSet;
        use rand::{Rng, SeedableRng, rngs::StdRng};
        // The BuildScratch pattern: one bitset, thousands of walk/clear
        // cycles. The sparse clear must leave no stale bit behind.
        let n: u32 = 100_000;
        let mut rng = StdRng::seed_from_u64(7);
        let mut bs = VisitedBitset::new(n as usize);
        for _ in 0..500 {
            let mut hs: AHashSet<u32> = AHashSet::new();
            for _ in 0..300 {
                let idx = rng.random_range(0..n);
                assert_eq!(bs.test_and_set(idx), !hs.insert(idx));
            }
            let mut want: Vec<u32> = hs.into_iter().collect();
            want.sort_unstable();
            let got: Vec<u32> = bs.iter().collect();
            assert_eq!(got, want);
            bs.clear();
            assert_eq!(bs.iter().count(), 0);
        }
    }

    #[test]
    fn matches_hashset_semantics_under_random_ops() {
        use ahash::AHashSet;
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let n: u32 = 10_000;
        let mut rng = StdRng::seed_from_u64(42);
        let mut bs = VisitedBitset::new(n as usize);
        let mut hs: AHashSet<u32> = AHashSet::new();

        // 10_000 random test_and_set operations.
        for _ in 0..10_000 {
            let idx = rng.random_range(0..n);
            let bs_was_set = bs.test_and_set(idx);
            let hs_was_set = !hs.insert(idx);
            assert_eq!(bs_was_set, hs_was_set, "diverge at idx {idx}");
        }

        // Same set of bits after all those ops.
        for i in 0..n {
            assert_eq!(bs.is_set(i), hs.contains(&i), "diverge at idx {i}");
        }
    }
}
