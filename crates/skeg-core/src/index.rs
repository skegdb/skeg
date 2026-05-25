#![deny(unsafe_code)]

use ahash::AHashMap;
use xxhash_rust::xxh3::xxh3_64;

/// Per-key location in the vLog. 16 bytes → 8 entries per M1 cache line (128 B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct IndexEntry {
    /// Low 32 bits of `xxh3_64(key)` - fast equality pre-check.
    pub fingerprint: u32,
    /// Segment file ID.
    pub segment_id: u16,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad: u16,
    /// Byte offset of the record within the segment.
    pub offset: u32,
    /// Padded on-disk record size in bytes.
    pub size: u32,
}

const _: () = assert!(std::mem::size_of::<IndexEntry>() == 16);

/// In-RAM index mapping key bytes → `IndexEntry`.
pub struct Index {
    map: AHashMap<Vec<u8>, IndexEntry>,
}

impl Index {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: AHashMap::new(),
        }
    }

    /// Insert or overwrite entry for `key`.
    pub fn set(&mut self, key: Vec<u8>, entry: IndexEntry) {
        self.map.insert(key, entry);
    }

    /// Look up entry for `key`.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&IndexEntry> {
        self.map.get(key)
    }

    /// Remove `key`. Returns `true` if it existed.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        self.map.remove(key).is_some()
    }

    /// Number of live entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over `(key, entry)` pairs. Order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &IndexEntry)> {
        self.map.iter().map(|(k, v)| (k.as_slice(), v))
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the fingerprint stored in an `IndexEntry`.
#[must_use]
pub fn fingerprint(key: &[u8]) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let h = xxh3_64(key) as u32;
    h
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seg: u16, off: u32, sz: u32, key: &[u8]) -> IndexEntry {
        IndexEntry {
            fingerprint: fingerprint(key),
            segment_id: seg,
            _pad: 0,
            offset: off,
            size: sz,
        }
    }

    #[test]
    fn index_entry_is_16_bytes() {
        assert_eq!(std::mem::size_of::<IndexEntry>(), 16);
    }

    #[test]
    fn test_index_set_get() {
        let mut idx = Index::new();
        let e = entry(0, 0, 128, b"mykey");
        idx.set(b"mykey".to_vec(), e);
        let got = idx.get(b"mykey").expect("should exist");
        assert_eq!(got.segment_id, 0);
        assert_eq!(got.offset, 0);
        assert_eq!(got.size, 128);
    }

    #[test]
    fn test_index_overwrite() {
        let mut idx = Index::new();
        idx.set(b"k".to_vec(), entry(0, 0, 128, b"k"));
        idx.set(b"k".to_vec(), entry(1, 256, 128, b"k"));
        let got = idx.get(b"k").expect("should exist");
        assert_eq!(got.segment_id, 1);
        assert_eq!(got.offset, 256);
    }

    #[test]
    fn test_index_remove() {
        let mut idx = Index::new();
        idx.set(b"k".to_vec(), entry(0, 0, 128, b"k"));
        assert!(idx.remove(b"k"));
        assert!(!idx.remove(b"k")); // already gone
        assert!(idx.get(b"k").is_none());
    }

    #[test]
    fn test_index_missing_key() {
        let idx = Index::new();
        assert!(idx.get(b"ghost").is_none());
    }

    #[test]
    fn test_index_len_and_is_empty() {
        let mut idx = Index::new();
        assert!(idx.is_empty());
        idx.set(b"a".to_vec(), entry(0, 0, 128, b"a"));
        idx.set(b"b".to_vec(), entry(0, 128, 128, b"b"));
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    // ── proptest ─────────────────────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn prop_index_lookup_consistency(
            ops in proptest::collection::vec(
                (
                    proptest::collection::vec(proptest::num::u8::ANY, 1..32),
                    proptest::num::u16::ANY,
                    proptest::num::u32::ANY,
                ),
                1..64,
            ),
        ) {
            let mut idx = Index::new();
            let mut expected: std::collections::HashMap<Vec<u8>, (u16, u32)> =
                std::collections::HashMap::new();

            for (key, seg, off) in &ops {
                let e = IndexEntry {
                    fingerprint: fingerprint(key),
                    segment_id:  *seg,
                    _pad:        0,
                    offset:      *off,
                    size:        128,
                };
                idx.set(key.clone(), e);
                expected.insert(key.clone(), (*seg, *off));
            }

            for (key, (seg, off)) in &expected {
                let got = idx.get(key).expect("must be present");
                proptest::prop_assert_eq!(got.segment_id, *seg);
                proptest::prop_assert_eq!(got.offset,     *off);
            }
        }
    }
}
