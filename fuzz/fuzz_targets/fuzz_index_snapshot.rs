//! Fuzz the index snapshot decoder.
//!
//! Entry point: `skeg_core::snapshot::decode`. The snapshot is read once
//! at server boot as the fast-recovery path. A malformed buffer must
//! return `None` (snapshot rejected, fall back to full scan) - never
//! panic or read past the slice.
//!
//! Adversarial inputs to discover: bad magic/version, hwm/max_ts that
//! mismatch the entry count, entries that lie about key length and
//! overflow the slice, truncated CRC, oversized counts that would
//! allocate excessively in `Vec::with_capacity`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use skeg_core::snapshot::decode;

fuzz_target!(|data: &[u8]| {
    let _ = decode(data);
});
