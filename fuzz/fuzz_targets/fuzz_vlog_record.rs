//! Fuzz the vLog record decoder.
//!
//! Entry point: `skeg_core::record::decode_record`. Records are written
//! one per append and parsed sequentially at recovery; a malformed buffer
//! must either return `Ok(_)` (good record) or `Err(_)` (bad CRC, short,
//! invalid kind), never panic or read past the slice.
//!
//! Adversarial inputs to discover: CRC mismatches with valid-looking
//! length fields, key/value sizes that overflow the slice, kind byte
//! outside the documented set, padding that lies about its size.

#![no_main]

use libfuzzer_sys::fuzz_target;
use skeg_core::record::decode_record;

fuzz_target!(|data: &[u8]| {
    let _ = decode_record(data);
});
