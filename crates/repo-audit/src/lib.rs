//! Repository hygiene checks (dev-only).
//!
//! This crate ships no code; its value is in `tests/hygiene.rs`, which scans
//! the workspace sources and fails `cargo test` if it finds internal project
//! codes, non-English (Italian) text, or em-dashes in the public tree. It is
//! the automated form of the manual audits that used to be done by grep.
