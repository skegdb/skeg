//! Gauges that are PULLED at dump time instead of pushed on a hot path.
//!
//! The closed [`crate::Gauge`] enum and the dynamic gauge pool both work the
//! same way: someone stores a number into a static, and the dumper reads the
//! static. That is right for a value produced by an event (a compaction
//! starts, a fold parks). It is wrong for a value that IS a live object's
//! state - a memory governor's headroom, an ingress class's held bytes -
//! because nothing on the write path wants to pay for republishing it, and a
//! number republished only when something happens is stale exactly when
//! nothing is happening.
//!
//! So those objects register themselves, and the dumper asks them. Three
//! rules make that safe:
//!
//! - **Keyed, and one series per key.** A source is registered under a
//!   `&'static str` key and the NEWEST LIVE source for that key is the one
//!   the dump reads. Two series with one name and one label set are not a
//!   Prometheus scrape, so a key cannot report twice; and a process that
//!   builds several servers - every integration test binary does - must not
//!   end with one series per server ever built.
//!
//!   Newest LIVE, not simply newest: an older source that is still alive
//!   takes over when a newer one is dropped, instead of the series vanishing
//!   because the last thing registered happened to go away first.
//! - **Held by [`Weak`].** The registry is a static and lives forever; the
//!   governor of a server that has been dropped must not. A dead source is
//!   skipped and pruned rather than reported as zero, because zero is a
//!   value and "gone" is not.
//! - **Sampled, never blocking.** `sample` is called on the dump path with
//!   the registry lock NOT held for the callee's own work: sources copy a
//!   few atomics.
//!
//! # State sets, not a single live series
//!
//! A gauge with three mutually exclusive states (`known` / `unlimited` /
//! `unknown`) used to be emitted only for the state that was true. That has
//! two failures an operator meets in production: an alert can only be written
//! with `absent()`, and the series that WAS true keeps its last value for the
//! scrape interval after a transition, so a dashboard shows two states at
//! once. A source that reports such a gauge emits ALL of its states every
//! time, exactly one of them at 1 - the Prometheus "state set" convention.

use std::sync::{Mutex, Weak};

/// One gauge reading: a metric name, its label text, and its value.
///
/// `labels` is the text that goes between the braces, `&'static str` on
/// purpose: a label built per scrape is a cardinality bomb waiting for the
/// first `format!` with an id in it. Empty means an unlabelled series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaugeSample {
    pub name: &'static str,
    pub labels: &'static str,
    pub value: u64,
}

impl GaugeSample {
    /// An unlabelled reading.
    #[must_use]
    pub const fn new(name: &'static str, value: u64) -> Self {
        Self {
            name,
            labels: "",
            value,
        }
    }

    /// A reading carrying label text, e.g. `state="known"`.
    #[must_use]
    pub const fn labelled(name: &'static str, labels: &'static str, value: u64) -> Self {
        Self {
            name,
            labels,
            value,
        }
    }
}

/// A live object that can report its gauges when asked.
pub trait GaugeSource: Send + Sync {
    /// Append this object's current readings to `out`.
    fn sample(&self, out: &mut Vec<GaugeSample>);
}

/// Register `source` under `key`. The newest LIVE source for a key is the one
/// the dump reads.
///
/// # Panics
///
/// If the registry mutex was poisoned by a panicking sampler.
pub fn register_gauge_source(key: &'static str, source: Weak<dyn GaugeSource>) {
    let mut sources = SOURCES.lock().expect("the gauge source registry");
    // Dead entries go on every registration, not only on the dump: a process
    // that builds a thousand servers must not carry a thousand dead weaks
    // around waiting for somebody to scrape. Live ones under the same key
    // STAY, and the newest of them is what gets read - see the module note.
    sources.retain(|(_, w)| w.strong_count() > 0);
    sources.push((key, source));
}

/// Every reading every live source reports, in registration order.
///
/// Sources whose object has been dropped are pruned rather than reported as
/// zero: zero is a value, and "gone" is not.
///
/// # Panics
///
/// If the registry mutex was poisoned by a panicking sampler.
#[must_use]
pub fn sample_all() -> Vec<GaugeSample> {
    // The live sources are taken under the lock and sampled OUTSIDE it. A
    // sampler is somebody else's code reached through a trait object; running
    // it while holding a process-wide mutex makes every future registration
    // wait on whatever it decides to do.
    let live: Vec<std::sync::Arc<dyn GaugeSource>> = {
        let mut sources = SOURCES.lock().expect("the gauge source registry");
        sources.retain(|(_, w)| w.strong_count() > 0);
        // One per key, the last registered, in the order the keys were first
        // seen: the dump stays grep-stable across releases and a key never
        // emits its series twice.
        distinct_keys(&sources)
            .into_iter()
            .filter_map(|key| {
                sources
                    .iter()
                    .rev()
                    .find(|(k, _)| *k == key)
                    .and_then(|(_, w)| w.upgrade())
            })
            .collect()
    };
    let mut out = Vec::new();
    for source in live {
        source.sample(&mut out);
    }
    out
}

/// Registered source keys, live ones only. For tests and for `SKEG.STATS`
/// diagnostics.
///
/// # Panics
///
/// If the registry mutex was poisoned by a panicking sampler.
#[must_use]
pub fn live_source_keys() -> Vec<&'static str> {
    let mut sources = SOURCES.lock().expect("the gauge source registry");
    sources.retain(|(_, w)| w.strong_count() > 0);
    distinct_keys(&sources)
}

/// The registered keys, each once, in the order they were first seen.
fn distinct_keys(sources: &[(&'static str, Weak<dyn GaugeSource>)]) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = Vec::new();
    for (k, _) in sources {
        if !keys.contains(k) {
            keys.push(k);
        }
    }
    keys
}

/// Append the pulled gauges to a Prometheus text dump.
///
/// One `# TYPE` line per metric name, emitted before its first sample, so a
/// state set's three series share one declaration.
pub fn dump_text(out: &mut String) {
    use core::fmt::Write;
    let mut typed: Vec<&'static str> = Vec::new();
    for sample in sample_all() {
        if !typed.contains(&sample.name) {
            let _ = writeln!(out, "# TYPE {} gauge", sample.name);
            typed.push(sample.name);
        }
        if sample.labels.is_empty() {
            let _ = writeln!(out, "{} {}", sample.name, sample.value);
        } else {
            let _ = writeln!(out, "{}{{{}}} {}", sample.name, sample.labels, sample.value);
        }
    }
}

/// The registry itself. A `Mutex<Vec<..>>` and not a map: the list is O(10)
/// entries and iteration order is the registration order, which keeps the
/// dump stable for `grep`.
static SOURCES: Mutex<Vec<(&'static str, Weak<dyn GaugeSource>)>> = Mutex::new(Vec::new());

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A source under the test's own key, so two tests in this binary cannot
    /// see each other's registrations.
    struct Fixed(Vec<GaugeSample>);

    impl GaugeSource for Fixed {
        fn sample(&self, out: &mut Vec<GaugeSample>) {
            out.extend_from_slice(&self.0);
        }
    }

    fn samples_of(key: &'static str) -> Vec<GaugeSample> {
        sample_all().into_iter().filter(|s| s.name == key).collect()
    }

    #[test]
    fn a_registered_source_is_asked_at_dump_time() {
        let src = Arc::new(Fixed(vec![GaugeSample::new("t_asked", 42)]));
        register_gauge_source("t_asked", Arc::downgrade(&src) as Weak<dyn GaugeSource>);
        assert_eq!(
            samples_of("t_asked"),
            vec![GaugeSample::new("t_asked", 42)],
            "the dumper must read the live object, not a copy taken earlier"
        );
    }

    #[test]
    fn registering_the_same_key_twice_reports_only_the_newest() {
        let first = Arc::new(Fixed(vec![GaugeSample::new("t_idem", 1)]));
        let second = Arc::new(Fixed(vec![GaugeSample::new("t_idem", 2)]));
        register_gauge_source("t_idem", Arc::downgrade(&first) as Weak<dyn GaugeSource>);
        register_gauge_source("t_idem", Arc::downgrade(&second) as Weak<dyn GaugeSource>);
        assert_eq!(
            samples_of("t_idem"),
            vec![GaugeSample::new("t_idem", 2)],
            "two series with one name and one label set are not a scrape; and \
             every integration test builds a server, so a registry that \
             reported all of them would report one per server ever built"
        );
        assert_eq!(
            live_source_keys()
                .iter()
                .filter(|k| **k == "t_idem")
                .count(),
            1,
            "one key, one series"
        );
    }

    #[test]
    fn an_older_live_source_takes_over_when_the_newest_is_dropped() {
        // Not a curiosity: a test binary builds several servers, and if the
        // series vanished whenever whichever registered LAST happened to be
        // dropped first, every assertion about the dump would be a race
        // against the test finishing next to it.
        let first = Arc::new(Fixed(vec![GaugeSample::new("t_fallback", 1)]));
        let second = Arc::new(Fixed(vec![GaugeSample::new("t_fallback", 2)]));
        register_gauge_source(
            "t_fallback",
            Arc::downgrade(&first) as Weak<dyn GaugeSource>,
        );
        register_gauge_source(
            "t_fallback",
            Arc::downgrade(&second) as Weak<dyn GaugeSource>,
        );
        assert_eq!(
            samples_of("t_fallback"),
            vec![GaugeSample::new("t_fallback", 2)]
        );
        drop(second);
        assert_eq!(
            samples_of("t_fallback"),
            vec![GaugeSample::new("t_fallback", 1)],
            "the older source is still alive and still reporting"
        );
        drop(first);
        assert!(samples_of("t_fallback").is_empty());
    }

    #[test]
    fn a_dropped_source_disappears_instead_of_reporting_zero() {
        let src = Arc::new(Fixed(vec![GaugeSample::new("t_dropped", 7)]));
        register_gauge_source("t_dropped", Arc::downgrade(&src) as Weak<dyn GaugeSource>);
        assert_eq!(samples_of("t_dropped").len(), 1);
        drop(src);
        assert!(
            samples_of("t_dropped").is_empty(),
            "zero is a value; a governor that no longer exists has none"
        );
        assert!(
            !live_source_keys().contains(&"t_dropped"),
            "the dead entry must be pruned, not merely skipped"
        );
    }

    #[test]
    fn a_three_state_gauge_reports_all_its_states_with_exactly_one_at_one() {
        let src = Arc::new(Fixed(vec![
            GaugeSample::labelled("t_state", "state=\"known\"", 1),
            GaugeSample::labelled("t_state", "state=\"unlimited\"", 0),
            GaugeSample::labelled("t_state", "state=\"unknown\"", 0),
        ]));
        register_gauge_source("t_state", Arc::downgrade(&src) as Weak<dyn GaugeSource>);
        let got = samples_of("t_state");
        assert_eq!(got.len(), 3, "all three series, always");
        assert_eq!(
            got.iter().filter(|s| s.value == 1).count(),
            1,
            "exactly one state is true"
        );
    }

    #[test]
    fn the_dump_declares_each_metric_name_once() {
        let src = Arc::new(Fixed(vec![
            GaugeSample::labelled("t_dump", "state=\"a\"", 1),
            GaugeSample::labelled("t_dump", "state=\"b\"", 0),
        ]));
        register_gauge_source("t_dump", Arc::downgrade(&src) as Weak<dyn GaugeSource>);
        let mut out = String::new();
        dump_text(&mut out);
        assert_eq!(
            out.matches("# TYPE t_dump gauge").count(),
            1,
            "a state set shares one TYPE line; two would make the scrape invalid"
        );
        assert!(out.contains("t_dump{state=\"a\"} 1\n"));
        assert!(out.contains("t_dump{state=\"b\"} 0\n"));
    }
}
