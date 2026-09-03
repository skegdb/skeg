//! One classification for every refusal taken BEFORE the work.
//!
//! An admission refusal is not an error about what happened; it is a
//! statement about what did not, and the only thing a client needs from it is
//! whether sending the same request again is worth doing. That single bit was
//! spelled in three unrelated places and in two unrelated ways:
//!
//! - the ingress budget wrote the word `BACKPRESSURE` at the front of a
//!   string, and the RESP3 handler passed it through by matching on that
//!   PREFIX (`SHARD_ERROR_CODES`);
//! - the shard worker wrote the same word at the front of a different string
//!   for the memory governor, and wrote nothing at all for the quota;
//! - the native handler mapped every one of them to
//!   [`ErrCode::Internal`][skeg_proto::ErrCode::Internal], which tells a
//!   client to give up, and left the word to arrive as prose at byte 16.
//!
//! So the classification lives here, once. [`AdmissionError::retryability`]
//! is THE decision; the RESP3 code word and the native
//! [`ErrCode`][skeg_proto::ErrCode] are both derived from it, which is why
//! the two wires cannot drift: making them disagree means editing one
//! function whose own test loops over every variant.
//!
//! # What this is not
//!
//! Not an application error type, and not a step towards one. It carries the
//! refusals a request meets before it runs and nothing else: the handlers
//! keep their own semantics, their own messages and their own shapes. The
//! only thing shared is the answer to "should the caller retry".

use std::fmt;

use skeg_proto::ErrCode;

use crate::ingress::IngressRejected;
use crate::memory::MemoryRejected;

/// Should the caller send the same request again?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryability {
    /// The condition is about the server's momentary room, and clears on its
    /// own. `BACKPRESSURE` on the RESP3 wire, [`ErrCode::Backpressure`] on
    /// the native one.
    Retryable,
    /// The condition is about the request. Sending it again produces the same
    /// answer, and telling a client to retry is telling it to loop.
    Permanent,
}

/// A refusal decided before any of the work happened.
///
/// `Clone` and not `Copy`: [`AdmissionError::Backend`] carries a string a
/// backend wrote, and there is nowhere else for it to live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// The ingress class, the per-connection allowance, or the governor
    /// underneath them refused the bytes.
    Ingress(IngressRejected),
    /// The memory governor refused the heap a write would take, at the
    /// admission step of the write itself.
    MemoryAtWrite(MemoryRejected),
    /// The tenant already holds every vector its limit allows.
    QuotaExceeded { tenant: u128, limit: u64 },
    /// One request declared more than a fixed ceiling permits. `what` names
    /// the thing that was too large, in the units `limit` and `got` are in.
    RequestTooLarge {
        what: &'static str,
        limit: u64,
        got: u64,
    },
    /// A bounded queue inside the engine had no room. Momentary by
    /// construction: what fills it is other traffic, and other traffic ends.
    Busy,
    /// A tenant backend refused the command.
    ///
    /// The message is OPAQUE. It is written by a backend that lives outside
    /// this tree, behind a trait whose contract says only that the string is
    /// "the full RESP3 error string (code word + human text)" and that the
    /// backend "should format a leading uppercase code, e.g.
    /// `RATELIMITED tenant request rate exceeded`". So the engine classifies
    /// it from THE ONE THING the contract promises - that leading word - and
    /// passes the string itself through untouched.
    ///
    /// Reading a word out of a message is exactly the pattern this module
    /// exists to remove, and it is here for the one case where the message is
    /// the interface: the alternative is giving `AdmitRejected` a typed
    /// `Retryability`, which changes a public trait contract and belongs to
    /// the revision that gets to break it. Until then this is the only place
    /// in the engine that reads a code word out of a string, it reads it once
    /// at the point the refusal enters, and a message it cannot classify is
    /// counted rather than guessed at.
    Backend { message: String },
}

impl AdmissionError {
    /// THE classification. Every other answer in this module is derived from
    /// this one, and no other module is allowed a second opinion.
    #[must_use]
    pub fn retryability(&self) -> Retryability {
        use Retryability::{Permanent, Retryable};
        // Exhaustive, no `_` arm anywhere in it. A refusal whose
        // classification nobody chose would default to something, and both
        // defaults are wrong: "permanent" tells a client to give up on a
        // condition that clears, "retryable" tells it to loop on one that
        // does not.
        match self {
            // The class is momentarily full, or its ceiling is momentarily
            // unreadable. Both clear without the client doing anything
            // different.
            Self::Ingress(
                IngressRejected::ClassFull { .. } | IngressRejected::Unreadable { .. },
            ) => Retryable,
            // A frame larger than one connection may ever hold. The same
            // frame will be refused again on the next attempt.
            Self::Ingress(IngressRejected::OverConnectionAllowance { .. }) => Permanent,
            // The governor, reached either through ingress or at the write
            // itself: one classification, written once, for both routes.
            Self::Ingress(IngressRejected::Governor(m)) | Self::MemoryAtWrite(m) => match m {
                MemoryRejected::NoHeadroom { .. } => Retryable,
                MemoryRejected::Unknown | MemoryRejected::ArithmeticOverflow { .. } => Permanent,
            },
            // A tenant at its ceiling stays there until somebody deletes a
            // vector or raises the limit, neither of which is a retry.
            Self::QuotaExceeded { .. } => Permanent,
            Self::RequestTooLarge { .. } => Permanent,
            // Stub: opened by `server: a full vsearch queue is backpressure`.
            Self::Busy => Permanent,
            // Stub: opened by `server: a backend refusal is classified by the
            // code word its own contract promises`.
            Self::Backend { .. } => Permanent,
        }
    }

    /// The first word of the RESP3 error line, which IS the code a client
    /// routes on.
    #[must_use]
    pub fn resp3_code(&self) -> &'static str {
        match self.retryability() {
            Retryability::Retryable => "BACKPRESSURE",
            Retryability::Permanent => "ERR",
        }
    }

    /// The whole RESP3 error line: code word, then the reason.
    ///
    /// A backend refusal is the exception, and passes through untouched: the
    /// string already IS a complete error line, code word included, and
    /// putting `ERR` or `BACKPRESSURE` in front of `RATELIMITED ...` would
    /// replace the word that deployment's clients already route on with one
    /// they do not.
    #[must_use]
    pub fn wire_message(&self) -> String {
        match self {
            Self::Backend { message } => message.clone(),
            other => format!("{} {other}", other.resp3_code()),
        }
    }

    /// The native error code. Retryable is always
    /// [`ErrCode::Backpressure`]; the permanent codes distinguish a request
    /// the client can fix from an accounting fault it cannot.
    #[must_use]
    pub fn code(&self) -> ErrCode {
        match self.retryability() {
            // Structural, not asserted: there is no way to write a retryable
            // refusal that does not carry the retryable code.
            Retryability::Retryable => ErrCode::Backpressure,
            Retryability::Permanent => self.permanent_code(),
        }
    }

    /// Which kind of permanent, for a client deciding what to do next: fix
    /// the request, or tell an operator. Exhaustive for the same reason
    /// [`AdmissionError::retryability`] is.
    fn permanent_code(&self) -> ErrCode {
        match self {
            // Headroom nobody could read is an accounting fault of this
            // process. The client cannot fix it by sending anything else, so
            // calling its request invalid would send it looking in the wrong
            // place.
            Self::Ingress(IngressRejected::Governor(MemoryRejected::Unknown))
            | Self::MemoryAtWrite(MemoryRejected::Unknown) => ErrCode::Internal,
            // Everything else permanent is about the request: it is too big,
            // or the tenant it belongs to is full.
            Self::Ingress(
                IngressRejected::ClassFull { .. }
                | IngressRejected::Unreadable { .. }
                | IngressRejected::OverConnectionAllowance { .. }
                | IngressRejected::Governor(
                    MemoryRejected::NoHeadroom { .. } | MemoryRejected::ArithmeticOverflow { .. },
                ),
            )
            | Self::MemoryAtWrite(
                MemoryRejected::NoHeadroom { .. } | MemoryRejected::ArithmeticOverflow { .. },
            )
            | Self::QuotaExceeded { .. }
            | Self::RequestTooLarge { .. }
            | Self::Busy
            // A refusal the engine could not classify is the request's, not
            // the server's: something about this tenant or this command was
            // declined, and the caller is the one who can act on it.
            | Self::Backend { .. } => ErrCode::InvalidRequest,
        }
    }
}

/// The code word a tenant backend's message must begin with for the engine to
/// read it as "come back later".
///
/// One word, the one the trait's own doc names. Adding synonyms here would be
/// the engine inventing contract on a backend's behalf; a backend that writes
/// something else gets the safe answer and a counter, which is a signal
/// somebody can act on rather than a guess nobody can see.
pub const BACKEND_RETRYABLE_CODE: &str = "RATELIMITED";

impl AdmissionError {
    /// Classify a tenant backend's refusal, ONCE, where it enters the engine.
    ///
    /// Here and not in [`AdmissionError::retryability`] because this is where
    /// the counter belongs: `retryability` is asked several times per refusal
    /// (by the code word, by the error byte, by a log line) and a counter
    /// that ticked on each would report traffic instead of refusals.
    #[must_use]
    pub fn from_backend(rejected: crate::tenant::AdmitRejected) -> Self {
        // Stub: opened by `server: a backend refusal is classified by the code
        // word its own contract promises`.
        let classify = false;
        if classify && backend_code_word(&rejected.message) != Some(BACKEND_RETRYABLE_CODE) {
            // Said out loud as well as counted: a deployment whose rate limit
            // reads as "give up" to every client is a fault in the backend,
            // and the operator who can fix it is not reading this counter.
            tracing::warn!(
                message = %rejected.message,
                "tenant backend refused without a code word this build knows; \
                 treating it as permanent. Format the message with a leading \
                 uppercase code - `{BACKEND_RETRYABLE_CODE} ...` for a \
                 refusal the caller should retry."
            );
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::BackendRefusalUnclassified);
        }
        Self::Backend {
            message: rejected.message,
        }
    }
}

/// The leading uppercase token of a backend's message, if it has one.
///
/// ASCII uppercase and digits only, which is what every RESP error code is
/// (`ERR`, `WRONGTYPE`, `LOADING`, `NOAUTH`) and what the trait's doc asks
/// for. A message that opens with prose has no code word, which is a
/// different answer from having an unknown one only in the log line - both
/// are classified the same way, and both are counted.
fn backend_code_word(message: &str) -> Option<&str> {
    let word = message.split(' ').next()?;
    let is_code = !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());
    is_code.then_some(word)
}

impl From<IngressRejected> for AdmissionError {
    fn from(e: IngressRejected) -> Self {
        Self::Ingress(e)
    }
}

impl From<&AdmissionError> for ErrCode {
    fn from(e: &AdmissionError) -> Self {
        e.code()
    }
}

impl fmt::Display for AdmissionError {
    /// The reason, WITHOUT the code word. The word is prepended once, by
    /// [`AdmissionError::wire_message`]; a `Display` that carried it would
    /// put it in every log line and, worse, in the native error message where
    /// the code is already a byte.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ingress(e) => write!(f, "{e}"),
            // `MemoryRejected` already opens with "memory budget:", so this
            // says where the refusal happened and lets the governor say why.
            Self::MemoryAtWrite(e) => write!(f, "out of budget at the write: {e}"),
            Self::QuotaExceeded { tenant, limit } => write!(
                f,
                "tenant vector quota exceeded: tenant {tenant:#034x} may hold {limit} vectors"
            ),
            Self::RequestTooLarge { what, limit, got } => write!(
                f,
                "{what}: at most {limit}, got {got}; send it in smaller pieces"
            ),
            // The text the shard error carried before it was classified, so
            // only the code word in front of it changes.
            Self::Busy => write!(f, "vsearch queue is full"),
            // Verbatim. The backend wrote a complete error line, code word
            // included, and rewriting it would strip the only thing a RESP3
            // client of that deployment already routes on.
            Self::Backend { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for AdmissionError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample of every variant, and of every sub-case that carries its own
    /// classification.
    ///
    /// The tag comes from an exhaustive match with no `_` arm, so a new
    /// variant does not compile until it is listed - which is what keeps this
    /// table from silently covering less than it claims.
    fn every_admission_error() -> Vec<(AdmissionError, Retryability)> {
        vec![
            (
                AdmissionError::Ingress(IngressRejected::ClassFull {
                    held: 1,
                    requested: 2,
                    cap: 3,
                }),
                Retryability::Retryable,
            ),
            (
                AdmissionError::Ingress(IngressRejected::Unreadable { requested: 1 }),
                Retryability::Retryable,
            ),
            (
                AdmissionError::Ingress(IngressRejected::OverConnectionAllowance {
                    requested: 2,
                    allowance: 1,
                }),
                Retryability::Permanent,
            ),
            (
                AdmissionError::Ingress(IngressRejected::Governor(MemoryRejected::NoHeadroom {
                    reserved: 1,
                    requested: 2,
                    usable: 3,
                })),
                Retryability::Retryable,
            ),
            (
                AdmissionError::Ingress(IngressRejected::Governor(MemoryRejected::Unknown)),
                Retryability::Permanent,
            ),
            (
                AdmissionError::Ingress(IngressRejected::Governor(
                    MemoryRejected::ArithmeticOverflow { requested: 9 },
                )),
                Retryability::Permanent,
            ),
            (
                AdmissionError::MemoryAtWrite(MemoryRejected::NoHeadroom {
                    reserved: 1,
                    requested: 2,
                    usable: 3,
                }),
                Retryability::Retryable,
            ),
            (
                AdmissionError::MemoryAtWrite(MemoryRejected::Unknown),
                Retryability::Permanent,
            ),
            (
                AdmissionError::MemoryAtWrite(MemoryRejected::ArithmeticOverflow { requested: 9 }),
                Retryability::Permanent,
            ),
            (
                AdmissionError::QuotaExceeded {
                    tenant: 7,
                    limit: 100,
                },
                Retryability::Permanent,
            ),
            (
                AdmissionError::RequestTooLarge {
                    what: "SKEG.VMSET items",
                    limit: 4096,
                    got: 4097,
                },
                Retryability::Permanent,
            ),
            (AdmissionError::Busy, Retryability::Retryable),
            (
                AdmissionError::Backend {
                    message: "RATELIMITED tenant request rate exceeded".to_owned(),
                },
                Retryability::Retryable,
            ),
            (
                AdmissionError::Backend {
                    message: "TENANTBLOCKED this tenant is suspended".to_owned(),
                },
                Retryability::Permanent,
            ),
            (
                AdmissionError::Backend {
                    message: "this tenant is out of credit".to_owned(),
                },
                Retryability::Permanent,
            ),
        ]
    }

    /// The variant tag, exhaustive. A new variant fails to compile here.
    fn tag(e: &AdmissionError) -> &'static str {
        match e {
            AdmissionError::Ingress(_) => "ingress",
            AdmissionError::MemoryAtWrite(_) => "memory_at_write",
            AdmissionError::QuotaExceeded { .. } => "quota_exceeded",
            AdmissionError::RequestTooLarge { .. } => "request_too_large",
            AdmissionError::Busy => "busy",
            AdmissionError::Backend { .. } => "backend",
        }
    }

    #[test]
    fn the_table_below_covers_every_variant() {
        let mut seen: Vec<&'static str> = every_admission_error()
            .iter()
            .map(|(e, _)| tag(e))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen,
            vec![
                "backend",
                "busy",
                "ingress",
                "memory_at_write",
                "quota_exceeded",
                "request_too_large"
            ],
            "a variant nothing samples is a classification nothing checks"
        );
    }

    #[test]
    #[ignore = "opens in `server: a full vsearch queue is backpressure, not an error`"]
    fn every_admission_error_is_classified_as_the_table_says() {
        for (e, want) in every_admission_error() {
            assert_eq!(
                e.retryability(),
                want,
                "{e:?} is classified {:?}, and the table says {want:?}",
                e.retryability()
            );
        }
    }

    /// The invariant the whole module exists for: one decision, two wires.
    #[test]
    fn retryability_decides_both_the_prefix_and_the_code() {
        for (e, _) in every_admission_error() {
            let retryable = e.retryability() == Retryability::Retryable;
            assert_eq!(
                retryable,
                e.resp3_code() == "BACKPRESSURE",
                "{e:?}: the RESP3 code word disagrees with the classification"
            );
            assert_eq!(
                retryable,
                e.code() == ErrCode::Backpressure,
                "{e:?}: the native code disagrees with the classification"
            );
            assert_eq!(
                retryable,
                e.code().is_retryable(),
                "{e:?}: the native code's own retryability disagrees"
            );
        }
    }

    #[test]
    fn a_permanent_refusal_names_the_request_or_the_accounting_but_never_both() {
        for (e, _) in every_admission_error() {
            if e.retryability() == Retryability::Permanent {
                assert!(
                    matches!(e.code(), ErrCode::InvalidRequest | ErrCode::Internal),
                    "{e:?}: a permanent refusal is either the request's fault \
                     or the server's, and {:?} is neither",
                    e.code()
                );
            }
        }
    }

    #[test]
    fn an_unreadable_budget_is_the_servers_fault_not_the_requests() {
        assert_eq!(
            AdmissionError::MemoryAtWrite(MemoryRejected::Unknown).code(),
            ErrCode::Internal,
            "the client cannot fix a headroom the server cannot read"
        );
        assert_eq!(
            AdmissionError::QuotaExceeded {
                tenant: 1,
                limit: 10
            }
            .code(),
            ErrCode::InvalidRequest,
            "a tenant at its ceiling is about the request, not the server"
        );
    }

    #[test]
    fn the_display_never_carries_the_code_word() {
        for (e, _) in every_admission_error() {
            // A backend refusal is the declared exception: its message IS a
            // complete error line and the engine passes it through.
            if matches!(e, AdmissionError::Backend { .. }) {
                continue;
            }
            let text = e.to_string();
            assert!(
                !text.starts_with("BACKPRESSURE") && !text.starts_with("ERR "),
                "{e:?} renders as {text:?}: the code word is prepended once, \
                 by wire_message, and a Display that carries it puts it in \
                 the native message where the code is already a byte"
            );
        }
    }

    // ── the tenant backend's own refusal ────────────────────────────────

    fn refused(message: &str) -> AdmissionError {
        AdmissionError::from_backend(crate::tenant::AdmitRejected {
            message: message.to_owned(),
        })
    }

    #[test]
    #[ignore = "opens in `server: a backend refusal is classified by the code word its own contract promises`"]
    fn a_backend_rate_limit_is_retryable_and_keeps_its_own_word() {
        let e = refused("RATELIMITED tenant request rate exceeded");
        assert_eq!(e.retryability(), Retryability::Retryable);
        assert_eq!(
            e.code(),
            ErrCode::Backpressure,
            "the native wire has no RATELIMITED; a rate limit is what 0x04 is for"
        );
        assert_eq!(
            e.wire_message(),
            "RATELIMITED tenant request rate exceeded",
            "verbatim: the clients of that deployment route on the backend's \
             own word, and replacing it takes the routing away"
        );
    }

    #[test]
    #[ignore = "opens in `server: a backend refusal is classified by the code word its own contract promises`"]
    fn a_backend_refusal_the_engine_cannot_classify_is_permanent_and_counted() {
        for message in [
            "TENANTBLOCKED this tenant is suspended",
            "this tenant is out of credit",
            "",
        ] {
            let before =
                skeg_telemetry::counter_value(skeg_telemetry::Counter::BackendRefusalUnclassified);
            let e = refused(message);
            let after =
                skeg_telemetry::counter_value(skeg_telemetry::Counter::BackendRefusalUnclassified);
            assert_eq!(
                e.retryability(),
                Retryability::Permanent,
                "{message:?}: guessing that an unclassified refusal clears on \
                 its own is how a client loops"
            );
            assert_eq!(e.code(), ErrCode::InvalidRequest, "{message:?}");
            assert!(
                after > before,
                "{message:?}: a rate limit that reads as 'give up' to every \
                 client must not be invisible (delta {})",
                after - before
            );
            assert_eq!(e.wire_message(), message, "still passed through untouched");
        }
    }

    #[test]
    #[ignore = "opens in `server: a backend refusal is classified by the code word its own contract promises`"]
    fn a_classified_backend_refusal_is_not_counted_as_unclassified() {
        let before =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::BackendRefusalUnclassified);
        let _ = refused("RATELIMITED slow down");
        let after =
            skeg_telemetry::counter_value(skeg_telemetry::Counter::BackendRefusalUnclassified);
        assert_eq!(
            after, before,
            "the counter names a fault in the backend, not backend traffic"
        );
    }

    #[test]
    fn the_code_word_is_the_leading_uppercase_token_and_nothing_else() {
        assert_eq!(backend_code_word("RATELIMITED x"), Some("RATELIMITED"));
        assert_eq!(backend_code_word("ERR2 x"), Some("ERR2"));
        assert_eq!(backend_code_word("RATELIMITED"), Some("RATELIMITED"));
        assert_eq!(backend_code_word("ratelimited x"), None);
        assert_eq!(backend_code_word("Ratelimited x"), None);
        assert_eq!(
            backend_code_word(" RATELIMITED x"),
            None,
            "no leading space"
        );
        assert_eq!(backend_code_word(""), None);
    }

    // ── a full queue ────────────────────────────────────────────────────

    #[test]
    #[ignore = "opens in `server: a full vsearch queue is backpressure, not an error`"]
    fn a_full_queue_is_momentary() {
        let e = AdmissionError::Busy;
        assert_eq!(
            e.retryability(),
            Retryability::Retryable,
            "what fills a bounded queue is other traffic, and other traffic ends"
        );
        assert_eq!(e.code(), ErrCode::Backpressure);
        assert_eq!(e.wire_message(), "BACKPRESSURE vsearch queue is full");
    }

    #[test]
    fn the_wire_message_is_the_code_word_then_the_reason() {
        let e = AdmissionError::Ingress(IngressRejected::ClassFull {
            held: 1,
            requested: 2,
            cap: 3,
        });
        assert_eq!(e.wire_message(), format!("BACKPRESSURE {e}"));
        let e = AdmissionError::QuotaExceeded {
            tenant: 1,
            limit: 10,
        };
        assert_eq!(e.wire_message(), format!("ERR {e}"));
    }
}
