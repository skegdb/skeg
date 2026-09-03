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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl AdmissionError {
    /// THE classification. Every other answer in this module is derived from
    /// this one, and no other module is allowed a second opinion.
    #[must_use]
    pub fn retryability(self) -> Retryability {
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
        }
    }

    /// The first word of the RESP3 error line, which IS the code a client
    /// routes on.
    #[must_use]
    pub fn resp3_code(self) -> &'static str {
        match self.retryability() {
            Retryability::Retryable => "BACKPRESSURE",
            Retryability::Permanent => "ERR",
        }
    }

    /// The whole RESP3 error line: code word, then the reason.
    #[must_use]
    pub fn wire_message(self) -> String {
        format!("{} {self}", self.resp3_code())
    }

    /// The native error code. Retryable is always
    /// [`ErrCode::Backpressure`]; the permanent codes distinguish a request
    /// the client can fix from an accounting fault it cannot.
    #[must_use]
    pub fn code(self) -> ErrCode {
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
    fn permanent_code(self) -> ErrCode {
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
            | Self::RequestTooLarge { .. } => ErrCode::InvalidRequest,
        }
    }
}

impl From<IngressRejected> for AdmissionError {
    fn from(e: IngressRejected) -> Self {
        Self::Ingress(e)
    }
}

impl From<AdmissionError> for ErrCode {
    fn from(e: AdmissionError) -> Self {
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
        ]
    }

    /// The variant tag, exhaustive. A new variant fails to compile here.
    fn tag(e: AdmissionError) -> &'static str {
        match e {
            AdmissionError::Ingress(_) => "ingress",
            AdmissionError::MemoryAtWrite(_) => "memory_at_write",
            AdmissionError::QuotaExceeded { .. } => "quota_exceeded",
            AdmissionError::RequestTooLarge { .. } => "request_too_large",
        }
    }

    #[test]
    fn the_table_below_covers_every_variant() {
        let mut seen: Vec<&'static str> = every_admission_error()
            .into_iter()
            .map(|(e, _)| tag(e))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen,
            vec![
                "ingress",
                "memory_at_write",
                "quota_exceeded",
                "request_too_large"
            ],
            "a variant nothing samples is a classification nothing checks"
        );
    }

    #[test]
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
            let text = e.to_string();
            assert!(
                !text.starts_with("BACKPRESSURE") && !text.starts_with("ERR "),
                "{e:?} renders as {text:?}: the code word is prepended once, \
                 by wire_message, and a Display that carries it puts it in \
                 the native message where the code is already a byte"
            );
        }
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
