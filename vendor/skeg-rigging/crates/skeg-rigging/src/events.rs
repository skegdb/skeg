//! Tenant event notifications (F.13).
//!
//! M4 roadmap item. Orchestrators that need to react to tenant state
//! changes without polling subscribe to an [`EventStream`] and pull
//! [`Event`]s as they arrive.
//!
//! ## Scope in v0.1
//!
//! - Write-side events ([`Event::RecordInserted`], [`Event::RecordDeleted`])
//!   produced by the adapter on `insert` / `delete`.
//! - Hansa-level events ([`Event::SagaRebuilt`], member join/leave) are
//!   emitted by the hansa crate itself when it touches the saga store
//!   or the registry.
//!
//! The [`TenantEvents`] trait is Provisional: the event surface will
//! grow in M4 (per-vault deltas, tier transitions) and an async
//! variant lands when F.10 ships.

use std::sync::mpsc;
use std::time::{Duration, SystemTime};

use crate::ids::{RecordId, TenantId};

/// One observable change in a tenant's state.
///
/// New variants may appear in minor releases; consumers should
/// `match` non-exhaustively when forward-compat matters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Event {
    /// A new record was written. The producing adapter emits this
    /// after the underlying engine has accepted the insert (delta-log
    /// or main index, engine-defined).
    RecordInserted {
        /// Record identifier.
        record_id: RecordId,
        /// Whether the record opted into peer sharing.
        shareable: bool,
    },
    /// A record was removed from the tenant. Idempotent: deleting a
    /// non-existent id does not emit an event.
    RecordDeleted {
        /// Record identifier that was removed.
        record_id: RecordId,
    },
    /// The tenant's saga digest was rebuilt. Emitted by hansa, not by
    /// the basic adapter - placed here for the shared type surface.
    SagaRebuilt {
        /// The tenant whose saga was rebuilt.
        tenant_id: TenantId,
        /// Wall-clock time of the rebuild.
        built_at: SystemTime,
    },
    /// A new member joined the hansa containing this tenant. Hansa-only.
    HansaMemberJoined {
        /// The joining tenant's id.
        tenant_id: TenantId,
    },
    /// A member left the hansa. Hansa-only.
    HansaMemberLeft {
        /// The leaving tenant's id.
        tenant_id: TenantId,
    },
}

impl Event {
    /// Discriminant used for [`EventFilter`] matching. Cheap; matches
    /// on the variant without copying payload.
    pub fn kind(&self) -> EventKind {
        match self {
            Event::RecordInserted { .. } => EventKind::RecordInserted,
            Event::RecordDeleted { .. } => EventKind::RecordDeleted,
            Event::SagaRebuilt { .. } => EventKind::SagaRebuilt,
            Event::HansaMemberJoined { .. } => EventKind::HansaMemberJoined,
            Event::HansaMemberLeft { .. } => EventKind::HansaMemberLeft,
        }
    }
}

/// Discriminator used by [`EventFilter`]. Matches the variant tags of
/// [`Event`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// Matches [`Event::RecordInserted`].
    RecordInserted,
    /// Matches [`Event::RecordDeleted`].
    RecordDeleted,
    /// Matches [`Event::SagaRebuilt`].
    SagaRebuilt,
    /// Matches [`Event::HansaMemberJoined`].
    HansaMemberJoined,
    /// Matches [`Event::HansaMemberLeft`].
    HansaMemberLeft,
}

/// Subscription filter. v0.1 is a per-kind allow-list; per-record-id
/// filtering is deferred to M4 alongside async streams.
#[derive(Debug, Clone, Copy)]
pub struct EventFilter {
    /// Allow [`Event::RecordInserted`].
    pub record_inserted: bool,
    /// Allow [`Event::RecordDeleted`].
    pub record_deleted: bool,
    /// Allow [`Event::SagaRebuilt`].
    pub saga_rebuilt: bool,
    /// Allow [`Event::HansaMemberJoined`].
    pub hansa_member_joined: bool,
    /// Allow [`Event::HansaMemberLeft`].
    pub hansa_member_left: bool,
}

impl EventFilter {
    /// Match every event variant. Convenient default for "tail
    /// everything" subscribers.
    pub const ALL: EventFilter = EventFilter {
        record_inserted: true,
        record_deleted: true,
        saga_rebuilt: true,
        hansa_member_joined: true,
        hansa_member_left: true,
    };

    /// Match nothing. Useful as a starting point for builder-style
    /// filter construction.
    pub const NONE: EventFilter = EventFilter {
        record_inserted: false,
        record_deleted: false,
        saga_rebuilt: false,
        hansa_member_joined: false,
        hansa_member_left: false,
    };

    /// True when `event` passes this filter.
    pub fn accepts(&self, event: &Event) -> bool {
        match event.kind() {
            EventKind::RecordInserted => self.record_inserted,
            EventKind::RecordDeleted => self.record_deleted,
            EventKind::SagaRebuilt => self.saga_rebuilt,
            EventKind::HansaMemberJoined => self.hansa_member_joined,
            EventKind::HansaMemberLeft => self.hansa_member_left,
        }
    }
}

impl Default for EventFilter {
    fn default() -> Self {
        Self::ALL
    }
}

/// Receiver end of a tenant subscription.
///
/// Wraps `std::sync::mpsc::Receiver` so consumers can pull events
/// blocking or non-blocking. Drop the `EventStream` to unsubscribe -
/// the producing tenant detects the disconnected sender on its next
/// emit and prunes the slot.
pub struct EventStream {
    rx: mpsc::Receiver<Event>,
    filter: EventFilter,
}

impl EventStream {
    /// Construct a stream from a raw receiver. Adapters that produce
    /// events use this; library consumers obtain streams through
    /// [`TenantEvents::subscribe`].
    pub fn new(rx: mpsc::Receiver<Event>, filter: EventFilter) -> Self {
        Self { rx, filter }
    }

    /// Block until an event arrives or the producer disconnects.
    /// Events that don't pass the filter are skipped transparently.
    pub fn recv(&self) -> Result<Event, mpsc::RecvError> {
        loop {
            let ev = self.rx.recv()?;
            if self.filter.accepts(&ev) {
                return Ok(ev);
            }
        }
    }

    /// Try to pull an event without blocking. Filtered-out events
    /// are drained without exposing them to the caller - `Empty`
    /// therefore means "no event passed the filter right now".
    pub fn try_recv(&self) -> Result<Event, mpsc::TryRecvError> {
        loop {
            match self.rx.try_recv() {
                Ok(ev) if self.filter.accepts(&ev) => return Ok(ev),
                Ok(_) => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Pull an event waiting at most `timeout`. Filtered-out events
    /// don't reset the budget - long bursts of filtered traffic
    /// can still result in `Timeout` even when raw events arrive.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Event, mpsc::RecvTimeoutError> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let ev = self.rx.recv_timeout(remaining)?;
            if self.filter.accepts(&ev) {
                return Ok(ev);
            }
        }
    }
}

impl Iterator for EventStream {
    type Item = Event;
    fn next(&mut self) -> Option<Event> {
        self.recv().ok()
    }
}

/// Push-mode notifications for tenant state changes.
///
/// Stability: Provisional.
pub trait TenantEvents {
    /// Subscribe to events. The returned [`EventStream`] is independent
    /// of any other subscriber - every subscriber sees every event
    /// that passes its own filter, broadcast-style.
    fn subscribe(&self, filter: EventFilter) -> EventStream;
}
