# ADR: the per-command admission seam (Admission + CommandKind)

Status: **accepted** (branch `feat/tenant-admit`).
Context: extends [multi-tenancy.md](multi-tenancy.md). Sibling of the tiering
seam ([adr-tenant-tiering-seam.md](adr-tenant-tiering-seam.md)) - same rule:
generic mechanism in the Apache engine, policy in the commercial backend.

## Context

`TenantBackend::admit` started as `admit(&self, id, cost)`: a per-command gate
that a backend uses for QoS (rate budget, concurrency cap). Two needs push on it:

- **RBAC per command** - an operator wants "this tenant may not run VINDEX.DROP".
  The gate must know *what kind* of command it is, not just who and how costly.
- **Per-operation metering** - billing/observability want counts per command
  class, not just per tenant.

Both are the same shape as admission: per-command, pre-execution, may refuse. The
question is how to extend the *public* seam without a churny migration later, and
without leaking engine internals.

## Decision

### One hook, carrying a typed command kind

`admit(&self, admission: Admission) -> Result<AdmitGuard, AdmitRejected>`, where

```rust
#[non_exhaustive]
pub struct Admission {
    pub tenant: TenantId,
    pub op: CommandKind,   // what the command is, for RBAC + per-op metering
    pub cost: u32,         // coarse QoS credits (the engine cost model)
}

#[non_exhaustive]
pub enum CommandKind {     // resource x action; lifecycle ops are individual
    KvRead, KvWrite,
    VectorRead, VectorWrite,
    VindexCreate, VindexDrop, VindexConsolidate, VindexList,
    Admin, Meta,
}
```

- **One gate, not two.** Authorization ("may run op?") and admission ("has
  budget?") are conceptually distinct but both are per-command, pre-execution,
  refusable. Keep a single `admit` hook now; split into `authorize` + `admit`
  only if a real need separates them. Premature decomposition is the more common
  mistake. The name `admit` still reads ("admit this command").
- **Pass a struct, not loose params.** `Admission` is `#[non_exhaustive]`, so
  future fields (payload size, a fairness key) are added without changing the
  signature - the seam is stable at the struct boundary, not just the arg list.
- **A typed, extensible enum, not a string.** `CommandKind` is an exhaustive
  match for the engine (a new command must classify itself) and
  `#[non_exhaustive]` for implementors (a new variant is not a breaking change).
  The grain is resource x action, with lifecycle ops individual because
  `VindexDrop` (destructive) is the natural RBAC target, distinct from
  `VindexCreate`.
- **Do not leak the internal `Command`.** The RESP3 `Command` enum is a parser
  type. The engine classifies it once (`command_kind(&Command)`) into the public
  `CommandKind`; the trait never sees the parser type.
- **The engine constructs `Admission`; a backend only reads it.** No public
  constructor: production backends receive it from the dispatcher. (Backends test
  their admission logic at the policy layer, or end-to-end through real dispatch.)

### Why stabilize the signature now, implement RBAC later

The hook is a *published* trait seam. Getting its shape right once is cheaper than
a v2 migration across implementors. So the signature lands now; the RBAC decision
logic (which tenant may run which op) is deferred until there is a policy to
enforce. The `op` field is simply unused by the default and current backends
until then.

## Consequences

- **Blast radius.** Adding a field to a default trait method is source-breaking
  only for *overriders*. The OSS engine ships the default no-op `admit`; the only
  override is the commercial backend - a one-line change. Pre-1.0, this is the
  right time to take it.
- The default `admit` still admits everything and ignores `Admission`, so
  single-tenant and existing backends are unaffected.
- `command_kind` is the single classification point; per-op metering and RBAC both
  consume it, so they never drift from how a command is dispatched.

## What this ADR does NOT decide

- The RBAC policy model (where rules live, how they are expressed/stored) - only
  that the seam carries enough to enforce them.
- The per-operation metering schema.
- Whether `admit` ever splits into `authorize` + `admit` - revisit only if a
  concrete need separates the two.
