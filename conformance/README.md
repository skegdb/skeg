# Conformance cases

Single source of truth for what the skeg server answers. Lives in the
engine repo, next to the server it describes, so every client repo - and
their CI - can read it. Every client repo
(skeg-py, skeg-client-rs, and anything built on them) runs its own runner over
these same files, driving its own SDK. A command added to the server is one new
line here, not one new test per client.

## Files

| file | what |
|---|---|
| `resp3-cases.jsonl` | RESP2/RESP3 wire, one case per line |
| `native-cases.jsonl` | native binary protocol, at the client-API level |
| `validate.py` | runs the RESP3 cases against a live server, over a raw socket |
| `validate_native.py` | same for the native protocol, building frames byte by byte |

The two validators exist to keep the case files honest. They speak the wire
directly and share no code with any client, so a case can never be "proved" by
the same client bug it was written to catch. They are **not** the client
runners: each client repo ships its own, driving its own SDK.

## Running

```sh
cargo build --release --bin skeg --bin skeg-resp3
python3 conformance/validate.py        --bin target/release/skeg-resp3
python3 conformance/validate_native.py --bin target/release/skeg
```

A client repo points `SKEG_CONFORMANCE_DIR` at this directory and runs
its own runner over the same files:

```sh
git clone --depth=1 https://github.com/skegdb/skeg /tmp/skeg
SKEG_CONFORMANCE_DIR=/tmp/skeg/conformance pytest      # or cargo test
```

Both spawn their own server on an ephemeral port with a throwaway data dir, and
exit non-zero on any failure.

## Case format

RESP3 case: `cmd` is the command as a list of arguments, `want` is a single
matcher.

```json
{"id":"kv.get.basic","cmd":["GET","cf:k1"],"want":{"bulk":"v1"}}
```

Native case: `op` names the client method, `args` its arguments, `version` the
frame version (1 or 2).

```json
{"id":"vindex.create.v2.tq2","op":"vindex_create","version":2,
 "args":{"name":"cfntq2","dim":8,"kind":4,"backend":1},"want":{"op":"ok"}}
```

**Argument escapes.** Arguments are text by default. `b64:<base64>` carries
arbitrary bytes; `f32:[1,0,0]` carries a little-endian f32 vector. Both work
inside `want` too, so a binary round-trip is one line.

**Order matters.** Cases run top to bottom on one connection, and a case may
rely on the state an earlier one left. `HELLO` cases get their own connection
(they renegotiate the protocol version, which would strand the shared one).

**Namespacing.** KV keys are prefixed `cf:`. VINDEX names cannot hold `:`
(the server allows only `[A-Za-z0-9._-]`), so indexes are prefixed `cfidx` (RESP3)
and `cfn` (native).

### Matchers

RESP3: `simple`, `bulk`, `bulk_contains` (list of fragments), `integer`,
`integer_min`, `double` (asserts the type, not the value), `null`,
`error_contains`, `array_len`, `items` (per position, each a nested matcher),
`map_has` (list of keys), `map_field` (key to expected value), `any` (anything
but an error).

Native: `op: "ok"`, `version`, `req_id`, `value`, `bool`, `mget` (list, `null`
for a miss), `rows_contain`, `hits_top_id`, `hits_len`, `error_contains`,
`error_code`, `closed_or_error`.

### Case flags

- `profile`: which server setup the case needs. `anon` (default) is a plain
  single-tenant server; `tenant` and `admin` need a multi-tenant backend.
- `bug`: the case documents a **server** defect and is expected to fail. When it
  starts passing, the validator reports it as loudly as a failure so the marker
  gets removed with the fix.
- `unvalidated`: written from the source but never actually run, with the reason.
  Treat as a draft, not as a contract.
- `note`: context for a surprising expectation.

## Rules

1. **The server is the truth.** A case that disagrees with the server is wrong
   until the server is proven wrong. When the server really is wrong, the case
   keeps the true expectation and gets a `bug` marker. Never bend it to match
   the defect.
2. **Add the case first.** New server command, or a client that needs to speak
   one: the case lands here first and is seen failing, then the client changes.
3. **Every case runs somewhere.** A case no runner executes is a comment. Mark
   it `unvalidated` with the reason, or delete it.

## Current state

| suite | result |
|---|---|
| `resp3-cases.jsonl`, profile `anon` | 106/106 pass |
| `resp3-cases.jsonl`, profiles `tenant` / `admin` | 12 cases, **unvalidated** |
| `native-cases.jsonl` | 49/49 pass |

**Unvalidated profiles**: `tenant` and `admin` need an `auth.kdb`, and nothing
ships a CLI that creates one (users are added only through the Rust `AuthStore`
API). Those 12 cases are written from the source and have never run.

**Native ops defined but not implemented**: `Op::Exists` (0x06), `Op::Mset`
(0x05), `Op::Mexists` (0x07) and `Op::Flush` (0x82) exist in `skeg-proto` and
are refused by the server with `op <Name> not implemented`. Cases pin that
refusal, so a client cannot ship them believing they work. No client calls them
today: in `skeg-py`, `OP_EXISTS` and `OP_STATS` are declared in `_wire.py` and
never used.
