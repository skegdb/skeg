# Filtered search

A vector in skeg can carry a small payload of typed fields, and a `SKEG.VSEARCH`
can restrict its results to vectors whose payload matches a filter, for example
"the nearest documents in this collection", "this user's vectors", or "anything
tagged `pdf` after 2024". This is the surface most RAG workloads need.

Filtered search is on the RESP3 protocol. The native binary protocol stays
payload and filter free.

## Attaching a payload

`SKEG.VSET <name> <id> <vector> [PAYLOAD <blob>]`. The blob is an opaque byte
buffer stored beside the vector and returned verbatim by a `WITHPAYLOAD` search.
Its `key=value` tokens (whitespace separated) are also parsed into a searchable
index:

```text
> SKEG.VSET docs 1 <vector> PAYLOAD "user=alice type=doc ts=20240115"
OK
```

- A value that parses as an integer is indexed as `i64` (so ranges work);
  anything else is a `keyword`. Datetimes fit as an integer such as `20240115`.
- A field repeated in one payload is multi-valued: `tag=a tag=b` makes the
  vector match both `tag = a` and `tag = b`.
- A `VSET` without `PAYLOAD` stores no blob and does no extra work. Overwriting a
  vector with a new payload replaces its indexed fields; `VDEL` drops them.

The blob lives in the KV vLog under a reserved, tenant-scoped key, not in the
vector index, so the quantized graph stays dense and the payload inherits the
vLog's crash-safety. After a restart the payload index is rebuilt from the blobs
on the first filtered search.

## Searching

`SKEG.VSEARCH <name> <k> <l_search> <query> [WITHPAYLOAD] [FILTER <expr>]`. The two
trailing modifiers are optional and may appear in either order.

- `WITHPAYLOAD` returns each hit's stored blob alongside its id and score.
- `FILTER <expr>` restricts the result to vectors whose payload matches `<expr>`.

```text
> SKEG.VSEARCH docs 10 100 <query> FILTER "user = alice AND type = doc"
> SKEG.VSEARCH docs 10 100 <query> WITHPAYLOAD FILTER "ts >= 20240101"
```

Without `FILTER` (and `WITHPAYLOAD`) the search is the ordinary nearest-neighbour
path, unchanged.

## Filter grammar

```text
expr    := or
or      := and ( OR and )*
and     := not ( AND not )*
not     := [ NOT ] atom
atom    := '(' or ')' | predicate
predicate :=
      field = value
    | field IN ( value , value , ... )
    | field >= value | field > value | field <= value | field < value
    | field BETWEEN value AND value
    | field EXISTS
```

- `=` and `IN` match keyword or integer values. `IN` is a shorthand for an `OR`
  of equalities.
- The range operators (`>=`, `>`, `<=`, `<`, `BETWEEN`) compare integer fields
  (including datetimes stored as integers).
- `EXISTS` matches any vector that has the field at all; `NOT field EXISTS`
  matches those that lack it.
- `AND`, `OR`, `NOT`, and parentheses combine predicates. Precedence is `NOT`
  then `AND` then `OR`; parentheses override it.

Examples:

```text
user = alice
type IN (doc, pdf, md)
ts BETWEEN 20240101 AND 20241231
user = alice AND (type = doc OR type = pdf) AND NOT archived = 1
source EXISTS
```

## How a filter is served

The planner picks the cheapest correct strategy from the size of the matching
set:

- **Selective filter** (few matches): the matching ids are scored exactly,
  full-precision, over just that set. Exact, no recall loss.
- **Broad filter** (many matches): a filtered graph search. Two complementary
  walks are merged so recall holds regardless of how the matching vectors sit in
  the embedding space: one walk explores only the matching subgraph (good when a
  filter selects a topic that clusters together, the common case), the other
  navigates the whole graph and filters at re-rank (good when the matches are
  scattered). The result is re-ranked exactly.

On real 1024-dim embeddings this holds recall@10 between 0.98 and 1.00 across
selectivities and metadata shapes, at query time, with no extra index build
cost.

## Scope

- RESP3 only; the native binary protocol is payload and filter free.
- Field types are `keyword` and `i64` (datetimes as integers). Floating-point
  fields, geo, and full-text are not indexed.
- Per-tenant indices keep filters scoped per tenant; see
  [`multi-tenancy.md`](multi-tenancy.md).
