# Getting started

Install skeg first (see the [README](../README.md#install)), then run it.

## Run

```sh
skeg --data-dir ./data --addr 127.0.0.1:7379 &       # native protocol
skeg-resp3 --data-dir ./data --addr 127.0.0.1:6379 & # Redis-compatible
```

## Key-value

Any Redis client works:

```text
$ redis-cli -3 -p 6379
> SET greeting "hello"
OK
> INCRBY counter 7
(integer) 7
```

## Vectors

Vector ops are namespaced under `SKEG.*` to stay out of the Redis command
surface:

```text
> SKEG.VINDEX.CREATE docs 1024 tq2 disk
OK
> SKEG.VSET docs 1 <1024-float vector as bytes>
OK
> SKEG.VSEARCH docs 10 100 <query vector bytes>
1) "1"
2) (double) 0.987
...
```

Runnable client examples (ingest, multi-tenant, filtered search) live in
[`skeg-bench/examples`](https://github.com/skegdb/skeg-bench/tree/main/examples).

## Command reference

`SKEG.VINDEX.CREATE <name> <dim> [<kind>] <backend>`. `kind` is optional and
defaults to `tq2`; it selects the index storage / tier:
`f32 | int8 | tq1 | tq2 | tq4 | binary` (see [`architecture.md`](architecture.md)
for what each tier costs). `backend` chooses in-RAM flat scan or on-disk Vamana:
`flat | disk`. The vector in `SKEG.VSET` / `SKEG.VSEARCH` is a raw byte buffer on
the native protocol, a bulk string on RESP3.

Filtered search (RESP3): `SKEG.VSET <name> <id> <vector> [PAYLOAD <blob>]`
attaches `key=value` fields; `SKEG.VSEARCH <name> <k> <l_search> <query>
[WITHPAYLOAD] [FILTER <expr>]` returns payloads and/or restricts to matches. The
grammar covers `=`, `IN (...)`, ranges `>= > <= < BETWEEN a AND b`, `EXISTS`, and
`AND` / `OR` / `NOT` with parentheses. See [`filtered-search.md`](filtered-search.md).

Multi-tenant setup (quotas, auth, key scoping) is in
[`multi-tenancy.md`](multi-tenancy.md).
