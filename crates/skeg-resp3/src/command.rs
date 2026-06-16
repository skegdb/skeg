//! Command parsing.
//!
//! Redis commands arrive on the wire as `Array<Bulk>`: the first bulk is the
//! command name (case-insensitive ASCII), the rest are positional arguments.
//! `parse_command` decodes a single `Frame` into a `Command`. Unrecognised
//! commands become `Command::Unknown` (caller decides whether to reject or
//! delegate to a higher-level dispatch).
//!
//! Pivot scope: only `HELLO` is parsed into a typed variant. `PING`/`ECHO`
//! and the KV layer (`GET`/`SET`/`DEL`/`EXISTS`) land in step 6.

use bytes::Bytes;

use crate::frame::Frame;

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Hello(HelloArgs),
    /// `PING` with an optional reply message. No msg => server responds with
    /// `+PONG`; with a msg, server echoes it back as a bulk string.
    Ping(Option<Bytes>),
    /// `ECHO msg` => bulk string reply identical to msg.
    Echo(Bytes),
    /// `GET key` => bulk reply with the stored value or null if missing.
    Get {
        key: Bytes,
    },
    /// `SET key value` => `+OK` reply. Per-call durability is the
    /// server's default; future EX/PX/NX/XX options are not yet typed.
    Set {
        key: Bytes,
        value: Bytes,
    },
    /// `DEL key [key ...]` => integer reply with the count of keys
    /// that existed and were removed.
    Del {
        keys: Vec<Bytes>,
    },
    /// `EXISTS key [key ...]` => integer reply with the count of
    /// keys present.
    Exists {
        keys: Vec<Bytes>,
    },
    /// `MGET key [key ...]` => array reply, one entry per key
    /// (bulk or null).
    Mget {
        keys: Vec<Bytes>,
    },
    /// `MSET key value [key value ...]` => `+OK` reply. The vec
    /// holds `(key, value)` tuples; the parser enforces even arity.
    Mset {
        pairs: Vec<(Bytes, Bytes)>,
    },
    /// `INCR key` => integer reply with the new value.
    Incr {
        key: Bytes,
    },
    /// `DECR key` => integer reply with the new value.
    Decr {
        key: Bytes,
    },
    /// `INCRBY key delta` => integer reply with the new value.
    IncrBy {
        key: Bytes,
        delta: i64,
    },
    /// `DECRBY key delta` => integer reply with the new value.
    DecrBy {
        key: Bytes,
        delta: i64,
    },
    /// `SELECT db` => `+OK` for `db == 0`, error otherwise. Skeg has
    /// a single logical DB; the parser captures the requested index
    /// so the dispatcher can decide whether to honour it.
    Select {
        db: i64,
    },

    // ── SKEG.* admin namespace ──────────────────────────────────────
    /// `SKEG.STATS` - cache + telemetry dump. No args.
    SkegStats,
    /// `SKEG.SHARDS` - per-shard metrics. No args.
    SkegShards,
    /// `SKEG.WHOAMI` - tenant identity bound to the connection.
    SkegWhoami,
    /// `SKEG.AUTH ...` - placeholder for future token-based auth. The
    /// parser preserves the raw arguments so the dispatcher (which
    /// currently emits a fixed `reserved` error) can evolve without a
    /// re-parse pass.
    SkegAuth {
        args: Vec<Bytes>,
    },

    // ── SKEG.* vector namespace ─────────────────────────────────────
    /// `SKEG.VINDEX.LIST` - enumerate vindexes for the calling tenant.
    SkegVindexList,
    /// `SKEG.VINDEX.CREATE name dim kind backend`. The parser checks
    /// arity (4 args) and forwards the raw bytes; inner argument
    /// parsing (UTF-8 name, u32 dim, kind/backend label) stays in the
    /// dispatcher because the error strings embed per-argument labels
    /// the existing clients rely on.
    SkegVindexCreate {
        args: Vec<Bytes>,
    },
    /// `SKEG.VINDEX.DROP name`. Arity 1; inner parsing in dispatcher.
    SkegVindexDrop {
        args: Vec<Bytes>,
    },
    /// `SKEG.VINDEX.CONSOLIDATE name`. Arity 1; fold the disk delta into the graph.
    SkegVindexConsolidate {
        args: Vec<Bytes>,
    },
    /// `SKEG.VSET name id vector [PAYLOAD blob]`. Arity 3 or 5.
    SkegVset {
        args: Vec<Bytes>,
    },
    /// `SKEG.VMSET name (id vector payload)+`. Bulk insert: one `name` then
    /// `(id, vector, payload)` triples (empty payload = none). The server fans
    /// the items out concurrently so the durable blob writes batch in the group
    /// committer - the bulk-ingest fast path. Arity 1 + 3k.
    SkegVmset {
        args: Vec<Bytes>,
    },
    /// `SKEG.VDEL name id`. Arity 2.
    SkegVdel {
        args: Vec<Bytes>,
    },
    /// `SKEG.VSEARCH name k l_search vector [WITHPAYLOAD] [FILTER expr]`.
    /// Arity 4 to 7; the handler validates the optional tail tokens.
    SkegVsearch {
        args: Vec<Bytes>,
    },

    /// `SKEG.QUOTA.SET tenant max_vectors max_disk_bytes`. Arity 3. Admin
    /// only; sets a tenant's hard quotas. Each limit is a u64, or `*` for
    /// unlimited. Inner parsing in the dispatcher.
    SkegQuotaSet {
        args: Vec<Bytes>,
    },
    /// `SKEG.QUOTA.GET tenant`. Arity 1. Admin only; reads a tenant's quotas.
    SkegQuotaGet {
        args: Vec<Bytes>,
    },

    /// Any command that was syntactically a valid array of bulks but whose
    /// name we have not wired into a typed variant yet. The dispatch layer
    /// gets the original name and args verbatim.
    Unknown {
        name: String,
        args: Vec<Bytes>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HelloArgs {
    /// Requested protocol version (2 or 3). `None` means the client asked
    /// `HELLO` with no positional argument and just wants the current state.
    pub protover: Option<u8>,
    /// `(username, password)` from the `AUTH` clause.
    pub auth: Option<(String, String)>,
    /// Client name from the `SETNAME` clause.
    pub setname: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CommandError {
    #[error("expected Array frame")]
    NotAnArray,
    #[error("empty command")]
    Empty,
    #[error("command name must be a bulk or simple string")]
    BadName,
    #[error("invalid utf-8 in command name")]
    NameInvalidUtf8,
    #[error("argument must be a bulk or simple string")]
    BadArg,
    #[error("HELLO: invalid protover, expected 2 or 3")]
    HelloBadProtover,
    #[error("HELLO: AUTH requires username and password")]
    HelloBadAuth,
    #[error("HELLO: SETNAME requires a name")]
    HelloBadSetname,
    #[error("HELLO: unknown clause '{0}'")]
    HelloUnknownClause(String),
    #[error("PING accepts 0 or 1 arguments, got {0}")]
    PingArity(usize),
    #[error("ECHO requires exactly 1 argument, got {0}")]
    EchoArity(usize),
    /// Wrong arity for a KV command. The `command` field carries the
    /// canonical name the server emits in its `ERR wrong number of
    /// arguments for '<command>'` reply.
    #[error("wrong number of arguments for '{command}'")]
    WrongArity { command: &'static str },
    /// `SELECT 1` (or any non-zero index): skeg has no DB number.
    #[error("DB index out of range (skeg only supports DB 0)")]
    SelectDbOutOfRange,
    /// `SELECT abc`: the DB index is not a valid integer.
    #[error("invalid DB index")]
    SelectInvalidIndex,
    /// `INCRBY foo bar`: the delta is not a valid i64.
    #[error("value is not an integer or out of range")]
    NotAnInteger,
    /// Wrong arity for a `SKEG.*` command whose error message names the
    /// expected positional argument labels (e.g. `want name dim kind
    /// backend`). Preserves the legacy server format byte-for-byte.
    #[error("wrong number of arguments for '{command}'; want {want}")]
    WrongAritySkeg {
        command: &'static str,
        want: &'static str,
    },
}

/// Parse one wire frame into a `Command`.
///
/// # Errors
///
/// Returns a `CommandError` variant if the frame is not an array of bulks,
/// is empty, has an invalid command name, or (for typed variants like
/// `HELLO`) violates the command's own argument rules.
pub fn parse_command(frame: Frame) -> Result<Command, CommandError> {
    let Frame::Array(items) = frame else {
        return Err(CommandError::NotAnArray);
    };
    let mut iter = items.into_iter();
    let Some(name_raw) = iter.next() else {
        return Err(CommandError::Empty);
    };
    let name = command_name(name_raw)?;
    let args: Vec<Bytes> = iter.map(arg_as_bytes).collect::<Result<_, _>>()?;
    let upper = name.to_ascii_uppercase();
    match upper.as_str() {
        "HELLO" => Ok(Command::Hello(parse_hello(args)?)),
        "PING" => Ok(Command::Ping(parse_ping(args)?)),
        "ECHO" => Ok(Command::Echo(parse_echo(args)?)),
        "GET" => parse_kv_get(args),
        "SET" => parse_kv_set(args),
        "DEL" => parse_kv_del(args),
        "EXISTS" => parse_kv_exists(args),
        "MGET" => parse_kv_mget(args),
        "MSET" => parse_kv_mset(args),
        "INCR" => parse_kv_incr(args),
        "DECR" => parse_kv_decr(args),
        "INCRBY" => parse_kv_incrby(args),
        "DECRBY" => parse_kv_decrby(args),
        "SELECT" => parse_kv_select(args),
        s if s.starts_with("SKEG.") => parse_skeg(&upper["SKEG.".len()..], args, name),
        _ => Ok(Command::Unknown { name, args }),
    }
}

fn parse_skeg(verb: &str, args: Vec<Bytes>, raw_name: String) -> Result<Command, CommandError> {
    match verb {
        "STATS" => {
            if !args.is_empty() {
                return Err(CommandError::WrongArity {
                    command: "SKEG.STATS",
                });
            }
            Ok(Command::SkegStats)
        }
        "SHARDS" => {
            if !args.is_empty() {
                return Err(CommandError::WrongArity {
                    command: "SKEG.SHARDS",
                });
            }
            Ok(Command::SkegShards)
        }
        "WHOAMI" => {
            if !args.is_empty() {
                return Err(CommandError::WrongArity {
                    command: "SKEG.WHOAMI",
                });
            }
            Ok(Command::SkegWhoami)
        }
        // Args preserved for forward-compat with the placeholder
        // `SKEG.AUTH is reserved` handler.
        "AUTH" => Ok(Command::SkegAuth { args }),
        "VINDEX.LIST" => {
            if !args.is_empty() {
                return Err(CommandError::WrongArity {
                    command: "SKEG.VINDEX.LIST",
                });
            }
            Ok(Command::SkegVindexList)
        }
        "VINDEX.CREATE" => {
            if args.len() != 4 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.VINDEX.CREATE",
                    want: "name dim kind backend",
                });
            }
            Ok(Command::SkegVindexCreate { args })
        }
        "VINDEX.DROP" => {
            if args.len() != 1 {
                return Err(CommandError::WrongArity {
                    command: "SKEG.VINDEX.DROP",
                });
            }
            Ok(Command::SkegVindexDrop { args })
        }
        "VINDEX.CONSOLIDATE" => {
            if args.len() != 1 {
                return Err(CommandError::WrongArity {
                    command: "SKEG.VINDEX.CONSOLIDATE",
                });
            }
            Ok(Command::SkegVindexConsolidate { args })
        }
        "VSET" => {
            // `name id vector` or `name id vector PAYLOAD <blob>`.
            if args.len() != 3 && args.len() != 5 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.VSET",
                    want: "name id vector [PAYLOAD blob]",
                });
            }
            Ok(Command::SkegVset { args })
        }
        "VMSET" => {
            // `name` then (id, vector, payload) triples: arity 1 + 3k, k >= 1.
            if args.len() < 4 || (args.len() - 1) % 3 != 0 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.VMSET",
                    want: "name (id vector payload)+",
                });
            }
            Ok(Command::SkegVmset { args })
        }
        "VDEL" => {
            if args.len() != 2 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.VDEL",
                    want: "name id",
                });
            }
            Ok(Command::SkegVdel { args })
        }
        "VSEARCH" => {
            // `name k l_search vector` plus optional trailing `WITHPAYLOAD`
            // and/or `FILTER <expr>` (in either order); the handler validates
            // the tail tokens.
            if !(4..=7).contains(&args.len()) {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.VSEARCH",
                    want: "name k l_search vector [WITHPAYLOAD] [FILTER expr]",
                });
            }
            Ok(Command::SkegVsearch { args })
        }
        "QUOTA.SET" => {
            if args.len() != 3 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.QUOTA.SET",
                    want: "tenant max_vectors max_disk_bytes",
                });
            }
            Ok(Command::SkegQuotaSet { args })
        }
        "QUOTA.GET" => {
            if args.len() != 1 {
                return Err(CommandError::WrongAritySkeg {
                    command: "SKEG.QUOTA.GET",
                    want: "tenant",
                });
            }
            Ok(Command::SkegQuotaGet { args })
        }
        // Unknown SKEG.* verb: pass through so the dispatcher emits
        // `ERR unknown command 'SKEG.<verb>'`.
        _ => Ok(Command::Unknown {
            name: raw_name,
            args,
        }),
    }
}

fn parse_kv_get(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 1 {
        return Err(CommandError::WrongArity { command: "GET" });
    }
    Ok(Command::Get {
        key: args.swap_remove(0),
    })
}

fn parse_kv_set(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 2 {
        return Err(CommandError::WrongArity { command: "SET" });
    }
    let value = args.swap_remove(1);
    let key = args.swap_remove(0);
    Ok(Command::Set { key, value })
}

fn parse_kv_del(args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.is_empty() {
        return Err(CommandError::WrongArity { command: "DEL" });
    }
    Ok(Command::Del { keys: args })
}

fn parse_kv_exists(args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.is_empty() {
        return Err(CommandError::WrongArity { command: "EXISTS" });
    }
    Ok(Command::Exists { keys: args })
}

fn parse_kv_mget(args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.is_empty() {
        return Err(CommandError::WrongArity { command: "MGET" });
    }
    Ok(Command::Mget { keys: args })
}

fn parse_kv_mset(args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.is_empty() || args.len() % 2 != 0 {
        return Err(CommandError::WrongArity { command: "MSET" });
    }
    let mut iter = args.into_iter();
    let mut pairs = Vec::with_capacity(iter.len() / 2);
    while let Some(k) = iter.next() {
        let v = iter.next().expect("even arity asserted above");
        pairs.push((k, v));
    }
    Ok(Command::Mset { pairs })
}

fn parse_kv_incr(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 1 {
        // The server formats arity errors for INCR and DECR with a
        // shared label `INCR/DECR` for historical reasons; preserved
        // byte-for-byte here.
        return Err(CommandError::WrongArity {
            command: "INCR/DECR",
        });
    }
    Ok(Command::Incr {
        key: args.swap_remove(0),
    })
}

fn parse_kv_decr(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 1 {
        return Err(CommandError::WrongArity {
            command: "INCR/DECR",
        });
    }
    Ok(Command::Decr {
        key: args.swap_remove(0),
    })
}

fn parse_kv_incrby(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 2 {
        return Err(CommandError::WrongArity {
            command: "INCRBY/DECRBY",
        });
    }
    let delta = parse_i64(&args[1])?;
    let key = args.swap_remove(0);
    Ok(Command::IncrBy { key, delta })
}

fn parse_kv_decrby(mut args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 2 {
        return Err(CommandError::WrongArity {
            command: "INCRBY/DECRBY",
        });
    }
    let delta = parse_i64(&args[1])?;
    let key = args.swap_remove(0);
    Ok(Command::DecrBy { key, delta })
}

fn parse_kv_select(args: Vec<Bytes>) -> Result<Command, CommandError> {
    if args.len() != 1 {
        return Err(CommandError::WrongArity { command: "SELECT" });
    }
    let s = std::str::from_utf8(&args[0]).map_err(|_| CommandError::SelectInvalidIndex)?;
    let db: i64 = s.parse().map_err(|_| CommandError::SelectInvalidIndex)?;
    Ok(Command::Select { db })
}

fn parse_i64(b: &Bytes) -> Result<i64, CommandError> {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or(CommandError::NotAnInteger)
}

fn parse_ping(mut args: Vec<Bytes>) -> Result<Option<Bytes>, CommandError> {
    match args.len() {
        0 => Ok(None),
        1 => Ok(Some(args.swap_remove(0))),
        n => Err(CommandError::PingArity(n)),
    }
}

fn parse_echo(mut args: Vec<Bytes>) -> Result<Bytes, CommandError> {
    match args.len() {
        1 => Ok(args.swap_remove(0)),
        n => Err(CommandError::EchoArity(n)),
    }
}

fn command_name(frame: Frame) -> Result<String, CommandError> {
    match frame {
        Frame::Bulk(b) => std::str::from_utf8(&b)
            .map(str::to_string)
            .map_err(|_| CommandError::NameInvalidUtf8),
        Frame::Simple(s) => Ok(s),
        _ => Err(CommandError::BadName),
    }
}

fn arg_as_bytes(frame: Frame) -> Result<Bytes, CommandError> {
    match frame {
        Frame::Bulk(b) => Ok(b),
        Frame::Simple(s) => Ok(Bytes::from(s)),
        _ => Err(CommandError::BadArg),
    }
}

fn parse_hello(args: Vec<Bytes>) -> Result<HelloArgs, CommandError> {
    let mut out = HelloArgs::default();
    let mut iter = args.into_iter();

    // First positional arg (optional): protover.
    if let Some(first) = iter.next() {
        let s = std::str::from_utf8(&first).map_err(|_| CommandError::HelloBadProtover)?;
        let v: u8 = s.parse().map_err(|_| CommandError::HelloBadProtover)?;
        if v != 2 && v != 3 {
            return Err(CommandError::HelloBadProtover);
        }
        out.protover = Some(v);
    }

    // Remaining: named clauses `AUTH user pass` and `SETNAME name`.
    while let Some(clause) = iter.next() {
        let upper = std::str::from_utf8(&clause)
            .map_err(|_| CommandError::HelloUnknownClause("<binary>".to_string()))?
            .to_ascii_uppercase();
        match upper.as_str() {
            "AUTH" => {
                let user_b = iter.next().ok_or(CommandError::HelloBadAuth)?;
                let pass_b = iter.next().ok_or(CommandError::HelloBadAuth)?;
                let user = std::str::from_utf8(&user_b)
                    .map_err(|_| CommandError::HelloBadAuth)?
                    .to_string();
                let pass = std::str::from_utf8(&pass_b)
                    .map_err(|_| CommandError::HelloBadAuth)?
                    .to_string();
                out.auth = Some((user, pass));
            }
            "SETNAME" => {
                let name_b = iter.next().ok_or(CommandError::HelloBadSetname)?;
                let name = std::str::from_utf8(&name_b)
                    .map_err(|_| CommandError::HelloBadSetname)?
                    .to_string();
                out.setname = Some(name);
            }
            other => return Err(CommandError::HelloUnknownClause(other.to_string())),
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(items: &[&[u8]]) -> Frame {
        Frame::Array(
            items
                .iter()
                .map(|b| Frame::Bulk(Bytes::copy_from_slice(b)))
                .collect(),
        )
    }

    #[test]
    fn hello_no_args() {
        let cmd = parse_command(arr(&[b"HELLO"])).unwrap();
        assert_eq!(cmd, Command::Hello(HelloArgs::default()));
    }

    #[test]
    fn hello_protover_3() {
        let cmd = parse_command(arr(&[b"HELLO", b"3"])).unwrap();
        let Command::Hello(args) = cmd else {
            panic!("expected Hello");
        };
        assert_eq!(args.protover, Some(3));
        assert!(args.auth.is_none());
        assert!(args.setname.is_none());
    }

    #[test]
    fn hello_protover_2() {
        let cmd = parse_command(arr(&[b"HELLO", b"2"])).unwrap();
        let Command::Hello(args) = cmd else {
            panic!("expected Hello");
        };
        assert_eq!(args.protover, Some(2));
    }

    #[test]
    fn hello_lowercase_name() {
        // Command name is case-insensitive.
        let cmd = parse_command(arr(&[b"hello", b"3"])).unwrap();
        assert!(matches!(cmd, Command::Hello(args) if args.protover == Some(3)));
    }

    #[test]
    fn hello_with_setname() {
        let cmd = parse_command(arr(&[b"HELLO", b"3", b"SETNAME", b"my-client"])).unwrap();
        let Command::Hello(args) = cmd else {
            panic!();
        };
        assert_eq!(args.protover, Some(3));
        assert_eq!(args.setname.as_deref(), Some("my-client"));
    }

    #[test]
    fn hello_with_auth() {
        let cmd = parse_command(arr(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"])).unwrap();
        let Command::Hello(args) = cmd else {
            panic!();
        };
        assert_eq!(args.protover, Some(3));
        assert_eq!(
            args.auth,
            Some(("alice".to_string(), "hunter2".to_string()))
        );
    }

    #[test]
    fn hello_with_auth_and_setname() {
        let cmd = parse_command(arr(&[
            b"HELLO", b"3", b"AUTH", b"u", b"p", b"SETNAME", b"c",
        ]))
        .unwrap();
        let Command::Hello(args) = cmd else {
            panic!();
        };
        assert_eq!(args.protover, Some(3));
        assert_eq!(args.auth, Some(("u".into(), "p".into())));
        assert_eq!(args.setname.as_deref(), Some("c"));
    }

    #[test]
    fn hello_clauses_case_insensitive() {
        let cmd = parse_command(arr(&[b"HELLO", b"3", b"auth", b"u", b"p"])).unwrap();
        let Command::Hello(args) = cmd else {
            panic!();
        };
        assert_eq!(args.auth, Some(("u".into(), "p".into())));
    }

    #[test]
    fn hello_protover_5_rejected() {
        let err = parse_command(arr(&[b"HELLO", b"5"])).unwrap_err();
        assert_eq!(err, CommandError::HelloBadProtover);
    }

    #[test]
    fn hello_protover_garbage_rejected() {
        let err = parse_command(arr(&[b"HELLO", b"abc"])).unwrap_err();
        assert_eq!(err, CommandError::HelloBadProtover);
    }

    #[test]
    fn hello_auth_missing_pass() {
        let err = parse_command(arr(&[b"HELLO", b"3", b"AUTH", b"alice"])).unwrap_err();
        assert_eq!(err, CommandError::HelloBadAuth);
    }

    #[test]
    fn hello_setname_missing_name() {
        let err = parse_command(arr(&[b"HELLO", b"3", b"SETNAME"])).unwrap_err();
        assert_eq!(err, CommandError::HelloBadSetname);
    }

    #[test]
    fn hello_unknown_clause() {
        let err = parse_command(arr(&[b"HELLO", b"3", b"WAT"])).unwrap_err();
        assert_eq!(err, CommandError::HelloUnknownClause("WAT".to_string()));
    }

    #[test]
    fn empty_command_rejected() {
        let err = parse_command(Frame::Array(vec![])).unwrap_err();
        assert_eq!(err, CommandError::Empty);
    }

    #[test]
    fn not_an_array_rejected() {
        let err = parse_command(Frame::Simple("PING".into())).unwrap_err();
        assert_eq!(err, CommandError::NotAnArray);
    }

    #[test]
    fn unknown_command_passthrough() {
        let cmd = parse_command(arr(&[b"FOO", b"bar", b"baz"])).unwrap();
        let Command::Unknown { name, args } = cmd else {
            panic!("expected Unknown");
        };
        assert_eq!(name, "FOO");
        assert_eq!(
            args,
            vec![Bytes::from_static(b"bar"), Bytes::from_static(b"baz")]
        );
    }

    #[test]
    fn ping_no_args() {
        let cmd = parse_command(arr(&[b"PING"])).unwrap();
        assert_eq!(cmd, Command::Ping(None));
    }

    #[test]
    fn ping_with_msg() {
        let cmd = parse_command(arr(&[b"PING", b"hello"])).unwrap();
        assert_eq!(cmd, Command::Ping(Some(Bytes::from_static(b"hello"))));
    }

    #[test]
    fn ping_lowercase() {
        let cmd = parse_command(arr(&[b"ping"])).unwrap();
        assert_eq!(cmd, Command::Ping(None));
    }

    #[test]
    fn ping_too_many_args_errors() {
        let err = parse_command(arr(&[b"PING", b"a", b"b"])).unwrap_err();
        assert_eq!(err, CommandError::PingArity(2));
    }

    #[test]
    fn echo_with_msg() {
        let cmd = parse_command(arr(&[b"ECHO", b"hi there"])).unwrap();
        assert_eq!(cmd, Command::Echo(Bytes::from_static(b"hi there")));
    }

    #[test]
    fn echo_binary_payload() {
        let cmd = parse_command(Frame::Array(vec![
            Frame::Bulk(Bytes::from_static(b"ECHO")),
            Frame::Bulk(Bytes::from_static(&[0u8, 1, 2, 0xFF])),
        ]))
        .unwrap();
        assert_eq!(cmd, Command::Echo(Bytes::from_static(&[0u8, 1, 2, 0xFF])));
    }

    #[test]
    fn echo_no_args_errors() {
        let err = parse_command(arr(&[b"ECHO"])).unwrap_err();
        assert_eq!(err, CommandError::EchoArity(0));
    }

    #[test]
    fn echo_too_many_args_errors() {
        let err = parse_command(arr(&[b"ECHO", b"a", b"b"])).unwrap_err();
        assert_eq!(err, CommandError::EchoArity(2));
    }

    #[test]
    fn arg_can_be_simple_string() {
        // RESP2 inline reply form: simple string. Accepted as arg too.
        let frame = Frame::Array(vec![
            Frame::Bulk(Bytes::from_static(b"HELLO")),
            Frame::Simple("3".into()),
        ]);
        let cmd = parse_command(frame).unwrap();
        assert!(matches!(cmd, Command::Hello(a) if a.protover == Some(3)));
    }

    // ── KV typed parsing ─────────────────────────────────────────

    #[test]
    fn get_one_arg() {
        let cmd = parse_command(arr(&[b"GET", b"foo"])).unwrap();
        assert_eq!(
            cmd,
            Command::Get {
                key: Bytes::from_static(b"foo")
            }
        );
    }

    #[test]
    fn get_wrong_arity_renders_existing_error() {
        let err = parse_command(arr(&[b"GET"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'GET'");
        let err = parse_command(arr(&[b"GET", b"a", b"b"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'GET'");
    }

    #[test]
    fn set_two_args() {
        let cmd = parse_command(arr(&[b"SET", b"k", b"v"])).unwrap();
        assert_eq!(
            cmd,
            Command::Set {
                key: Bytes::from_static(b"k"),
                value: Bytes::from_static(b"v"),
            }
        );
    }

    #[test]
    fn set_wrong_arity() {
        let err = parse_command(arr(&[b"SET", b"k"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'SET'");
    }

    #[test]
    fn del_one_or_more() {
        let cmd = parse_command(arr(&[b"DEL", b"a"])).unwrap();
        assert_eq!(
            cmd,
            Command::Del {
                keys: vec![Bytes::from_static(b"a")]
            }
        );
        let cmd = parse_command(arr(&[b"DEL", b"a", b"b", b"c"])).unwrap();
        assert_eq!(
            cmd,
            Command::Del {
                keys: vec![
                    Bytes::from_static(b"a"),
                    Bytes::from_static(b"b"),
                    Bytes::from_static(b"c"),
                ]
            }
        );
    }

    #[test]
    fn del_zero_args_errors() {
        let err = parse_command(arr(&[b"DEL"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'DEL'");
    }

    #[test]
    fn exists_one_or_more() {
        let cmd = parse_command(arr(&[b"EXISTS", b"a", b"b"])).unwrap();
        assert_eq!(
            cmd,
            Command::Exists {
                keys: vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]
            }
        );
    }

    #[test]
    fn mget_one_or_more() {
        let cmd = parse_command(arr(&[b"MGET", b"a", b"b"])).unwrap();
        assert_eq!(
            cmd,
            Command::Mget {
                keys: vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]
            }
        );
    }

    #[test]
    fn mset_pairs() {
        let cmd = parse_command(arr(&[b"MSET", b"k1", b"v1", b"k2", b"v2"])).unwrap();
        assert_eq!(
            cmd,
            Command::Mset {
                pairs: vec![
                    (Bytes::from_static(b"k1"), Bytes::from_static(b"v1")),
                    (Bytes::from_static(b"k2"), Bytes::from_static(b"v2")),
                ]
            }
        );
    }

    #[test]
    fn mset_odd_arity_errors() {
        let err = parse_command(arr(&[b"MSET", b"k", b"v", b"x"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'MSET'");
        let err = parse_command(arr(&[b"MSET"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'MSET'");
    }

    #[test]
    fn incr_decr_arity_error_string_matches_server() {
        // Historical: the server emits 'INCR/DECR' shared label.
        let err = parse_command(arr(&[b"INCR"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'INCR/DECR'");
        let err = parse_command(arr(&[b"DECR", b"a", b"b"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'INCR/DECR'");
    }

    #[test]
    fn incrby_parses_delta() {
        let cmd = parse_command(arr(&[b"INCRBY", b"counter", b"7"])).unwrap();
        assert_eq!(
            cmd,
            Command::IncrBy {
                key: Bytes::from_static(b"counter"),
                delta: 7,
            }
        );
        let cmd = parse_command(arr(&[b"INCRBY", b"counter", b"-3"])).unwrap();
        assert_eq!(
            cmd,
            Command::IncrBy {
                key: Bytes::from_static(b"counter"),
                delta: -3,
            }
        );
    }

    #[test]
    fn incrby_non_integer_errors() {
        let err = parse_command(arr(&[b"INCRBY", b"k", b"abc"])).unwrap_err();
        assert_eq!(err.to_string(), "value is not an integer or out of range");
    }

    #[test]
    fn incrby_decrby_arity_error() {
        let err = parse_command(arr(&[b"INCRBY", b"k"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'INCRBY/DECRBY'"
        );
        let err = parse_command(arr(&[b"DECRBY", b"k"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'INCRBY/DECRBY'"
        );
    }

    #[test]
    fn select_zero_ok() {
        let cmd = parse_command(arr(&[b"SELECT", b"0"])).unwrap();
        assert_eq!(cmd, Command::Select { db: 0 });
    }

    #[test]
    fn select_nonzero_typed_but_dispatcher_decides() {
        // Parser accepts any valid integer; the dispatcher emits the
        // "DB index out of range" error string when db != 0.
        let cmd = parse_command(arr(&[b"SELECT", b"1"])).unwrap();
        assert_eq!(cmd, Command::Select { db: 1 });
    }

    #[test]
    fn select_invalid_index() {
        let err = parse_command(arr(&[b"SELECT", b"abc"])).unwrap_err();
        assert_eq!(err.to_string(), "invalid DB index");
    }

    #[test]
    fn select_wrong_arity() {
        let err = parse_command(arr(&[b"SELECT"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'SELECT'");
        let err = parse_command(arr(&[b"SELECT", b"0", b"1"])).unwrap_err();
        assert_eq!(err.to_string(), "wrong number of arguments for 'SELECT'");
    }

    #[test]
    fn command_name_case_insensitive_for_kv() {
        // Lowercase should match too (Redis convention).
        let cmd = parse_command(arr(&[b"get", b"k"])).unwrap();
        assert!(matches!(cmd, Command::Get { .. }));
        let cmd = parse_command(arr(&[b"SeT", b"k", b"v"])).unwrap();
        assert!(matches!(cmd, Command::Set { .. }));
    }

    proptest::proptest! {
        /// Equivalence: for any KV command frame built from random
        /// byte arguments, the typed parser preserves every argument
        /// byte-for-byte. Catches subtle gather/swap_remove mistakes
        /// that would surface as silent data corruption.
        #[test]
        fn prop_get_preserves_key(key in proptest::collection::vec(proptest::num::u8::ANY, 0..256)) {
            let key_b = Bytes::copy_from_slice(&key);
            let frame = Frame::Array(vec![
                Frame::Bulk(Bytes::from_static(b"GET")),
                Frame::Bulk(key_b.clone()),
            ]);
            let Command::Get { key: got } = parse_command(frame).unwrap() else {
                panic!("expected Get");
            };
            proptest::prop_assert_eq!(got, key_b);
        }

        #[test]
        fn prop_set_preserves_kv(
            key in proptest::collection::vec(proptest::num::u8::ANY, 0..256),
            value in proptest::collection::vec(proptest::num::u8::ANY, 0..512),
        ) {
            let key_b = Bytes::copy_from_slice(&key);
            let val_b = Bytes::copy_from_slice(&value);
            let frame = Frame::Array(vec![
                Frame::Bulk(Bytes::from_static(b"SET")),
                Frame::Bulk(key_b.clone()),
                Frame::Bulk(val_b.clone()),
            ]);
            let Command::Set { key: gk, value: gv } = parse_command(frame).unwrap() else {
                panic!("expected Set");
            };
            proptest::prop_assert_eq!(gk, key_b);
            proptest::prop_assert_eq!(gv, val_b);
        }

        #[test]
        fn prop_del_preserves_all_keys(
            keys in proptest::collection::vec(
                proptest::collection::vec(proptest::num::u8::ANY, 0..128),
                1..16,
            ),
        ) {
            let mut frame_items: Vec<Frame> = vec![Frame::Bulk(Bytes::from_static(b"DEL"))];
            let expected: Vec<Bytes> = keys
                .iter()
                .map(|k| Bytes::copy_from_slice(k))
                .collect();
            for b in &expected {
                frame_items.push(Frame::Bulk(b.clone()));
            }
            let Command::Del { keys: got } = parse_command(Frame::Array(frame_items)).unwrap() else {
                panic!("expected Del");
            };
            proptest::prop_assert_eq!(got, expected);
        }

        #[test]
        fn prop_mset_preserves_pair_order(
            pairs in proptest::collection::vec(
                (
                    proptest::collection::vec(proptest::num::u8::ANY, 0..64),
                    proptest::collection::vec(proptest::num::u8::ANY, 0..64),
                ),
                1..8,
            ),
        ) {
            let mut frame_items: Vec<Frame> = vec![Frame::Bulk(Bytes::from_static(b"MSET"))];
            let expected: Vec<(Bytes, Bytes)> = pairs
                .iter()
                .map(|(k, v)| (Bytes::copy_from_slice(k), Bytes::copy_from_slice(v)))
                .collect();
            for (k, v) in &expected {
                frame_items.push(Frame::Bulk(k.clone()));
                frame_items.push(Frame::Bulk(v.clone()));
            }
            let Command::Mset { pairs: got } = parse_command(Frame::Array(frame_items)).unwrap() else {
                panic!("expected Mset");
            };
            proptest::prop_assert_eq!(got, expected);
        }

        #[test]
        fn prop_incrby_roundtrips_any_i64(delta in proptest::num::i64::ANY) {
            let s = delta.to_string();
            let frame = Frame::Array(vec![
                Frame::Bulk(Bytes::from_static(b"INCRBY")),
                Frame::Bulk(Bytes::from_static(b"k")),
                Frame::Bulk(Bytes::copy_from_slice(s.as_bytes())),
            ]);
            let Command::IncrBy { delta: got, .. } = parse_command(frame).unwrap() else {
                panic!("expected IncrBy");
            };
            proptest::prop_assert_eq!(got, delta);
        }

        /// Equivalence: every wrong-arity input emits the byte-identical
        /// string the server has been emitting since v0.1, regardless of
        /// the arity offset.
        #[test]
        fn prop_wrong_arity_strings_stable(
            extra in 0usize..32,
        ) {
            let too_many = 2 + extra;
            // GET wants exactly 1.
            let mut items = vec![Frame::Bulk(Bytes::from_static(b"GET"))];
            for _ in 0..too_many {
                items.push(Frame::Bulk(Bytes::from_static(b"x")));
            }
            let err = parse_command(Frame::Array(items)).unwrap_err();
            proptest::prop_assert_eq!(
                err.to_string(),
                "wrong number of arguments for 'GET'"
            );
        }
    }

    // ── SKEG.* typed parsing ─────────────────────────────────────

    #[test]
    fn skeg_stats_no_args() {
        let cmd = parse_command(arr(&[b"SKEG.STATS"])).unwrap();
        assert_eq!(cmd, Command::SkegStats);
    }

    #[test]
    fn skeg_stats_with_args_errors() {
        let err = parse_command(arr(&[b"SKEG.STATS", b"x"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.STATS'"
        );
    }

    #[test]
    fn skeg_shards_no_args() {
        let cmd = parse_command(arr(&[b"SKEG.SHARDS"])).unwrap();
        assert_eq!(cmd, Command::SkegShards);
    }

    #[test]
    fn skeg_whoami_no_args() {
        let cmd = parse_command(arr(&[b"SKEG.WHOAMI"])).unwrap();
        assert_eq!(cmd, Command::SkegWhoami);
    }

    #[test]
    fn skeg_auth_preserves_args() {
        let cmd = parse_command(arr(&[b"SKEG.AUTH", b"token-abc"])).unwrap();
        assert_eq!(
            cmd,
            Command::SkegAuth {
                args: vec![Bytes::from_static(b"token-abc")]
            }
        );
    }

    #[test]
    fn skeg_vindex_list_no_args() {
        let cmd = parse_command(arr(&[b"SKEG.VINDEX.LIST"])).unwrap();
        assert_eq!(cmd, Command::SkegVindexList);
    }

    #[test]
    fn skeg_vindex_create_four_args() {
        let cmd = parse_command(arr(&[
            b"SKEG.VINDEX.CREATE",
            b"x",
            b"1024",
            b"int8",
            b"flat",
        ]))
        .unwrap();
        let Command::SkegVindexCreate { args } = cmd else {
            panic!("expected SkegVindexCreate");
        };
        assert_eq!(args.len(), 4);
        assert_eq!(args[0], Bytes::from_static(b"x"));
        assert_eq!(args[3], Bytes::from_static(b"flat"));
    }

    #[test]
    fn skeg_vindex_create_wrong_arity_error_string() {
        let err = parse_command(arr(&[b"SKEG.VINDEX.CREATE", b"x"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.VINDEX.CREATE'; want name dim kind backend"
        );
    }

    #[test]
    fn skeg_vindex_drop_one_arg() {
        let cmd = parse_command(arr(&[b"SKEG.VINDEX.DROP", b"x"])).unwrap();
        let Command::SkegVindexDrop { args } = cmd else {
            panic!("expected SkegVindexDrop");
        };
        assert_eq!(args, vec![Bytes::from_static(b"x")]);
    }

    #[test]
    fn skeg_vindex_drop_wrong_arity_error_string() {
        // No `; want ...` suffix for VINDEX.DROP.
        let err = parse_command(arr(&[b"SKEG.VINDEX.DROP"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.VINDEX.DROP'"
        );
    }

    #[test]
    fn skeg_vset_three_args() {
        let cmd = parse_command(arr(&[b"SKEG.VSET", b"x", b"42", &[0u8, 0, 0, 0]])).unwrap();
        let Command::SkegVset { args } = cmd else {
            panic!("expected SkegVset");
        };
        assert_eq!(args.len(), 3);
    }

    #[test]
    fn skeg_vset_wrong_arity_error_string() {
        let err = parse_command(arr(&[b"SKEG.VSET", b"x"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.VSET'; want name id vector [PAYLOAD blob]"
        );
    }

    #[test]
    fn skeg_quota_set_three_args() {
        let cmd = parse_command(arr(&[b"SKEG.QUOTA.SET", b"acme", b"1000", b"*"])).unwrap();
        let Command::SkegQuotaSet { args } = cmd else {
            panic!("expected SkegQuotaSet");
        };
        assert_eq!(args.len(), 3);
    }

    #[test]
    fn skeg_quota_get_one_arg() {
        let cmd = parse_command(arr(&[b"SKEG.QUOTA.GET", b"acme"])).unwrap();
        let Command::SkegQuotaGet { args } = cmd else {
            panic!("expected SkegQuotaGet");
        };
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn skeg_quota_set_wrong_arity_error_string() {
        let err = parse_command(arr(&[b"SKEG.QUOTA.SET", b"acme"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.QUOTA.SET'; \
             want tenant max_vectors max_disk_bytes"
        );
    }

    #[test]
    fn skeg_vdel_two_args() {
        let cmd = parse_command(arr(&[b"SKEG.VDEL", b"x", b"42"])).unwrap();
        let Command::SkegVdel { args } = cmd else {
            panic!("expected SkegVdel");
        };
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn skeg_vdel_wrong_arity_error_string() {
        let err = parse_command(arr(&[b"SKEG.VDEL", b"x"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.VDEL'; want name id"
        );
    }

    #[test]
    fn skeg_vsearch_four_args() {
        let cmd = parse_command(arr(&[b"SKEG.VSEARCH", b"x", b"10", b"300", &[0u8; 4]])).unwrap();
        let Command::SkegVsearch { args } = cmd else {
            panic!("expected SkegVsearch");
        };
        assert_eq!(args.len(), 4);
    }

    #[test]
    fn skeg_vsearch_wrong_arity_error_string() {
        let err = parse_command(arr(&[b"SKEG.VSEARCH", b"x"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "wrong number of arguments for 'SKEG.VSEARCH'; want name k l_search vector [WITHPAYLOAD] [FILTER expr]"
        );
    }

    #[test]
    fn unknown_skeg_verb_falls_through_to_unknown() {
        // Verbs we have not typed yet must round-trip through Unknown
        // so the dispatcher emits `ERR unknown command 'SKEG.FOO'`.
        let cmd = parse_command(arr(&[b"SKEG.FOO", b"a"])).unwrap();
        let Command::Unknown { name, args } = cmd else {
            panic!("expected Unknown for SKEG.FOO");
        };
        assert_eq!(name, "SKEG.FOO");
        assert_eq!(args, vec![Bytes::from_static(b"a")]);
    }

    #[test]
    fn skeg_namespace_case_insensitive() {
        let cmd = parse_command(arr(&[b"skeg.stats"])).unwrap();
        assert_eq!(cmd, Command::SkegStats);
        let cmd = parse_command(arr(&[b"Skeg.VindEx.List"])).unwrap();
        assert_eq!(cmd, Command::SkegVindexList);
    }
}
