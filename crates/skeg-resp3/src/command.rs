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
    match name.to_ascii_uppercase().as_str() {
        "HELLO" => Ok(Command::Hello(parse_hello(args)?)),
        "PING" => Ok(Command::Ping(parse_ping(args)?)),
        "ECHO" => Ok(Command::Echo(parse_echo(args)?)),
        _ => Ok(Command::Unknown { name, args }),
    }
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
}
