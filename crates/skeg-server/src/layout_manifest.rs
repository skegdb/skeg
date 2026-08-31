//! The store's layout, declared on disk instead of deduced from it.
//!
//! Discovery by directory scan answers "how many shards are there?" with a
//! guess that is usually right. Usually is the problem: serve mode once
//! assumed one shard over an eight-shard store and answered every query with
//! complete confidence over an eighth of the index - 597 rows of 5,000, recall
//! 0.115, no error and no warning. The scan is fail-closed now, which turns
//! that into a refusal, but a refusal still means the reader is deducing
//! something the writer already knew and simply never wrote down.
//!
//! `LAYOUT` writes it down. It holds only what is IMMUTABLE about the store:
//! the format version, the shard count, the store's identity and the feature
//! flags its files were written with. Base generations and router epochs are
//! deliberately absent - those advance per vindex, many times a minute, and
//! already have their own atomic per-vindex files. Putting them here would
//! turn one rarely-written file into a contended one and reintroduce the
//! two-places-one-rule shape that produced every P0 this engine has had.
//!
//! Every ambiguity is an error. A checksum that does not match, a version this
//! build does not know, a flag it cannot honour, a shard count that disagrees
//! with the caller or with the directories on disk: all refuse to open. The
//! one accommodation is for stores written before this file existed, which
//! migrate on a writable open and are read as-is on a read-only one.

use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

/// The manifest's file name, at the root of the store.
const FILE: &str = "LAYOUT";
const MAGIC: &[u8; 8] = b"SKEGLAY1";
/// The version of the ENCODING below, bumped when its bytes change meaning.
const FORMAT_VERSION: u32 = 1;
/// Fixed: 40 bytes of body plus a 4-byte checksum over exactly those 40.
const BODY_LEN: usize = 40;
const ENCODED_LEN: usize = BODY_LEN + 4;
/// No flags are defined yet. Any bit set is a store this build cannot honour.
const KNOWN_FLAGS: u64 = 0;

/// What the store declares about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutManifest {
    format_version: u32,
    shard_count: NonZeroUsize,
    store_uuid: [u8; 16],
    feature_flags: u64,
}

/// How the caller intends to use the store, which decides whether a legacy
/// layout may be migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    /// The caller will write. A legacy store adopts a manifest; the requested
    /// count must match what is already there.
    ReadWrite { requested_shards: NonZeroUsize },
    /// The caller will not write, and neither will this. Serve mode runs over
    /// copies an operator may have mounted read-only.
    ReadOnly,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl LayoutManifest {
    pub fn shard_count(&self) -> NonZeroUsize {
        self.shard_count
    }

    pub fn store_uuid(&self) -> [u8; 16] {
        self.store_uuid
    }

    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Open the store's declared layout, migrating a legacy one if the mode
    /// permits it.
    ///
    /// The rules, in order:
    ///
    /// - `LAYOUT` present: validate it, then check it against the directories
    ///   and against the caller's request. Any disagreement refuses.
    /// - absent, writable, empty root: mint one from the requested count.
    /// - absent, writable, legacy root: scan, require the request to match,
    ///   write the manifest.
    /// - absent, read-only: scan, change nothing on disk.
    /// - any I/O error while enumerating: propagate. Never treat an unreadable
    ///   entry as an absent shard.
    pub fn open_or_migrate(root: &Path, mode: OpenMode) -> io::Result<Self> {
        let path = root.join(FILE);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let m = Self::decode(&bytes, root)?;
                m.check_directories(root)?;
                if let OpenMode::ReadWrite { requested_shards } = mode
                    && requested_shards != m.shard_count
                {
                    return Err(invalid(format!(
                        "{} declares {} shards, but this open asked for {}: \
                         refusing to run against a layout it disagrees with",
                        path.display(),
                        m.shard_count,
                        requested_shards
                    )));
                }
                Ok(m)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Self::migrate(root, mode),
            Err(e) => Err(e),
        }
    }

    /// A store with no manifest: either brand new, or written before this file
    /// existed.
    fn migrate(root: &Path, mode: OpenMode) -> io::Result<Self> {
        let scanned = scan_shard_count(root);
        match mode {
            OpenMode::ReadOnly => {
                // Read what is there and touch nothing. A store mounted
                // read-only must still serve, and a store this build did not
                // write must not be silently rewritten by having been read.
                let shard_count = scanned?;
                Ok(Self {
                    format_version: FORMAT_VERSION,
                    shard_count,
                    // Not persisted, so not an identity: a legacy store has
                    // none until it migrates. Zero says exactly that.
                    store_uuid: [0u8; 16],
                    feature_flags: 0,
                })
            }
            OpenMode::ReadWrite { requested_shards } => {
                let shard_count = match scanned {
                    Ok(found) => {
                        if found != requested_shards {
                            return Err(invalid(format!(
                                "{} holds {} shard directories but this open \
                                 asked for {}: refusing to strand the \
                                 difference",
                                root.display(),
                                found,
                                requested_shards
                            )));
                        }
                        found
                    }
                    // An empty root is a new store, and only a new store.
                    Err(e) if e.kind() == io::ErrorKind::NotFound => requested_shards,
                    Err(e) => return Err(e),
                };
                let m = Self {
                    format_version: FORMAT_VERSION,
                    shard_count,
                    store_uuid: fresh_uuid()?,
                    feature_flags: 0,
                };
                m.write_atomic(root)?;
                Ok(m)
            }
        }
    }

    /// The manifest and the directories are two statements about one fact. If
    /// they disagree the only safe move is to refuse: picking a winner is how
    /// a store gets served with a shard missing.
    fn check_directories(&self, root: &Path) -> io::Result<()> {
        for i in 0..self.shard_count.get() {
            let d = root.join(format!("shard-{i}"));
            // A store whose shards have not been created yet is legitimate -
            // the manifest is written BEFORE them, deliberately, so a crash
            // between the two leaves a declared layout rather than a guessed
            // one. Only a PARTIAL set is a contradiction.
            if !d.is_dir() && i > 0 && root.join("shard-0").is_dir() {
                return Err(invalid(format!(
                    "{} declares {} shards but {} is missing: refusing to \
                     serve around a hole",
                    root.join(FILE).display(),
                    self.shard_count,
                    d.display()
                )));
            }
        }
        Ok(())
    }

    fn encode(&self) -> [u8; ENCODED_LEN] {
        let mut b = [0u8; ENCODED_LEN];
        b[..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        // u32 on the wire: a store with more than 4 billion shards is not a
        // layout question. Checked on the way back in.
        b[12..16].copy_from_slice(&(self.shard_count.get() as u32).to_le_bytes());
        b[16..32].copy_from_slice(&self.store_uuid);
        b[32..40].copy_from_slice(&self.feature_flags.to_le_bytes());
        let sum = crc32c::crc32c(&b[..BODY_LEN]);
        b[BODY_LEN..].copy_from_slice(&sum.to_le_bytes());
        b
    }

    fn decode(bytes: &[u8], root: &Path) -> io::Result<Self> {
        let where_ = root.join(FILE);
        // Length first, and exactly: a longer file is not a newer format - the
        // version field is what says the format. Tolerating a tail is how a
        // future writer's extra data gets silently ignored by an older reader.
        if bytes.len() != ENCODED_LEN {
            return Err(invalid(format!(
                "{} is {} bytes, expected exactly {}",
                where_.display(),
                bytes.len(),
                ENCODED_LEN
            )));
        }
        // Checksum before ANY field is believed, including the version - the
        // version of a corrupt file is itself corrupt.
        let want = u32::from_le_bytes(bytes[BODY_LEN..].try_into().unwrap());
        let got = crc32c::crc32c(&bytes[..BODY_LEN]);
        if got != want {
            return Err(invalid(format!(
                "{} fails its checksum (stored {want:#010x}, computed \
                 {got:#010x}): the layout on disk cannot be trusted",
                where_.display()
            )));
        }
        if &bytes[..8] != MAGIC {
            return Err(invalid(format!(
                "{} does not start with the layout magic: not a skeg layout \
                 manifest",
                where_.display()
            )));
        }
        let format_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if format_version != FORMAT_VERSION {
            return Err(invalid(format!(
                "{} declares layout format {format_version}, this build reads \
                 {FORMAT_VERSION}: refusing to guess what it means",
                where_.display()
            )));
        }
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let shard_count = NonZeroUsize::new(count as usize)
            .ok_or_else(|| invalid(format!("{} declares zero shards", where_.display())))?;
        let feature_flags = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        if feature_flags & !KNOWN_FLAGS != 0 {
            return Err(invalid(format!(
                "{} sets layout flags {feature_flags:#018x} this build does \
                 not know: the store uses something it cannot honour",
                where_.display()
            )));
        }
        Ok(Self {
            format_version,
            shard_count,
            store_uuid: bytes[16..32].try_into().unwrap(),
            feature_flags,
        })
    }

    /// Publish by rename, the only single-file operation that is actually
    /// atomic - the same discipline the base generation slots use.
    fn write_atomic(&self, root: &Path) -> io::Result<()> {
        std::fs::create_dir_all(root)?;
        let tmp = root.join("LAYOUT.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&self.encode())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, root.join(FILE))?;
        // The rename itself needs to reach the disk, or a crash can leave the
        // directory entry unpublished with the data already durable.
        skeg_platform::sync_dir(root)?;
        Ok(())
    }
}

/// 16 bytes of identity. `/dev/urandom` rather than a dependency: this is the
/// only randomness the server needs, and it is read exactly once per store.
fn fresh_uuid() -> io::Result<[u8; 16]> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf)
}

/// The legacy path: deduce the count from `shard-N` directories, fail-closed.
///
/// Kept private. It is a migration, not an API - every server constructor goes
/// through [`LayoutManifest::open_or_migrate`], so there is exactly one place
/// that decides how many shards a store has.
fn scan_shard_count(root: &Path) -> io::Result<NonZeroUsize> {
    let mut ids: Vec<usize> = Vec::new();
    for e in std::fs::read_dir(root)? {
        // NOT `.flatten()`: swallowing an unreadable entry lets discovery
        // conclude the layout simply HAS fewer shards, which is the exact
        // failure this function exists to refuse.
        let e = e?;
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(rest) = name.strip_prefix("shard-")
            && let Ok(id) = rest.parse::<usize>()
            && e.path().is_dir()
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    let Some(&highest) = ids.last() else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no shard-N directory under {} and no {FILE}: refusing to \
                 serve an unknown layout",
                root.display()
            ),
        ));
    };
    if ids.len() != highest + 1 || ids[0] != 0 {
        return Err(invalid(format!(
            "shard directories under {} are not contiguous from 0 (found \
             {ids:?}): refusing to serve around a hole",
            root.display()
        )));
    }
    NonZeroUsize::new(highest + 1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no shards"))
}
