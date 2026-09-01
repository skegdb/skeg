//! cgroup CPU quota/cpuset parsing (v1 and v2), for container-aware pool
//! sizing and thread affinity.
//!
//! `std::thread::available_parallelism` reports the *host's* CPU count -
//! inside a cgroup-limited container that over-sizes a rayon/tokio pool to
//! more workers than the kernel will ever schedule concurrently for this
//! process. The parsing here is plain string handling with no OS-specific
//! calls, so it runs on any platform; only the call sites in `lib.rs` /
//! `affinity.rs` gate *use* of it to Linux, where these cgroup paths
//! actually exist.
//!
//! The parsing helpers below have no production caller outside
//! `#[cfg(target_os = "linux")]` call sites (`lib.rs`, `affinity.rs`), so a
//! non-Linux build never exercises them - that's a legitimate
//! cross-platform shape, not dead code, hence the blanket allow. Their own
//! tests still run on every platform.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::path::Path;

/// Parse cgroup v2 `cpu.max` content: `"<quota> <period>"` or `"max <period>"`.
/// Returns the quota in whole vCPUs, `None` for `"max"`/unset/unparseable.
fn parse_cpu_max(s: &str) -> Option<f64> {
    let mut parts = s.split_whitespace();
    let quota = parts.next()?;
    let period: f64 = parts.next()?.parse().ok()?;
    if quota == "max" {
        return None;
    }
    let quota: f64 = quota.parse().ok()?;
    if quota <= 0.0 || period <= 0.0 {
        return None;
    }
    Some(quota / period)
}

/// Parse cgroup v1 `cpu.cfs_quota_us` (`-1` = unlimited) and `cpu.cfs_period_us`.
fn parse_cfs_quota(quota_us: &str, period_us: &str) -> Option<f64> {
    let quota: i64 = quota_us.trim().parse().ok()?;
    let period: i64 = period_us.trim().parse().ok()?;
    if quota <= 0 || period <= 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    Some(quota as f64 / period as f64)
}

/// Expand a cgroup `cpuset` list like `"0-3,8,10-11"` into explicit CPU ids.
/// Unparseable items are skipped rather than failing the whole list - a
/// malformed entry should not blind the caller to the CPUs it can parse.
fn parse_cpu_list(s: &str) -> Vec<usize> {
    s.trim()
        .split(',')
        .filter(|item| !item.is_empty())
        .flat_map(|item| match item.split_once('-') {
            Some((lo, hi)) => {
                let lo: usize = lo.trim().parse().unwrap_or(0);
                let hi: usize = hi.trim().parse().unwrap_or(lo);
                lo..=hi
            }
            None => {
                let v: usize = item.trim().parse().unwrap_or(0);
                v..=v
            }
        })
        .collect()
}

fn read(path: std::path::PathBuf) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// The cgroup's effective cpuset (v2 `cpuset.cpus.effective`, falling back
/// to plain `cpuset.cpus`, then v1 `cpuset/cpuset.cpus`) as explicit CPU
/// ids. `None` when no cpuset file is present (unconfined, or not running
/// under a cgroup at all) or it lists no CPUs.
pub(crate) fn effective_cpu_ids(root: &Path) -> Option<Vec<usize>> {
    let list = read(root.join("cpuset.cpus.effective"))
        .or_else(|| read(root.join("cpuset.cpus")))
        .or_else(|| read(root.join("cpuset/cpuset.cpus")))?;
    let ids = parse_cpu_list(&list);
    (!ids.is_empty()).then_some(ids)
}

/// Effective CPU count for the cgroup rooted at `root` (normally
/// `/sys/fs/cgroup`), combining the CPU quota and the cpuset - whichever is
/// tighter. `None` when neither is set (unbounded, or not running under a
/// cgroup at all); the caller should fall back to `available_parallelism`
/// in that case.
pub(crate) fn cpu_quota(root: &Path) -> Option<usize> {
    let quota_vcpu = read(root.join("cpu.max"))
        .and_then(|s| parse_cpu_max(&s))
        .or_else(|| {
            let quota = read(root.join("cpu,cpuacct/cpu.cfs_quota_us"))?;
            let period = read(root.join("cpu,cpuacct/cpu.cfs_period_us"))?;
            parse_cfs_quota(&quota, &period)
        });

    let cpuset_vcpu = effective_cpu_ids(root).map(|ids| ids.len());

    #[allow(clippy::cast_precision_loss)]
    let effective = match (quota_vcpu, cpuset_vcpu) {
        (Some(q), Some(c)) => Some(q.min(c as f64)),
        (Some(q), None) => Some(q),
        (None, Some(c)) => Some(c as f64),
        (None, None) => None,
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    effective.map(|v| (v.floor() as usize).max(1))
}

/// What the cgroup says about memory: the hard limit, and current usage.
///
/// Both are `Option` and both mean what they say. `None` is "not known",
/// never "zero" - a governor told its limit is zero rejects every allocation
/// forever, and one told its usage is zero believes it has the whole machine.
/// A file that is missing, unreadable, malformed, zero, or set to one of the
/// two "unlimited" spellings yields `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryStatus {
    /// The tightest hard limit on this process's cgroup chain.
    pub limit_bytes: Option<u64>,
    /// Usage charged to the cgroup that set `limit_bytes`.
    pub current_bytes: Option<u64>,
    /// What this process may still allocate before the FIRST cgroup on its
    /// chain is saturated: `min(limit - current)` over the whole hierarchy.
    ///
    /// A separate number because one limit/current pair cannot describe a
    /// hierarchy. Every cgroup from the process's own to the root applies at
    /// once, and the one that runs out first is the one with the least
    /// headroom, which is NOT the one with the smallest limit: a leaf at
    /// 256 MiB holding 10 MiB has 246 MiB free, but inside a parent at
    /// 512 MiB already holding 500 MiB, twelve more megabytes end the
    /// process.
    ///
    /// `None` means unknown, never zero and never unlimited: no limit exists
    /// anywhere, or one does and its usage could not be read. A budget cannot
    /// be derived from either, and guessing is what makes a governor unsafe.
    pub available_bytes: Option<u64>,
}

/// cgroup v1 spells "unlimited" as PAGE_SIZE-rounded `i64::MAX` rather than a
/// word. Anything at or above this is the sentinel, not a limit.
const V1_UNLIMITED: u64 = 0x7FFF_FFFF_FFFF_F000;

/// Parse one of these files: a bare decimal, or v2's literal `max`.
/// Zero and the v1 sentinel are refused - neither is a usable limit.
fn parse_mem_limit(s: &str) -> Option<u64> {
    let s = s.trim();
    if s == "max" {
        return None;
    }
    let v: u64 = s.parse().ok()?;
    (v > 0 && v < V1_UNLIMITED).then_some(v)
}

/// A usage counter: a bare decimal. Zero is a legitimate reading here.
fn parse_mem_usage(s: &str) -> Option<u64> {
    s.trim().parse().ok()
}

/// The cgroup path this process actually belongs to, relative to the cgroup
/// mount, parsed from `/proc/self/cgroup`.
///
/// This is the piece that makes the difference between a memory budget and a
/// decoration. Under systemd a service is not at the cgroup root - the file
/// reads `0::/system.slice/skeg.service` - and the limit that will kill the
/// process lives THERE, while the root's `memory.max` says `max`. A governor
/// reading only the root reports "no limit" and admits everything right up to
/// the OOM kill.
///
/// v2 lines look like `0::/path`; v1 lists one line per controller and only
/// the `memory` one is relevant. Returns `None` when the file is absent
/// (not Linux, /proc not mounted) or names nothing usable - the caller then
/// reads the root, which is the honest guess.
fn self_cgroup_path(proc_self_cgroup: &Path) -> Option<Membership> {
    let text = read(proc_self_cgroup.to_path_buf())?;
    let mut v1: Option<&str> = None;
    let mut v2: Option<String> = None;
    for line in text.lines() {
        let mut f = line.splitn(3, ':');
        let (_hier, controllers, path) = (f.next()?, f.next()?, f.next());
        let Some(path) = path else { continue };
        if controllers.is_empty() {
            // v2 unified line. Remembered, not returned: on a HYBRID host both
            // hierarchies are mounted, and the v2 line can exist while the
            // memory controller lives in v1. Returning here would hide the
            // only limit there is.
            v2 = sanitise_cgroup_path(path);
            continue;
        }
        if controllers.split(',').any(|c| c == "memory") {
            v1 = Some(path);
        }
    }
    // v1 first when both are present: on a hybrid host the memory controller
    // is the one in v1, and it is the limit that applies.
    v1.and_then(sanitise_cgroup_path)
        .map(Membership::V1)
        .or(v2.map(Membership::V2))
}

/// Which hierarchy the process belongs to, and where in it.
///
/// The two are not interchangeable on disk: under v2 the path hangs directly
/// off the mount, while under v1 the CONTROLLER is part of the path
/// (`/sys/fs/cgroup/memory/<path>`) and the files are named differently.
/// Treating them as one layout reads a v1 limit from a directory that does
/// not exist, and reports "unlimited" for a capped process.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Membership {
    V2(String),
    V1(String),
}

/// Accept only a plain absolute cgroup path. The value comes from a FILE, and
/// joining `..` onto the cgroup mount would read arbitrary files as if they
/// were limits. Anything suspicious yields `None`, and the caller falls back
/// to the root.
fn sanitise_cgroup_path(path: &str) -> Option<String> {
    let p = path.trim();
    let p = p.strip_prefix('/')?;
    if p.is_empty() {
        return None; // the root itself; nothing to descend into
    }
    if p.split('/').any(|c| c.is_empty() || c == "." || c == "..") {
        return None;
    }
    Some(p.to_string())
}

/// v2 files, read from one directory.
fn v2_at(dir: &Path) -> MemoryStatus {
    MemoryStatus {
        limit_bytes: read(dir.join("memory.max")).and_then(|s| parse_mem_limit(&s)),
        current_bytes: read(dir.join("memory.current")).and_then(|s| parse_mem_usage(&s)),
        // One directory cannot know the chain's headroom.
        available_bytes: None,
    }
}

/// v1 files, read from one directory (already including the controller).
fn v1_at(dir: &Path) -> MemoryStatus {
    MemoryStatus {
        limit_bytes: read(dir.join("memory.limit_in_bytes")).and_then(|s| parse_mem_limit(&s)),
        current_bytes: read(dir.join("memory.usage_in_bytes")).and_then(|s| parse_mem_usage(&s)),
        available_bytes: None,
    }
}

/// Memory limit and usage for the process's OWN cgroup.
///
/// Reads membership from `proc_self_cgroup`, then walks from the process's own
/// directory up to the hierarchy root, taking the first limit it finds.
/// Climbing is deliberate: a nested cgroup often sets nothing itself and is
/// capped by an ancestor, and the limit that matters is the tightest one that
/// applies - reading only the leaf reports "unlimited" for a process capped
/// one level up.
///
/// Usage comes from the same directory as the limit, so the two numbers
/// describe one accounting domain rather than two.
pub(crate) fn memory_status_rooted(proc_self_cgroup: &Path, mount: &Path) -> MemoryStatus {
    let (mut dir, top, read_at): (_, _, fn(&Path) -> MemoryStatus) =
        match self_cgroup_path(proc_self_cgroup) {
            Some(Membership::V2(rel)) => (mount.join(rel), mount.to_path_buf(), v2_at),
            Some(Membership::V1(rel)) => {
                let top = mount.join("memory");
                (top.join(rel), top, v1_at)
            }
            // No membership to read: not Linux, /proc absent, or a path this
            // build will not follow. The mount root is the honest guess.
            None => return memory_status_at(mount),
        };
    // Walk the WHOLE chain and keep two things: the tightest limit (for
    // reporting) and the least headroom (for deciding).
    //
    // Every cgroup from the process's own up to the root applies at once, so
    // what runs out first is the one with the least `limit - current`. That is
    // not the one with the smallest limit: a leaf at 256 MiB holding 10 MiB
    // has 246 MiB free, but inside a parent at 512 MiB already holding
    // 500 MiB - siblings' memory - twelve more megabytes end the process.
    //
    // A limit whose usage cannot be read makes the headroom UNKNOWN, and
    // unknown must not be substituted with zero (rejects everything) or with
    // the limit (invents room that may not exist).
    let mut tightest: Option<(u64, MemoryStatus)> = None;
    let mut headroom: Option<u64> = None;
    let mut saw_limit = false;
    let mut headroom_unknown = false;
    loop {
        let m = read_at(&dir);
        if let Some(limit) = m.limit_bytes {
            saw_limit = true;
            if tightest.as_ref().is_none_or(|(b, _)| limit < *b) {
                tightest = Some((limit, m));
            }
            match m.current_bytes {
                Some(cur) => {
                    let free = limit.saturating_sub(cur);
                    headroom = Some(headroom.map_or(free, |h: u64| h.min(free)));
                }
                None => headroom_unknown = true,
            }
        }
        if dir == top {
            break;
        }
        match dir.parent() {
            Some(p) if p.starts_with(&top) => dir = p.to_path_buf(),
            _ => break,
        }
    }
    let available_bytes = if !saw_limit || headroom_unknown {
        None
    } else {
        headroom
    };
    match tightest {
        Some((_, m)) => MemoryStatus {
            available_bytes,
            ..m
        },
        // Nothing in the chain sets a limit. Report usage from the process's
        // own cgroup, which is the one that describes it.
        None => MemoryStatus {
            available_bytes: None,
            ..read_at(&match self_cgroup_path(proc_self_cgroup) {
                Some(Membership::V2(rel)) => mount.join(rel),
                Some(Membership::V1(rel)) => mount.join("memory").join(rel),
                None => mount.to_path_buf(),
            })
        },
    }
}

/// Memory limit and usage for the cgroup rooted at `root`, v2 first then v1.
///
/// v2 wins when both exist: a host running v2 with v1 compatibility files
/// present would otherwise be read through the older, coarser pair.
pub(crate) fn memory_status_at(root: &Path) -> MemoryStatus {
    let (limit, current) = match read(root.join("memory.max")) {
        Some(s) => (
            parse_mem_limit(&s),
            read(root.join("memory.current")).and_then(|s| parse_mem_usage(&s)),
        ),
        None => (
            read(root.join("memory/memory.limit_in_bytes")).and_then(|s| parse_mem_limit(&s)),
            read(root.join("memory/memory.usage_in_bytes")).and_then(|s| parse_mem_usage(&s)),
        ),
    };
    MemoryStatus {
        limit_bytes: limit,
        current_bytes: current,
        // One directory is not a chain; `memory_status_rooted` computes this.
        available_bytes: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_max_numeric_quota() {
        assert_eq!(parse_cpu_max("400000 100000"), Some(4.0));
        assert_eq!(parse_cpu_max("150000 100000\n"), Some(1.5));
    }

    #[test]
    fn parse_cpu_max_unlimited() {
        assert_eq!(parse_cpu_max("max 100000"), None);
    }

    #[test]
    fn parse_cpu_max_malformed_is_none() {
        assert_eq!(parse_cpu_max(""), None);
        assert_eq!(parse_cpu_max("garbage"), None);
        assert_eq!(parse_cpu_max("0 100000"), None);
    }

    #[test]
    fn parse_cfs_quota_numeric() {
        assert_eq!(parse_cfs_quota("400000", "100000"), Some(4.0));
    }

    #[test]
    fn parse_cfs_quota_unlimited_is_negative_one() {
        assert_eq!(parse_cfs_quota("-1", "100000"), None);
    }

    #[test]
    fn parse_cpu_list_ranges_and_singletons() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-3,8"), vec![0, 1, 2, 3, 8]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
    }

    #[test]
    fn parse_cpu_list_empty_is_empty() {
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    }

    fn write(dir: &std::path::Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn effective_cpu_ids_prefers_v2_effective_file() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpuset.cpus.effective", "0-3");
        assert_eq!(effective_cpu_ids(dir.path()), Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn effective_cpu_ids_falls_back_to_v1_path() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpuset/cpuset.cpus", "0,2");
        assert_eq!(effective_cpu_ids(dir.path()), Some(vec![0, 2]));
    }

    #[test]
    fn effective_cpu_ids_none_when_unset() {
        let dir = tempfile::TempDir::new().unwrap();
        assert_eq!(effective_cpu_ids(dir.path()), None);
    }

    #[test]
    fn cpu_quota_v2_quota_tighter_than_cpuset() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpu.max", "400000 100000"); // 4 vCPU
        write(dir.path(), "cpuset.cpus.effective", "0-7"); // 8 CPUs
        assert_eq!(cpu_quota(dir.path()), Some(4));
    }

    #[test]
    fn cpu_quota_v2_cpuset_tighter_than_quota() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpu.max", "800000 100000"); // 8 vCPU
        write(dir.path(), "cpuset.cpus.effective", "0-1"); // 2 CPUs
        assert_eq!(cpu_quota(dir.path()), Some(2));
    }

    #[test]
    fn cpu_quota_v2_unlimited_falls_back_to_cpuset() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpu.max", "max 100000");
        write(dir.path(), "cpuset.cpus.effective", "0-3");
        assert_eq!(cpu_quota(dir.path()), Some(4));
    }

    #[test]
    fn cpu_quota_v1_fallback() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpu,cpuacct/cpu.cfs_quota_us", "300000");
        write(dir.path(), "cpu,cpuacct/cpu.cfs_period_us", "100000");
        assert_eq!(cpu_quota(dir.path()), Some(3));
    }

    #[test]
    fn cpu_quota_nothing_set_is_none() {
        let dir = tempfile::TempDir::new().unwrap();
        assert_eq!(cpu_quota(dir.path()), None);
    }

    #[test]
    fn cpu_quota_floors_a_fractional_vcpu_but_never_below_one() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "cpu.max", "150000 100000"); // 1.5 vCPU
        assert_eq!(cpu_quota(dir.path()), Some(1));
    }

    // ---- where the process actually lives ----

    #[test]
    fn v2_membership_comes_from_proc_self_cgroup() {
        // Under systemd a service is NOT at the cgroup root. `/proc/self/cgroup`
        // reads `0::/system.slice/skeg.service`, and the limit lives there -
        // the root's `memory.max` is `max`. Reading only the root reports "no
        // limit" for a process that has one, which makes a memory governor a
        // placebo: it admits everything right up to the OOM kill.
        let dir = tempfile::TempDir::new().unwrap();
        write(
            dir.path(),
            "proc/self/cgroup",
            "0::/system.slice/skeg.service\n",
        );
        write(dir.path(), "sys/fs/cgroup/memory.max", "max\n");
        write(
            dir.path(),
            "sys/fs/cgroup/system.slice/skeg.service/memory.max",
            "268435456\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/system.slice/skeg.service/memory.current",
            "1000\n",
        );
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(
            m.limit_bytes,
            Some(268_435_456),
            "must read the LEAF, not the root"
        );
        assert_eq!(m.current_bytes, Some(1000));
    }

    #[test]
    fn a_process_at_the_v2_root_reads_the_root() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/\n");
        write(dir.path(), "sys/fs/cgroup/memory.max", "123\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(123));
    }

    #[test]
    fn what_runs_out_first_is_the_least_headroom_not_the_lowest_limit() {
        // The counterexample that the "smallest limit" rule gets wrong.
        //
        //   leaf   256 MiB limit, 10 MiB used  -> 246 MiB of headroom
        //   parent 512 MiB limit, 500 MiB used ->  12 MiB of headroom
        //
        // The smallest LIMIT is the leaf, and reading it promises 246 MiB.
        // But the parent is shared with sibling processes that have already
        // taken 500 MiB of it, so twelve more megabytes saturate it and this
        // process is killed. What a budget needs is the least headroom on the
        // chain, not the lowest number on it.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/parent/leaf\n");
        write(dir.path(), "sys/fs/cgroup/parent/memory.max", "536870912\n");
        write(
            dir.path(),
            "sys/fs/cgroup/parent/memory.current",
            "524288000\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/parent/leaf/memory.max",
            "268435456\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/parent/leaf/memory.current",
            "10485760\n",
        );
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(
            m.available_bytes,
            Some(12 * 1024 * 1024),
            "the parent has 12 MiB left; the leaf's 246 MiB is unreachable"
        );
    }

    #[test]
    fn a_limit_whose_usage_cannot_be_read_yields_no_headroom_at_all() {
        // A limit with unknown usage cannot produce a safe number. Falling
        // back to zero would reject everything; falling back to RSS, or to the
        // limit itself, would invent headroom that may not exist. `None` says
        // what is true: this build does not know.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.max", "268435456\n");
        // no memory.current
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(268_435_456), "the limit is still known");
        assert_eq!(m.available_bytes, None, "the headroom is not");
    }

    #[test]
    fn no_limit_anywhere_means_no_headroom_figure_either() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.current", "999\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, None);
        assert_eq!(m.available_bytes, None, "unlimited is not zero headroom");
    }

    #[test]
    fn a_cgroup_already_over_its_limit_has_no_headroom_not_negative() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.max", "1000\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.current", "1500\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.available_bytes, Some(0));
    }

    #[test]
    fn the_effective_limit_is_the_tightest_in_the_chain_not_the_nearest() {
        // Every cgroup from the process's own up to the root applies at once,
        // so the one that kills the process is the tightest - and it is not
        // always the nearest. A leaf at 1 GiB inside a parent at 256 MiB is a
        // 256 MiB process; reporting the leaf claims four times the memory it
        // has, which is worse than reporting none: a governor then admits work
        // right up to the kill.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a/b\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.max", "268435456\n"); // 256 MiB
        write(dir.path(), "sys/fs/cgroup/a/memory.current", "1\n");
        write(dir.path(), "sys/fs/cgroup/a/b/memory.max", "1073741824\n"); // 1 GiB
        write(dir.path(), "sys/fs/cgroup/a/b/memory.current", "999\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(268_435_456), "must take the tightest");
        assert_eq!(
            m.current_bytes,
            Some(1),
            "usage must come from the cgroup that set the limit, not another"
        );
    }

    #[test]
    fn a_hybrid_host_does_not_let_the_v2_line_hide_the_v1_limit() {
        // Both hierarchies mounted: the v2 unified line exists but carries no
        // memory controller, while the real limit sits in v1. Returning on the
        // first v2 line hides the only limit there is.
        let dir = tempfile::TempDir::new().unwrap();
        write(
            dir.path(),
            "proc/self/cgroup",
            "0::/user.slice\n9:memory:/docker/abc\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
            "268435456\n",
        );
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(268_435_456));
    }

    #[test]
    fn a_chain_with_no_limit_anywhere_reports_none_and_the_leafs_usage() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a/b\n");
        write(dir.path(), "sys/fs/cgroup/a/b/memory.current", "4242\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, None);
        assert_eq!(m.current_bytes, Some(4242));
    }

    #[test]
    fn a_leaf_without_its_own_limit_climbs_to_the_nearest_ancestor_that_has_one() {
        // A nested cgroup often sets nothing itself; the limit that will kill
        // the process is the tightest one above it. Reading only the leaf
        // reports "unlimited" for a process that is capped one level up.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/a/b/c\n");
        write(dir.path(), "sys/fs/cgroup/a/memory.max", "999\n");
        write(dir.path(), "sys/fs/cgroup/a/b/memory.max", "max\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(999));
    }

    #[test]
    fn v1_membership_uses_the_memory_controller_line() {
        // v1 lists one line per controller; only the `memory` one matters here.
        let dir = tempfile::TempDir::new().unwrap();
        write(
            dir.path(),
            "proc/self/cgroup",
            "12:cpu,cpuacct:/other\n9:memory:/docker/abc\n3:devices:/x\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
            "536870912\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes",
            "77\n",
        );
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(536_870_912));
        assert_eq!(m.current_bytes, Some(77));
    }

    #[test]
    fn a_missing_proc_file_falls_back_to_the_root() {
        // Not Linux, or /proc not mounted. The root is the honest guess, and
        // "no limit found" is reported as None rather than invented.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "sys/fs/cgroup/memory.max", "42\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(m.limit_bytes, Some(42));
    }

    #[test]
    fn a_malformed_proc_file_does_not_escape_the_cgroup_root() {
        // The path comes from a file. `..` in it must not walk out of
        // /sys/fs/cgroup and read arbitrary files as if they were limits.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "proc/self/cgroup", "0::/../../../../etc\n");
        write(dir.path(), "sys/fs/cgroup/memory.max", "7\n");
        let m = memory_status_rooted(
            &dir.path().join("proc/self/cgroup"),
            &dir.path().join("sys/fs/cgroup"),
        );
        assert_eq!(
            m.limit_bytes,
            Some(7),
            "must fall back to the root, not climb out"
        );
    }

    // ---- memory ----

    #[test]
    fn memory_v2_reports_limit_and_current() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory.max", "268435456\n");
        write(dir.path(), "memory.current", "12345678\n");
        let m = memory_status_at(dir.path());
        assert_eq!(m.limit_bytes, Some(268_435_456));
        assert_eq!(m.current_bytes, Some(12_345_678));
    }

    #[test]
    fn memory_v2_max_means_unlimited_not_zero() {
        // The difference matters: `Some(0)` would tell the governor it has no
        // memory at all and reject every write.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory.max", "max\n");
        write(dir.path(), "memory.current", "999\n");
        let m = memory_status_at(dir.path());
        assert_eq!(m.limit_bytes, None);
        assert_eq!(m.current_bytes, Some(999));
    }

    #[test]
    fn memory_falls_back_to_v1() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory/memory.limit_in_bytes", "536870912\n");
        write(dir.path(), "memory/memory.usage_in_bytes", "4096\n");
        let m = memory_status_at(dir.path());
        assert_eq!(m.limit_bytes, Some(536_870_912));
        assert_eq!(m.current_bytes, Some(4096));
    }

    #[test]
    fn memory_v1_unlimited_sentinel_is_not_a_limit() {
        // v1 spells "unlimited" as a huge number rather than a word: PAGE_SIZE
        // rounded i64::MAX. Believing it would set a limit of 8 exabytes,
        // which is not wrong so much as meaningless - and it would make the
        // reserve arithmetic below overflow-adjacent for no reason.
        let dir = tempfile::TempDir::new().unwrap();
        write(
            dir.path(),
            "memory/memory.limit_in_bytes",
            "9223372036854771712",
        );
        assert_eq!(memory_status_at(dir.path()).limit_bytes, None);
    }

    #[test]
    fn memory_malformed_is_none_never_zero() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory.max", "not-a-number");
        write(dir.path(), "memory.current", "");
        let m = memory_status_at(dir.path());
        assert_eq!(m.limit_bytes, None);
        assert_eq!(m.current_bytes, None);
    }

    #[test]
    fn memory_zero_limit_is_rejected_as_nonsense() {
        // A zero limit is not a limit, it is a broken file. Honouring it would
        // refuse every allocation forever.
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory.max", "0");
        assert_eq!(memory_status_at(dir.path()).limit_bytes, None);
    }

    #[test]
    fn memory_nothing_set_is_all_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let m = memory_status_at(dir.path());
        assert_eq!(m.limit_bytes, None);
        assert_eq!(m.current_bytes, None);
    }

    #[test]
    fn memory_v2_wins_over_v1_when_both_exist() {
        let dir = tempfile::TempDir::new().unwrap();
        write(dir.path(), "memory.max", "111");
        write(dir.path(), "memory/memory.limit_in_bytes", "222");
        assert_eq!(memory_status_at(dir.path()).limit_bytes, Some(111));
    }
}
