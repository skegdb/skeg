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
    /// The hard limit this process will be killed for crossing.
    pub limit_bytes: Option<u64>,
    /// Current charged usage, as the kernel accounts it.
    pub current_bytes: Option<u64>,
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
