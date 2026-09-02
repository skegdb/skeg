//! `skeg-platform` - HAL: `AsyncFile`, aligned buffers, core affinity.
//!
//! unsafe is intentionally allowed here; every unsafe block carries a
//! comment explaining the invariant that makes it sound.

pub mod affinity;
pub mod aligned;
mod cgroup;
pub mod durability;
pub mod file;
pub mod lock;
mod process;
pub use process::{cpu_seconds, fd_limit, open_fd_count, raise_fd_limit, rss_bytes};
pub mod uring;

pub use affinity::{QosClass, current_thread_qos, pin_current_thread_to_performance_core};
pub use aligned::AlignedBytes;
pub use durability::{DURABILITY_MODEL, DurabilityModel, resolve_durability_model};
pub use file::{
    BUFFER_ALIGNMENT, MappedFile, PlatformFile, advise_sequential_file, read_bounded,
    read_small_bytes, read_small_file, sync_dir,
};
pub use lock::{DirLock, LOCK_FILE};
#[cfg(all(target_os = "linux", feature = "uring"))]
pub use uring::UringBatchReader;
pub use uring::{BatchReader, BlockingBatchReader, best_batch_reader};

/// Return the number of performance (P-) cores available.
///
/// On macOS reads `hw.perflevel0.physicalcpu` via sysctl. On Linux reads the
/// cgroup CPU quota (v2 `cpu.max`, falling back to v1
/// `cpu,cpuacct/cpu.cfs_quota_us` + `cpu.cfs_period_us`), intersected with
/// the cgroup's `cpuset` when both are set - `available_parallelism` alone
/// reports the *host's* CPU count, which over-sizes a rayon/tokio pool
/// inside a container that only gets a fraction of the host. Falls back to
/// `std::thread::available_parallelism()` when no quota is set (or on
/// error, or on other platforms).
pub use cgroup::{Headroom, MemoryStatus};

/// What this process may use, and what it is using.
///
/// On Linux this is the cgroup's own accounting - the numbers the kernel will
/// actually kill the process over. Everywhere else there is no such limit to
/// read, so the limit is `None` and usage falls back to RSS.
///
/// RSS is a POOR substitute and the difference matters: it excludes what a
/// cgroup charges to the page cache for files this process mapped, and on
/// macOS it is unreliable enough that the same process has reported 3 MB and
/// 338 MB minutes apart. It is a signal, not an accounting. Callers that need
/// a real budget need Linux.
pub fn memory_status() -> MemoryStatus {
    #[cfg(target_os = "linux")]
    {
        // From the process's OWN cgroup, not the mount root. Under systemd a
        // service lives at /system.slice/<name>.service and the root reports
        // `max`, so reading the root makes the budget a decoration.
        let mut m = cgroup::memory_status_rooted(
            std::path::Path::new("/proc/self/cgroup"),
            std::path::Path::new("/sys/fs/cgroup"),
        );
        if m.current_bytes.is_none() {
            m.current_bytes = Some(rss_bytes());
        }
        m
    }
    #[cfg(not(target_os = "linux"))]
    {
        MemoryStatus {
            limit_bytes: None,
            current_bytes: Some(rss_bytes()),
            // No cgroup sets a limit here, which is what `Unlimited` means -
            // "nothing to be killed for" - and it is the truth: there is no
            // ceiling for a governor to respect. It used to report `Unknown`,
            // reasoning that an unmeasurable platform should not hand back a
            // number that looks safe. But `Unknown` means a ceiling APPLIES and
            // could not be read, and a fail-closed caller refused every write
            // on this platform as a result.
            //
            // No number is handed back either way: `Unlimited` is not a size.
            // An operator who wants a budget where the platform imposes none
            // sets `SKEG_MEMORY_LIMIT_BYTES`, which is the same lever the
            // refusal message names. RSS stays a signal and is still not an
            // accounting: nothing derives a budget from it.
            available: cgroup::Headroom::Unlimited,
        }
    }
}

#[cfg(test)]
mod platform_headroom_tests {
    use super::*;

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn a_platform_without_cgroups_reports_no_ceiling_not_an_unreadable_one() {
        // `Unknown` is fail-closed for its callers, so reporting it where no
        // ceiling exists refuses every write on a machine that has no limit to
        // respect. Caught by 25 tests at once, which is the good version of
        // finding out.
        assert_eq!(memory_status().available, cgroup::Headroom::Unlimited);
    }
}

pub fn num_performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        macos_perf_cores().unwrap_or_else(fallback_parallelism)
    }
    #[cfg(target_os = "linux")]
    {
        cgroup::cpu_quota(std::path::Path::new("/sys/fs/cgroup"))
            .unwrap_or_else(fallback_parallelism)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        fallback_parallelism()
    }
}

fn fallback_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn macos_perf_cores() -> Option<usize> {
    use std::mem;
    // SAFETY: sysctlbyname is a standard POSIX sysctl call.
    // `val` is a stack-allocated i32; `len` is set to sizeof(i32) so the
    // kernel cannot write past the end of `val`. All pointers are valid for
    // the duration of the call. Return value is checked before use.
    // The sign-loss cast is safe: we guard with `val > 0` before converting.
    #[allow(clippy::cast_sign_loss)]
    unsafe {
        let name = b"hw.perflevel0.physicalcpu\0";
        let mut val: libc::c_int = 0;
        let mut len = mem::size_of::<libc::c_int>() as libc::size_t;
        let ret = libc::sysctlbyname(
            name.as_ptr().cast(),
            std::ptr::addr_of_mut!(val).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        );
        if ret == 0 && val > 0 {
            Some(val as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_performance_cores_at_least_one() {
        assert!(num_performance_cores() >= 1);
    }
}
