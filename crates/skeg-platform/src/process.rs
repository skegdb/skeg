//! Self-reported process resource usage, for the telemetry surface: the
//! engine says what it costs instead of every operator re-deriving it from
//! ps. RSS is the platform's current resident set; CPU is cumulative
//! user+system seconds from getrusage (a monotone counter - consumers take
//! window deltas for a percentage).

/// Current resident set size in bytes, or 0 where unsupported.
#[must_use]
pub fn rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // /proc/self/statm field 2 is resident pages.
        if let Ok(s) = std::fs::read_to_string("/proc/self/statm")
            && let Some(res) = s.split_whitespace().nth(1)
            && let Ok(pages) = res.parse::<u64>()
        {
            // SAFETY: sysconf is a pure libc query.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            return pages * u64::try_from(page.max(0)).unwrap_or(4096);
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // MACH_TASK_BASIC_INFO.resident_size, the same number Activity
        // Monitor reports as memory.
        let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        // SAFETY: task_info fills `info` up to `count` natural_t words for
        // the current task; both pointers are valid for the call.
        #[allow(deprecated)] // mach_task_self: the mach2 crate is not worth a dep for one call
        let kr = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                info.as_mut_ptr().cast(),
                &mut count,
            )
        };
        if kr == libc::KERN_SUCCESS {
            // SAFETY: KERN_SUCCESS means the struct was written.
            return unsafe { info.assume_init() }.resident_size;
        }
        0
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Cumulative CPU seconds (user + system) of this process.
#[must_use]
pub fn cpu_seconds() -> f64 {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage fills the struct for RUSAGE_SELF.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return 0.0;
    }
    // SAFETY: rc == 0 means the struct was written.
    let ru = unsafe { ru.assume_init() };
    #[allow(clippy::cast_precision_loss)] // sub-second fields, exact in f64
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

/// The process's file-descriptor limits as `(soft, hard)`.
///
/// One descriptor per vlog segment and per vindex segment file means a large
/// store holds hundreds; macOS ships a soft limit of 256, which a parallel
/// test run or a multi-shard store at scale exhausts.
#[must_use]
pub fn fd_limit() -> (u64, u64) {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes into a fully-owned, correctly-typed struct.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut lim) } != 0 {
        return (0, 0);
    }
    // rlim_t is u64 on our targets; the casts keep it explicit where it is not.
    #[allow(clippy::unnecessary_cast)]
    (lim.rlim_cur as u64, lim.rlim_max as u64)
}

/// Raise the soft descriptor limit toward `desired`, capped by the hard limit.
/// Returns the soft limit in force afterwards.
///
/// What every production database does at boot: the default soft limit is a
/// shell convention, not a capacity decision. Never lowers an already-higher
/// limit, and a refusal is reported, not hidden - the caller logs it so an
/// operator can raise the hard limit themselves.
pub fn raise_fd_limit(desired: u64) -> u64 {
    let (soft, hard) = fd_limit();
    if soft == 0 || soft >= desired {
        return soft;
    }
    let want = desired.min(hard);
    let lim = libc::rlimit {
        rlim_cur: want as libc::rlim_t,
        rlim_max: hard as libc::rlim_t,
    };
    // SAFETY: setrlimit reads a fully-owned, correctly-typed struct.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &raw const lim) } == 0 {
        want
    } else {
        soft
    }
}

/// Descriptors this process currently holds, when the platform can say.
/// `None` where it cannot - the caller reports the limit either way.
#[must_use]
pub fn open_fd_count() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|d| d.filter_map(Result::ok).count() as u64)
    }
    #[cfg(target_os = "macos")]
    {
        // proc_pidinfo(PROC_PIDLISTFDS) with a null buffer returns the byte
        // size of the list; divide by the record size for the count.
        const PROC_PIDLISTFDS: libc::c_int = 1;
        // SAFETY: the documented "size query" form - null buffer, zero size.
        let bytes = unsafe {
            libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                PROC_PIDLISTFDS,
                0,
                std::ptr::null_mut(),
                0,
            )
        };
        if bytes <= 0 {
            return None;
        }
        Some(bytes as u64 / std::mem::size_of::<libc::proc_fdinfo>() as u64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_limit_and_count_are_plausible() {
        let (soft, hard) = fd_limit();
        assert!(soft > 0 && hard >= soft, "limits: soft {soft} hard {hard}");
        // The test process holds at least stdin/stdout/stderr.
        if let Some(n) = open_fd_count() {
            assert!(n >= 3, "only {n} descriptors open");
            assert!(n <= soft, "{n} open exceeds the soft limit {soft}");
        }
    }

    #[test]
    fn raising_never_lowers_the_limit() {
        let (before, _) = fd_limit();
        let after = raise_fd_limit(1);
        assert!(
            after >= before,
            "raise_fd_limit(1) lowered {before} to {after}"
        );
    }

    #[test]
    fn rss_and_cpu_report_plausible_numbers() {
        let rss = rss_bytes();
        assert!(
            rss > 1 << 20,
            "a running test process holds > 1MB, got {rss}"
        );
        let c0 = cpu_seconds();
        // Burn a little CPU; the counter must be monotone non-decreasing.
        let mut x = 0u64;
        for i in 0..5_000_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        std::hint::black_box(x);
        assert!(cpu_seconds() >= c0);
    }
}
