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
    let tv = |t: libc::timeval| t.tv_sec as f64 + f64::from(t.tv_usec as i32) / 1e6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_and_cpu_report_plausible_numbers() {
        let rss = rss_bytes();
        assert!(rss > 1 << 20, "a running test process holds > 1MB, got {rss}");
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
