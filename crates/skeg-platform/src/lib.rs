//! `skeg-platform` - HAL: `AsyncFile`, aligned buffers, core affinity.
//!
//! unsafe is intentionally allowed here; every unsafe block carries a
//! comment explaining the invariant that makes it sound.

pub mod affinity;
pub mod aligned;
pub mod durability;
pub mod file;
pub mod lock;

pub use affinity::{QosClass, current_thread_qos, pin_current_thread_to_performance_core};
pub use aligned::AlignedBytes;
pub use durability::{DURABILITY_MODEL, DurabilityModel, resolve_durability_model};
pub use file::{BUFFER_ALIGNMENT, MappedFile, PlatformFile};
pub use lock::{DirLock, LOCK_FILE};

/// Return the number of performance (P-) cores available.
///
/// On macOS reads `hw.perflevel0.physicalcpu` via sysctl.
/// Falls back to `std::thread::available_parallelism()` on error or other OS.
#[must_use]
pub fn num_performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        macos_perf_cores().unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
    }
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
