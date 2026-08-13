//! Thread scheduling hints.
//!
//! macOS does not expose explicit CPU pinning. Instead the `QoS` class steers
//! the scheduler: `QOS_CLASS_USER_INTERACTIVE` threads are placed on
//! performance (P-) cores.
//!
//! Linux has no portable P-/E-core split to hint at (hybrid core
//! detection is Intel-specific and not exposed uniformly), so the pinning
//! call there does something different in kind: it restricts the calling
//! thread's affinity mask to the cgroup's effective cpuset via
//! `sched_setaffinity`, when one is configured. Full NUMA-node-aware
//! distribution of the tokio/rayon worker pool across a multi-socket box
//! (so workers stop bouncing between sockets) is *not* attempted here -
//! that needs real multi-socket hardware to validate the topology parsing
//! against, which has not been available; only the cpuset boundary is
//! enforced.
//!
//! On other platforms these calls are no-ops.
//!
//! unsafe here is intentional for the pthread `QoS` / `sched_setaffinity`
//! syscalls - see SAFETY comments.

/// Thread `QoS` class - mirrors macOS `qos_class_t` discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosClass {
    UserInteractive,
    UserInitiated,
    Default,
    Utility,
    Background,
    Unspecified,
}

/// Hint the scheduler to run the calling thread on a performance core.
///
/// On macOS sets the thread `QoS` to `USER_INTERACTIVE`. On Linux, restricts
/// the calling thread's affinity mask to the cgroup's effective cpuset (a
/// no-op if none is configured - see the module docs for what this does and
/// does not do there). No-op on other platforms.
pub fn pin_current_thread_to_performance_core() {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `pthread_set_qos_class_self_np` sets the QoS of the *calling*
        // thread only. `QOS_CLASS_USER_INTERACTIVE` with relative priority 0 is
        // a valid argument pair. The call does not touch caller memory; the
        // return value carries no safety obligation, so it is ignored.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
    #[cfg(target_os = "linux")]
    {
        let Some(cpus) = crate::cgroup::effective_cpu_ids(std::path::Path::new("/sys/fs/cgroup"))
        else {
            return; // no cpuset configured - nothing to restrict to
        };
        // SAFETY: `set` is a stack-local `cpu_set_t` fully initialised by
        // `CPU_ZERO` before any `CPU_SET` call. `sched_setaffinity(0, ...)`
        // targets the calling thread (pid 0 means "self" as an affinity
        // target); the size argument matches `size_of::<cpu_set_t>()`
        // exactly, matching what `&set` points to. The return value carries
        // no safety obligation - a failure (e.g. `EINVAL` from a cpu id at
        // or past `CPU_SETSIZE`) just leaves the thread's affinity as the
        // kernel had already set it, which is always still valid.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            for cpu in cpus {
                if cpu < libc::CPU_SETSIZE as usize {
                    libc::CPU_SET(cpu, &mut set);
                }
            }
            libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

/// Read the `QoS` class of the calling thread.
#[must_use]
pub fn current_thread_qos() -> QosClass {
    #[cfg(target_os = "macos")]
    {
        use libc::qos_class_t::{
            QOS_CLASS_BACKGROUND, QOS_CLASS_DEFAULT, QOS_CLASS_UNSPECIFIED,
            QOS_CLASS_USER_INITIATED, QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_UTILITY,
        };
        let mut class: libc::qos_class_t = QOS_CLASS_UNSPECIFIED;
        let mut prio: libc::c_int = 0;
        // SAFETY: `pthread_get_qos_class_np` writes the QoS class and relative
        // priority of the given thread into the two out-pointers. `&mut class`
        // and `&mut prio` are valid, properly aligned stack locations live for
        // the full duration of the call.
        unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &raw mut class, &raw mut prio);
        }
        match class {
            QOS_CLASS_USER_INTERACTIVE => QosClass::UserInteractive,
            QOS_CLASS_USER_INITIATED => QosClass::UserInitiated,
            QOS_CLASS_DEFAULT => QosClass::Default,
            QOS_CLASS_UTILITY => QosClass::Utility,
            QOS_CLASS_BACKGROUND => QosClass::Background,
            QOS_CLASS_UNSPECIFIED => QosClass::Unspecified,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        QosClass::Unspecified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_affinity_p_core() {
        // Run on a fresh thread so we do not perturb the test runner's QoS.
        let handle = std::thread::spawn(|| {
            pin_current_thread_to_performance_core();
            current_thread_qos()
        });
        let qos = handle.join().expect("thread join");

        #[cfg(target_os = "macos")]
        assert_eq!(
            qos,
            QosClass::UserInteractive,
            "P-core hint must set USER_INTERACTIVE QoS"
        );
        #[cfg(not(target_os = "macos"))]
        assert_eq!(qos, QosClass::Unspecified);
    }
}
