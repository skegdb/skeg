//! Password verification must not occupy an async worker or escape memory admission.
use crate::{TenantBackend, TenantId, memory::MemoryGovernor};
use std::sync::{Arc, LazyLock};

// Process-wide, including multiple listeners. No unbounded blocking-pool queue.
static PASSWORD_JOBS: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(2)));

pub(crate) async fn verify(
    backend: Arc<dyn TenantBackend>,
    memory: Arc<MemoryGovernor>,
    user: String,
    password: Vec<u8>,
) -> Result<Option<TenantId>, &'static str> {
    verify_with(Arc::clone(&PASSWORD_JOBS), backend, memory, user, password).await
}

async fn verify_with(
    workers: Arc<tokio::sync::Semaphore>,
    backend: Arc<dyn TenantBackend>,
    memory: Arc<MemoryGovernor>,
    user: String,
    password: Vec<u8>,
) -> Result<Option<TenantId>, &'static str> {
    let bytes = backend.login_memory_bytes(&user);
    if bytes == u64::MAX {
        return Err("ERR invalid authentication memory cost");
    }
    let permit = workers
        .try_acquire_owned()
        .map_err(|_| "BACKPRESSURE authentication workers busy")?;
    let reservation = memory
        .try_reserve(bytes)
        .map_err(|_| "BACKPRESSURE authentication memory busy")?;
    tokio::task::spawn_blocking(move || {
        // Cancelling the connection cannot release either permit while Argon2 runs.
        let (_permit, _reservation) = (permit, reservation);
        backend.verify_login(&user, &password)
    })
    .await
    .map_err(|_| "ERR authentication worker failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Headroom, MemorySource};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    // Separate fixture gate: parallel unit tests cannot consume each other's
    // capacity. They execute exactly the implementation used by the listener.
    async fn verify(
        backend: Arc<dyn TenantBackend>,
        memory: Arc<MemoryGovernor>,
        user: String,
        password: Vec<u8>,
    ) -> Result<Option<TenantId>, &'static str> {
        verify_with(
            Arc::new(tokio::sync::Semaphore::new(2)),
            backend,
            memory,
            user,
            password,
        )
        .await
    }

    #[derive(Debug)]
    struct Unlimited;
    impl MemorySource for Unlimited {
        fn headroom(&self) -> Headroom {
            Headroom::Unlimited
        }
    }
    struct Slow {
        entered: AtomicBool,
        released: AtomicBool,
    }
    impl TenantBackend for Slow {
        fn has_tenant(&self, _: TenantId) -> bool {
            true
        }
        fn verify_login(&self, _: &str, _: &[u8]) -> Option<TenantId> {
            self.entered.store(true, Ordering::Release);
            while !self.released.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Some(TenantId::ZERO)
        }
    }
    fn memory(limit: u64) -> Arc<MemoryGovernor> {
        Arc::new(MemoryGovernor::new(Arc::new(Unlimited), Some(limit), Some(0)).unwrap())
    }
    fn slow() -> Arc<Slow> {
        Arc::new(Slow {
            entered: AtomicBool::new(false),
            released: AtomicBool::new(false),
        })
    }

    #[tokio::test]
    async fn invalid_password_cost_is_rejected_even_without_a_memory_limit() {
        struct InvalidCost;
        impl TenantBackend for InvalidCost {
            fn has_tenant(&self, _: TenantId) -> bool {
                true
            }
            fn login_memory_bytes(&self, _: &str) -> u64 {
                u64::MAX
            }
            fn verify_login(&self, _: &str, _: &[u8]) -> Option<TenantId> {
                Some(TenantId::ZERO)
            }
        }
        let governor = Arc::new(MemoryGovernor::new(Arc::new(Unlimited), None, Some(0)).unwrap());
        let result = verify(Arc::new(InvalidCost), governor.clone(), "u".into(), vec![]).await;
        assert_eq!(result, Err("ERR invalid authentication memory cost"));
        assert_eq!(governor.reserved_bytes(), 0);
    }

    #[tokio::test]
    async fn full_worker_pool_refuses_without_starting_or_reserving() {
        let workers = Arc::new(tokio::sync::Semaphore::new(2));
        let one = workers.clone().acquire_owned().await.unwrap();
        let two = workers.clone().acquire_owned().await.unwrap();
        let backend = slow();
        backend.released.store(true, Ordering::Release);
        let governor = memory(256 << 20);
        let result = verify_with(
            workers.clone(),
            backend.clone(),
            governor.clone(),
            "u".into(),
            vec![],
        )
        .await;
        assert_eq!(result, Err("BACKPRESSURE authentication workers busy"));
        assert!(!backend.entered.load(Ordering::Acquire));
        assert_eq!(governor.reserved_bytes(), 0);
        drop((one, two));
        assert!(
            verify_with(workers, backend, governor, "u".into(), vec![])
                .await
                .unwrap()
                .is_some()
        );
    }
    #[tokio::test]
    async fn password_work_does_not_stall_the_async_worker() {
        let backend = slow();
        let release = backend.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            release.released.store(true, Ordering::Release);
        });
        let start = Instant::now();
        let job = tokio::spawn(verify(backend, memory(256 << 20), "u".into(), vec![]));
        tokio::time::sleep(Duration::from_millis(20)).await;
        let elapsed = start.elapsed();
        job.await.unwrap().unwrap();
        releaser.join().unwrap();
        assert!(
            elapsed < Duration::from_millis(200),
            "async worker stalled for {elapsed:?}"
        );
    }
    #[tokio::test]
    async fn insufficient_memory_refuses_before_password_work() {
        let backend = slow();
        backend.released.store(true, Ordering::Release);
        let result = verify(backend.clone(), memory(1), "u".into(), vec![]).await;
        assert!(result.is_err(), "auth was admitted without memory");
        assert!(!backend.entered.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn cancelled_waiter_retains_memory_until_password_work_finishes() {
        let backend = slow();
        let release = backend.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            release.released.store(true, Ordering::Release);
        });
        let governor = memory(256 << 20);
        let job = tokio::spawn(verify(
            backend.clone(),
            governor.clone(),
            "u".into(),
            vec![],
        ));
        tokio::time::timeout(Duration::from_secs(2), async {
            while !backend.entered.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        job.abort();
        let _ = job.await;
        assert!(governor.reserved_bytes() > 0);
        releaser.join().unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while governor.reserved_bytes() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
