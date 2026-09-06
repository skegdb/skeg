//! F.13 - `TenantEvents` on the DiskVamana-backed adapter.

use std::time::Duration;

use skeg_rigging::{Event, EventFilter, RecordId, TenantEvents, TenantId};
use skeg_rigging_skeg::Tenant;

const DIM: u32 = 4;

fn unit(at: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[at] = 1.0;
    v
}

#[test]
fn subscribe_receives_record_inserted_on_adapter() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let stream = t.subscribe(EventFilter::ALL);
    t.insert(RecordId(1), unit(0), true, vec![], b"x".to_vec())
        .unwrap();
    let ev = stream
        .recv_timeout(Duration::from_millis(200))
        .expect("insert event should arrive");
    match ev {
        Event::RecordInserted {
            record_id,
            shareable,
        } => {
            assert_eq!(record_id, RecordId(1));
            assert!(shareable);
        }
        other => panic!("expected RecordInserted, got {other:?}"),
    }
}

#[test]
fn subscribe_receives_record_deleted_on_adapter() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    t.insert(RecordId(7), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    let stream = t.subscribe(EventFilter::ALL);
    assert!(t.delete(RecordId(7)).unwrap());
    let ev = stream
        .recv_timeout(Duration::from_millis(200))
        .expect("delete event should arrive");
    assert!(matches!(ev, Event::RecordDeleted { record_id } if record_id == RecordId(7)));
}

#[test]
fn filter_drops_unwanted_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let stream = t.subscribe(EventFilter {
        record_inserted: false,
        record_deleted: true,
        ..EventFilter::NONE
    });
    t.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
        .unwrap();
    // Insert is filtered out - should time out.
    assert!(stream.recv_timeout(Duration::from_millis(30)).is_err());
    assert!(t.delete(RecordId(1)).unwrap());
    let ev = stream
        .recv_timeout(Duration::from_millis(200))
        .expect("delete passes filter");
    assert!(matches!(ev, Event::RecordDeleted { .. }));
}

#[test]
fn multiple_subscribers_each_get_events() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let s1 = t.subscribe(EventFilter::ALL);
    let s2 = t.subscribe(EventFilter::ALL);
    t.insert(RecordId(42), unit(1), true, vec![], b"x".to_vec())
        .unwrap();
    let e1 = s1.recv_timeout(Duration::from_millis(200)).unwrap();
    let e2 = s2.recv_timeout(Duration::from_millis(200)).unwrap();
    assert!(matches!(e1, Event::RecordInserted { record_id, .. } if record_id == RecordId(42)));
    assert!(matches!(e2, Event::RecordInserted { record_id, .. } if record_id == RecordId(42)));
}

#[test]
fn dropping_subscription_prunes_sender() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    {
        let _s = t.subscribe(EventFilter::ALL);
        t.insert(RecordId(1), unit(0), false, vec![], b"x".to_vec())
            .unwrap();
    }
    // After the stream is dropped, the sender is invalidated; the
    // next emit prunes it. Just need the second insert to not panic.
    t.insert(RecordId(2), unit(1), false, vec![], b"y".to_vec())
        .unwrap();
}

#[test]
fn delete_of_missing_record_emits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let t = Tenant::create_new(dir.path(), TenantId::ZERO, DIM).unwrap();
    let stream = t.subscribe(EventFilter::ALL);
    assert!(!t.delete(RecordId(99)).unwrap());
    assert!(stream.recv_timeout(Duration::from_millis(30)).is_err());
}
