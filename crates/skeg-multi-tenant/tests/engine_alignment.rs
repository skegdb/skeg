//! The multi-tenant layer must embed the workspace engine, not an older
//! `skeg-vector` pulled transitively from crates.io through
//! `skeg-rigging-skeg`. Two checks, both mechanical:
//!
//! 1. The resolved dependency graph contains exactly one `skeg-vector`,
//!    and it is the workspace path crate.
//! 2. A tenant written through `MultiTenantRoot` reopens with the
//!    workspace `skeg_vector::DiskVamanaIndex` and answers a search.

use std::process::Command;

use skeg_multi_tenant::{MultiTenantRoot, SkegTenantId};
use skeg_rigging::prelude::*;

const DIM: u32 = 4;

fn cargo_tree(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO"))
        .args(["tree", "-p", "skeg-multi-tenant", "-e", "normal"])
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn assert_single_workspace_engine(tree: &str) {
    let lines: Vec<&str> = tree
        .lines()
        .filter(|l| l.contains("skeg-vector v"))
        .collect();
    assert!(
        !lines.is_empty(),
        "skeg-vector absent from the graph:\n{tree}"
    );
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let ws_vector = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .join("skeg-vector");
    for l in &lines {
        assert!(
            l.contains(&ws_vector.display().to_string()),
            "skeg-vector resolved from a registry, not the workspace: {l}\n{tree}"
        );
    }
}

#[test]
fn dependency_graph_has_one_engine() {
    assert_single_workspace_engine(&cargo_tree(&[]));
    // `-d` prints each duplicated package at column 0, followed by the
    // indented path that pulls it in; skeg-vector legitimately shows up
    // indented under other crates' duplicates (getrandom, rand_core).
    let dup = cargo_tree(&["-d"]);
    assert!(
        !dup.lines().any(|l| l.starts_with("skeg-vector v")),
        "duplicate skeg-vector in the graph:\n{dup}"
    );
}

#[test]
fn dependency_graph_has_one_engine_with_live_attach() {
    assert_single_workspace_engine(&cargo_tree(&["--features", "live-attach"]));
}

#[test]
fn tenant_written_here_reopens_with_the_workspace_engine() {
    let dir = tempfile::tempdir().unwrap();
    let root = MultiTenantRoot::new(dir.path());
    let id = SkegTenantId::from_bytes([0x42; 16]);
    let tenant_dir;
    {
        let tenant = root.open(id, DIM).unwrap();
        for i in 0..DIM as usize {
            let mut v = vec![0.0f32; DIM as usize];
            v[i] = 1.0;
            tenant
                .insert(RecordId(i as u64 + 1), v, true, vec![], vec![])
                .unwrap();
        }
        tenant.flush().unwrap();
        tenant_dir = root.tenant_dir(id);
    }
    let idx = skeg_vector::DiskVamanaIndex::open(&tenant_dir).unwrap();
    assert_eq!(idx.dim(), DIM as usize);
    assert_eq!(idx.len(), DIM as usize);
    let hits = idx.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
    assert_eq!(hits[0].0, 3, "nearest to e3 must be record 3: {hits:?}");
}
