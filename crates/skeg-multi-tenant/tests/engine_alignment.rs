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
    let mut command = Command::new(env!("CARGO"));
    command
        .args(["tree", "-p", "skeg-multi-tenant", "-e", "normal"])
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    if let Some(config) = std::env::var_os("SKEG_RELEASE_CARGO_CONFIG") {
        command.arg("--config").arg(config);
    }
    let out = command.output().expect("cargo tree");
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
    let expected_source = std::env::var_os("SKEG_EXPECTED_VECTOR_SOURCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("skeg-vector")
        });
    for l in &lines {
        assert!(
            l.contains(&expected_source.display().to_string()),
            "skeg-vector did not resolve from {}: {l}\n{tree}",
            expected_source.display()
        );
    }
}

fn assert_single_version(tree: &str, package: &str, version: &str) {
    let prefix = format!("{package} v");
    let versions: std::collections::BTreeSet<&str> = tree
        .lines()
        .filter_map(|line| line.find(&prefix).map(|at| &line[at + prefix.len()..]))
        .filter_map(|tail| tail.split_whitespace().next())
        .collect();
    assert_eq!(
        versions,
        std::collections::BTreeSet::from([version]),
        "expected only {package} v{version} in the release graph:\n{tree}"
    );
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
    let tree = cargo_tree(&["--features", "live-attach"]);
    assert_single_workspace_engine(&tree);
    assert_single_version(&tree, "skeg-rigging", "0.1.5");
    assert_single_version(&tree, "skeg-rigging-skeg", "0.1.5");
    assert_single_version(&tree, "skeg-rigging-net", "0.1.2");
    assert_single_version(&tree, "skeg-rigging-net-resp3", "0.1.2");
    assert_single_version(&tree, "skeg-resp3", "0.3.0");
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
