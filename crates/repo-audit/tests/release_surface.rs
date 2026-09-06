/// The product rule, pinned where it can be broken: one public RESP3
/// executable, and nothing that tells an operator to start another one.
///
/// A second binary is not a documentation problem - it is a deployment whose
/// authentication depends on which of two commands someone typed. This test
/// reads the manifests, the Dockerfile and the release workflows, so adding
/// one back fails here rather than in production.
#[test]
fn nothing_ships_a_second_resp3_binary() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let shim = std::fs::read_to_string(root.join("crates/skeg-server-tenant/Cargo.toml"))
        .expect("read the compatibility crate's manifest");
    assert!(
        !shim.contains("[[bin]]"),
        "skeg-server-tenant declares a binary again: the multi-tenant profile \
         belongs to skeg-resp3"
    );

    for (file, what) in [
        ("Dockerfile", "the image"),
        (
            ".github/workflows/build-artifacts.yml",
            "the artifact build",
        ),
        (".github/workflows/release.yml", "the release build"),
    ] {
        let text = std::fs::read_to_string(root.join(file)).expect("read");
        // The crate name may still appear in the publish list - the shim is
        // still published. What must not appear is a second server BINARY.
        assert!(
            !text.contains("--bin skeg-server"),
            "{what} builds a second server binary: {file}"
        );
        assert!(
            !text.contains("-p skeg-server-tenant"),
            "{what} builds the compatibility crate as a binary: {file}"
        );
    }

    let dockerfile = std::fs::read_to_string(root.join("Dockerfile")).expect("read Dockerfile");
    assert!(
        dockerfile.contains("skeg-resp3"),
        "the image must carry the one RESP3 binary"
    );
}
