//! Generate disposable smoke credentials; never use these known passwords in deployment.
use skeg_tenant::auth::hash_password;
use skeg_tenant::{AuthStore, TenantId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: artifact_auth OUTPUT")?;
    let mut store = AuthStore::open(&path)?;
    for user in ["admin", "alice", "bob"] {
        store.upsert(
            user,
            TenantId::from_name(user),
            hash_password(format!("{user}-smoke-only").as_bytes())?,
        );
    }
    store.save()?;
    for byte in std::fs::read(path)? {
        print!("{byte:02x}");
    }
    println!();
    Ok(())
}
