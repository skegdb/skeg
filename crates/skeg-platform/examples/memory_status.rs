//! What this process can see of its own memory limit.
//!
//! The cgroup reading has unit tests over tempdirs on every platform, and had
//! never once run inside a real cgroup. This prints what it finds, so the
//! answer comes from execution rather than from the tests agreeing with
//! themselves.
//!
//!   docker run --rm --memory 256m -v "$PWD:/src" -w /src rust:1.88 \
//!       cargo run -q -p skeg-platform --example memory_status

fn main() {
    let m = skeg_platform::memory_status();
    println!("limit_bytes   = {:?}", m.limit_bytes);
    println!("current_bytes = {:?}", m.current_bytes);
    println!("available     = {:?}", m.available);
    if let Some(l) = m.limit_bytes {
        println!("limite letto  = {:.1} MiB", l as f64 / (1024.0 * 1024.0));
    }
    match m.available {
        skeg_platform::Headroom::Known(h) => {
            println!("headroom      = {:.1} MiB", h as f64 / (1024.0 * 1024.0));
        }
        other => println!("headroom      = {other:?}"),
    }
    // What the kernel actually says, for comparison: if these two disagree the
    // parser is wrong, and no unit test over a tempdir would have shown it.
    for p in [
        "/proc/self/cgroup",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
    ] {
        match std::fs::read_to_string(p) {
            Ok(s) => println!("{p} -> {}", s.trim()),
            Err(e) => println!("{p} -> ({e})"),
        }
    }
}
