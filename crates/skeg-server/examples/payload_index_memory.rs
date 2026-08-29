//! What the payload index actually weighs, isolated from everything else.
//!
//! Two modes, and they must run as separate processes: RSS does not fall when
//! memory is freed, so building the second index after dropping the first
//! reuses pages already counted and reads as costing nothing. Run it once to
//! build the file, then again with DISK_ONLY=1.
//!
//! Point it at a directory holding ONE vindex. Two vindexes share an id space,
//! so loading both into one index silently overwrites and the per-vector
//! figure comes out against the wrong denominator.
use skeg_server::payload::{parse_fields, PayloadIndex};

fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}

fn main() {
    let dir = std::env::args().nth(1).expect("directory holding one vindex");
    let mut files = vec![];
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path().join("payload.cache.bin");
        if p.exists() { files.push(p); }
    }
    // the raw blobs: the same content the warm puts into the index
    let mut blobs: Vec<(u64, Vec<u8>)> = vec![];
    for f in &files {
        let buf = std::fs::read(f).unwrap();
        let count = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
        let mut o = 36usize;   // HEADER: magic+version+hwm+offset+count+crc
        for _ in 0..count {
            let id = u64::from_le_bytes(buf[o..o+8].try_into().unwrap());
            let len = u32::from_le_bytes(buf[o+8..o+12].try_into().unwrap()) as usize;
            o += 12;
            blobs.push((id, buf[o..o+len].to_vec()));
            o += len;
        }
    }
    let raw: usize = blobs.iter().map(|(_, b)| b.len()).sum();
    println!("  {} blobs, {:.1} MB of raw text", blobs.len(), raw as f64 / 1e6);

    // See the module note: the two modes must not share a process.
    if std::env::var("DISK_ONLY").is_ok() {
        let tmp = std::env::temp_dir().join("pidx_probe");
        let file = std::fs::metadata(tmp.join(skeg_server::payload_disk::FILE))
            .expect("run the in-memory mode first")
            .len();
        let base = rss_kb();
        let disk = PayloadIndex::from_disk(
            skeg_server::payload_disk::DiskPostings::open(&tmp, (1, 1)).expect("open"),
        );
        let after = rss_kb();
        let dm = (after - base) as f64;
        println!("  from disk: +{:6.1} MB   {:5.1} B/vector   (file {:.1} MB)",
                 dm / 1024.0, dm * 1024.0 / disk.len() as f64, file as f64 / 1e6);
        println!("  ids covered: {}", disk.len());
        std::hint::black_box(&disk);
        return;
    }

    let base = rss_kb();
    let mut idx = PayloadIndex::default();
    for (id, b) in &blobs {
        idx.upsert(*id, parse_fields(b));
    }
    let after = rss_kb();
    let mem = (after - base) as f64;
    println!("  in memory: +{:6.0} MB   {:5.0} B/vector   ({} ids covered)", mem / 1024.0,
             mem * 1024.0 / idx.len() as f64, idx.len());

    // write the file the second mode reads
    let tmp = std::env::temp_dir().join("pidx_probe");
    let _ = std::fs::create_dir_all(&tmp);
    idx.persist(&tmp, (1, 1)).unwrap();
    println!("  file written: {:.1} MB   (rerun with DISK_ONLY=1)",
             std::fs::metadata(tmp.join(skeg_server::payload_disk::FILE)).unwrap().len() as f64 / 1e6);
    std::hint::black_box(&idx);
}
