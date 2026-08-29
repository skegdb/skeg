//! Quanto pesa davvero l'indice dei payload, isolato da tutto il resto.
use skeg_server::payload::{parse_fields, PayloadIndex};

fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
}

fn main() {
    let dir = std::env::args().nth(1).expect("dir con payload.cache.bin");
    let mut files = vec![];
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path().join("payload.cache.bin");
        if p.exists() { files.push(p); }
    }
    // legge i blob grezzi: stesso contenuto che il warm mette nell'indice
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
    println!("  {} blob, {:.1} MB di testo grezzo", blobs.len(), raw as f64 / 1e6);

    let base = rss_kb();
    let mut idx = PayloadIndex::default();
    for (id, b) in &blobs {
        idx.upsert(*id, parse_fields(b));
    }
    let after = rss_kb();
    println!("  indice costruito: RSS +{:.0} MB", (after - base) as f64 / 1024.0);
    println!("  per vettore: {:.0} byte", (after - base) as f64 * 1024.0 / blobs.len() as f64);
    std::hint::black_box(&idx);
}
