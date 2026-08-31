//! An index served from disk must answer exactly like one held in memory.
//!
//! The whole point of moving the id lists out of RAM is that nothing else
//! changes. A filter that returns a different set is a silent wrong answer:
//! rows quietly missing from a search, no error anywhere. So the gate is not
//! "the disk path works" but "the two are indistinguishable", checked over a
//! generated corpus and every shape the filter grammar has.

use skeg_server::payload::{PayloadIndex, parse_fields, parse_filter};
use skeg_server::payload_disk::DiskPostings;

/// Deterministic pseudo-random: reproducible failures beat lucky passes.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn upto(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const LICS: [&str; 4] = ["mit", "apache-2.0", "bsd-3-clause", "gpl-3.0"];
const TASKS: [&str; 3] = [
    "text-generation",
    "automatic-speech-recognition",
    "text-to-image",
];

fn payload(rng: &mut Rng, id: u64) -> String {
    let mut s = format!(
        "dl={} likes={} params_m={} lic={} task={}",
        rng.upto(100_000),
        rng.upto(500),
        rng.upto(70_000),
        LICS[rng.upto(4) as usize],
        TASKS[rng.upto(3) as usize],
    );
    // a third of the records carry an extra field, so EXISTS has something to
    // distinguish and some ids are absent from a posting entirely
    if id % 3 == 0 {
        s.push_str(" quant=gguf");
    }
    s
}

const FILTERS: [&str; 16] = [
    "lic = mit",
    "lic IN (mit, apache-2.0)",
    "task = text-generation",
    "params_m <= 9000",
    "params_m < 1",
    "params_m >= 60000",
    "dl > 50000",
    "dl BETWEEN 10000 AND 20000",
    "quant EXISTS",
    "NOT quant EXISTS",
    "lic = mit AND params_m <= 9000",
    "lic = mit AND NOT task = text-generation",
    "lic = nonexistent",
    "nosuchfield = x",
    "nosuchfield EXISTS",
    "likes >= 250 AND lic IN (mit, gpl-3.0) AND NOT quant EXISTS",
];

fn build(n: u64, seed: u64) -> (PayloadIndex, Vec<(u64, String)>) {
    let mut rng = Rng(seed);
    let mut idx = PayloadIndex::default();
    let mut rows = Vec::new();
    for id in 0..n {
        let p = payload(&mut rng, id);
        idx.upsert(id, parse_fields(p.as_bytes()));
        rows.push((id, p));
    }
    (idx, rows)
}

fn compare(mem: &PayloadIndex, disk: &PayloadIndex, label: &str) {
    let mut matched = 0usize;
    for f in FILTERS {
        let filter = parse_filter(f).expect("filtro valido");
        let a = filter.evaluate(mem);
        let b = filter.evaluate(disk);
        assert_eq!(
            a,
            b,
            "{label}: `{f}` differs (memory {} ids, disk {} ids)",
            a.len(),
            b.len()
        );
        matched += a.len();
    }
    // Without this, two indexes that both answer nothing would pass every
    // comparison above and the test would be green having proved nothing.
    assert!(
        matched > 1000,
        "{label}: the filters matched only {matched} ids in total, so the \
         comparison is not exercising anything"
    );
}

#[test]
fn a_disk_backed_index_answers_exactly_like_one_in_memory() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mem, _) = build(3000, 0x5EED);
    mem.persist(tmp.path(), (7, 4096)).unwrap();
    let disk = PayloadIndex::from_disk(
        DiskPostings::open(tmp.path(), (7, 4096)).expect("il file deve aprirsi"),
    );
    assert_eq!(disk.len(), mem.len(), "the two must cover the same ids");
    compare(&mem, &disk, "fresh");
}

#[test]
fn writes_after_the_file_was_built_are_visible_and_shadow_it() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mut mem, _) = build(2000, 0xC0FFEE);
    mem.persist(tmp.path(), (1, 1)).unwrap();
    let mut disk = PayloadIndex::from_disk(
        DiskPostings::open(tmp.path(), (1, 1)).expect("il file deve aprirsi"),
    );

    // The same churn applied to both: overwrites, deletes, and new ids.
    let mut rng = Rng(0xABCD);
    for _ in 0..400 {
        let id = rng.upto(2000);
        let p = payload(&mut rng, id);
        mem.upsert(id, parse_fields(p.as_bytes()));
        disk.upsert(id, parse_fields(p.as_bytes()));
    }
    for _ in 0..200 {
        let id = rng.upto(2000);
        mem.remove(id);
        disk.remove(id);
    }
    for id in 2000..2300u64 {
        let p = payload(&mut rng, id);
        mem.upsert(id, parse_fields(p.as_bytes()));
        disk.upsert(id, parse_fields(p.as_bytes()));
    }
    assert_eq!(
        disk.len(),
        mem.len(),
        "the two must still cover the same ids"
    );
    compare(&mem, &disk, "after churn");
}

#[test]
fn persisting_again_folds_the_previous_file_in_rather_than_chaining() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mut mem, _) = build(1500, 0x1234);
    mem.persist(tmp.path(), (1, 1)).unwrap();
    let mut disk = PayloadIndex::from_disk(
        DiskPostings::open(tmp.path(), (1, 1)).expect("il file deve aprirsi"),
    );

    let mut rng = Rng(0x9999);
    for _ in 0..300 {
        let id = rng.upto(1500);
        let p = payload(&mut rng, id);
        mem.upsert(id, parse_fields(p.as_bytes()));
        disk.upsert(id, parse_fields(p.as_bytes()));
    }
    // Second generation, written from an index that already had a disk part.
    disk.persist(tmp.path(), (2, 2)).unwrap();
    let reloaded = PayloadIndex::from_disk(
        DiskPostings::open(tmp.path(), (2, 2)).expect("il file deve aprirsi"),
    );
    assert_eq!(reloaded.len(), mem.len());
    compare(&mem, &reloaded, "second generation");
}

#[test]
fn a_file_written_for_another_log_position_is_refused() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mem, _) = build(50, 1);
    mem.persist(tmp.path(), (5, 100)).unwrap();
    assert!(DiskPostings::open(tmp.path(), (5, 200)).is_none());
    assert!(DiskPostings::open(tmp.path(), (6, 100)).is_none());
    assert!(DiskPostings::open(tmp.path(), (5, 100)).is_some());
}

#[test]
fn truncation_at_every_length_is_refused_and_never_panics() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mem, _) = build(60, 2);
    mem.persist(tmp.path(), (3, 3)).unwrap();
    let path = tmp.path().join(skeg_server::payload_disk::FILE);
    let full = std::fs::read(&path).unwrap();
    for cut in 0..full.len() {
        std::fs::write(&path, &full[..cut]).unwrap();
        assert!(
            DiskPostings::open(tmp.path(), (3, 3)).is_none(),
            "a file cut at {cut} of {} was accepted",
            full.len()
        );
    }
}

#[test]
fn a_flipped_bit_anywhere_is_refused() {
    let tmp = tempfile::TempDir::new().unwrap();
    let (mem, _) = build(40, 4);
    mem.persist(tmp.path(), (2, 2)).unwrap();
    let path = tmp.path().join(skeg_server::payload_disk::FILE);
    let full = std::fs::read(&path).unwrap();
    for byte in (0..full.len()).step_by(7) {
        let mut buf = full.clone();
        buf[byte] ^= 0x40;
        std::fs::write(&path, &buf).unwrap();
        assert!(
            DiskPostings::open(tmp.path(), (2, 2)).is_none(),
            "a bit flipped at byte {byte} was accepted"
        );
    }
}
