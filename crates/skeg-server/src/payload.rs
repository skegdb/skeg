//! Typed payload fields, the per-vindex payload index, and the filter AST that
//! turns a `SKEG.VSEARCH ... FILTER` clause into the set of matching vector ids.
//!
//! Pure and synchronous: the shard owns one [`PayloadIndex`] per vindex, feeds it
//! the fields parsed from a VSET payload, and queries it on a filtered VSEARCH.
//! The blob itself is still stored verbatim (see `shard::payload_key`) for
//! WITHPAYLOAD return; this module only derives the searchable index from it.
//!
//! Grammar: keyword and i64 fields; predicates `=`, `IN (...)`, the ranges
//! `>=`, `>`, `<=`, `<`, `BETWEEN a AND b`, and `field EXISTS`; combined with
//! `AND`, `OR`, `NOT` and parentheses. Wider types (f64) and roaring-bitmap
//! postings are deferred.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;

/// A typed payload value. A token that parses as an `i64` is an `Int`, otherwise
/// a `Keyword`. `Ord` (derived) gives ranges their order and keys a `BTreeMap`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Value {
    Keyword(String),
    Int(i64),
}

impl Value {
    fn parse(tok: &str) -> Value {
        match tok.parse::<i64>() {
            Ok(n) => Value::Int(n),
            Err(_) => Value::Keyword(tok.to_owned()),
        }
    }
}

/// Parse a payload blob into `(field, value)` pairs. Format: `key=value` tokens
/// separated by whitespace. A token without `=`, an empty key, or non-UTF-8
/// content is skipped rather than rejected: the same blob is returned verbatim to
/// clients, so the index must tolerate free-form content it cannot field-ize.
#[must_use]
pub fn parse_fields(blob: &[u8]) -> Vec<(String, Value)> {
    let Ok(text) = std::str::from_utf8(blob) else {
        return Vec::new();
    };
    text.split_whitespace()
        .filter_map(|tok| tok.split_once('='))
        .filter(|(k, _)| !k.is_empty())
        .map(|(k, v)| (k.to_owned(), Value::parse(v)))
        .collect()
}

/// Per-vindex payload index. For each field, `value -> set of vector ids`; plus
/// `by_id` so an overwrite or delete drops an id's previous values without a
/// scan. Postings are `BTreeSet<u64>` (stdlib); roaring bitmaps are the upgrade
/// if a scale bench asks for it.
#[derive(Default)]
pub struct PayloadIndex {
    /// The id lists built before this process started, read back from
    /// `payload.idx`. Immutable: nothing is ever inserted into or removed from
    /// it. Changes since it was built live in the maps below, and `shadowed`
    /// says which of its ids to disregard.
    ///
    /// This is what keeps the index off the heap. Held as `BTreeSet`s the
    /// postings cost 178 bytes per vector on a real corpus; the same ids
    /// delta-encoded on disk cost 12,2.
    disk: Option<crate::payload_disk::DiskPostings>,
    /// Ids written or deleted since `disk` was built. A shadowed id is ignored
    /// wherever `disk` mentions it, so an overwrite never has to rewrite the
    /// file, and a delete never has to know which postings to withdraw.
    shadowed: std::collections::HashSet<u64>,
    by_field: BTreeMap<String, BTreeMap<Value, BTreeSet<u64>>>,
    /// What `id` currently contributes to `by_field`, kept as the canonical
    /// `key=value` text rather than as parsed pairs.
    ///
    /// This map answers no query. It exists only so an overwrite knows which
    /// postings to withdraw. Holding `Vec<(String, Value)>` for that cost 488
    /// bytes per vector on a real corpus of 103-byte payloads: ten `String`
    /// allocations per record for ten field names shared by every record, plus
    /// a `Value` each. The text is the same information, and re-parsing it on
    /// the rare path (an overwrite or a delete) is cheaper than carrying the
    /// parse for every record for the life of the process.
    by_id: BTreeMap<u64, Box<[u8]>>,
}

impl PayloadIndex {
    /// Index `id`'s fields, replacing anything previously indexed for that id.
    /// Indexing the same id again is how an overwrite VSET stays consistent.
    pub fn upsert(&mut self, id: u64, fields: Vec<(String, Value)>) {
        self.remove(id);
        if self.disk.is_some() {
            self.shadowed.insert(id);
        }
        if fields.is_empty() {
            return;
        }
        let mut canonical = String::new();
        for (f, v) in &fields {
            self.by_field
                .entry(f.clone())
                .or_default()
                .entry(v.clone())
                .or_default()
                .insert(id);
            if !canonical.is_empty() {
                canonical.push(' ');
            }
            canonical.push_str(f);
            canonical.push('=');
            match v {
                Value::Keyword(s) => canonical.push_str(s),
                Value::Int(n) => {
                    use std::fmt::Write;
                    let _ = write!(canonical, "{n}");
                }
            }
        }
        self.by_id
            .insert(id, canonical.into_bytes().into_boxed_slice());
    }

    /// Drop all of `id`'s postings. No-op if `id` was never indexed.
    pub fn remove(&mut self, id: u64) {
        if self.disk.is_some() {
            self.shadowed.insert(id);
        }
        let Some(blob) = self.by_id.remove(&id) else {
            return;
        };
        // Re-parsed rather than stored parsed: see the note on `by_id`. The
        // text is canonical, produced by `upsert` from the same pairs, so this
        // yields exactly the fields that were indexed.
        for (f, v) in parse_fields(&blob) {
            if let Some(values) = self.by_field.get_mut(&f) {
                if let Some(ids) = values.get_mut(&v) {
                    ids.remove(&id);
                    if ids.is_empty() {
                        values.remove(&v);
                    }
                }
                if values.is_empty() {
                    self.by_field.remove(&f);
                }
            }
        }
    }

    /// `id`'s payload in the blob form `parse_fields` accepts.
    ///
    /// Lets the index be persisted without reading a single blob back from the
    /// log: everything the index knows about an id is already here, in the form
    /// the parser accepts. Tokens the parser skipped when the blob first
    /// arrived are not here and are not missed, since they were never indexed,
    /// so an index built from this is the same index.
    ///
    /// An id with nothing indexed yields an empty blob, which is the honest
    /// answer and lets a caller record that the id is covered rather than
    /// unknown.
    #[must_use]
    pub fn field_blob(&self, id: u64) -> Vec<u8> {
        self.by_id.get(&id).map(|b| b.to_vec()).unwrap_or_default()
    }

    /// Build an index whose id lists come from `disk`.
    ///
    /// Nothing is copied out of the file: the maps here start empty and take
    /// only what changes afterwards.
    #[must_use]
    pub fn from_disk(disk: crate::payload_disk::DiskPostings) -> Self {
        Self {
            disk: Some(disk),
            ..Self::default()
        }
    }

    /// Number of ids this index currently answers for.
    #[must_use]
    pub fn len(&self) -> usize {
        self.all_ids().len()
    }

    /// Whether this index answers for no ids.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the current state to `payload.idx` in `dir`, stamped with the log
    /// position it reflects.
    ///
    /// The lists come from the same merge the reads use, so a file written from
    /// an index that already had a disk part folds that part in rather than
    /// layering another one: reopening never has to walk a chain.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written, synced or renamed.
    pub fn persist(
        &self,
        dir: &std::path::Path,
        stamp: (u64, u64),
        generation: u128,
    ) -> std::io::Result<()> {
        let mut keys: BTreeMap<&str, BTreeMap<Value, ()>> = BTreeMap::new();
        if let Some(d) = &self.disk {
            for (f, vs) in d.directory() {
                let e = keys.entry(f.as_str()).or_default();
                for v in vs.keys() {
                    e.insert(v.clone(), ());
                }
            }
        }
        for (f, vs) in &self.by_field {
            let e = keys.entry(f.as_str()).or_default();
            for v in vs.keys() {
                e.insert(v.clone(), ());
            }
        }
        let mut lists: Vec<(&str, Value, Vec<u64>)> = Vec::new();
        for (f, vs) in &keys {
            for v in vs.keys() {
                let ids = self.postings(f, v);
                if !ids.is_empty() {
                    lists.push((f, v.clone(), ids));
                }
            }
        }
        let all = self.all_ids();
        crate::payload_disk::write(
            dir,
            stamp,
            generation,
            &all,
            lists.iter().map(|(f, v, ids)| (*f, v, ids.as_slice())),
        )
    }

    /// Merge a disk id list with the in-memory one for the same key.
    ///
    /// Order matters only in that the result must come out sorted; the two
    /// sources are each sorted already. A shadowed id is one the memory side
    /// has an opinion about, so the disk side does not get a vote on it.
    fn merge(&self, from_disk: Vec<u64>, mem: Option<&BTreeSet<u64>>) -> Vec<u64> {
        let mut out: Vec<u64> = if self.shadowed.is_empty() {
            from_disk
        } else {
            from_disk
                .into_iter()
                .filter(|id| !self.shadowed.contains(id))
                .collect()
        };
        if let Some(m) = mem {
            out.extend(m.iter().copied());
        }
        sort_dedup(out)
    }

    /// Ids for one `(field, value)`, sorted.
    fn postings(&self, field: &str, value: &Value) -> Vec<u64> {
        let mem = self.by_field.get(field).and_then(|vs| vs.get(value));
        match &self.disk {
            Some(d) => self.merge(d.postings(field, value), mem),
            None => mem.map(|s| s.iter().copied().collect()).unwrap_or_default(),
        }
    }

    /// Every id that has any value for `field` (the `EXISTS` predicate), sorted.
    fn field_ids(&self, field: &str) -> Vec<u64> {
        let mut mem = Vec::new();
        if let Some(vs) = self.by_field.get(field) {
            for ids in vs.values() {
                mem.extend(ids.iter().copied());
            }
        }
        match &self.disk {
            Some(d) => {
                let mut out: Vec<u64> = d
                    .field_ids(field)
                    .into_iter()
                    .filter(|id| !self.shadowed.contains(id))
                    .collect();
                out.extend(mem);
                sort_dedup(out)
            }
            None => sort_dedup(mem),
        }
    }

    /// Ids whose `field` value lies in `[lo, hi]` (per the bounds), sorted.
    fn range_ids(&self, field: &str, lo: &Bound<Value>, hi: &Bound<Value>) -> Vec<u64> {
        let mut mem = Vec::new();
        if let Some(vs) = self.by_field.get(field) {
            for (_, ids) in vs.range((lo.as_ref(), hi.as_ref())) {
                mem.extend(ids.iter().copied());
            }
        }
        match &self.disk {
            Some(d) => {
                let mut out: Vec<u64> = d
                    .range_ids(field, lo, hi)
                    .into_iter()
                    .filter(|id| !self.shadowed.contains(id))
                    .collect();
                out.extend(mem);
                sort_dedup(out)
            }
            None => sort_dedup(mem),
        }
    }

    /// Every indexed id (the universe `NOT` complements against), sorted.
    fn all_ids(&self) -> Vec<u64> {
        let mem = self.by_id.keys().copied();
        match &self.disk {
            Some(d) => {
                let mut out: Vec<u64> = d
                    .all_ids()
                    .filter(|id| !self.shadowed.contains(id))
                    .collect();
                out.extend(mem);
                sort_dedup(out)
            }
            None => mem.collect(),
        }
    }
}

/// Sort + dedup an id list into the canonical sorted form the planner expects.
fn sort_dedup(mut v: Vec<u64>) -> Vec<u64> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Intersection of two SORTED id lists (two-pointer, cache-friendly).
fn intersect_sorted(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// A filter over payload fields. Built by [`parse_filter`], evaluated against a
/// [`PayloadIndex`] into the set of matching ids.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    Eq(String, Value),
    In(String, Vec<Value>),
    /// `field` value within `[lo, hi]` (bounds carry inclusivity). Covers `>=`,
    /// `>`, `<=`, `<`, and `BETWEEN`.
    Range {
        field: String,
        lo: Bound<Value>,
        hi: Bound<Value>,
    },
    /// `field EXISTS`: the id has any value for `field`.
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    /// The set of vector ids matching this filter against `idx`. An unknown
    /// field or value contributes the empty set; `NOT` complements against the
    /// indexed universe.
    #[must_use]
    pub fn evaluate(&self, idx: &PayloadIndex) -> Vec<u64> {
        match self {
            // A posting is already sorted (BTreeSet iteration order).
            Filter::Eq(f, v) => idx.postings(f, v),
            Filter::In(f, vs) => {
                let mut out = Vec::new();
                for v in vs {
                    out.extend(idx.postings(f, v));
                }
                sort_dedup(out)
            }
            Filter::Range { field, lo, hi } => idx.range_ids(field, lo, hi),
            Filter::Exists(field) => idx.field_ids(field),
            Filter::And(parts) => {
                // Split direct `NOT x` children from the rest: intersect the
                // positives, then SUBTRACT each negated set (`acc \ B`). This
                // turns `A AND NOT B` into a difference, never materialising a
                // full universe-complement just to intersect it away.
                let mut positives: Vec<&Filter> = Vec::new();
                let mut negated: Vec<&Filter> = Vec::new();
                for p in parts {
                    match p {
                        Filter::Not(inner) => negated.push(inner),
                        other => positives.push(other),
                    }
                }
                let mut acc: Vec<u64> = if positives.is_empty() {
                    // Only negations: start from the universe, then subtract.
                    idx.all_ids()
                } else {
                    // Intersect smallest-first so the running set only shrinks.
                    let mut sets: Vec<Vec<u64>> =
                        positives.iter().map(|p| p.evaluate(idx)).collect();
                    sets.sort_by_key(Vec::len);
                    let mut iter = sets.into_iter();
                    let mut a = iter.next().unwrap();
                    for s in iter {
                        a = intersect_sorted(&a, &s);
                        if a.is_empty() {
                            break;
                        }
                    }
                    a
                };
                for neg in negated {
                    if acc.is_empty() {
                        break;
                    }
                    let b = neg.evaluate(idx); // sorted
                    acc.retain(|id| b.binary_search(id).is_err());
                }
                acc
            }
            Filter::Or(parts) => sort_dedup(parts.iter().flat_map(|p| p.evaluate(idx)).collect()),
            Filter::Not(inner) => {
                // A standalone `NOT` (top level, or inside `OR`) genuinely
                // returns the universe minus the excluded set. `A AND NOT B`
                // does NOT take this path: `And` subtracts instead (see above).
                let excluded = inner.evaluate(idx); // sorted
                idx.all_ids()
                    .into_iter()
                    .filter(|id| excluded.binary_search(id).is_err())
                    .collect()
            }
        }
    }
}

// ── parser ────────────────────────────────────────────────────────────────────

fn is_keyword(t: &str, kw: &str) -> bool {
    t.eq_ignore_ascii_case(kw)
}

fn is_reserved(t: &str) -> bool {
    matches!(t, "=" | ">" | "<" | ">=" | "<=" | "(" | ")" | ",")
        || ["AND", "OR", "NOT", "IN", "BETWEEN", "EXISTS"]
            .iter()
            .any(|k| is_keyword(t, k))
}

/// Split a filter string into tokens. `( ) , = > < >= <=` are standalone tokens
/// so punctuation needs no surrounding spaces.
fn tokenize(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let flush = |cur: &mut String, toks: &mut Vec<String>| {
        if !cur.is_empty() {
            toks.push(std::mem::take(cur));
        }
    };
    while let Some(c) = chars.next() {
        match c {
            '(' | ')' | ',' | '=' => {
                flush(&mut cur, &mut toks);
                toks.push(c.to_string());
            }
            '>' | '<' => {
                flush(&mut cur, &mut toks);
                if chars.peek() == Some(&'=') {
                    chars.next();
                    toks.push(format!("{c}="));
                } else {
                    toks.push(c.to_string());
                }
            }
            c if c.is_whitespace() => flush(&mut cur, &mut toks),
            c => cur.push(c),
        }
    }
    flush(&mut cur, &mut toks);
    toks
}

/// Parse a `FILTER` clause into a [`Filter`]. Grammar (lowest to highest
/// precedence): `OR` of `AND` of `NOT`-able atoms; an atom is a parenthesised
/// expression or a predicate (`field <op> value`, `field IN (...)`,
/// `field BETWEEN a AND b`).
///
/// # Errors
///
/// Returns a human-readable message on malformed input.
pub fn parse_filter(s: &str) -> Result<Filter, String> {
    let toks = tokenize(s);
    if toks.is_empty() {
        return Err("empty filter".to_owned());
    }
    let mut i = 0;
    let f = parse_or(&toks, &mut i)?;
    if i != toks.len() {
        return Err(format!("unexpected trailing token '{}'", toks[i]));
    }
    Ok(f)
}

fn parse_or(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    let mut parts = vec![parse_and(toks, i)?];
    while toks.get(*i).is_some_and(|t| is_keyword(t, "OR")) {
        *i += 1;
        parts.push(parse_and(toks, i)?);
    }
    Ok(if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Filter::Or(parts)
    })
}

fn parse_and(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    let mut parts = vec![parse_not(toks, i)?];
    while toks.get(*i).is_some_and(|t| is_keyword(t, "AND")) {
        *i += 1;
        parts.push(parse_not(toks, i)?);
    }
    Ok(if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Filter::And(parts)
    })
}

fn parse_not(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    if toks.get(*i).is_some_and(|t| is_keyword(t, "NOT")) {
        *i += 1;
        return Ok(Filter::Not(Box::new(parse_not(toks, i)?)));
    }
    parse_atom(toks, i)
}

fn parse_atom(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    if toks.get(*i).map(String::as_str) == Some("(") {
        *i += 1;
        let f = parse_or(toks, i)?;
        if toks.get(*i).map(String::as_str) != Some(")") {
            return Err("expected ')'".to_owned());
        }
        *i += 1;
        return Ok(f);
    }
    parse_predicate(toks, i)
}

fn take_value(toks: &[String], i: &mut usize, what: &str) -> Result<Value, String> {
    let tok = toks.get(*i).ok_or_else(|| format!("expected {what}"))?;
    if is_reserved(tok) {
        return Err(format!("expected {what}, got '{tok}'"));
    }
    *i += 1;
    Ok(Value::parse(tok))
}

fn parse_predicate(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    let field = toks.get(*i).ok_or("expected a field name")?.clone();
    if is_reserved(&field) {
        return Err(format!("expected a field name, got '{field}'"));
    }
    *i += 1;
    let op = toks
        .get(*i)
        .ok_or("expected an operator after the field")?
        .clone();
    *i += 1;
    match op.as_str() {
        "=" => Ok(Filter::Eq(field, take_value(toks, i, "a value")?)),
        ">=" => Ok(range(
            field,
            Bound::Included(take_value(toks, i, "a value")?),
            Bound::Unbounded,
        )),
        ">" => Ok(range(
            field,
            Bound::Excluded(take_value(toks, i, "a value")?),
            Bound::Unbounded,
        )),
        "<=" => Ok(range(
            field,
            Bound::Unbounded,
            Bound::Included(take_value(toks, i, "a value")?),
        )),
        "<" => Ok(range(
            field,
            Bound::Unbounded,
            Bound::Excluded(take_value(toks, i, "a value")?),
        )),
        _ if is_keyword(&op, "EXISTS") => Ok(Filter::Exists(field)),
        _ if is_keyword(&op, "IN") => parse_in(field, toks, i),
        _ if is_keyword(&op, "BETWEEN") => {
            let lo = take_value(toks, i, "the BETWEEN lower bound")?;
            if !toks.get(*i).is_some_and(|t| is_keyword(t, "AND")) {
                return Err("expected AND in BETWEEN".to_owned());
            }
            *i += 1;
            let hi = take_value(toks, i, "the BETWEEN upper bound")?;
            Ok(range(field, Bound::Included(lo), Bound::Included(hi)))
        }
        _ => Err(format!("expected an operator, got '{op}'")),
    }
}

fn range(field: String, lo: Bound<Value>, hi: Bound<Value>) -> Filter {
    Filter::Range { field, lo, hi }
}

fn parse_in(field: String, toks: &[String], i: &mut usize) -> Result<Filter, String> {
    if toks.get(*i).map(String::as_str) != Some("(") {
        return Err("expected '(' after IN".to_owned());
    }
    *i += 1;
    let mut vals = Vec::new();
    loop {
        vals.push(take_value(toks, i, "a value in the IN list")?);
        match toks.get(*i).map(String::as_str) {
            Some(",") => *i += 1,
            Some(")") => {
                *i += 1;
                break;
            }
            _ => return Err("expected ',' or ')' in the IN list".to_owned()),
        }
    }
    Ok(Filter::In(field, vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw(s: &str) -> Value {
        Value::Keyword(s.to_owned())
    }

    #[test]
    fn parse_fields_types_and_skips() {
        let f = parse_fields(b"user=alice age=42 loose tag=");
        assert_eq!(
            f,
            vec![
                ("user".to_owned(), kw("alice")),
                ("age".to_owned(), Value::Int(42)),
                ("tag".to_owned(), kw("")),
            ]
        );
        assert!(parse_fields(&[0xff, 0xfe]).is_empty());
    }

    #[test]
    fn index_upsert_remove_overwrite() {
        let mut idx = PayloadIndex::default();
        idx.upsert(1, vec![("user".into(), kw("alice"))]);
        idx.upsert(2, vec![("user".into(), kw("alice"))]);
        idx.upsert(3, vec![("user".into(), kw("bob"))]);
        assert_eq!(idx.postings("user", &kw("alice")), vec![1, 2]);
        idx.upsert(1, vec![("user".into(), kw("carol"))]);
        assert_eq!(idx.postings("user", &kw("alice")), vec![2]);
        idx.remove(3);
        assert!(idx.postings("user", &kw("bob")).is_empty());
    }

    #[test]
    fn parse_grammar() {
        assert_eq!(
            parse_filter("user = alice").unwrap(),
            Filter::Eq("user".into(), kw("alice"))
        );
        assert_eq!(
            parse_filter("doc IN (a,b,c)").unwrap(),
            Filter::In("doc".into(), vec![kw("a"), kw("b"), kw("c")])
        );
        assert_eq!(
            parse_filter("age >= 18").unwrap(),
            Filter::Range {
                field: "age".into(),
                lo: Bound::Included(Value::Int(18)),
                hi: Bound::Unbounded
            }
        );
        assert_eq!(
            parse_filter("ts BETWEEN 100 AND 200").unwrap(),
            Filter::Range {
                field: "ts".into(),
                lo: Bound::Included(Value::Int(100)),
                hi: Bound::Included(Value::Int(200))
            }
        );
        // Precedence: NOT > AND > OR; parens override.
        assert_eq!(
            parse_filter("a = 1 OR b = 2 AND NOT c = 3").unwrap(),
            Filter::Or(vec![
                Filter::Eq("a".into(), Value::Int(1)),
                Filter::And(vec![
                    Filter::Eq("b".into(), Value::Int(2)),
                    Filter::Not(Box::new(Filter::Eq("c".into(), Value::Int(3)))),
                ]),
            ])
        );
        assert!(matches!(
            parse_filter("(a = 1 OR b = 2)").unwrap(),
            Filter::Or(_)
        ));
        assert_eq!(
            parse_filter("source EXISTS").unwrap(),
            Filter::Exists("source".into())
        );
        // Errors.
        assert!(parse_filter("").is_err());
        assert!(parse_filter("a =").is_err());
        assert!(parse_filter("doc IN (a,").is_err());
        assert!(parse_filter("a = 1 b = 2").is_err()); // missing connective
        assert!(parse_filter("(a = 1").is_err()); // unbalanced paren
    }

    #[test]
    fn evaluate_all_predicates() {
        let mut idx = PayloadIndex::default();
        idx.upsert(
            1,
            vec![("u".into(), kw("a")), ("age".into(), Value::Int(20))],
        );
        idx.upsert(
            2,
            vec![("u".into(), kw("a")), ("age".into(), Value::Int(40))],
        );
        idx.upsert(
            3,
            vec![("u".into(), kw("b")), ("age".into(), Value::Int(60))],
        );
        idx.upsert(4, vec![("u".into(), kw("c"))]); // no `age` field

        // evaluate returns a sorted Vec; collect to a set for order-free asserts.
        let eval = |s: &str| {
            parse_filter(s)
                .unwrap()
                .evaluate(&idx)
                .into_iter()
                .collect::<BTreeSet<u64>>()
        };
        assert_eq!(eval("u = a"), BTreeSet::from([1, 2]));
        assert_eq!(eval("age >= 40"), BTreeSet::from([2, 3]));
        assert_eq!(eval("age > 40"), BTreeSet::from([3]));
        assert_eq!(eval("age < 40"), BTreeSet::from([1]));
        assert_eq!(eval("age BETWEEN 20 AND 40"), BTreeSet::from([1, 2]));
        assert_eq!(eval("u = a AND age >= 40"), BTreeSet::from([2]));
        assert_eq!(eval("u = b OR age < 40"), BTreeSet::from([1, 3]));
        // A AND NOT B is the difference A \ B (no full complement built).
        assert_eq!(eval("u = a AND NOT age = 20"), BTreeSet::from([2]));
        // Pure-negative AND starts from the universe {1,2,3,4} then subtracts.
        assert_eq!(eval("NOT u = a AND NOT u = b"), BTreeSet::from([4]));
        assert_eq!(eval("age EXISTS"), BTreeSet::from([1, 2, 3]));
        assert_eq!(eval("NOT age EXISTS"), BTreeSet::from([4]));
        // NOT complements against the indexed universe {1,2,3,4}.
        assert_eq!(eval("NOT u = a"), BTreeSet::from([3, 4]));
        assert!(eval("u = nobody").is_empty());
    }
}

#[cfg(test)]
mod blob_roundtrip_tests {
    use super::*;

    /// Persisting the index depends on this: the reconstructed blob must parse
    /// back to exactly the fields it came from, or a restart would serve a
    /// different index than the one that was running.
    #[test]
    fn field_blob_round_trips_through_the_parser() {
        let cases: &[&[u8]] = &[
            b"dl=10 likes=0 lic=apache-2.0 task=text-generation",
            b"params_m=7000 alive=1 enr=0",
            b"k=-42 neg=-0 big=9223372036854775807",
            b"single=x",
            b"",
            b"no_equals_token dl=5", // the parser skips the bare token
            b"=novalue dl=5",        // and an empty key
        ];
        for blob in cases {
            let mut idx = PayloadIndex::default();
            let fields = parse_fields(blob);
            idx.upsert(1, fields.clone());
            let again = parse_fields(&idx.field_blob(1));
            assert_eq!(
                again,
                fields,
                "blob {:?} did not round trip",
                String::from_utf8_lossy(blob)
            );
        }
    }

    /// A value that looks like an integer must not come back as a keyword, and
    /// the other way round: the two are ordered differently, so a swap would
    /// quietly change what a range filter matches.
    #[test]
    fn field_blob_preserves_the_value_type() {
        let mut idx = PayloadIndex::default();
        idx.upsert(1, parse_fields(b"n=7 s=seven z=007"));
        let back = parse_fields(&idx.field_blob(1));
        assert_eq!(back[0], ("n".into(), Value::Int(7)));
        assert_eq!(back[1], ("s".into(), Value::Keyword("seven".into())));
        assert_eq!(
            back[2],
            ("z".into(), Value::Int(7)),
            "leading zeros parse as the number"
        );
    }
}
