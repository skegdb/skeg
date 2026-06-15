//! Typed payload fields, the per-vindex payload index, and the filter AST that
//! turns a `SKEG.VSEARCH ... FILTER` clause into the set of matching vector ids.
//!
//! Pure and synchronous: the shard owns one [`PayloadIndex`] per vindex, feeds it
//! the fields parsed from a VSET payload, and queries it on a filtered VSEARCH.
//! The blob itself is still stored verbatim (see `shard::payload_key`) for
//! WITHPAYLOAD return; this module only derives the searchable index from it.
//!
//! Grammar: keyword and i64 fields; predicates `=`, `IN (...)`, the ranges
//! `>=`, `>`, `<=`, `<`, `BETWEEN a AND b`; combined with `AND`, `OR`, `NOT`
//! and parentheses. Wider types and roaring-bitmap postings are deferred.

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
    by_field: BTreeMap<String, BTreeMap<Value, BTreeSet<u64>>>,
    by_id: BTreeMap<u64, Vec<(String, Value)>>,
}

impl PayloadIndex {
    /// Index `id`'s fields, replacing anything previously indexed for that id.
    /// Indexing the same id again is how an overwrite VSET stays consistent.
    pub fn upsert(&mut self, id: u64, fields: Vec<(String, Value)>) {
        self.remove(id);
        for (f, v) in &fields {
            self.by_field
                .entry(f.clone())
                .or_default()
                .entry(v.clone())
                .or_default()
                .insert(id);
        }
        if !fields.is_empty() {
            self.by_id.insert(id, fields);
        }
    }

    /// Drop all of `id`'s postings. No-op if `id` was never indexed.
    pub fn remove(&mut self, id: u64) {
        let Some(fields) = self.by_id.remove(&id) else {
            return;
        };
        for (f, v) in fields {
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

    fn postings(&self, field: &str, value: &Value) -> Option<&BTreeSet<u64>> {
        self.by_field.get(field).and_then(|vs| vs.get(value))
    }

    /// Union of ids whose `field` value lies in `[lo, hi]` (per the bounds).
    fn range_ids(&self, field: &str, lo: &Bound<Value>, hi: &Bound<Value>) -> BTreeSet<u64> {
        let mut out = BTreeSet::new();
        if let Some(vs) = self.by_field.get(field) {
            for (_, ids) in vs.range((lo.as_ref(), hi.as_ref())) {
                out.extend(ids.iter().copied());
            }
        }
        out
    }

    /// Every indexed id (the universe `NOT` complements against).
    fn all_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.by_id.keys().copied()
    }
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
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    /// The set of vector ids matching this filter against `idx`. An unknown
    /// field or value contributes the empty set; `NOT` complements against the
    /// indexed universe.
    #[must_use]
    pub fn evaluate(&self, idx: &PayloadIndex) -> BTreeSet<u64> {
        match self {
            Filter::Eq(f, v) => idx.postings(f, v).cloned().unwrap_or_default(),
            Filter::In(f, vs) => {
                let mut out = BTreeSet::new();
                for v in vs {
                    if let Some(p) = idx.postings(f, v) {
                        out.extend(p.iter().copied());
                    }
                }
                out
            }
            Filter::Range { field, lo, hi } => idx.range_ids(field, lo, hi),
            Filter::And(parts) => {
                // Intersect smallest-first so the running set only shrinks.
                let mut sets: Vec<BTreeSet<u64>> = parts.iter().map(|p| p.evaluate(idx)).collect();
                sets.sort_by_key(BTreeSet::len);
                let mut iter = sets.into_iter();
                let Some(mut acc) = iter.next() else {
                    return BTreeSet::new();
                };
                for s in iter {
                    acc = acc.intersection(&s).copied().collect();
                    if acc.is_empty() {
                        break;
                    }
                }
                acc
            }
            Filter::Or(parts) => parts.iter().flat_map(|p| p.evaluate(idx)).collect(),
            Filter::Not(inner) => {
                let excluded = inner.evaluate(idx);
                idx.all_ids().filter(|id| !excluded.contains(id)).collect()
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
        || ["AND", "OR", "NOT", "IN", "BETWEEN"]
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
        ">=" => Ok(range(field, Bound::Included(take_value(toks, i, "a value")?), Bound::Unbounded)),
        ">" => Ok(range(field, Bound::Excluded(take_value(toks, i, "a value")?), Bound::Unbounded)),
        "<=" => Ok(range(field, Bound::Unbounded, Bound::Included(take_value(toks, i, "a value")?))),
        "<" => Ok(range(field, Bound::Unbounded, Bound::Excluded(take_value(toks, i, "a value")?))),
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
        assert_eq!(
            idx.postings("user", &kw("alice")),
            Some(&BTreeSet::from([1, 2]))
        );
        idx.upsert(1, vec![("user".into(), kw("carol"))]);
        assert_eq!(idx.postings("user", &kw("alice")), Some(&BTreeSet::from([2])));
        idx.remove(3);
        assert_eq!(idx.postings("user", &kw("bob")), None);
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
        idx.upsert(1, vec![("u".into(), kw("a")), ("age".into(), Value::Int(20))]);
        idx.upsert(2, vec![("u".into(), kw("a")), ("age".into(), Value::Int(40))]);
        idx.upsert(3, vec![("u".into(), kw("b")), ("age".into(), Value::Int(60))]);

        let eval = |s: &str| parse_filter(s).unwrap().evaluate(&idx);
        assert_eq!(eval("u = a"), BTreeSet::from([1, 2]));
        assert_eq!(eval("age >= 40"), BTreeSet::from([2, 3]));
        assert_eq!(eval("age > 40"), BTreeSet::from([3]));
        assert_eq!(eval("age < 40"), BTreeSet::from([1]));
        assert_eq!(eval("age BETWEEN 20 AND 40"), BTreeSet::from([1, 2]));
        assert_eq!(eval("u = a AND age >= 40"), BTreeSet::from([2]));
        assert_eq!(eval("u = b OR age < 40"), BTreeSet::from([1, 3]));
        assert_eq!(eval("u = a AND NOT age = 20"), BTreeSet::from([2]));
        // NOT complements against the indexed universe {1,2,3}.
        assert_eq!(eval("NOT u = a"), BTreeSet::from([3]));
        assert!(eval("u = nobody").is_empty());
    }
}
