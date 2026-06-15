//! Typed payload fields, the per-vindex payload index, and the filter AST that
//! turns a `SKEG.VSEARCH ... FILTER` clause into the set of matching vector ids.
//!
//! Pure and synchronous: the shard owns one [`PayloadIndex`] per vindex, feeds it
//! the fields parsed from a VSET payload, and queries it on a filtered VSEARCH.
//! The blob itself is still stored verbatim (see `shard::payload_key`) for
//! WITHPAYLOAD return; this module only derives the searchable index from it.
//!
//! MVP scope: keyword and i64 fields; `=`, `IN`, and `AND`. OR, NOT, ranges,
//! wider types, and roaring-bitmap postings are deferred (see the gate doc).

use std::collections::{BTreeMap, BTreeSet};

/// A typed payload value. A token that parses as an `i64` is an `Int`, otherwise
/// a `Keyword`. `Ord` is derived only so it can key a `BTreeMap`.
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
}

/// A filter over payload fields. MVP grammar: AND of equality / membership
/// predicates. OR, NOT, and range predicates come later.
#[derive(Debug, PartialEq)]
pub enum Filter {
    Eq(String, Value),
    In(String, Vec<Value>),
    And(Vec<Filter>),
}

impl Filter {
    /// The set of vector ids matching this filter against `idx`. An unknown
    /// field or value contributes the empty set, which `AND` then propagates.
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
            Filter::And(parts) => {
                // Intersect smallest-first so the running set only shrinks and
                // the cost tracks the most selective predicate.
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
        }
    }
}

fn is_reserved(t: &str) -> bool {
    matches!(t, "=" | "(" | ")" | ",")
        || t.eq_ignore_ascii_case("AND")
        || t.eq_ignore_ascii_case("IN")
}

/// Split a filter string into tokens, treating `= ( ) ,` as standalone tokens so
/// `doc IN (a,b)` needs no spaces around the punctuation.
fn tokenize(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '=' | '(' | ')' | ',' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks
}

/// Parse a `FILTER` clause: `pred (AND pred)*`, where a `pred` is
/// `field = value` or `field IN (v1, v2, ...)`.
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
    let mut preds = Vec::new();
    loop {
        preds.push(parse_predicate(&toks, &mut i)?);
        match toks.get(i) {
            None => break,
            Some(t) if t.eq_ignore_ascii_case("AND") => i += 1,
            Some(t) => return Err(format!("expected AND or end of filter, got '{t}'")),
        }
    }
    Ok(if preds.len() == 1 {
        preds.pop().unwrap()
    } else {
        Filter::And(preds)
    })
}

fn parse_predicate(toks: &[String], i: &mut usize) -> Result<Filter, String> {
    let field = toks.get(*i).ok_or("expected a field name")?.clone();
    if is_reserved(&field) {
        return Err(format!("expected a field name, got '{field}'"));
    }
    *i += 1;
    let op = toks.get(*i).ok_or("expected '=' or IN after the field")?;
    if op == "=" {
        *i += 1;
        let val = toks.get(*i).ok_or("expected a value after '='")?;
        if is_reserved(val) {
            return Err(format!("expected a value, got '{val}'"));
        }
        let v = Value::parse(val);
        *i += 1;
        Ok(Filter::Eq(field, v))
    } else if op.eq_ignore_ascii_case("IN") {
        *i += 1;
        if toks.get(*i).map(String::as_str) != Some("(") {
            return Err("expected '(' after IN".to_owned());
        }
        *i += 1;
        let mut vals = Vec::new();
        loop {
            let val = toks.get(*i).ok_or("unterminated IN list")?;
            if is_reserved(val) {
                return Err(format!("expected a value in the IN list, got '{val}'"));
            }
            vals.push(Value::parse(val));
            *i += 1;
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
    } else {
        Err(format!("expected '=' or IN, got '{op}'"))
    }
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
        // `loose` has no '=' and is skipped; non-UTF-8 yields nothing.
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

        // Overwrite id 1 to a new value: no phantom posting under the old one.
        idx.upsert(1, vec![("user".into(), kw("carol"))]);
        assert_eq!(idx.postings("user", &kw("alice")), Some(&BTreeSet::from([2])));
        assert_eq!(idx.postings("user", &kw("carol")), Some(&BTreeSet::from([1])));

        // Remove drops the id and prunes the now-empty value/field maps.
        idx.remove(3);
        assert_eq!(idx.postings("user", &kw("bob")), None);
    }

    #[test]
    fn parse_filter_grammar() {
        assert_eq!(
            parse_filter("user = alice").unwrap(),
            Filter::Eq("user".into(), kw("alice"))
        );
        assert_eq!(
            parse_filter("doc IN (a,b,c)").unwrap(),
            Filter::In("doc".into(), vec![kw("a"), kw("b"), kw("c")])
        );
        assert_eq!(
            parse_filter("user=alice AND type=doc").unwrap(),
            Filter::And(vec![
                Filter::Eq("user".into(), kw("alice")),
                Filter::Eq("type".into(), kw("doc")),
            ])
        );
        // Errors.
        assert!(parse_filter("").is_err());
        assert!(parse_filter("user ==").is_err());
        assert!(parse_filter("doc IN (a,").is_err());
        assert!(parse_filter("user = alice type = doc").is_err()); // missing AND
    }

    #[test]
    fn evaluate_eq_in_and() {
        let mut idx = PayloadIndex::default();
        idx.upsert(1, vec![("u".into(), kw("a")), ("t".into(), kw("doc"))]);
        idx.upsert(2, vec![("u".into(), kw("a")), ("t".into(), kw("img"))]);
        idx.upsert(3, vec![("u".into(), kw("b")), ("t".into(), kw("doc"))]);

        assert_eq!(
            parse_filter("u = a").unwrap().evaluate(&idx),
            BTreeSet::from([1, 2])
        );
        assert_eq!(
            parse_filter("t IN (doc,img)").unwrap().evaluate(&idx),
            BTreeSet::from([1, 2, 3])
        );
        assert_eq!(
            parse_filter("u = a AND t = doc").unwrap().evaluate(&idx),
            BTreeSet::from([1])
        );
        // Unknown field/value -> empty, and AND with empty stays empty.
        assert!(parse_filter("u = zzz").unwrap().evaluate(&idx).is_empty());
        assert!(
            parse_filter("u = a AND missing = x")
                .unwrap()
                .evaluate(&idx)
                .is_empty()
        );
    }
}
