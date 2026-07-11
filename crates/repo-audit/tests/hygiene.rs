//! Repository hygiene gate: scans the workspace `crates/` sources and fails if
//! any public file carries an internal project code, non-English (Italian)
//! text, or an em-dash. Runs under `cargo test` (and the pre-push gate / CI),
//! so the manual grep audits are now an enforced, automatic check.
//!
//! If this test fails, fix the flagged comment/string (do not weaken a rule
//! unless it is a genuine false positive, in which case tighten the regex or
//! add to the hardware allow-list below).

use std::path::{Path, PathBuf};

use regex::Regex;

/// Recursively collect `*.rs` files under `dir`, skipping `target/`.
fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Recursively collect `*.md` files under `dir`, skipping `target/` and dot
/// directories (`.git`, `.github`, ...). Used for the docs em-dash check.
fn md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if name == "target" || name.starts_with('.') {
                continue;
            }
            md_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "md") {
            out.push(p);
        }
    }
}

#[test]
fn no_internal_codes_or_non_english() {
    // CARGO_MANIFEST_DIR is `crates/repo-audit`; its parent is `crates/`.
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .to_path_buf();

    let mut files = Vec::new();
    rs_files(&crates_dir, &mut files);
    // Skip this audit crate's own sources: they hold the denylist literals.
    files.retain(|p| !p.components().any(|c| c.as_os_str() == "repo-audit"));

    // Apple-Silicon hardware mentions (M1 Pro, M3 Max, M1/M3/M4, ...) are legit
    // and excused from the milestone / Q-code rules only.
    let hw_allow = Regex::new(r"(Pro|Max|Ultra|M-series|Apple M|/M[0-9]|M[0-9]/)").unwrap();

    // (rule name, pattern, whether the hardware allow-list applies)
    let rules: Vec<(&str, Regex, bool)> = vec![
        ("italian", Regex::new(r" e' |piu'|gia'|cosi'|perche'").unwrap(), false),
        ("em-dash", Regex::new("\u{2014}").unwrap(), false),
        ("slice-code", Regex::new(r"\bP0[a-z]\b").unwrap(), false),
        ("gate-code", Regex::new(r"\bG-[A-Z0-9]").unwrap(), false),
        (
            "internal-doc",
            Regex::new(
                r"optimizations/PLAN|OBSERVATIONS|turboquant/PLAN|skeg-internal|PLAN-POST|BLOCK-KERNEL|FEATURES\.md|STEP-[0-9]|/PLAN\.md",
            )
            .unwrap(),
            false,
        ),
        ("position-code", Regex::new(r"\bPosition [0-9]").unwrap(), false),
        ("tier-code", Regex::new(r"\bTier [0-9]").unwrap(), false),
        ("slice-letter", Regex::new(r"\bslice [A-D]\b").unwrap(), false),
        ("milestone-code", Regex::new(r"\bM[2-9]\b").unwrap(), true),
        ("q-code", Regex::new(r"\bQ[0-9]+\b").unwrap(), true),
    ];

    let mut violations = Vec::new();
    for f in &files {
        let Ok(content) = std::fs::read_to_string(f) else {
            continue;
        };
        for (lineno, line) in content.lines().enumerate() {
            for (name, re, uses_hw) in &rules {
                if re.is_match(line) && !(*uses_hw && hw_allow.is_match(line)) {
                    let rel = f.strip_prefix(&crates_dir).unwrap_or(f);
                    violations.push(format!(
                        "crates/{}:{}: [{name}] {}",
                        rel.display(),
                        lineno + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    // Em-dash is banned in docs too. Scan `*.md` from the workspace root (README,
    // docs/, per-crate READMEs, CHANGELOG). Only the em-dash rule applies here -
    // prose legitimately uses words the source-code rules (slice/tier/Q-codes)
    // would flag, so those stay `.rs`-only.
    let workspace_root = crates_dir.parent().expect("workspace root");
    // Em-dash in any form: the Unicode char plus the HTML entities that render
    // as one (`&mdash;`, `&#8212;`). Markdown passes raw HTML through, so the
    // entity forms slip past a bare `\u{2014}` scan. En-dash stays allowed:
    // numeric ranges (`9-17x`, `p50/p99`) use it legitimately.
    let em_dash = Regex::new(r"\u{2014}|&mdash;|&#8212;").unwrap();
    let mut md = Vec::new();
    md_files(workspace_root, &mut md);
    for f in &md {
        let Ok(content) = std::fs::read_to_string(f) else {
            continue;
        };
        for (lineno, line) in content.lines().enumerate() {
            if em_dash.is_match(line) {
                let rel = f.strip_prefix(workspace_root).unwrap_or(f);
                violations.push(format!(
                    "{}:{}: [em-dash] {}",
                    rel.display(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "source hygiene: {} violation(s) (internal codes / non-English / em-dash):\n{}",
        violations.len(),
        violations.join("\n")
    );
}
