"""Check segmented abstracts against structural invariants and the booklet TOC.

Hard failures (count vs TOC, ids, title/body/authors present) must be zero.
Author-format anomalies come from how submitters typed their author lists
(missing parentheses, no affiliations), so they are reported as a rate and
only fail above SOFT_MAX_RATE.

Usage: python src/ashg/validate.py CONFIG.yaml SPANS.parquet ABSTRACTS.parquet
Exit status is non-zero if validation fails.
"""

import argparse
import re
import sys

import pandas as pd
import yaml

from authors import is_plausible_name, load_overrides, parse_authors, repair_raw, senior_author
from segment import iter_lines

BODY_MIN, BODY_MAX = 300, 7000  # chars
SOFT_MAX_RATE = 0.03


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_completeness(df, spans, cfg):
    """Parsed abstracts vs. the TOC and the "Authors:" labels on abstract pages.

    Poster TOCs list Board IDs, so counts are compared directly. Talk TOC entries
    wrap unpredictably (dot leaders on their own line, bold session lines), so
    instead every parsed title must appear in the TOC text (only the first talk
    of each session with toc_first_per_session, as in the 2023 platform TOC).
    """
    problems = []
    lines = list(iter_lines(spans, cfg))
    n_labels = sum(text.startswith("Authors:") for _, _, _, text in lines)
    if n_labels != len(df):
        problems.append(f"count: parsed {len(df)} abstracts but found {n_labels} 'Authors:' labels")
    toc_lines = [text for _, _, _, text in iter_lines(spans, cfg, toc=True)]
    if cfg.get("id_regex"):
        id_re = re.compile(cfg["id_regex"])
        n_toc = sum(bool(id_re.match(t)) for t in toc_lines)
        if n_toc != len(df):
            problems.append(f"count: parsed {len(df)} abstracts but TOC lists {n_toc}")
    else:
        toc = norm(" ".join(toc_lines))
        listed = df[~df.session.duplicated()] if cfg.get("toc_first_per_session") else df
        missing = [r.abstract_id for r in listed.itertuples() if norm(r.title)[:40] not in toc]
        if missing:
            problems.append(f"{len(missing)} parsed title(s) not in TOC: {missing[:5]}")
    return problems


def check(df, spans, cfg, overrides):
    problems = check_completeness(df, spans, cfg)
    if df.abstract_id.isna().any():
        problems.append(f"missing id: {df.abstract_id.isna().sum()} records")
    dup = df.abstract_id[df.abstract_id.duplicated()]
    if len(dup):
        problems.append(f"duplicate ids: {sorted(dup)[:10]}")

    soft, no_affils = [], 0
    for r in df[~df.truncated].itertuples(index=False):
        where = f"{r.abstract_id} (p{r.page_start})"
        if not isinstance(r.title, str) or len(r.title) < 10:
            problems.append(f"{where}: short/missing title {r.title!r}")
        body = r.body if isinstance(r.body, str) else ""
        if not BODY_MIN <= len(body) <= BODY_MAX:
            problems.append(f"{where}: body length {len(body)}")
        raw = r.authors_raw if isinstance(r.authors_raw, str) else ""
        authors = parse_authors(raw)
        if not authors:
            problems.append(f"{where}: no authors")
            continue
        if r.abstract_id in overrides:
            continue  # fixed by hand
        if not any(a["affils"] for a in authors):
            no_affils += 1  # submitter listed names only; nothing to parse
            continue
        senior = senior_author(authors)
        fixed = repair_raw(raw)
        if fixed.count("(") > fixed.count(")"):  # an unclosed "(" may have swallowed authors
            soft.append(f"{where}: unclosed parenthesis in author list")
        elif senior is None or not senior["affils"] or not is_plausible_name(senior["name"]):
            s = senior or authors[-1]
            soft.append(f"{where}: senior author {s['name'][:50]!r} affils={s['affils'][:2]}")
    return problems, soft, no_affils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("spans")
    ap.add_argument("abstracts")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--overrides", default="overrides.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = pd.read_parquet(args.abstracts)
    problems, soft, no_affils = check(df, pd.read_parquet(args.spans), cfg, load_overrides(args.overrides))
    n_ok = int((~df.truncated).sum())
    rate = len(soft) / max(1, n_ok)
    ok = not problems and rate <= SOFT_MAX_RATE
    print(f"{args.config}: {'PASS' if ok else 'FAIL'} — {len(df)} abstracts ({len(df) - n_ok} truncated in PDF, "
          f"{no_affils} without any affiliations), {len(problems)} hard problem(s), "
          f"senior-author parse anomalies {len(soft)} ({rate:.1%}, max {SOFT_MAX_RATE:.0%})")
    for p in problems[: args.show]:
        print("  HARD " + p)
    for p in soft[: args.show]:
        print("  soft " + p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
