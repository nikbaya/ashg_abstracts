"""Pass 2: stitch spans (from extract_spans.py) into one record per abstract.

Streams lines in page order through a small state machine driven by a per-file
YAML config: a title-styled line opens an abstract, labelled lines ("Authors:",
"Abstract:", ...) switch the field being filled, and other lines continue the
current field.

Usage: python src/ashg/segment.py CONFIG.yaml SPANS.parquet OUT.parquet
"""

import argparse
import re

import pandas as pd
import pyarrow.parquet as pq
import yaml

LABELS = {
    "Location:": "location",
    "Session Time:": "session_time",
    "Subsession Time:": "time",
    "Session:": "session",
    "Authors:": "authors_raw",
    "Abstract:": "body",
}
LABEL_RE = re.compile(r"^(%s)\s*" % "|".join(re.escape(k) for k in LABELS))
BOLD = 16
TRUNCATED_BELOW = 300  # body chars


def iter_lines(spans, cfg, toc=False):
    """Group spans into visual lines; yield (page, size, bold, text).

    Covers the abstract pages, or with toc=True the table-of-contents pages.
    """
    in_range = spans.page < cfg["start_page"] if toc else spans.page >= cfg["start_page"]
    spans = spans[in_range & (spans.y0 > cfg["header_y_max"]) & (spans.y0 < cfg["footer_y_min"])]
    spans = spans.sort_values(["page", "y0", "x0"], kind="stable")
    line, key = [], None
    for row in spans.itertuples(index=False):
        if line and (row.page != key[0] or abs(row.y0 - key[1]) > 3):
            yield finish_line(line)
            line = []
        if not line:
            key = (row.page, row.y0)
        line.append(row)
    if line:
        yield finish_line(line)


def finish_line(spans):
    spans = sorted(spans, key=lambda s: s.x0)
    first = spans[0]
    text = "".join(s.text for s in spans).strip()
    return first.page, first.size, bool(first.flags & BOLD), text


def matches(styles, size, bold):
    """`styles` is one {size, bold} dict, a list of them, or None."""
    if styles is None:
        return False
    if isinstance(styles, dict):
        styles = [styles]
    return any(abs(size - s["size"]) < 0.5 and bold == s["bold"] for s in styles)


def join(lines):
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out += ln  # hyphenated word or compound wrapped across lines
        elif out:
            out += " " + ln
        else:
            out = ln
    return re.sub(r"\s+", " ", out).strip()


def segment(spans, cfg):
    id_re = re.compile(cfg["id_regex"]) if cfg.get("id_regex") else None
    records, cur, field, session = [], None, None, None
    session_lines = None  # collecting a multi-line session header

    def close():
        if cur is not None:
            records.append({k: join(v) if isinstance(v, list) else v for k, v in cur.items()})

    for page, size, bold, text in iter_lines(spans, cfg):
        if matches(cfg.get("session_header"), size, bold):
            if session_lines is None:
                close()
                cur, field, session_lines = None, None, []
            session_lines.append(text)
            session = join(session_lines)
            continue
        session_lines = None

        if matches(cfg["title"], size, bold):
            is_id_line = id_re is not None and id_re.match(text)
            if field == "title" and is_id_line and not id_re.match(cur["title"][0]):
                # lines so far were a poster category header in title style; drop them
                cur["title"] = []
            elif field != "title" or is_id_line:
                close()
                cur = {"page_start": page, "session": [session] if session else [], "title": []}
                field = "title"
            cur["title"].append(text)
            cur["page_end"] = page
            continue

        if cur is None:
            continue  # session-level lines (Location, Session Time) before the first title
        m = LABEL_RE.match(text)
        if m:
            field = LABELS[m.group(1)]
            cur[field] = []
            text = text[m.end():]
        cur.setdefault(field, []).append(text)
        cur["page_end"] = page
    close()

    df = pd.DataFrame(records)
    df.insert(0, "year", cfg["year"])
    df.insert(1, "pres_type", cfg["pres_type"])
    if id_re is not None:
        ids = df.title.str.extract(id_re, expand=False)
        df["title"] = df.title.str.replace(id_re, "", n=1, regex=True)
        df.insert(0, "abstract_id", f"{cfg['year']}-{cfg['pres_type']}-" + ids)
    else:
        df.insert(0, "abstract_id", [f"{cfg['year']}-{cfg['pres_type']}-{i + 1:04d}" for i in range(len(df))])
    # Some booklet entries are cut off mid-word in the PDF itself (e.g. 2025
    # Platform pp 101-137), authors included; flag them so analyses can skip them.
    body = df.body.fillna("")
    df["truncated"] = (body.str.len() < TRUNCATED_BELOW) & ~body.str.rstrip().str[-1:].isin(list(".!?)"))
    cols = ["abstract_id", "year", "pres_type", "session", "title", "time", "session_time",
            "location", "authors_raw", "body", "truncated", "page_start", "page_end"]
    return df[[c for c in cols if c in df.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("spans")
    ap.add_argument("out")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = segment(pq.read_table(args.spans).to_pandas(), cfg)
    df.to_parquet(args.out, index=False)
    print(f"{args.config}: {len(df)} abstracts -> {args.out}")


if __name__ == "__main__":
    main()
