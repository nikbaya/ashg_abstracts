"""Resolve first- and senior-author affiliations to institutes via aliases.csv.

Each alias pattern is searched in the author's full affiliation text (not comma
segments, which would split "University of California, San Diego"). Patterns
are tried in file order and every match is blanked out, so "MIT" inside
"Broad Institute of MIT and Harvard" is not counted again.

Usage: python src/ashg/institutions.py ALIASES.csv OUT.parquet ABSTRACTS.parquet [...]
"""

import argparse
import collections
import csv
import re

import pandas as pd

from authors import key_authors, load_overrides, parse_authors, senior_author
from seed_institutions import affil_segments


def load_aliases(path):
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    bad = [r["institute"] for r in rows if None in r]
    if bad:
        raise ValueError(f"aliases.csv rows with extra fields (quote patterns containing commas): {bad}")
    return [(r["institute"], re.compile(r["pattern"], re.I)) for r in rows]


# The 2023-2024 booklets abbreviate affiliation words ("Harvard Med. Sch.",
# "Natl. Inst.s of Hlth."); expand them so aliases.csv needs only full names.
ABBREVIATIONS = [
    (r"\bMed\.\s*(?=(Ctr|Coll|Sch|Campus|Inst|Res|Univ|Genet|Sci|Branch|Biol))", "Medical "),
    (r"\b(Women|Children|Men) s\b", r"\1's"),  # apostrophe lost in the PDF text
    (r"\bGen\.\s*(?=Hosp\b)", "General "),
    (r"\bMed\.", "Medicine"), (r"\bUniv\b\.?", "University"), (r"\bInst\.s\b", "Institutes"), (r"\bCtr\.s\b", "Centers"),
    (r"\bInst\.", "Institute"), (r"\bCtr\.", "Center"), (r"\bRes\.", "Research"), (r"\bHosp\.", "Hospital"),
    (r"\bColl\.", "College"), (r"\bSch\.", "School"), (r"\bNatl\.", "National"), (r"\bHlth\.", "Health"),
    (r"\bBioSci\.s?", "Biosciences"),  # 2022 "Parse BioSci.s", "Pacific BioSci.s of California"
    (r"\bSci\.", "Science"), (r"\bTechnol\.", "Technology"), (r"\bLab\.", "Laboratory"),
]


def normalize(text):
    text = text.replace("’", "'").replace("‘", "'")
    for pat, full in ABBREVIATIONS:
        text = re.sub(pat, full + " ", text)
    return re.sub(r"\s+", " ", re.sub(r" +(?=[,.;)])", "", text)).strip()


def match_institutes(author, aliases):
    """Distinct institutes in an author's affiliations, in alias-file order."""
    if not author or not author["affils"]:
        return []
    text = normalize(", ".join(author["affils"]))  # rejoin as written
    found = []
    for name, pat in aliases:
        if pat.search(text):
            found.append(name)
            text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return found


def resolve(df, aliases, overrides):
    """One row per abstract. Abstracts cut off in the booklet keep their first
    author (the list is cut at the end) but have no known senior author."""
    rows = []
    for r in df.itertuples(index=False):
        first, senior = key_authors(r.abstract_id, parse_authors(r.authors_raw), overrides)
        if r.truncated:
            senior = None
            if first and "(" in first["name"]:  # cut inside the first author's affiliations
                first = None
        rows.append({
            "abstract_id": r.abstract_id, "year": r.year, "pres_type": r.pres_type, "truncated": r.truncated,
            "first_author": first["name"] if first else None,
            "senior_author": senior["name"] if senior else None,
            "first_insts": match_institutes(first, aliases),
            "senior_insts": match_institutes(senior, aliases),
            "senior_has_affils": bool(senior and senior["affils"]),
            "first_has_affils": bool(first and first["affils"]),
        })
    return pd.DataFrame(rows)


def report(out, df_by_file, top_unmatched=30):
    for role in ("senior", "first"):
        has = out[f"{role}_has_affils"]
        matched = out[f"{role}_insts"].str.len() > 0
        print(f"{role:6s}: {matched[has].mean():.1%} of {has.sum()} abstracts with affiliations matched ≥1 institute")
    both = (out.first_insts.str.len() > 0) & (out.senior_insts.str.len() > 0)
    agree = out[both].apply(lambda r: bool(set(r.first_insts) & set(r.senior_insts)), axis=1)
    print(f"first/senior institute sets overlap in {agree.mean():.1%} of {both.sum()} abstracts with both matched")

    # Most frequent affiliation segments of unmatched senior authors -> candidates for aliases.csv
    unmatched = set(out.abstract_id[out.senior_has_affils & (out.senior_insts.str.len() == 0)])
    counts = collections.Counter()
    for df in df_by_file:
        for r in df[df.abstract_id.isin(unmatched)].itertuples():
            counts.update(set(affil_segments(senior_author(parse_authors(r.authors_raw)))))
    print(f"\ntop affiliation segments among {len(unmatched)} unmatched senior authors:")
    for seg, n in counts.most_common(top_unmatched):
        print(f"  {n:3d} {seg[:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aliases")
    ap.add_argument("out")
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--top-unmatched", type=int, default=30)
    ap.add_argument("--overrides", default="overrides.csv")
    args = ap.parse_args()
    aliases = load_aliases(args.aliases)
    dfs = [pd.read_parquet(p) for p in args.abstracts]
    overrides = load_overrides(args.overrides)
    out = pd.concat([resolve(df, aliases, overrides) for df in dfs], ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out)} abstracts -> {args.out}")
    report(out, dfs, args.top_unmatched)


if __name__ == "__main__":
    main()
