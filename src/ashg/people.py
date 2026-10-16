"""Resolve senior-author occurrences into people.

Occurrences with the same normalized name (first + last name; middle
names/initials and suffixes dropped) are the same person when:
- the surname is distinctive: always (collisions are rare, while the same
  person's affiliations are often spelled differently: VUMC vs Vanderbilt);
- the surname is common (seen with many distinct first names across all
  authors in the corpus): only if they share at least one institution, either
  an alias-table institute or a raw affiliation segment naming an institution.
  Merging is transitive (union-find), and affiliation-less occurrences join
  the name's cluster only when that cluster is unambiguous.

Usage: python src/ashg/people.py INSTITUTES.parquet OUT.parquet ABSTRACTS.parquet [...]
"""

import argparse
import collections
import re
import unicodedata

import pandas as pd

from authors import INSTITUTION_WORDS, key_authors, load_overrides, parse_authors
from seed_institutions import affil_segments

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "md", "phd", "mph", "ms", "msc", "dr"}
COMMON_MIN_FIRST_NAMES = 5  # surname seen with >= this many distinct first names


def name_key(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    words = [w for w in re.sub(r"[^a-z\s-]", " ", s).split() if w.strip("-") and w not in SUFFIXES]
    if len(words) < 2:
        return None
    return f"{words[0]} {words[-1]}"


def raw_institution_segments(author):
    out = set()
    for seg in affil_segments(author):
        if len(seg.split()) >= 2 and INSTITUTION_WORDS.search(seg):
            out.add(re.sub(r"\W+", " ", seg.lower()).strip())
    return out


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def common_surnames(all_name_keys, min_first_names=COMMON_MIN_FIRST_NAMES):
    firsts = collections.defaultdict(set)
    for key in all_name_keys:
        first, last = key.split(" ", 1)
        firsts[last].add(first)
    return {last for last, f in firsts.items() if len(f) >= min_first_names}


def resolve_people(occ, common):
    """occ: DataFrame with name_key and `tokens` (set of institution tokens). Adds person_id."""
    occ = occ.reset_index(drop=True)
    uf = UnionFind(len(occ))
    for key, idx in occ.groupby("name_key").groups.items():
        if key.split(" ", 1)[1] not in common:
            for i in idx[1:]:
                uf.union(i, idx[0])
            continue
        first_with = {}
        for i in idx:
            for t in occ.at[i, "tokens"]:
                if t in first_with:
                    uf.union(i, first_with[t])
                else:
                    first_with[t] = i
        # affiliation-less occurrences join the name's cluster if it is unambiguous
        roots = {uf.find(i) for i in idx if occ.at[i, "tokens"]}
        if len(roots) == 1:
            root = roots.pop()
            for i in idx:
                if not occ.at[i, "tokens"]:
                    uf.union(i, root)
    roots = [uf.find(i) for i in range(len(occ))]
    ids = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    occ["person_id"] = [ids[r] for r in roots]
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("institutes")
    ap.add_argument("out")
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--overrides", default="overrides.csv")
    args = ap.parse_args()

    overrides = load_overrides(args.overrides)
    insts = pd.read_parquet(args.institutes).set_index("abstract_id")
    rows, all_keys = [], []
    for path in args.abstracts:
        df = pd.read_parquet(path)
        for r in df[~df.truncated].itertuples(index=False):
            authors = parse_authors(r.authors_raw)
            all_keys += [k for k in map(name_key, (a["name"] for a in authors)) if k]
            _, senior = key_authors(r.abstract_id, authors, overrides)
            key = name_key(senior["name"]) if senior else None
            if key is None:
                continue
            tokens = set(insts.at[r.abstract_id, "senior_insts"]) | raw_institution_segments(senior)
            rows.append({"abstract_id": r.abstract_id, "year": r.year, "pres_type": r.pres_type,
                         "name": senior["name"], "name_key": key, "tokens": tokens})
    common = common_surnames(all_keys)
    occ = resolve_people(pd.DataFrame(rows), common)
    print(f"{len(common)} common surnames (>= {COMMON_MIN_FIRST_NAMES} distinct first names among "
          f"{len(set(all_keys))} author names), e.g. {sorted(common)[:8]}")

    # display name = most common spelling; institutes = matched institutes across the person's abstracts
    occ["display_name"] = occ.groupby("person_id").name.transform(lambda s: s.value_counts().index[0])
    occ["institutes"] = occ.abstract_id.map(lambda a: list(insts.at[a, "senior_insts"]))
    occ.drop(columns="tokens").to_parquet(args.out, index=False)

    n_people = occ.person_id.nunique()
    split = occ.groupby("name_key").person_id.nunique()
    sizes = occ.groupby("person_id").size()
    print(f"{len(occ)} senior-author occurrences -> {n_people} people -> {args.out}")
    print(f"people with >=5 abstracts: {(sizes >= 5).sum()}, >=4: {(sizes >= 4).sum()}")
    print(f"names split into >1 person: {(split > 1).sum()} (possible missed merges; largest below)")
    for key in split[split > 1].index:
        sub = occ[occ.name_key == key]
        if len(sub) >= 4:
            parts = [f"{len(g)}×{collections.Counter(i for l in g.institutes for i in l).most_common(1)[0][0] if any(len(l) for l in g.institutes) else '?'}"
                     for _, g in sub.groupby("person_id")]
            print(f"  {sub.display_name.iloc[0]}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
