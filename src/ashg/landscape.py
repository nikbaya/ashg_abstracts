"""Input file for the site's Landscape page: one node per 2026 abstract at its t-SNE position.

The layout is the one network.py makes for Find talks (same BM25 vectors, same TruncatedSVD + TSNE settings), rescaled
to about -100..100. Each node lists its authors in order with their position (first, middle, last), the institutions
the institutions table found for the first and senior author, and keywords: the synonym groups of its highest-weighted BM25 terms (src/ashg/synonyms.py), definition groups first, with
the spellings in each group written to <out>_synonyms.json. "similar" holds its SIMILAR nearest abstracts as [id, match %], ranked by raw
BM25 cosine and scored on the same 0-100 scale as Find talks (95th percentile of best-match cosines = 100); matches
below SIM_FLOOR (10%) are dropped.
Titles and abstract text are never written out (same rule as network.json), so "title" is left out; the page paints
keywords from the terms.

Authors are merged on author_key (first + last name, lowercased) and shown under the longest spelling seen, so one
person's abstracts end up under a single autocomplete entry.

Usage: python src/ashg/landscape.py outputs/tsne_landscape.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from network import SIM_FLOOR, author_key, bm25_vectors, load_abstracts, top_neighbors
from synonyms import Keywords

KEYWORDS = 25  # paint-by-keyword only finds an abstract through these, so a topic ranked 10th still counts
SIMILAR = 15  # text neighbors per node, for the page's similar-abstracts web
SCALE = 100  # coordinates run -SCALE..SCALE
# Degrees, titles and suffixes the author-block parsing leaves as their own "authors" or glued to a name. Degrees match
# case-sensitively so a surname like Ma is kept.
DEGREES = {"PhD", "Ph.D", "MD", "M.D", "MS", "M.S", "MA", "M.A", "MSc", "M.Sc", "BS", "BSc", "BA", "MPH", "DPhil", "MBA",
           "MHS", "DDS", "RN", "FACMG", "MBBS", "PharmD", "DVM", "MSN", "Jr", "Sr", "II", "III"}
TITLES = {"dr", "prof", "professor"}
FILLER = {"candidate", "scholar", "student", "fellow", "postdoctoral", "researcher", "resident", "trainee"}


def tidy(name):
    """A name without trailing degrees ("Kun Zhao Ph.D"), a leading title ("Dr. Eliot Shearer") or Jr/Sr; None when no
    name is left ("PhD", "PhD candidate")."""
    bare = lambda t: t.strip(".,;")
    toks = name.split()
    while toks and bare(toks[-1]) in DEGREES:
        toks.pop()
    while toks and bare(toks[0]).lower() in TITLES:
        toks.pop(0)
    if all(bare(t) in DEGREES or bare(t).lower() in FILLER for t in toks):
        return None
    return " ".join(toks)


def people(fulls):
    """[(author_key, name)] for the real names in an author list, in order."""
    out = []
    for full in fulls:
        name = tidy(full)
        if name and (key := author_key(name)):
            out.append((key, name))
    return out


def positions(n):
    return ["first"] + ["middle"] * (n - 2) + ["last"] if n > 1 else ["first"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    d = load_abstracts()
    vec, vocab = bm25_vectors(d.text)
    kw = Keywords(vec, vocab, d.text, KEYWORDS)

    similar = top_neighbors(vec @ vec.T, SIMILAR)
    ref = np.percentile([nb[0][1] for nb in similar if nb], 95)  # as network.py's match scale
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
        TruncatedSVD(50, random_state=0).fit_transform(vec))
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))  # as network.py, so the layouts agree
    xy = SCALE * (2 * xy - 1)

    spelled = {}  # author_key -> longest spelling
    listed = [people(fulls) for fulls in d.full]
    for ps in listed:
        for k, name in ps:
            spelled[k] = max(spelled.get(k, ""), name, key=len)

    insts = pd.read_parquet("data/institutions/2026.parquet").set_index("abstract_id")
    nodes = []
    for i, r in enumerate(d.itertuples()):
        ks = [k for k, _ in listed[i]]
        seen, authors = set(), []
        for k, pos in zip(ks, positions(len(ks))):
            if k not in seen:  # a name listed twice keeps its earliest (highest-ranked) position
                seen.add(k)
                authors.append({"name": spelled[k], "position": pos})
        found = []
        if r.id in insts.index:
            row = insts.loc[r.id]
            found = sorted(set(row.first_insts) | set(row.senior_insts))
        nodes.append({"id": i + 1, "x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2), "type": r.type,
                      "authors": authors, "institutions": found, "keywords": kw.of(i),
                      "similar": [[j + 1, p] for j, c in similar[i] if (p := min(100, round(100 * c / ref))) >= SIM_FLOOR]})
    json.dump(nodes, open(args.out, "w"), separators=(",", ":"), ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + "_synonyms.json", "w") as f:  # label -> the other spellings grouped under it
        json.dump(kw.synonyms, f, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(nodes)} nodes, {sum(bool(n['institutions']) for n in nodes)} with institutions -> {args.out}")


if __name__ == "__main__":
    main()
