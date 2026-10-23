"""Author centrality in the 2026 co-authorship network, for the hidden Centrality page (not in the site menu).

Nodes are authors, matched on first + last name (network.author_key). Two authors are linked when they share an
abstract, and the link's weight is the number of abstracts they share. Four rankings, all over the same network:
degree (co-authors), strength (shared abstracts summed over co-authors), PageRank (weighted, damping 0.85) and
betweenness (unweighted shortest paths, estimated from BETWEENNESS_SOURCES sampled starting authors).

Affiliation strings, consortia and other non-person author entries are dropped before the graph is built, as are
names with one word. Each author gets a talk count (abstracts that are not posters), which decides whether the page
lists them: like everywhere on the site, nobody is named for having no talks. Ranks count every author in the graph.

Usage: python src/ashg/centrality.py outputs/2026_author_centrality.csv
"""

import argparse
import itertools
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

import network as N

MEASURES = ("degree", "strength", "pagerank", "betweenness")
BETWEENNESS_SOURCES = 1000  # sampled starting authors; the exact computation is slow for ~17k authors
NOT_PEOPLE = re.compile(r"universit|hospital|department|institut|center|centre|school|college|clinic|laborator|medicine|"
                        r"genetics|\bof\b|foundation|health|research|consortium|collaborat|group|study|project|"
                        r"investigators|team|;", re.I)


def build(d):
    """One row per author with their counts, rankings and scores. d is network.load_abstracts()."""
    ids, spelled, nabs, talks, edges = {}, {}, Counter(), Counter(), Counter()
    for ks, fulls, typ in zip(d.authors, d.full, d.type):
        pairs = {k: f for k, f in zip(ks, fulls) if k and " " in k and not NOT_PEOPLE.search(f)}
        keys = sorted(ids.setdefault(k, len(ids)) for k in pairs)
        for k, f in pairs.items():
            i = ids[k]
            spelled[i] = max(spelled.get(i, ""), f, key=len)
            nabs[i] += 1
            talks[i] += typ != "poster"
        for a, b in itertools.combinations(keys, 2):
            edges[(a, b)] += 1

    n = len(ids)
    r, c = zip(*edges)
    A = sparse.coo_matrix((list(edges.values()), (r, c)), shape=(n, n))
    A = (A + A.T).tocsr().astype(float)  # weight = shared abstracts
    B = (A > 0).astype(float)  # unweighted
    G = nx.from_scipy_sparse_array(A)
    df = pd.DataFrame({
        "author": [spelled[i] for i in range(n)],
        "abstracts": [nabs[i] for i in range(n)],
        "talks": [talks[i] for i in range(n)],
        "degree": np.asarray(B.sum(axis=1)).ravel(),
        "strength": np.asarray(A.sum(axis=1)).ravel(),
        "pagerank": pd.Series(nx.pagerank(G, weight="weight")).sort_index().to_numpy(),
        "betweenness": pd.Series(nx.betweenness_centrality(nx.from_scipy_sparse_array(B), k=BETWEENNESS_SOURCES,
                                                           seed=0)).sort_index().to_numpy(),
    })
    for m in MEASURES:
        df[f"{m}_rank"] = df[m].rank(ascending=False, method="min").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    df = build(N.load_abstracts())
    df.to_csv(args.out, index=False)
    print(f"{len(df)} authors, {int((df.talks > 0).sum())} with a talk -> {args.out}")


if __name__ == "__main__":
    main()
