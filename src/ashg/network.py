"""Similarity network of the 2026 abstracts, for the site's "Find talks" page.

Per abstract: its nearest neighbors by text (BM25-weighted cosine on title + body) and by shared authors (Jaccard on
the author sets), and a 2-D layout of all of them for the minimap. Titles and abstract text are used here but never
written out: the JSON holds only the first author, track, type, board, session time, coordinates and neighbor ids.

Poster tracks come from the session name ("Cancer Poster Friday Session"). Talk sessions have no track in the program,
so a talk gets the majority track of its nearest posters and is flagged as inferred.

Usage: python src/ashg/network.py outputs/network.json
"""

import argparse
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import floq
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

YEAR = 2026
TYPES = ("plenary", "platform", "poster")
TEXT_K, AUTHOR_K, SHOWN_K = 30, 15, 15  # stored text neighbors leave SHOWN_K after "hide past sessions" drops some
BM25_K1, BM25_B = 1.5, 0.75
TZ = ZoneInfo("America/Toronto")  # Montréal


def load_abstracts():
    """One row per abstract, in program order: parquet text and authors joined to the manual YAML's floq id and board."""
    raw = json.load(open("data/raw/floq_2026.json"))
    talks = [t for kids in raw["children"] for t in kids]
    times = {t["id"]: (t["start"], t["end"]) for t in talks}
    speakers = {t["id"]: [r["actor"] for r in json.loads(t["roles"] or "[]")] for t in talks}  # Floq person ids
    sessions = {s["name"]: (s["start"], s["end"]) for s in raw["sessions"]}
    rows = []
    for t in TYPES:
        df = pd.read_parquet(f"data/abstracts/{YEAR}_{t}.parquet")
        recs = yaml.safe_load(open(f"data/manual/{YEAR}_{t}.yaml"))["abstracts"]
        assert len(df) == len(recs)
        for r, rec in zip(df.itertuples(), recs):
            start, end = times.get(rec["floq_id"]) or sessions[r.session]
            names = [n for n, _ in floq.split_authors(r.authors_raw)]
            rows.append({"id": r.abstract_id, "first": names[0] if names else "", "authors": [author_key(n) for n in names],
                          "type": t, "session": r.session, "board": rec.get("board") or "",
                         "text": f"{r.title} {r.body}", "start": start, "end": end,
                         "speakers": speakers.get(rec["floq_id"], [])})
    return pd.DataFrame(rows)


def clean_name(name):
    """An author-block name without list numbering ("1. Ann Lee"), an "Authors:" label, affiliation fragments and
    footnote markers (digits, *, †, ‡, §) glued to the surname ("Lee1", "Lee*")."""
    name = re.sub(r"^\s*(?:authors?\s*:|\d+\s*[.)])\s*", "", name, flags=re.I)
    name = name.split("(")[0] if "(" in name else name.split(")")[-1]
    name = re.sub(r"[\d*†‡§#+¹²³]+", "", name).strip(" ,;.\n")
    return name if len(name) <= 40 else ""  # a whole unparsed author block, not a name


def author_key(name):
    """First + last name, lowercased, so "Yash K. Pershad" and "Yash Pershad" match (people.py only resolves senior authors)."""
    parts = clean_name(name).lower().split()
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else " ".join(parts)


def bm25_vectors(texts):
    """L2-normalized BM25 document vectors; their dot product is a symmetric BM25-style similarity."""
    tf = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.4,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]+\b").fit_transform(texts).tocsr().astype(float)
    n = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    length = np.asarray(tf.sum(axis=1)).ravel()
    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / length.mean())
    tf = tf.tocoo()
    tf.data = tf.data * (BM25_K1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
    return normalize(tf.tocsr())


def top_neighbors(sim, k):
    """Per row, the k most similar other rows with a positive score: [[(j, score), ...], ...]."""
    out = []
    for i in range(sim.shape[0]):
        row = np.asarray(sim[i].todense()).ravel() if sparse.issparse(sim) else sim[i].copy()
        row[i] = 0
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.argsort(-row[idx])]
        out.append([(int(j), float(row[j])) for j in idx if row[j] > 0])
    return out


def jaccard(keys):
    """Sparse Jaccard matrix of the abstracts' author sets."""
    vocab = {}
    ids = [sorted({vocab.setdefault(k, len(vocab)) for k in ks if k}) for ks in keys]
    n = len(ids)
    m = sparse.csr_matrix((np.ones(sum(map(len, ids))), (np.repeat(np.arange(n), list(map(len, ids))),
                                                         [p for s in ids for p in s])))
    m.data[:] = 1
    inter = (m @ m.T).tocoo()
    size = np.asarray(m.sum(axis=1)).ravel()
    keep = inter.row != inter.col
    r, c, v = inter.row[keep], inter.col[keep], inter.data[keep]
    return sparse.csr_matrix((v / (size[r] + size[c] - v), (r, c)), shape=(n, n))


def short_name(name):
    """"Alexander G Bick" -> "Bick, A."; names without a space (consortia) stay as they are."""
    name = name if isinstance(name, str) else ""  # abstracts the people table has no author for come through as NaN
    parts = clean_name(name).split()
    return f"{parts[-1]}, {parts[0][0]}." if len(parts) > 1 else name


def ts(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    d = load_abstracts()
    n = len(d)

    vec = bm25_vectors(d.text)
    text_nb = top_neighbors(vec @ vec.T, TEXT_K)
    ref = np.percentile([nb[0][1] for nb in text_nb if nb], 95)  # cosines are small; 100 = a very close pair
    author_nb = top_neighbors(jaccard(list(d.authors)), AUTHOR_K)

    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
        TruncatedSVD(50, random_state=0).fit_transform(vec))
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))

    d["track"] = d.session.str.replace(r" Poster \w+ Session$", "", regex=True).where(d.type == "poster")
    poster = d.track.notna().to_numpy()
    for i in np.flatnonzero(~poster):  # talks: majority track among their ten nearest posters
        votes = [d.track[j] for j, _ in top_neighbors_to(vec, i, poster)]
        d.loc[i, "track"] = pd.Series(votes).mode().iloc[0]
    tracks = sorted(d.track.unique())

    local = lambda iso: datetime.fromtimestamp(ts(iso), TZ)
    nodes = []
    for i, r in enumerate(d.itertuples()):
        t0 = local(r.start)
        nodes.append({
            "a": short_name(r.first), "k": tracks.index(r.track), "t": r.type,
            "b": r.board or None, "p": r.speakers, "s": ts(r.start), "e": ts(r.end), "w": f"{t0:%a} {t0:%-I:%M %p}",
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "nt": [[j, min(100, round(100 * s / ref))] for j, s in text_nb[i]],
            "na": [[j, round(100 * s)] for j, s in author_nb[i]],
        })
        if r.type != "poster":
            nodes[-1]["inf"] = 1  # track inferred from the nearest posters
    json.dump({"tracks": tracks, "shown": SHOWN_K, "nodes": nodes}, open(args.out, "w"), separators=(",", ":"),
              ensure_ascii=False)
    print(f"{n} abstracts, {sum(bool(x) for x in author_nb)} with a co-author link -> {args.out}")


def top_neighbors_to(vec, i, mask, k=10):
    """The k nearest rows of `vec` to row i among rows where mask is true."""
    s = (vec @ vec[i].T).toarray().ravel()
    s[~mask] = -1
    idx = np.argsort(-s)[:k]
    return [(int(j), float(s[j])) for j in idx]


if __name__ == "__main__":
    main()
