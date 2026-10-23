"""Similarity network of the 2026 abstracts, for the site's "Find talks" page.

Per abstract: its nearest neighbors by text (BM25-weighted cosine on title + body) and by shared authors (Jaccard on
the author sets), and a 2-D layout of all of them for the minimap. Titles and abstract text are used here but never
written out: the JSON holds only the first author, track, type, board, session time, coordinates and neighbor ids, plus
"pp", the DOI of a high-confidence preprint match (see preprint_dois).

Besides the abstracts, the featured symposia (one node per talk, speaker as author) and the other CME sessions the program
describes only as a whole (workshops, short courses, symposia without per-talk entries) are nodes too; their text is the
program description. Speaker names come from data/raw/floq_2026_people.json (`floq.py people`). Lightning talks
(new in 2026) are not nodes: each one is a poster's abstract, so the poster just gets its lightning-talk time.

Poster tracks come from the session name ("Cancer Poster Friday Session"). Talk sessions have no track in the program,
so a talk gets the majority track of its nearest posters and is flagged as inferred.

Usage: python src/ashg/network.py outputs/network.json
"""

import argparse
import gzip
import json
import os
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
PEOPLE = f"data/raw/floq_{YEAR}_people.json"
TEXT_K, AUTHOR_K, SHOWN_K = 30, 15, 15  # stored text neighbors leave SHOWN_K after "hide past sessions" drops some
SIM_FLOOR = 10  # matches below this are stored as 0 in the pairwise matrix
BM25_K1, BM25_B = 1.5, 0.75
TZ = ZoneInfo("America/Toronto")  # Montréal
PREPRINTS = "data/preprints/matches.csv"  # from preprints.py analyze
PREPRINT_TIERS = ("exact", "A")  # tier B has not been hand-checked, so it is not shown


def norm_title(title):
    return re.sub(r"\W+", " ", title.lower()).strip()


def program_rows(raw, speakers):
    """Nodes for talks and sessions with no author block: symposium talks (one per talk with a description), and
    sessions that carry CME credit but no abstracts or talk entries. A multi-part short course is one node (part 1)."""
    names = json.load(open(PEOPLE)) if os.path.exists(PEOPLE) else {}
    if not names:
        print(f"warning: {PEOPLE} missing, symposium speakers stay unnamed (python src/ashg/floq.py people ...)")
    who = lambda item: [n for n in (names.get(a, "") for a in floq.role_ids(item, "role:speaker")) if n]
    rows, seen = [], set()
    for s, kids in zip(raw["sessions"], raw["children"]):
        if "Poster" in s["name"] or any("Author Block" in (t["info"] or "") for t in kids):
            continue
        talks = [t for t in kids if len(t["info"] or "") > 80 and not t.get("deleted")]
        if not talks and "CME" in (s["info"] or ""):
            base = re.sub(r"\s*\(Part \d of \d\)", "", s["name"])
            if base not in seen:
                seen.add(base)
                rows.append(program_row(s, s, "session", who(s), speakers))
        rows += [program_row(s, t, "symposium", who(t), speakers) for t in talks]
    return rows


def program_row(session, item, type_, speaker_names, speakers):
    """A session-level node with no named speaker is labelled by the session name."""
    label = speaker_names or ([session["name"]] if type_ == "session" else [""])
    return {"id": f"{YEAR}-{type_}-{item['id']}", "first": label[0], "authors": [author_key(n) for n in speaker_names],
            "full": [clean_name(n) for n in speaker_names], "type": type_, "session": session["name"], "board": "",
            "text": f"{item['name']} {text_of(item['info'])}", "start": item["start"], "end": item["end"],
            "speakers": speakers.get(item["id"], []), "url": item["id"], "lightning": None}


def text_of(info):
    return floq.text(re.sub(r"<b>(Learning Objectives|CME Credits)</b>.*", "", info or "", flags=re.S))


def load_abstracts():
    """One row per abstract, in program order: parquet text and authors joined to the manual YAML's floq id and board."""
    raw = json.load(open("data/raw/floq_2026.json"))
    talks = [t for kids in raw["children"] for t in kids]
    times = {t["id"]: (t["start"], t["end"]) for t in talks}
    speakers = {t["id"]: floq.role_ids(t) for t in talks + raw["sessions"]}  # Floq person ids
    lightning = {norm_title(re.sub(r"^Board \S+:\s*", "", t["name"])): t["start"]
                 for s, kids in zip(raw["sessions"], raw["children"]) if "Poster" not in s["name"] and len(kids) > 10
                 for t in kids if "Author Block" in (t["info"] or "")}
    sessions = {s["name"]: (s["start"], s["end"]) for s in raw["sessions"]}
    rows = []
    for t in TYPES:
        df = pd.read_parquet(f"data/abstracts/{YEAR}_{t}.parquet")
        recs = yaml.safe_load(open(f"data/manual/{YEAR}_{t}.yaml"))["abstracts"]
        assert len(df) == len(recs)
        for r, rec in zip(df.itertuples(), recs):
            start, end = times.get(rec["floq_id"]) or sessions[r.session]
            names = [n for n, _ in floq.split_authors(r.authors_raw)]
            rows.append({"id": r.abstract_id, "first": names[0] if names else "", "authors": [author_key(n) for n in names], "full": [clean_name(n) for n in names],
                          "type": t, "session": r.session, "board": rec.get("board") or "",
                         "text": f"{r.title} {r.body}", "start": start, "end": end,
                         "speakers": speakers.get(rec["floq_id"], []), "url": None,
                         "lightning": lightning.get(norm_title(r.title)) if t == "poster" else None})
    return pd.DataFrame(rows + program_rows(raw, speakers))


def preprint_dois():
    """abstract_id -> DOI of the matched bioRxiv/medRxiv preprint, for the high-confidence tiers only. Skipped (no
    buttons) if the matches file is missing."""
    if not os.path.exists(PREPRINTS):
        print(f"note: {PREPRINTS} missing, no preprint buttons")
        return {}
    m = pd.read_csv(PREPRINTS)
    m = m[m.tier.isin(PREPRINT_TIERS)]
    return dict(zip(m.abstract_id, m.doi))


def clean_name(name):
    """An author-block name without list numbering ("1. Ann Lee"), an "Authors:" label, affiliation fragments and
    footnote markers (digits, *, †, ‡, §) glued to the surname ("Lee1", "Lee*")."""
    name = re.sub(r"^\s*(?:authors?(?:\s+block)?\s*:|\d+\s*[.)])\s*", "", name, flags=re.I)
    name = name.split("(")[0] if "(" in name else name.split(")")[-1]
    name = re.sub(r"[\d*†‡§#+¹²³]+", "", name).strip(" ,;.\n")
    return name if len(name) <= 40 else ""  # a whole unparsed author block, not a name


def author_key(name):
    """First + last name, lowercased, so "Yash K. Pershad" and "Yash Pershad" match (people.py only resolves senior authors)."""
    parts = clean_name(name).lower().split()
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else " ".join(parts)


def bm25_vectors(texts):
    """L2-normalized BM25 document vectors; their dot product is a symmetric BM25-style similarity."""
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.4,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]+\b")
    tf = cv.fit_transform(texts).tocsr().astype(float)
    n = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    length = np.asarray(tf.sum(axis=1)).ravel()
    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / length.mean())
    tf = tf.tocoo()
    tf.data = tf.data * (BM25_K1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
    return normalize(tf.tocsr()), cv.get_feature_names_out()


def term_vectors(vec, vocab):
    """Every abstract's terms as flat [term id, weight in thousandths, ...] rows, and the terms the ids point to. The
    weights are the L2-normalized BM25 ones, so two abstracts' score is the sum over shared terms of weight * weight."""
    used, rows = {}, []
    for i in range(vec.shape[0]):
        row = vec.getrow(i)
        rows.append([x for c, w in zip(row.indices, row.data) if round(1000 * w) for x in (used.setdefault(int(c), len(used)), int(round(1000 * w)))])
    terms = [""] * len(used)
    for col, new in used.items():
        terms[new] = str(vocab[col])
    return rows, terms


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

    vec, vocab = bm25_vectors(d.text)
    term_rows, terms = term_vectors(vec, vocab)
    with gzip.open(os.path.splitext(args.out)[0] + "_terms.json.gz", "wt", compresslevel=9) as f:  # only the compare page fetches it
        json.dump({"terms": terms, "rows": term_rows}, f, separators=(",", ":"))
    cos = vec @ vec.T
    text_nb = top_neighbors(cos, TEXT_K)
    ref = np.percentile([nb[0][1] for nb in text_nb if nb], 95)  # cosines are small; 100 = a very close pair
    scaled = np.minimum(100, np.rint(100 * cos.toarray() / ref)).astype(np.uint8)  # same scale as "nt"
    scaled[scaled < SIM_FLOOR] = 0  # weak pairs read as "under SIM_FLOOR%"; zeros are what make the file compress to ~2 MB
    with gzip.open(os.path.splitext(args.out)[0] + "_sim.bin.gz", "wb", compresslevel=9) as f:
        f.write(scaled.tobytes())  # n*n bytes, row-major: any pair's match, for center-to-node scores at degrees 2-3
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

    people = {}  # author_key -> id, so each node can list its authors compactly ("au") for the page's hide-co-authorship box ("fa" is the first author)
    authors = [sorted({people.setdefault(k, len(people)) for k in ks if k}) for ks in d.authors]
    spelled, count = {}, {}  # names for the compare page: only authors on 2+ abstracts can be shared, so only they get one
    for ks, fulls in zip(d.authors, d.full):
        for k, full in set(zip(ks, fulls)):
            if k:
                count[people[k]] = count.get(people[k], 0) + 1
                spelled[people[k]] = max(spelled.get(people[k], ""), full, key=len)
    names = {str(i): spelled[i] for i, c in count.items() if c > 1}

    local = lambda iso: datetime.fromtimestamp(ts(iso), TZ)
    preprints = preprint_dois()
    nodes = []
    for i, r in enumerate(d.itertuples()):
        t0 = local(r.start)
        nodes.append({
            "a": short_name(r.first), "k": tracks.index(r.track), "t": r.type,
            "b": r.board or None, "p": r.speakers, "s": ts(r.start), "e": ts(r.end), "w": f"{t0:%a} {t0:%-I:%M %p}",
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "nt": [[j, min(100, round(100 * s / ref))] for j, s in text_nb[i]],
            "na": [[j, round(100 * s)] for j, s in author_nb[i]], "au": authors[i],
            **({"fa": people[d.authors[i][0]]} if d.authors[i] and d.authors[i][0] else {}),
        })
        full = clean_name(r.first) if isinstance(r.first, str) else ""
        if len(full.split()) > 1:
            nodes[-1]["f"] = full  # first author's full name, shown in the result list (the graph keeps "Surname, I.")
        if r.type != "poster":
            nodes[-1]["inf"] = 1  # track inferred from the nearest posters
        if isinstance(r.url, str):  # NaN for abstracts
            nodes[-1]["u"] = r.url  # symposium talks and sessions link to their own program entry
            nodes[-1]["n"] = r.session
        if r.id in preprints:
            nodes[-1]["pp"] = preprints[r.id]
        if isinstance(r.lightning, str):
            t1 = local(r.lightning)
            nodes[-1]["lw"] = f"{t1:%a} {t1:%-I:%M %p}"
    json.dump({"tracks": tracks, "shown": SHOWN_K, "nodes": nodes, "names": names, "co": co_authors(d)}, open(args.out, "w"),
              separators=(",", ":"), ensure_ascii=False)
    print(f"{n} abstracts, {sum(bool(x) for x in author_nb)} with a co-author link -> {args.out}")


def co_authors(d):
    """Every non-first author as [name, [node ids of the abstracts they are on], n_last], so the page can search
    co-authors. The ids list the abstracts where they are last author first (n_last of them), then the rest.
    People are merged on author_key; the name shown is the longest spelling seen."""
    people = {}
    last = {}
    for i, r in enumerate(d.itertuples()):
        for key, full in list(zip(r.authors, r.full))[1:]:
            if not key:
                continue
            name, ids = people.setdefault(key, ["", []])
            people[key][0] = max(name, full, key=len)
            if i not in ids:
                ids.append(i)
                if key == r.authors[-1]:
                    last.setdefault(key, []).append(i)
    out = []
    for key, (name, ids) in people.items():
        lead = last.get(key, [])
        out.append([name, lead + [i for i in ids if i not in lead], len(lead)])
    return sorted(out, key=lambda p: p[0].lower())


def top_neighbors_to(vec, i, mask, k=10):
    """The k nearest rows of `vec` to row i among rows where mask is true."""
    s = (vec @ vec[i].T).toarray().ravel()
    s[~mask] = -1
    idx = np.argsort(-s)[:k]
    return [(int(j), float(s[j])) for j in idx]


if __name__ == "__main__":
    main()
