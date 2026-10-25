"""Input file for the site's Landscape page: one node per abstract from 2022 to 2026, all on one shared 2-D map.

Every abstract is embedded with SPECTER2 (title [SEP] abstract, CLS pooling) and projected with UMAP over the whole
corpus, so a position means the same thing in every year and an author's abstracts from different years can be
joined into a trajectory. Each node lists its year, its authors in order with their position (first, middle, last), the
institutions the institutions table found for the first and senior author, and keywords: the synonym groups of its
highest-weighted BM25 terms (src/ashg/synonyms.py), definition groups first, with the spellings in each group written to
<out>_synonyms.json. "similar" holds its SIMILAR nearest abstracts as [id, match %], ranked by raw BM25 cosine and scored
on the same 0-100 scale as Find talks (95th percentile of best-match cosines = 100); matches below SIM_FLOOR (10%) are
dropped. The BM25 vectors are kept for the keywords and the similar web (see the comment in main()).
Titles and abstract text are never written out (same rule as network.json), so "title" is left out; the page paints
keywords from the terms.

Authors are merged on person_key (first name + surname, lowercased, with apostrophes and hyphens dropped and initials
left out of the surname) across all years and shown under the longest spelling seen, so one person's abstracts end up
under a single autocomplete entry. Initial-only names from the older author blocks ("E. Kun") are merged into the one full
name with the same surname and initials when that is unique (link_initials), which is a heuristic and can join two people.

Usage: python src/ashg/landscape.py outputs/tsne_landscape.json
"""

import argparse
import json
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conda's libomp and torch's clash on macOS; must precede the torch import

import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer, models

import floq
from network import SIM_FLOOR, TYPES, YEAR, author_key, bm25_vectors, clean_name, load_abstracts, top_neighbors
from synonyms import Keywords

YEARS = range(2022, 2027)
MODEL = "allenai/specter2_base"  # SPECTER2 base: scientific-paper embeddings trained on title [SEP] abstract
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
    name = re.sub(r"(?<![A-Za-z])([A-Za-z])\.(?=[A-Za-z])", r"\1. ", name)  # "M.A. Pericak" -> "M. A. Pericak"
    toks = name.split()
    while toks and bare(toks[-1]) in DEGREES:
        toks.pop()
    while toks and bare(toks[0]).lower() in TITLES:
        toks.pop(0)
    if all(bare(t) in DEGREES or bare(t).lower() in FILLER for t in toks):
        return None
    return " ".join(toks)


def person_key(name):
    """First name + surname, lowercased, with apostrophes, hyphens and spaces dropped from the surname and initials left out, so
    "Anne O'Donnell-Luria", "Anne O’Donnell Luria" and "Anne ODonnell-Luria" are one person. Middle names are dropped too
    when they are initials ("Yash K. Pershad"), but a spelled-out middle name is kept ("Mary Ann Lee" != "Mary Lee")."""
    sq = lambda s: re.sub(r"['’‘\-‐‑]", "", s).lower()  # ASCII and Unicode hyphens (U+2010, U+2011)
    parts = name.split()
    if len(parts) < 2:
        return sq(name)
    tail = [t for t in parts[1:] if not re.fullmatch(r"[A-Za-z]\.?", t)] or parts[-1:]
    return f"{sq(parts[0])} {re.sub(r' ', '', sq(' '.join(tail)))}"


def people(fulls):
    """[(person_key, name)] for the real names in an author list, in order."""
    out = []
    for full in fulls:
        name = tidy(full)
        if name and (key := person_key(name)):
            out.append((key, name))
    return out


def given_initials(name):
    """The initial of each given name, in order: "Benjamin M. Neale" and "B. M. Neale" are both "bm", "Ben Neale" is "b"."""
    return "".join(t[0].lower() for t in name.split()[:-1])


def link_initials(listed, institutions):
    """Merge initial-only keys ("m. daly", from 2022-2024 author blocks) into the full-name key with the same surname and
    initials ("mark daly"; "b. m. neale" into "benjamin m. neale"). A full name matches when its given-name initials agree
    with the initial-only name's, and the full name has at least as many: a match on every initial beats a match on the
    first only ("Ben Neale" does not take "B. M. Neale" when "Benjamin M. Neale" exists). A unique match is taken as is;
    when several share the surname and initials, the one whose abstracts share an institution with the initial-only
    name's abstracts is taken, if exactly one does. Otherwise the initial stays separate. institutions maps person_key ->
    the institutions of its abstracts. This is a heuristic: a match can still be a different person with the same
    surname and initials, or at the same institution."""
    keys = {k for ps in listed for k, _ in ps}
    seqs = {}  # person_key -> the given-name initials of each spelling seen under it
    for ps in listed:
        for k, name in ps:
            seqs.setdefault(k, set()).add(given_initials(name))
    full = {}  # surname -> full-name keys
    for k in keys:
        first, _, sur = k.partition(" ")
        if sur and len(first.rstrip(".")) > 1:
            full.setdefault(sur, set()).add(k)
    canon, by_institution = {}, 0
    for k, want in {(k, given_initials(n)) for ps in listed for k, n in ps}:
        first, _, sur = k.partition(" ")
        if sur and len(first.rstrip(".")) == 1:  # keyed on the spelling's own initials, so "M. Hill" and "M. S. Hill" can differ
            agree = lambda s, n: s[:n] == want[:n]
            cands = [j for j in full.get(sur, ()) if any(agree(s, min(len(s), len(want))) for s in seqs[j])]
            strong = [j for j in cands if any(len(s) >= len(want) and agree(s, len(want)) for s in seqs[j])]
            cands = strong or cands
            if len(cands) > 1:
                shared = [c for c in cands if institutions.get(c, set()) & institutions.get(k, set())]
                if len(shared) == 1:
                    cands, by_institution = shared, by_institution + 1
            if len(cands) == 1:
                canon[(k, want)] = cands[0]
    # an unmatched initial-only spelling joins a longer one that matched ("B. Neale" -> "B. M. Neale")
    for k, want in {(k, given_initials(n)) for ps in listed for k, n in ps}:
        first, _, sur = k.partition(" ")
        if (k, want) in canon or not (sur and len(first.rstrip(".")) == 1):
            continue
        sibs = {canon[(k2, w)] for (k2, w) in canon if k2 == k and len(w) > len(want) and w.startswith(want)}
        if len(sibs) == 1 and institutions.get(sibs.copy().pop(), set()) & institutions.get(k, set()):
            canon[(k, want)] = sibs.pop()

    def relabel(k, name):
        seq = given_initials(name)
        if (k, seq) in canon:
            return canon[(k, seq)]
        first, _, sur = k.partition(" ")
        if sur and len(first.rstrip(".")) == 1 and len(seq) > 1:  # unmatched: "B. M. Neale" must not share a key with "B. N. Neale"
            return " ".join(f"{c}." for c in seq) + f" {sur}"
        return k
    print(f"{by_institution} of them by shared institution")
    return [[(relabel(k, n), n) for k, n in ps] for ps in listed], len(canon)


def positions(n):
    return ["first"] + ["middle"] * (n - 2) + ["last"] if n > 1 else ["first"]


def author_names(raw, year):
    """The names in an author block. Through 2024 the block is "Name, Name; Affiliation, Affiliation", so the affiliations
    after the first semicolon outside parentheses are cut off; from 2025 the affiliations are in parentheses and semicolons
    also separate authors, so the whole block is parsed."""
    if year <= 2024:
        depth = 0
        for j, ch in enumerate(raw):
            depth += (ch == "(") - (ch == ")")
            if ch == ";" and depth == 0:
                raw = raw[:j]
                break
    # Superscript footnotes written as "2,3" split into bare numbers ("3"), which are not names
    return [n for n, _ in floq.split_authors(raw) if re.search(r"[A-Za-z]", n)]


def load_years():
    """One row per abstract for every year in YEARS, plus the 2026 symposium and session nodes. The abstract rows are read
    from each year's parquet (load_abstracts() only reads 2026); the same author parsing as load_abstracts() is used, so
    the years line up. "title" and "body" are None for the program rows, whose text is the program description."""
    rows = []
    for y in YEARS:
        for t in TYPES:
            for r in pd.read_parquet(f"data/abstracts/{y}_{t}.parquet").itertuples():
                names = author_names(r.authors_raw, y)
                rows.append({"id": r.abstract_id, "year": y, "type": t, "authors": [author_key(n) for n in names],
                             "full": [clean_name(n) for n in names], "title": r.title, "body": r.body,
                             "text": f"{r.title} {r.body}"})
    program = load_abstracts()
    program = program[~program.type.isin(TYPES)]  # the abstract rows above already cover them
    for r in program.itertuples():
        rows.append({"id": r.id, "year": YEAR, "type": r.type, "authors": r.authors, "full": r.full,
                     "title": None, "body": None, "text": r.text})
    return pd.DataFrame(rows)


def embed(d):
    """Unit-length SPECTER2 vectors: [CLS] of title [SEP] abstract, or of the program text for the symposium and session
    nodes. A bare HF checkpoint gets mean pooling from sentence-transformers, but SPECTER2 is trained on CLS, so the
    pooling is set explicitly."""
    word = models.Transformer(MODEL, max_seq_length=512)
    model = SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")])
    sep = model.tokenizer.sep_token
    texts = [f"{t}{sep}{b}" if isinstance(t, str) else x for t, b, x in zip(d.title, d.body, d.text)]
    return model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    d = load_years()
    vec, vocab = bm25_vectors(d.text)
    kw = Keywords(vec, vocab, d.text, KEYWORDS)

    # Similar web: BM25 cosine, not the dense SPECTER2 vectors, because the 0-100 match scale and SIM_FLOOR are calibrated
    # on BM25 in network.py (Find talks) and the page reads them as web strength. Dense neighbors were not evaluated here.
    similar = top_neighbors(vec @ vec.T, SIMILAR)
    ref = np.percentile([nb[0][1] for nb in similar if nb], 95)  # as network.py's match scale

    print("Generating SPECTER2 embeddings...")
    dense = embed(d)
    print("Running UMAP...")
    xy = umap.UMAP(n_components=2, metric="cosine", random_state=0).fit_transform(dense)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))  # as network.py's layout scaling
    xy = SCALE * (2 * xy - 1)

    insts = pd.concat([pd.read_parquet(f"data/institutions/{y}.parquet") for y in YEARS]).set_index("abstract_id")
    people_insts = {}  # person_key -> institutions of their abstracts, for linking initials
    for aid, fulls in zip(d.id, d.full):
        if aid in insts.index:
            row = insts.loc[aid]
            found = set(row.first_insts) | set(row.senior_insts)
            for k, _ in people(fulls):
                people_insts.setdefault(k, set()).update(found)
    listed, linked = link_initials([people(fulls) for fulls in d.full], people_insts)
    print(f"{linked} initial-only author keys merged into a full name")
    spelled = {}  # person_key -> longest spelling, across all years
    for ps in listed:
        for k, name in ps:
            spelled[k] = max(spelled.get(k, ""), name, key=len)

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
        nodes.append({"id": i + 1, "year": int(r.year), "x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2),
                      "type": r.type, "authors": authors, "institutions": found, "keywords": kw.of(i),
                      "similar": [[j + 1, p] for j, c in similar[i] if (p := min(100, round(100 * c / ref))) >= SIM_FLOOR]})
    json.dump(nodes, open(args.out, "w"), separators=(",", ":"), ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + "_synonyms.json", "w") as f:  # label -> the other spellings grouped under it
        json.dump(kw.synonyms, f, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(nodes)} nodes, {sum(bool(n['institutions']) for n in nodes)} with institutions -> {args.out}")


if __name__ == "__main__":
    main()
