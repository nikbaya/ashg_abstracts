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
under a single autocomplete entry. Initial-only names from the older author blocks ("E. Kun", "P-R. Loh") are linked by
surname and initials, including initials for each part of a hyphenated given name; when author-specific affiliation
evidence conflicts with a candidate full name, that occurrence stays separate (link_initials).

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
from institutions import load_aliases, match_institutes
from network import SIM_FLOOR, TYPES, YEAR, author_key, bm25_vectors, clean_name, load_abstracts, top_neighbors
from synonyms import Keywords

YEARS = range(2022, 2027)
MODEL = "allenai/specter2_base"  # SPECTER2 base: scientific-paper embeddings trained on title [SEP] abstract
KEYWORDS = 25  # paint-by-keyword only finds an abstract through these, so a topic ranked 10th still counts
SIMILAR = 15  # text neighbors per node, for the page's similar-abstracts web
SCALE = 100  # coordinates run -SCALE..SCALE
CHITINASE_CLUSTER = {
    (2023, "t. kanaizumi"), (2023, "k. suzuki"), (2023, "k. takahashi"),
    (2024, "f. oyama"), (2024, "k. suzuki"),
    (2025, "fumitaka oyama"), (2025, "keita suzuki"),
}
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
    when they are initials ("Yash K. Pershad"), but a spelled-out middle name is kept ("Mary Ann Lee" != "Mary Lee").
    When the first name is only an initial, later given names are treated as middle names ("F. Kyle Satterstrom")."""
    sq = lambda s: re.sub(r"['’‘\-‐‑]", "", s).lower()  # ASCII and Unicode hyphens (U+2010, U+2011)
    parts = name.split()
    if len(parts) < 2:
        return sq(name)
    first = sq(parts[0]).rstrip(".")
    if len(first) == 1:
        surname = parts[-1]
    else:
        tail = [t for t in parts[1:] if not re.fullmatch(r"[A-Za-z]\.?", t)] or parts[-1:]
        surname = " ".join(tail)
    return f"{first} {re.sub(r' ', '', sq(surname))}"


def people(fulls):
    """[(person_key, name)] for the real names in an author list, in order."""
    out = []
    for full in fulls:
        name = tidy(full)
        if name and (key := person_key(name)):
            out.append((key, name))
    return out


def given_initials(name):
    """The initials of each given-name part, including hyphenated names: "P-R. Loh" and "Po-Ru Loh" are both "pr"."""
    return "".join(part[0].lower() for token in name.split()[:-1] for part in re.split(r"[-‐‑]", token) if part)


def initial_only(name):
    """Whether the first given-name token contains only initials, optionally hyphenated ("P-R. Loh")."""
    parts = re.split(r"[-‐‑]", name.split()[0].rstrip("."))
    return all(len(part) == 1 and part.isalpha() for part in parts)


def link_initials(listed, institutions, abstract_ids=None, author_institutions=None):
    """Merge initial-only keys ("m. daly", from 2022-2024 author blocks) into the full-name key with the same surname and
    initials ("mark daly"; "b. m. neale" into "benjamin m. neale"; "P-R. Loh" into "Po-Ru Loh"). Hyphen-separated
    given-name parts each contribute an initial. A full name matches when its given-name initials agree with the
    initial-only name's, and the full name has at least as many: a match on every initial beats a match on the first only
    ("Ben Neale" does not take "B. M. Neale" when "Benjamin M. Neale" exists). A unique match is taken as is;
    when several share the surname and initials, the one whose abstracts share an institution with the initial-only
    name's abstracts is taken, if exactly one does. If an abstract's own listed affiliations conflict with the candidate's,
    that occurrence stays under the initials instead of being merged. When author-specific affiliation evidence is absent,
    the existing unique-match heuristic is retained."""
    seqs = {}  # person_key -> the given-name initials of each spelling seen under it
    for ps in listed:
        for k, name in ps:
            seqs.setdefault(k, set()).add(given_initials(name))
    full = {}  # surname -> full-name keys
    for ps in listed:
        for k, name in ps:
            _, _, sur = k.partition(" ")
            if sur and not initial_only(name):
                full.setdefault(sur, set()).add(k)
    canon, by_institution = {}, 0
    initials = {(k, given_initials(name)) for ps in listed for k, name in ps if initial_only(name)}
    for k, want in initials:
        _, _, sur = k.partition(" ")
        if sur:
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
    for k, want in initials:
        _, _, sur = k.partition(" ")
        if (k, want) in canon or not sur:
            continue
        sibs = {canon[(k2, w)] for (k2, w) in canon if k2 == k and len(w) > len(want) and w.startswith(want)}
        if len(sibs) == 1 and institutions.get(sibs.copy().pop(), set()) & institutions.get(k, set()):
            canon[(k, want)] = sibs.pop()

    def relabel(k, name):
        seq = given_initials(name)
        if (k, seq) in canon:
            return canon[(k, seq)]
        _, _, sur = k.partition(" ")
        if sur and initial_only(name) and len(seq) > 1:  # unmatched: "B. M. Neale" must not share a key with "B. N. Neale"
            return " ".join(f"{c}." for c in seq) + f" {sur}"
        return k
    linked = 0
    conflicts = []
    resolved = []
    for i, ps in enumerate(listed):
        abstract_id = abstract_ids[i] if abstract_ids is not None else None
        found_by_author = (author_institutions or {}).get(abstract_id, {})
        row = []
        for k, name in ps:
            target = "emilie wigdor" if k == "e wigdor" else relabel(k, name)
            candidate_insts = institutions.get(target, set())
            found = found_by_author.get(k, set())
            if k != "e wigdor" and target != k and found and candidate_insts and not found.intersection(candidate_insts):
                conflicts.append((i, k, name, target))
                target = k
            if target != k:
                linked += 1
            row.append((target, name))
        resolved.append(row)
    print(f"{by_institution} initial-only keys disambiguated by shared institution")
    return resolved, linked, conflicts


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


def author_institutions(raw, aliases):
    """Map each author key to institutions from their numbered author-block affiliations."""
    authors_raw, separator, affiliations_raw = raw.partition(";")
    if not separator:
        result = {}
        for name, affils in floq.split_authors(raw):
            key = person_key(clean_name(name))
            if key and affils:
                result.setdefault(key, set()).update(match_institutes({"affils": [affils]}, aliases))
        return result

    affiliations = {}
    markers = list(re.finditer(r"(?:^\s*|,\s*)(\d+)(?=\s|[A-Za-z])\s*", affiliations_raw))
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(affiliations_raw)
        text = affiliations_raw[marker.end():end].strip(" ,;")
        affiliations[marker.group(1)] = set(match_institutes({"affils": [text]}, aliases))

    author_affiliations = {}
    chunks = authors_raw.split(",")
    for chunk in chunks:
        refs = re.search(r"((?:\d+\s*,\s*)+\d+|\d+)\s*$", chunk)
        name = clean_name(chunk)
        if not name or not re.search(r"[A-Za-z]", name):
            if refs and author_affiliations:
                key = next(reversed(author_affiliations))
                author_affiliations[key].update(
                    inst for ref in re.findall(r"\d+", refs.group(1)) for inst in affiliations.get(ref, set())
                )
            continue
        key = person_key(name)
        found = set()
        if refs:
            found.update(inst for ref in re.findall(r"\d+", refs.group(1)) for inst in affiliations.get(ref, set()))
        author_affiliations.setdefault(key, set()).update(found)
    return author_affiliations


def load_years():
    """One row per abstract for every year in YEARS, plus the 2026 symposium and session nodes. The abstract rows are read
    from each year's parquet (load_abstracts() only reads 2026); the same author parsing as load_abstracts() is used, so
    the years line up. "title" and "body" are None for the program rows, whose text is the program description."""
    rows = []
    for y in YEARS:
        for t in TYPES:
            for r in pd.read_parquet(f"data/abstracts/{y}_{t}.parquet").itertuples():
                names = author_names(r.authors_raw, y)
                rows.append({"id": r.abstract_id, "year": y, "type": t, "authors_raw": r.authors_raw,
                             "authors": [author_key(n) for n in names],
                             "full": [clean_name(n) for n in names], "title": r.title, "body": r.body,
                             "text": f"{r.title} {r.body}"})
    program = load_abstracts()
    program = program[~program.type.isin(TYPES)]  # the abstract rows above already cover them
    for r in program.itertuples():
        rows.append({"id": r.id, "year": YEAR, "type": r.type, "authors_raw": None,
                     "authors": r.authors, "full": r.full,
                     "title": None, "body": None, "text": r.text})
    return pd.DataFrame(rows)


def author_institution_evidence(d, insts, aliases):
    """Return institution evidence keyed to each author, not every coauthor on the abstract."""
    by_person = {}
    by_abstract = {aid: author_institutions(raw, aliases)
                   for aid, raw in zip(d.id, d.authors_raw) if isinstance(raw, str)}
    for aid, fulls in zip(d.id, d.full):
        for key, _ in people(fulls):
            by_person.setdefault(key, set()).update(by_abstract.get(aid, {}).get(key, set()))
        if aid not in insts.index:
            continue
        row = insts.loc[aid]
        first = person_key(clean_name(row.first_author)) if isinstance(row.first_author, str) else ""
        senior = person_key(clean_name(row.senior_author)) if isinstance(row.senior_author, str) else ""
        if first:
            by_person.setdefault(first, set()).update(row.first_insts)
            by_abstract.setdefault(aid, {}).setdefault(first, set()).update(row.first_insts)
        if senior:
            by_person.setdefault(senior, set()).update(row.senior_insts)
            by_abstract.setdefault(aid, {}).setdefault(senior, set()).update(row.senior_insts)
    return by_person, by_abstract


def chitinase_cluster_mask(d):
    """Identify the seven-post abstract outlier cluster to project into the shared map."""
    first = d.full.map(lambda names: names[0].strip().casefold() if names else "")
    mask = d.type.eq("poster") & pd.Series(list(zip(d.year, first)), index=d.index).isin(CHITINASE_CLUSTER)
    found = set(zip(d.loc[mask, "year"], first[mask]))
    if found != CHITINASE_CLUSTER or int(mask.sum()) != len(CHITINASE_CLUSTER):
        raise ValueError(f"Expected one chitinase poster for each selected year and first author; found {found}")
    return mask


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
    cluster = chitinase_cluster_mask(d)
    vec, vocab = bm25_vectors(d.text)
    kw = Keywords(vec, vocab, d.text, KEYWORDS)

    # Similar web: BM25 cosine, not the dense SPECTER2 vectors, because the 0-100 match scale and SIM_FLOOR are calibrated
    # on BM25 in network.py (Find talks) and the page reads them as web strength. Dense neighbors were not evaluated here.
    similar = top_neighbors(vec @ vec.T, SIMILAR)
    ref = np.percentile([nb[0][1] for nb in similar if nb], 95)  # as network.py's match scale

    print("Generating SPECTER2 embeddings...")
    dense = embed(d)
    print(f"Running UMAP (projecting {int(cluster.sum())} chitinase-cluster posters after fitting)...")
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0)
    xy = np.empty((len(d), 2))
    xy[~cluster] = reducer.fit_transform(dense[~cluster])
    xy[cluster] = reducer.transform(dense[cluster])
    lo, hi = xy[~cluster].min(axis=0), xy[~cluster].max(axis=0)
    xy = (xy - lo) / (hi - lo)  # keep held-out projections from changing the map's scale
    xy = np.clip(xy, 0, 1)
    xy = SCALE * (2 * xy - 1)

    insts = pd.concat([pd.read_parquet(f"data/institutions/{y}.parquet") for y in YEARS]).set_index("abstract_id")
    people_insts, authors_insts = author_institution_evidence(d, insts, load_aliases("aliases.csv"))
    for mapping in authors_insts.values():
        for key, found in mapping.items():
            people_insts.setdefault(key, set()).update(found)
    listed, linked, _ = link_initials([people(fulls) for fulls in d.full], people_insts, d.id, authors_insts)
    print(f"{linked} initial-only author entries linked using affiliation evidence")
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
