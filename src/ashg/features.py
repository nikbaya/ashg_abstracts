"""Text features: n-gram talk-vs-poster log odds ratios, and typed sample sizes.

N-grams: document frequency (does an abstract body contain it), 1-3 grams from
a genomics-aware tokenizer. Stop words are kept while forming n-grams so
"loss of function" survives; n-grams starting/ending in a stop word are dropped.
Each n-gram gets a 2x2 table (talk/poster x contains/not), +0.5 in every cell,
log OR, SE and z = logOR/SE. YEAR "all" pools every year given, with a 2x2 table
per year combined by Mantel-Haenszel (talk rate differs by year, so a phrase's
rise or fall over time doesn't masquerade as a talk premium).

Usage: python src/ashg/features.py YEAR|all OUTDIR ABSTRACTS.parquet [...]
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# numbers like 1,078 / 3.4 are one token (but "23andMe" is a word); words keep internal hyphens/apostrophes
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?![\w-])|\w+(?:[-'’]\w+)*")
NUM_RE = re.compile(r"^[\d.,]+$")
MIN_TALK_DOCS = 5


def tokenize(text):
    return ["<num>" if NUM_RE.match(t) else t.replace("’", "'") for t in TOKEN_RE.findall(text.lower())]


def ngrams(text, n_max=3):
    toks = tokenize(text)
    out = set()
    for n in range(1, n_max + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if gram[0] in ENGLISH_STOP_WORDS or gram[-1] in ENGLISH_STOP_WORDS:
                continue
            if "<num>" in (gram[0], gram[-1]) or len(gram[0]) < 2 or len(gram[-1]) < 2:
                continue
            out.add(" ".join(gram))
    return list(out)


def mantel_haenszel(a, b, c, d):
    """Log Mantel-Haenszel odds ratio and its Robins-Breslow-Greenland SE over strata (rows), one
    column per term; +0.5 in every cell. With one stratum this is the plain log OR and Woolf SE."""
    a, b, c, d = (np.atleast_2d(x) + .5 for x in (a, b, c, d))
    n = a + b + c + d
    r, s = a * d / n, b * c / n
    p, q = (a + d) / n, (b + c) / n
    R, S = r.sum(axis=0), s.sum(axis=0)
    var = ((p * r).sum(axis=0) / (2 * R ** 2) + (p * s + q * r).sum(axis=0) / (2 * R * S)
           + (q * s).sum(axis=0) / (2 * S ** 2))
    return np.log(R / S), np.sqrt(var)


def premium(docs, is_talk, min_talk_docs=MIN_TALK_DOCS, strata=None):
    """Per-n-gram 2x2 stats, combined over `strata` (e.g. year per doc) if given. Returns
    (stats DataFrame, binary doc-term matrix, vocab)."""
    vec = CountVectorizer(analyzer=ngrams, binary=True, min_df=min_talk_docs)
    X = vec.fit_transform(docs).tocsc()
    vocab = np.array(vec.get_feature_names_out())
    talk = np.asarray(is_talk, dtype=bool)
    n_talk, n_poster = talk.sum(), (~talk).sum()
    a = np.asarray(X[talk].sum(axis=0)).ravel()   # talks containing term
    c = np.asarray(X[~talk].sum(axis=0)).ravel()  # posters containing term
    # symmetric support: >= min_talk_docs talks, or the poster-side equivalent
    # (so a term in 0 talks but hundreds of posters is kept)
    keep = (a >= min_talk_docs) | (c >= min_talk_docs * n_poster / n_talk)
    X, vocab, a, c = X[:, keep], vocab[keep], a[keep], c[keep]
    strata = np.zeros(len(talk)) if strata is None else np.asarray(strata)
    cells = []  # per stratum: talks with, talks without, posters with, posters without
    for k in np.unique(strata):
        t, p = talk & (strata == k), ~talk & (strata == k)
        ak, ck = (np.asarray(X[m].sum(axis=0)).ravel() for m in (t, p))
        cells.append((ak, t.sum() - ak, ck, p.sum() - ck))
    log_or, se = mantel_haenszel(*(np.array(x) for x in zip(*cells)))
    stats = pd.DataFrame({
        "ngram": vocab, "n": [g.count(" ") + 1 for g in vocab],
        "talk_docs": a, "poster_docs": c, "talk_frac": a / n_talk, "poster_frac": c / n_poster,
        "log_or": log_or, "se": se, "z": log_or / se,
        "or": np.exp(log_or), "or_lo": np.exp(log_or - 1.96 * se), "or_hi": np.exp(log_or + 1.96 * se),
    })
    return stats, X, vocab


def initials(phrase):
    return "".join(w[0] for w in phrase.split())


def lexically_related(g, h):
    """Share a content word, or one is an acronym of the other ("mpra" / "massively parallel reporter")."""
    if (set(g.split()) & set(h.split())) - ENGLISH_STOP_WORDS:
        return True
    return any(" " not in short and " " in long and len(initials(long)) >= 3 and initials(long) in short.rstrip("s")
               for short, long in ((g, h), (h, g)))


def is_acronym(g, h):
    """ "qtls" / "quantitative trait loci": one word equal to the other's (3+) initials, plural allowed."""
    return any(" " not in short and len(initials(long)) >= 3 and short.rstrip("s") == initials(long)
               for short, long in ((g, h), (h, g)))


def group_related(ranked, X, vocab, threshold=0.45):
    """Group n-grams linked by an exact acronym, or by being lexically related and found in mostly
    the same abstracts (Jaccard >= threshold); groups are connected components, so "massively
    parallel", "parallel reporter assay" and "mpra" are one row. 0.45, not 0.5, so "reporter" (half
    its abstracts are MPRA, the rest luciferase etc.: Jaccard 0.47 pooled) joins that row. A row keeps the stats of its group's
    best-ranked n-gram (a union could be swamped by a broad word like "trait"); `related` lists the
    rest, best first."""
    col = {g: i for i, g in enumerate(vocab)}
    X = X.tocsc()
    grams = list(ranked.ngram)
    docs = [set(X[:, col[g]].indices) for g in grams]
    parent = list(range(len(grams)))  # union-find; a root is always its group's best-ranked member

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, g in enumerate(grams):
        for j in range(i):
            h = grams[j]
            if is_acronym(g, h) or (lexically_related(g, h)
                                    and len(docs[i] & docs[j]) >= threshold * len(docs[i] | docs[j])):
                a, b = sorted((root(i), root(j)))
                parent[b] = a
    groups = {}
    for i, g in enumerate(grams):
        groups.setdefault(root(i), []).append(g)
    related = {grp[0]: "; ".join(grp[1:]) for grp in groups.values()}
    out = ranked[ranked.ngram.isin(related)].copy()
    out.insert(out.columns.get_loc("ngram") + 1, "related", out.ngram.map(related))
    return out


# --- sample sizes -----------------------------------------------------------

HUMAN_NOUNS = (r"individuals|participants|cases|controls|patients|subjects|people|persons|probands"
               r"|families|children|adults|women|men|donors|veterans|genomes|exomes|samples|newborns|infants")
NUMBER = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|thousand|[mk])?\b"
TYPED_RE = re.compile(NUMBER + r"\s+(?:[\w-]+\s+){0,3}?(" + HUMAN_NOUNS + r")\b", re.I)
N_EQ_RE = re.compile(r"\b[nN]\s*(?:=|≈|~|>|≥)\s*" + NUMBER)
SCALE = {"million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def to_number(num, scale):
    return float(num.replace(",", "")) * SCALE.get((scale or "").lower(), 1)


def sample_size(text):
    """Largest human-subject N mentioned: '<number> [<=3 words] <human noun>', or 'N = <number>'
    within 80 chars after a human noun. Returns (max_n, noun) or (nan, None)."""
    best = (np.nan, None)
    for m in TYPED_RE.finditer(text):
        v = to_number(m.group(1), m.group(2))
        if not best[0] >= v:
            best = (v, m.group(3).lower())
    for m in N_EQ_RE.finditer(text):
        before = text[max(0, m.start() - 80):m.start()]
        noun = re.findall(HUMAN_NOUNS, before, re.I)
        v = to_number(m.group(1), m.group(2))
        if noun and not best[0] >= v:
            best = (v, noun[-1].lower())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", help="a year, or 'all' to pool every year in ABSTRACTS")
    ap.add_argument("outdir")
    ap.add_argument("abstracts", nargs="+")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.abstracts], ignore_index=True)
    df = df[~df.truncated & ((args.year == "all") | (df.year.astype(str) == args.year))].reset_index(drop=True)
    is_talk = (df.pres_type != "poster").to_numpy()

    stats, X, vocab = premium(df.body.tolist(), is_talk, strata=df.year if args.year == "all" else None)
    stats.insert(0, "year", args.year if args.year == "all" else int(args.year))
    stats.sort_values("z", ascending=False).to_parquet(f"{args.outdir}/premium_{args.year}.parquet", index=False)

    ss = pd.DataFrame([sample_size(t) for t in df.body], columns=["max_n", "noun"])
    ss.loc[~(ss.max_n > 0), ["max_n", "noun"]] = [np.nan, None]  # e.g. "0 individuals"
    ss.insert(0, "abstract_id", df.abstract_id)
    ss.insert(1, "year", df.year)
    ss.insert(2, "pres_type", df.pres_type)
    ss.to_parquet(f"{args.outdir}/sample_size_{args.year}.parquet", index=False)

    sparse.save_npz(f"{args.outdir}/doc_term_{args.year}.npz", X)
    pd.Series(vocab).to_frame("ngram").to_parquet(f"{args.outdir}/vocab_{args.year}.parquet")
    print(f"{args.year}: {len(df)} abstracts ({is_talk.sum()} talks), {len(stats)} n-grams passing support; "
          f"sample size found in {ss.max_n.notna().mean():.0%} of abstracts")


if __name__ == "__main__":
    main()
