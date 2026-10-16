"""Text features: n-gram talk-vs-poster log odds ratios, and typed sample sizes.

N-grams: document frequency (does an abstract body contain it), 1-3 grams from
a genomics-aware tokenizer. Stop words are kept while forming n-grams so
"loss of function" survives; n-grams starting/ending in a stop word are dropped.
Each n-gram gets a 2x2 table (talk/poster x contains/not), +0.5 in every cell,
log OR, SE and z = logOR/SE.

Usage: python src/ashg/features.py YEAR OUTDIR ABSTRACTS.parquet [...]
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


def premium(docs, is_talk, min_talk_docs=MIN_TALK_DOCS):
    """Per-n-gram 2x2 stats. Returns (stats DataFrame, binary doc-term matrix, vocab)."""
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
    b, d = n_talk - a, n_poster - c
    log_or = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    stats = pd.DataFrame({
        "ngram": vocab, "n": [g.count(" ") + 1 for g in vocab],
        "talk_docs": a, "poster_docs": c, "talk_frac": a / n_talk, "poster_frac": c / n_poster,
        "log_or": log_or, "se": se, "z": log_or / se,
        "or": np.exp(log_or), "or_lo": np.exp(log_or - 1.96 * se), "or_hi": np.exp(log_or + 1.96 * se),
    })
    return stats, X, vocab


def drop_subsumed(ranked, X, vocab, threshold=0.9):
    """Drop an n-gram when a longer kept-or-candidate n-gram containing it covers >= threshold of its docs."""
    col = {g: i for i, g in enumerate(vocab)}
    df = np.asarray(X.sum(axis=0)).ravel()
    grams = list(ranked.ngram)
    drop = set()
    for g in grams:
        padded = f" {g} "
        for h in grams:
            if h != g and padded in f" {h} " and df[col[h]] >= threshold * df[col[g]]:
                drop.add(g)
                break
    return ranked[~ranked.ngram.isin(drop)]


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
    ap.add_argument("year", type=int)
    ap.add_argument("outdir")
    ap.add_argument("abstracts", nargs="+")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.abstracts], ignore_index=True)
    df = df[~df.truncated & (df.year == args.year)].reset_index(drop=True)
    is_talk = (df.pres_type != "poster").to_numpy()

    stats, X, vocab = premium(df.body.tolist(), is_talk)
    stats.insert(0, "year", args.year)
    stats.sort_values("z", ascending=False).to_parquet(f"{args.outdir}/premium_{args.year}.parquet", index=False)

    ss = pd.DataFrame([sample_size(t) for t in df.body], columns=["max_n", "noun"])
    ss.loc[~(ss.max_n > 0), ["max_n", "noun"]] = [np.nan, None]  # e.g. "0 individuals"
    ss.insert(0, "abstract_id", df.abstract_id)
    ss.insert(1, "year", args.year)
    ss.insert(2, "pres_type", df.pres_type)
    ss.to_parquet(f"{args.outdir}/sample_size_{args.year}.parquet", index=False)

    sparse.save_npz(f"{args.outdir}/doc_term_{args.year}.npz", X)
    pd.Series(vocab).to_frame("ngram").to_parquet(f"{args.outdir}/vocab_{args.year}.parquet")
    print(f"{args.year}: {len(df)} abstracts ({is_talk.sum()} talks), {len(stats)} n-grams passing support; "
          f"sample size found in {ss.max_n.notna().mean():.0%} of abstracts")


if __name__ == "__main__":
    main()
