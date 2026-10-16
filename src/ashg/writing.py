"""Writing style across all years: likely typos and lexical diversity.

Typos are found with the corpus as its own dictionary: a lowercase word that
occurs at most twice in all abstracts, is not an English word (system word
list, allowing simple inflections), and is one edit away from a word >= 50x as
common ("anlaysis" -> "analysis"). Spelling variants (tumour/tumor) and a
letter glued to either end (a lost space next to an italic "p") don't count.
This favours precision over recall, so rates are a lower bound.

Lexical diversity is MATTR: the share of distinct words in each 50-word
window, averaged over the abstract, which unlike raw distinct-word counts
does not grow with length. Outliers (mattr_z) use MATTR adjusted for log length.

Writes data/writing/abstracts.parquet (per abstract, not committed) and
outputs/writing_summary.csv, outputs/typos_top.csv (aggregates).

Usage: python src/ashg/writing.py ABSTRACTS.parquet [...] [--data data] [--out outputs]
"""

import argparse
import collections
import pathlib
import re

import numpy as np
import pandas as pd

WORD = re.compile(r"(?<![\w-])[A-Za-z]+(?:'[a-z]+)?(?![\w-])")  # plain words, no hyphen/digit neighbours
DICT_PATH = "/usr/share/dict/words"
MAX_TYPO_COUNT, MIN_CORRECT_COUNT, MIN_RATIO, MIN_LEN = 2, 30, 50, 6
MATTR_WINDOW = 50
LETTERS = "abcdefghijklmnopqrstuvwxyz"
VARIANTS = [("s", ""), ("es", ""), ("ies", "y"), ("our", "or"), ("ae", "e"), ("oe", "e"), ("ise", "ize"),
            ("yse", "yze"), ("isation", "ization"), ("ised", "ized"), ("ising", "izing"), ("tre", "ter"),
            ("lled", "led"), ("lling", "ling")]


def edits1(w):
    s = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in s if b} | {a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1}
            | {a + c + b[1:] for a, b in s if b for c in LETTERS} | {a + c + b for a, b in s for c in LETTERS})


def is_variant(w, v):
    """Spelling variants (tumour/tumor) and inflections of one stem (downregulates/downregulated)."""
    stem = lambda x: re.sub(r"(es|ed|ing|s|d|e)$", "", x)
    return stem(w) == stem(v) or any(w.replace(a, b) == v or v.replace(a, b) == w for a, b in VARIANTS)


def is_glued(w, v):
    return len(w) == len(v) + 1 and (w[1:] == v or w[:-1] == v)


def find_typos(docs, bodies, dictionary):
    """{typo: correction} over tokenized docs (original case).

    Words ever capitalized in a body (names, genes, acronyms) are skipped; titles
    don't count, since they are often in Title Case.
    """
    counts = collections.Counter(w.lower() for doc in docs for w in doc)
    capitalized = {w.lower() for body in bodies for w in body if not w.islower()}
    def in_dict(w):
        return any(x in dictionary for x in (w, w[:-1], w[:-2], w[:-3], w[:-2] + "e", w[:-3] + "e", w[:-3] + "y"))
    typos = {}
    for w, n in counts.items():
        if n > MAX_TYPO_COUNT or len(w) < MIN_LEN or "'" in w or w in capitalized or in_dict(w):
            continue
        best = max((v for v in edits1(w) if counts.get(v, 0) >= MIN_CORRECT_COUNT), key=counts.get, default=None)
        if best and counts[best] >= MIN_RATIO * n and not is_variant(w, best) and not is_glued(w, best):
            typos[w] = best
    return typos, counts


def mattr(words, window=MATTR_WINDOW):
    if len(words) <= window:
        return len(set(words)) / max(1, len(words))
    seen = collections.Counter(words[:window])
    total = len(seen)
    for i in range(window, len(words)):
        out, inn = words[i - window], words[i]
        seen[out] -= 1
        if not seen[out]:
            del seen[out]
        seen[inn] += 1
        total += len(seen)
    return total / ((len(words) - window + 1) * window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    df = pd.concat([pd.read_parquet(p) for p in args.abstracts], ignore_index=True)
    df = df[~df.truncated].reset_index(drop=True)
    docs = [WORD.findall(t) for t in df.title.fillna("") + " . " + df.body.fillna("")]
    bodies = [WORD.findall(t) for t in df.body.fillna("")]
    typos, counts = find_typos(docs, bodies, {w.strip().lower() for w in open(DICT_PATH)})

    lower = [[w.lower() for w in doc] for doc in docs]
    per = df[["abstract_id", "year", "pres_type", "title"]].assign(
        talk=df.pres_type != "poster",
        words=[len(d) for d in lower], unique=[len(set(d)) for d in lower], mattr=[mattr(d) for d in lower],
        typos=[[w for w in d if w in typos] for d in lower])
    per["n_typos"] = per.typos.str.len()
    # MATTR still drifts with length (short abstracts repeat less), so outliers use the residual on log(words)
    x = np.log(per.words.clip(lower=1))
    resid = per.mattr - np.polyval(np.polyfit(x, per.mattr, 1), x)
    z = per["mattr_z"] = (resid - resid.mean()) / resid.std()
    pathlib.Path(args.data, "writing").mkdir(parents=True, exist_ok=True)
    per.to_parquet(pathlib.Path(args.data, "writing", "abstracts.parquet"), index=False)

    per["type"] = np.where(per.talk, "talk", "poster")
    summary = per.groupby(["year", "type"]).agg(
        abstracts=("words", "size"), median_words=("words", "median"), median_unique=("unique", "median"),
        median_mattr=("mattr", "median"), with_typo=("n_typos", lambda x: int((x > 0).sum())),
        typos=("n_typos", "sum"), words_total=("words", "sum")).reset_index()
    summary["with_typo_share"] = summary.with_typo / summary.abstracts
    summary["typos_per_10k_words"] = 1e4 * summary.typos / summary.words_total
    top = pd.Series([w for t in per.typos for w in t]).value_counts().rename_axis("typo").reset_index(name="occurrences")
    top["correction"] = top.typo.map(typos)
    out = pathlib.Path(args.out)
    summary.drop(columns="words_total").to_csv(out / "writing_summary.csv", index=False, float_format="%.4g")
    top.head(50).to_csv(out / "typos_top.csv", index=False)
    print(f"{len(per)} abstracts: {len(typos)} typo types in {(per.n_typos > 0).sum()} abstracts; "
          f"MATTR median {per.mattr.median():.3f}, {(z > 3).sum()} abstracts with z > 3")
    print(summary.drop(columns="words_total").to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
