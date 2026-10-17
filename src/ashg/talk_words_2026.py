"""Which phrases are over-represented in ASHG 2026 talks?

2026 has no booklet, so its abstracts come from the online program (data/manual/2026_*.yaml,
built by floq.py); earlier years come from data/abstracts/*.parquet. As in earlier years, talks
are plenary + platform. The 90 lightning talks (a new 2026 format) are left out of both sides.
  vs_posters     2026 talks vs 2026 posters (the same talk premium the site shows per year)
  vs_past_talks  2026 talks vs 2022-2025 talks (what is new in this year's talks)
Stats are features.premium (document frequency, 2x2 log OR, +0.5 cells, >= MIN_TALK_DOCS
2026 talks); group_related merges near-duplicates. Each phrase also gets a one-sided
hypergeometric (Fisher) p and a Benjamini-Hochberg q within its comparison, and its pooled
2022-2025 talk-vs-poster OR (data/features/premium_all.parquet) for context.

Usage: python src/ashg/talk_words_2026.py OUT.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd
import yaml
from scipy.stats import hypergeom

from features import MIN_TALK_DOCS, group_related, premium

TALK_TYPES = ("plenary", "platform")
TOP = 30


def load_2026():
    rows = []
    for ptype in ["plenary", "platform", "lightning", "poster"]:
        for r in yaml.safe_load(open(f"data/manual/2026_{ptype}.yaml"))["abstracts"]:
            rows.append({"year": 2026, "pres_type": ptype, "body": r["body"]})
    return pd.DataFrame(rows)


def load_past():
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/abstracts/20*.parquet"))], ignore_index=True)
    return df[~df.truncated & (df.year < 2026)][["year", "pres_type", "body"]]


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def compare(name, target, baseline, past_or):
    docs = pd.concat([target, baseline], ignore_index=True)
    is_target = np.r_[np.ones(len(target), bool), np.zeros(len(baseline), bool)]
    stats, X, vocab = premium(docs.body.tolist(), is_target)
    stats = stats.rename(columns={"talk_docs": "talk_docs_2026", "poster_docs": "baseline_docs",
                                  "talk_frac": "talk_frac_2026", "poster_frac": "baseline_frac"})
    stats = stats[stats.talk_docs_2026 >= MIN_TALK_DOCS]
    k, K = stats.talk_docs_2026, stats.talk_docs_2026 + stats.baseline_docs
    stats["p"] = hypergeom.sf(k - 1, len(docs), K, len(target))  # P(>= k of the talks | K abstracts with it)
    stats["q"] = bh(stats.p)
    ranked = stats.sort_values(["z", "talk_docs_2026"], ascending=False).head(TOP * 4)
    out = group_related(ranked, X, vocab).head(TOP)
    out["past_talk_or"] = out.ngram.map(past_or)
    out.insert(0, "comparison", name)
    out.insert(1, "n_talks_2026", len(target))
    out.insert(2, "n_baseline", len(baseline))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    now, past = load_2026(), load_past()
    talks = now[now.pres_type.isin(TALK_TYPES)]
    past_or = pd.read_parquet("data/features/premium_all.parquet").set_index("ngram")["or"]
    results = pd.concat([
        compare("vs_posters", talks, now[now.pres_type == "poster"], past_or),
        compare("vs_past_talks", talks, past[past.pres_type != "poster"], past_or),
    ], ignore_index=True)
    cols = ["comparison", "n_talks_2026", "n_baseline", "ngram", "related", "talk_docs_2026", "baseline_docs",
            "talk_frac_2026", "baseline_frac", "or", "or_lo", "or_hi", "z", "p", "q", "past_talk_or"]
    results[cols].to_csv(args.out, index=False, float_format="%.4g")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
