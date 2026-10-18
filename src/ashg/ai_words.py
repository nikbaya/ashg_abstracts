"""Are words linked to AI-generated text more common in recent ASHG abstracts?

Terms come from Matsui K. "Delving Into PubMed Records: How AI-Influenced Vocabulary has Transformed Medical
Writing since ChatGPT." Perspectives on Medical Education 2025; 14(1): 882-890, https://doi.org/10.5334/pme.1929
(ai_terms.csv): 135 potentially AI-influenced terms and 84 common medical-academic phrases used there as controls.

A term is "used" by an abstract if its title or text contains it (verbs and nouns also in -s/-ed/-ing forms; adjectives,
adverbs and phrases exactly). Abstracts cut off in a booklet are left out. For a set of terms, an abstract's count is
how many different terms of the set it uses, so a year's mean count is the sum of the terms' shares of abstracts.

Tests, year by year against the first year (2022, the last before ChatGPT) and as a trend over all years:
  - rate ratio of the mean count, with a 95% bootstrap CI over abstracts;
  - the same ratio for the paper's control phrases, and for random sets of ordinary words matched one-to-one to the
    AI terms on their share of 2022 abstracts (2,000 draws). That matched set is the real null: wording in general
    shifts from year to year (see trends_words_enriched.csv), so a rise only counts if it beats words as common as
    these. The empirical p is the share of draws with a ratio at least as large;
  - per term, a Cochran-Armitage test for a rise across years (one-sided) with Benjamini-Hochberg q, and the ratio of
    its 2026 to its 2022 share; across terms, the share with a rising trend and the mean trend z against the same
    matched draws.

Writes outputs/ai_words_terms.csv, ai_words_years.csv, ai_words_summary.json, trends_ai_terms.png.

Usage: python src/ashg/ai_words.py [--years 2022 ... 2026] [--data data] [--out outputs] [--terms ai_terms.csv]
"""
import argparse
import json
import pathlib
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from features import tokenize
from figures import INK, INK_2, TALK, new_figure, style

DRAWS, BOOT, MIN_WORD_DOCS, NEAREST, SEED = 2000, 1000, 5, 25, 0  # MIN_WORD_DOCS: in the first year
CITATION = ("Matsui K. Delving Into PubMed Records: How AI-Influenced Vocabulary has Transformed Medical Writing "
            "since ChatGPT. Perspectives on Medical Education 2025; 14(1): 882-890. https://doi.org/10.5334/pme.1929")


def forms(term, category):
    """Surface forms that count as the term: verbs and nouns also inflected, the rest exactly."""
    if category not in ("verb", "noun"):
        return {term}
    out = {term, term + "s"}
    if category == "verb":
        stem = term[:-1] if term.endswith("e") else term
        out |= {term + "es", term + "ed", term + "ing", stem + "ed", stem + "ing"}
        if term.endswith("y"):
            out |= {term[:-1] + "ies", term[:-1] + "ied"}
    return out


def load_abstracts(data, years):
    df = pd.concat([pd.read_parquet(p) for y in years for p in sorted((data / "abstracts").glob(f"{y}_*.parquet"))],
                   ignore_index=True)
    return df[~df.truncated].reset_index(drop=True)


def term_hits(docs, terms):
    """Abstracts x terms 0/1 matrix, and each abstract's token set (reused for the matched-word vocabulary)."""
    index = {}  # surface form -> term rows; phrases are keyed by their words joined with spaces
    for i, t in enumerate(terms.itertuples()):
        for f in forms(t.term, t.category):
            index.setdefault(f, []).append(i)
    n_max = max(len(f.split()) for f in index)
    hits = np.zeros((len(docs), len(terms)), dtype=np.uint8)
    token_sets = []
    for d, text in enumerate(docs):
        toks = tokenize(text)
        token_sets.append(set(toks))
        grams = set(toks)
        for n in range(2, n_max + 1):
            grams |= {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}
        for g in grams & index.keys():
            hits[d, index[g]] = 1
    return hits, token_sets


def share_ratio(k, m, n1, n0):
    """Share of abstracts using a term in one year (k of n1) over another (m of n0), each share nudged by 1/(n1+n0) so
    a term unused in both years has ratio 1 (a pseudocount of 0.5 in each year's count would give n0/n1)."""
    c = 1 / (n1 + n0)
    return (k / n1 + c) / (m / n0 + c)


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def word_docs(token_sets, year, years, exclude):
    """Abstracts per year using each ordinary word (not a term form, not a stop word), in >= MIN_WORD_DOCS first-year
    abstracts: chosen on the first year alone, so the choice does not depend on how the word later did."""
    used = pd.DataFrame({"year": np.repeat(year, [len(s) for s in token_sets]), "word": [w for s in token_sets for w in s]})
    used = used[used.word.str[0].isin(list(string.ascii_lowercase)) & (used.word.str.len() > 1)
                & ~used.word.isin(ENGLISH_STOP_WORDS) & ~used.word.isin(exclude)]
    docs = used.groupby(["word", "year"]).size().unstack(fill_value=0).reindex(columns=years, fill_value=0)
    return docs[docs[years[0]] >= MIN_WORD_DOCS]


def trend_z(docs, n):
    """Cochran-Armitage trend z (years as scores) for each row of `docs` (abstracts using it per year, of n per year);
    positive when the share rises across years."""
    x = np.arange(len(n), dtype=float)
    N, R = n.sum(), docs.sum(axis=1)
    p = R / N
    num = (docs * x).sum(axis=1) - R * (n * x).sum() / N
    var = p * (1 - p) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / N)
    return np.divide(num, np.sqrt(var), out=np.zeros(len(num)), where=var > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--terms", default="ai_terms.csv")
    args = ap.parse_args()
    years, out = sorted(args.years), pathlib.Path(args.out)
    first, latest = years[0], years[-1]
    terms = pd.read_csv(args.terms, comment="#")
    df = load_abstracts(pathlib.Path(args.data), years)
    text = (df.title.fillna("") + " . " + df.body).tolist()
    year = df.year.to_numpy()
    n_year = pd.Series(year).value_counts().sort_index()
    n = n_year.to_numpy().astype(float)
    n_words = np.array([len(t.split()) for t in text])
    hits, token_sets = term_hits(text, terms)
    rng = np.random.default_rng(SEED)
    ai = (terms.group == "ai").to_numpy()

    # per term: abstracts per year, trend, and 2026 vs 2022
    docs = pd.DataFrame({y: hits[year == y].sum(axis=0) for y in years}, index=terms.term)
    t = terms.copy()
    for y in years:
        t[f"docs_{y}"] = docs[y].to_numpy()
    t["share_first"], t["share_latest"] = docs[first].to_numpy() / n[0], docs[latest].to_numpy() / n[-1]
    t["share_ratio"] = share_ratio(docs[latest].to_numpy(), docs[first].to_numpy(), n[-1], n[0])
    t["trend_z"] = trend_z(docs.to_numpy(), n)
    t["p"] = norm.sf(t.trend_z)
    t["q"] = np.nan
    for g in ("ai", "control"):
        sel = (t.group == g).to_numpy()
        t.loc[sel, "q"] = bh(t.p[sel])
    t["abstracts_total"] = docs.sum(axis=1).to_numpy()
    t.to_csv(out / "ai_words_terms.csv", index=False, float_format="%.4g")

    # matched ordinary words: one drawn per AI term from the words nearest in share of first-year abstracts
    reserved = {f for r in terms.itertuples() for f in forms(r.term, r.category)}
    wd = word_docs(token_sets, year, years, reserved)
    W = wd.to_numpy()
    w_first = W[:, 0] / n[0]
    near = [np.argsort(np.abs(np.log(w_first) - np.log(max(s, 1 / n[0]))))[:NEAREST] for s in t.share_first[ai]]
    wz = trend_z(W, n)
    draws = np.empty((DRAWS, len(years)))  # summed share per year
    up, mz = np.empty(DRAWS), np.empty(DRAWS)
    for d in range(DRAWS):
        pick = np.unique([rng.choice(c) for c in near])
        draws[d] = (W[pick] / n).sum(axis=0) * (len(near) / len(pick))
        up[d], mz[d] = (wz[pick] > 0).mean(), wz[pick].mean()

    # observed sets, bootstrap CIs over abstracts
    sets = {"ai": ai, "control": (terms.group == "control").to_numpy()}
    rows = []
    summary = {"citation": CITATION, "n_abstracts": {int(y): int(c) for y, c in n_year.items()}, "draws": DRAWS,
               "n_ai_terms": int(ai.sum()), "n_ai_terms_used": int((t.abstracts_total[ai] > 0).sum()),
               "n_control_terms": int(sets["control"].sum()), "first": first, "latest": latest}
    idx = {y: np.flatnonzero(year == y) for y in years}
    seeds = rng.integers(0, 2**31, BOOT)
    obs = {}
    for name, sel in sets.items():
        cnt = hits[:, sel].sum(axis=1).astype(float)
        mean = {y: cnt[idx[y]].mean() for y in years}
        boot = {y: np.empty(BOOT) for y in years}
        for i, sd in enumerate(seeds):
            r = np.random.default_rng(sd)
            for y in years:
                boot[y][i] = cnt[r.choice(idx[y], len(idx[y]))].mean()
        obs[name] = {"mean": mean, "boot": boot}
        for y in years:
            lo, hi = np.percentile(boot[y] / boot[first], [2.5, 97.5])
            rows.append({"set": name, "year": y, "abstracts": int(n_year[y]), "mean_terms": mean[y],
                         "per_1000_words": 1000 * cnt[idx[y]].sum() / n_words[idx[y]].sum(),
                         "share_any": float((cnt[idx[y]] > 0).mean()), "ratio_to_first": mean[y] / mean[first],
                         "ratio_lo": lo, "ratio_hi": hi})
        sel_t = t[sel]
        summary[name] = {"mean_terms": {int(y): mean[y] for y in years}, "ratio_to_first": {int(y): mean[y] / mean[first] for y in years},
                         "terms_rising_trend": int((sel_t.trend_z > 0).sum()), "terms_q05": int((sel_t.q < .05).sum()),
                         "mean_trend_z": float(sel_t.trend_z.mean()),
                         "ratio_ci": {int(y): np.percentile(boot[y] / boot[first], [2.5, 97.5]).tolist() for y in years}}
    summary["ai"]["share_terms_rising"] = float((t.trend_z[ai] > 0).mean())
    # matched null per year, relative to the first year; empirical p for each year's AI ratio
    null = draws / draws[:, [0]]
    for j, y in enumerate(years):
        rows.append({"set": "matched", "year": y, "abstracts": int(n_year[y]), "mean_terms": float(draws[:, j].mean()),
                     "ratio_to_first": float(np.median(null[:, j])), "ratio_lo": float(np.percentile(null[:, j], 2.5)),
                     "ratio_hi": float(np.percentile(null[:, j], 97.5))})
    summary["matched"] = {
        "ratio_to_first_median": {int(y): float(np.median(null[:, j])) for j, y in enumerate(years)},
        "ratio_95": {int(y): np.percentile(null[:, j], [2.5, 97.5]).tolist() for j, y in enumerate(years)},
        "p_ratio_ge_observed": {int(y): float((1 + (null[:, j] >= summary["ai"]["ratio_to_first"][int(y)]).sum()) / (1 + DRAWS))
                                for j, y in enumerate(years)},
        "share_terms_rising_median": float(np.median(up)),
        "p_share_rising_ge_observed": float((1 + (up >= summary["ai"]["share_terms_rising"]).sum()) / (1 + DRAWS)),
        "mean_trend_z_median": float(np.median(mz)),
        "p_mean_trend_z_ge_observed": float((1 + (mz >= summary["ai"]["mean_trend_z"]).sum()) / (1 + DRAWS))}
    # AI against control: ratio of rate ratios, bootstrap over abstracts
    summary["ai_vs_control"] = {int(y): [float(np.median(a)), *np.percentile(a, [2.5, 97.5]).tolist()]
                                for y in years for a in [(obs["ai"]["boot"][y] / obs["ai"]["boot"][first])
                                                         / (obs["control"]["boot"][y] / obs["control"]["boot"][first])]}
    pd.DataFrame(rows).to_csv(out / "ai_words_years.csv", index=False, float_format="%.5g")
    (out / "ai_words_summary.json").write_text(json.dumps(summary, indent=1))

    # headline PNG: one series, AI-associated terms per abstract by year
    ys = pd.DataFrame(rows).query("set == 'ai'")
    fig, _ = new_figure((6.4, 3.4))
    ax = fig.add_subplot()
    style(ax, "y")
    ax.plot(ys.year, ys.mean_terms, marker="o", color=TALK, lw=2)
    ax.set_xticks(years)
    ax.set_ylim(0)
    ax.set_ylabel("distinct AI-associated terms\nper abstract", color=INK_2)
    ax.set_title(f"{int(ai.sum())} terms linked to AI-generated text (Matsui 2025)", loc="left", color=INK, fontsize=11)
    fig.savefig(out / "trends_ai_terms.png", dpi=200, bbox_inches="tight")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
