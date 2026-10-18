"""Score a text by how many phrases overrepresented in talks it uses, and place it among ASHG abstracts.

Overrepresented phrases: the pooled n-grams from features.py (every year, Mantel-Haenszel OR stratified by
year) more common in talks than posters after a Bonferroni correction over every n-gram tested (two-sided
5%). Variants found in mostly the same abstracts are grouped as on the site (features.group_related), and
a group counts once however many of its variants a text uses. The score is the number of groups a text
contains; it is computed on abstract bodies, as the phrase statistics are.

- outputs/talk_score.json           the phrase groups, the score distribution of every complete abstract
                                    (abstracts per score, talks and posters; for the site's percentile tool)
                                    and an out-of-sample check
- outputs/trends_talk_score.png     histogram of scores, talks vs posters
- data/features/talk_score.parquet  per-abstract scores (kept out of outputs/, like other per-abstract data)

The check rebuilds the list from the years before the last and scores the last year's abstracts: AUC is
the chance a random talk that year outscores a random poster (ties count half). Distinct words per
abstract (length alone) is the baseline.

Usage: python src/ashg/talk_score.py              build
       python src/ashg/talk_score.py TEXT.txt     score a text against the built list
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import sparse
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from features import group_related, ngrams, premium, tokenize
from figures import INK_2, POSTER, SURFACE, TALK, header, new_figure, style


def talk_groups(stats, X, vocab):
    """[[best n-gram, *related], ...] for the n-grams more common in talks, Bonferroni-significant."""
    z = norm.isf(0.025 / len(stats))
    ranked = stats[stats.z >= z].sort_values("z", ascending=False)
    grouped = group_related(ranked, X, vocab)
    return [[g, *(r.split("; ") if isinstance(r, str) and r else [])] for g, r in zip(grouped.ngram, grouped.related)], grouped


def score(text, groups):
    """Number of phrase groups the text contains (any variant)."""
    grams = set(ngrams(text))
    return sum(any(g in grams for g in group) for group in groups)


def percentile(s, talks, posters):
    """Whole-number percentile of score s among abstracts (counts per score), ties counted half; rounded half
    up, as the site's Math.round does."""
    counts = np.asarray(talks) + np.asarray(posters)
    below, equal = counts[:min(s, len(counts))].sum(), counts[s] if s < len(counts) else 0
    return int(np.floor(100 * (below + 0.5 * equal) / counts.sum() + 0.5))


def histogram_figure(p, path):
    """Share of talks and of posters at each score, side by side, with each group's median."""
    talks, posters = np.array(p["talks"]), np.array(p["posters"])
    h = 4.2
    fig, gs = new_figure((10, h), left=0.07, right=0.98, top=1 - 1.3 / h, bottom=0.85 / h)
    ax = fig.add_subplot(gs[0])
    style(ax, grid_axis="y")
    ax.tick_params(axis="y", colors=INK_2)
    xs = np.arange(len(talks))
    for share, color, dx, label in ((talks / talks.sum(), TALK, -0.21, "talks"),
                                    (posters / posters.sum(), POSTER, 0.21, "posters")):
        ax.bar(xs + dx, share, width=0.4, color=color, lw=0, label=label)
    top = max(talks.max() / talks.sum(), posters.max() / posters.sum())
    for key, color, side in (("talks", TALK, 1), ("posters", POSTER, -1)):  # talks score higher: label right
        m = p["median"][key]
        ax.axvline(m, color=color, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.annotate(f"median {key[:-1]} {m:g}", (m, top * 1.12), xytext=(4 * side, 0), textcoords="offset points",
                    ha="left" if side > 0 else "right", va="bottom", color=INK_2, fontsize=8.5)
    ax.set_ylim(0, top * 1.22)
    ax.set_xlim(-0.8, len(talks) - 0.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel(f"Phrases overrepresented in talks found in the abstract (of {len(p['groups'])})", color=INK_2,
                  fontsize=9)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 1 - 0.9 / h), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK_2, handlelength=1.2)
    header(fig, f"Overrepresented phrases per abstract, ASHG {p['years'][0]}–{p['years'][1]}",
           "Share of talks and of posters by how many of the phrases overrepresented in talks each abstract uses.",
           "Source: ASHG abstracts. Phrases: Bonferroni-significant over every phrase tested, variants grouped. "
           "Accepted abstracts only: association, not acceptance odds.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def load_abstracts(data):
    """Complete abstracts in features.py's row order (every data/abstracts/*.parquet in glob order)."""
    df = pd.concat([pd.read_parquet(p) for p in sorted((data / "abstracts").glob("*.parquet"))], ignore_index=True)
    return df[~df.truncated].reset_index(drop=True)


def build(data, out):
    df = load_abstracts(data)
    talk = (df.pres_type != "poster").to_numpy()
    X = sparse.load_npz(data / "features/doc_term_all.npz").tocsc()
    if X.shape[0] != len(df):
        raise ValueError("doc_term_all.npz is stale: rerun `make pooled`")
    vocab = pd.read_parquet(data / "features/vocab_all.parquet").ngram.to_numpy()
    stats = pd.read_parquet(data / "features/premium_all.parquet")
    groups, grouped = talk_groups(stats, X, vocab)
    scores = np.array([score(t, groups) for t in df.body])
    df[["abstract_id", "year", "pres_type"]].assign(talk_score=scores).to_parquet(
        data / "features/talk_score.parquet", index=False)

    # out of sample: the list from the earlier years, scored on the last year
    last = df.year.max()
    train, test = df.year < last, (df.year == last).to_numpy()
    tr_stats, tr_X, tr_vocab = premium(df.body[train].tolist(), talk[train], strata=df.year[train])
    tr_groups, _ = talk_groups(tr_stats, tr_X, tr_vocab)
    test_scores = np.array([score(t, tr_groups) for t in df.body[test]])
    length = np.array([len(set(tokenize(t))) for t in df.body[test]])
    check = {"train": [int(df.year[train].min()), int(df.year[train].max())], "test": int(last),
             "groups": len(tr_groups), "talks": int(talk[test].sum()), "posters": int((~talk[test]).sum()),
             "auc": round(roc_auc_score(talk[test], test_scores), 3),
             "auc_length": round(roc_auc_score(talk[test], length), 3),
             "auc_in_sample": round(roc_auc_score(talk, scores), 3)}

    n = scores.max() + 1
    payload = {
        "years": [int(df.year.min()), int(df.year.max())], "n_tested": len(stats),
        "z": round(float(norm.isf(0.025 / len(stats))), 2),
        # [best n-gram, [the rest of its group], pooled OR, talks with it, posters with it]
        "groups": [[grp[0], grp[1:], round(float(r["or"]), 2), int(r.talk_docs), int(r.poster_docs)]
                   for grp, (_, r) in zip(groups, grouped.iterrows())],
        "talks": np.bincount(scores[talk], minlength=n).tolist(),
        "posters": np.bincount(scores[~talk], minlength=n).tolist(),
        "median": {"talks": float(np.median(scores[talk])), "posters": float(np.median(scores[~talk]))},
        "check": check,
    }
    (out / "talk_score.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    histogram_figure(payload, out / "trends_talk_score.png")
    print(f"{len(groups)} phrase groups overrepresented in talks ({sum(map(len, groups))} n-grams, z >= {payload['z']}); "
          f"median score talks {payload['median']['talks']:g}, posters {payload['median']['posters']:g}\n"
          f"check: list from {check['train'][0]}-{check['train'][1]} ({check['groups']} groups) on {last}: "
          f"AUC {check['auc']} (distinct words alone {check['auc_length']}; in-sample, every year {check['auc_in_sample']})"
          f"\n-> {out}/talk_score.json, {data}/features/talk_score.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="a text file to score (omit to build)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    data, out = pathlib.Path(args.data), pathlib.Path(args.out)
    if not args.text:
        build(data, out)
        return
    built = json.loads((out / "talk_score.json").read_text())
    groups = [[label, *also] for label, also, *_ in built["groups"]]
    s = score(pathlib.Path(args.text).read_text(), groups)
    print(f"score {s} of {len(groups)}: higher than about {percentile(s, built['talks'], built['posters'])}% of "
          f"ASHG {built['years'][0]}–{built['years'][1]} abstracts")


if __name__ == "__main__":
    main()
