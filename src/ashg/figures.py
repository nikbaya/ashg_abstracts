"""Headline figure: phrases over-represented in talks vs posters (one year).

Dot = odds ratio that an abstract containing the phrase is a talk rather than
a poster; line = 95% CI; log x-axis centred on OR = 1.

Usage: python src/ashg/figures.py YEAR [--out outputs] [--k 15]
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

# Reference palette (dataviz skill): diverging poles + recessive ink, validated light mode
TALK, POSTER = "#2a78d6", "#e34948"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def headline(premium, summary, year, k, path):
    talk = premium[premium.direction == "talk"].head(k).sort_values("or_")
    poster = premium[premium.direction == "poster"].head(k).sort_values("or_")
    rows = pd.concat([poster, talk])  # posters at the bottom, talks on top
    gap = 1.2
    y = list(range(len(poster))) + [len(poster) + gap + i for i in range(len(talk))]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 10.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for yi, r in zip(y, rows.itertuples()):
        color = TALK if r.direction == "talk" else POSTER
        ax.plot([r.or_lo, r.or_hi], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(r.or_, yi, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.axvline(1, color=INK_2, lw=1, zorder=1)

    ax.set_xscale("log")
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    lo, hi = rows.or_lo.min() / 1.15, rows.or_hi.max() * 1.15
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    ax.set_xlim(lo, hi)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)

    ax.set_yticks(y, rows.ngram.str.replace("<num>", "N", regex=False), color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(-0.8, y[-1] + 1.6)

    # section labels (identity carried by text + position, not color alone)
    for label, top, color in [("More common in talks", y[-1] + 1, TALK),
                              ("More common in posters", len(poster) - 1 + 1, POSTER)]:
        ax.annotate("●", (0, top), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                    textcoords="offset points", ha="right", va="center", color=color, fontsize=11)
        ax.annotate(label, (0, top), xycoords=("axes fraction", "data"), xytext=(2, 0),
                    textcoords="offset points", ha="left", va="center", color=INK, fontweight="bold")

    s = summary.iloc[0]
    n_talk, n_all = int(s.talks), int(s.abstracts)
    ax.set_xlabel("Odds ratio of being a talk (vs poster), with vs without the phrase · 95% CI",
                  color=INK_2, labelpad=8)
    fig.suptitle(f"What gets an ASHG {year} abstract a talk?", x=0.02, ha="left", fontsize=16,
                 fontweight="bold", color=INK)
    fig.text(0.02, 0.945,
             f"Phrases most over-represented in talks vs posters. {n_talk:,} talks (platform + plenary) "
             f"vs {n_all - n_talk:,} posters.\nAccepted abstracts only, and authors can opt for a poster, "
             f"so this is association, not acceptance odds.", ha="left", va="top", color=INK_2, fontsize=9.5)
    fig.text(0.02, 0.01, f"Source: ASHG {year} abstract booklets. Phrases: 1–3-word n-grams in abstract text, "
             "ranked by z-score; nested duplicates removed.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.935))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    premium = pd.read_csv(out / f"{args.year}_premium.csv").rename(columns={"or": "or_"})
    summary = pd.read_csv(out / f"{args.year}_summary.csv")
    path = out / f"{args.year}_headline.png"
    headline(premium, summary, args.year, args.k, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
