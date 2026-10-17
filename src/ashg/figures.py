"""Summary figures for one year (PNG, for the GitHub post and the site).

- {year}_headline.png      phrases over-represented in talks vs posters (odds ratio, 95% CI),
                           plus a small panel of stated sample sizes (median, IQR)
- {year}_institutions.png  the 20 highest talk rates among the 50 largest institutions (first-author credit)
- {year}_metros.png        the same, with institutions grouped by metro area
- {year}_regions.png       talk rate by UN world region of the first author
  (each also as {name}_senior.png, crediting the senior author)
- {year}_senior_authors.png talk rate of the top-10 senior authors
YEAR "all" draws the same figures from outputs/all_*.csv (every year pooled), as all_*.png.

Usage: python src/ashg/figures.py YEAR|all [--out outputs] [--k 15] [--n-inst 20] [--n-metros 20]
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

# Reference palette (dataviz skill): diverging poles + recessive ink, validated light mode
TALK, POSTER = "#2a78d6", "#e34948"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAVEAT = "Accepted abstracts only, and authors can opt for a poster, so this is association, not acceptance odds."
CAVEAT_2LINE = "Accepted abstracts only, and authors can opt for a poster,\nso this is association, not acceptance odds."

HOST_METRO = {2022: "Los Angeles", 2023: "Washington, DC", 2024: "Denver", 2025: "Boston",
              2026: "Montreal"}  # where each meeting was held

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def new_figure(size, **kw):
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    return fig, fig.add_gridspec(**kw) if kw else None


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=6, labelcolor=INK)
    ax.tick_params(axis="x", colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def log_axis(ax, lo, hi, ticks, fmt):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(lo, hi)


def dot_ci(ax, x, lo, hi, y, color):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(x, y, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)


def header(fig, title, subtitle, source, top=0.965):
    fig.text(0.02, top, title, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.02, top - 0.032 * 10.5 / fig.get_figheight(), subtitle, ha="left", va="top", color=INK_2, fontsize=9.5)
    fig.text(0.02, 0.012, source, color=INK_2, fontsize=8)


def section_label(ax, y, text, color):
    ax.annotate("●", (0, y), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                textcoords="offset points", ha="right", va="center", color=color, fontsize=11)
    ax.annotate(text, (0, y), xycoords=("axes fraction", "data"), xytext=(2, 0),
                textcoords="offset points", ha="left", va="center", color=INK, fontweight="bold")


def short_number(v):
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:.3g}{suffix}"
    return f"{v:.0f}"


# --- headline ----------------------------------------------------------------

def phrases_panel(ax, premium, k):
    talk = premium[premium.direction == "talk"].head(k).sort_values("or_")
    poster = premium[premium.direction == "poster"].head(k).sort_values("or_")
    rows = pd.concat([poster, talk])  # posters at the bottom, talks on top
    y = list(range(len(poster))) + [len(poster) + 1.2 + i for i in range(len(talk))]
    for yi, r in zip(y, rows.itertuples()):
        dot_ci(ax, r.or_, r.or_lo, r.or_hi, yi, TALK if r.direction == "talk" else POSTER)
    ax.axvline(1, color=INK_2, lw=1, zorder=1)
    style(ax)
    log_axis(ax, rows.or_lo.min() / 1.15, rows.or_hi.max() * 1.15,
             [0.1, 0.2, 0.5, 1, 2, 5, 10, 20], lambda v, _: f"{v:g}×")
    ax.set_yticks(y, rows.ngram.str.replace("<num>", "N", regex=False))
    ax.set_ylim(-0.8, y[-1] + 1.6)
    section_label(ax, y[-1] + 1, "More common in talks", TALK)
    section_label(ax, len(poster), "More common in posters", POSTER)
    ax.set_xlabel("Odds ratio of being a talk (vs poster), with vs without the phrase · 95% CI",
                  color=INK_2, labelpad=8)


def sample_size_panel(ax, ss):
    ss = ss.set_index("type")
    for yi, t in enumerate(["poster", "talk"]):
        r = ss.loc[t]
        dot_ci(ax, r.median_max_n, r.q25, r.q75, yi, TALK if t == "talk" else POSTER)
    style(ax)
    log_axis(ax, 10, 1e6 * 1.5, [10, 100, 1e3, 1e4, 1e5, 1e6], lambda v, _: short_number(v))
    ax.set_yticks([0, 1], [f"{label}: median {short_number(ss.loc[t].median_max_n)}"
                           for t, label in [("poster", "Posters"), ("talk", "Talks")]])
    ax.set_ylim(-0.7, 1.7)
    # sub-figure header at the figure's left margin, in line with the title
    ax.annotate("Sample size: talks report larger cohorts", (0.02, 1), xycoords=("figure fraction", "axes fraction"),
                xytext=(0, 12), textcoords="offset points", ha="left", va="bottom", color=INK,
                fontweight="bold", fontsize=12)
    ax.set_xlabel(f"Largest stated sample size per abstract (\"12,000 individuals\") · median and IQR\n"
                  f"(an N is stated in {ss.loc['talk'].coverage:.0%} of talks and {ss.loc['poster'].coverage:.0%} "
                  f"of posters)", color=INK_2, labelpad=6)


def headline(premium, summary, ss, year, k, path, pooled=False):
    """Phrase dot plot, plus a talk-vs-poster sample-size panel for a single year (the overview page shows
    sample size per year in its own figure, trends_sample_size.png, so the pooled version has none)."""
    if pooled:
        fig, gs = new_figure((8, 10.8), nrows=1, left=0.3, right=0.97, top=0.86, bottom=0.11)
    else:
        fig, gs = new_figure((8, 12), nrows=2, height_ratios=[15, 1.5], hspace=0.3,
                             left=0.3, right=0.97, top=0.875, bottom=0.085)
        sample_size_panel(fig.add_subplot(gs[1]), ss)
    phrases_panel(fig.add_subplot(gs[0]), premium, k)
    s = summary.iloc[0]
    header(fig, f"What gets an ASHG {year} abstract a talk?",
           f"Phrases most over-represented in talks vs posters: {int(s.text_talks):,} talks (platform + plenary) "
           f"vs {int(s.text_abstracts - s.text_talks):,}\nposters with complete text. {CAVEAT_2LINE}",
           f"Source: ASHG {year} abstracts. Phrases: 1–3-word n-grams in abstract text, ranked by "
           + ("z-score of the\nodds ratio combined across years (Mantel–Haenszel); " if pooled else "z-score; ")
           + "nested duplicates removed.")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- leaderboards ------------------------------------------------------------

def share_figure(rows, labels, counts, baseline, title, subtitle, source, path, row_h=0.36, left=0.36):
    """Dot + Wilson CI of talk rate per row (top row first), baseline line, k/n column at right."""
    n = len(rows)
    h = 2.1 + row_h * n
    fig, gs = new_figure((8, h), nrows=1, left=left, right=0.85, top=1 - 1.3 / h, bottom=0.8 / h)
    ax = fig.add_subplot(gs[0])
    y = list(range(n))[::-1]
    for yi, r in zip(y, rows.itertuples()):
        dot_ci(ax, r.talk_rate, r.ci_lo, r.ci_hi, yi, TALK)
        ax.annotate(counts[n - 1 - yi], (1, yi), xycoords=("axes fraction", "data"), xytext=(10, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
    ax.annotate("talks / abstracts", (1, n - 0.3), xycoords=("axes fraction", "data"), xytext=(10, 0),
                textcoords="offset points", va="bottom", color=INK_2, fontsize=8)
    ax.axvline(baseline, color=INK_2, lw=1, zorder=1)
    ax.annotate(f"all abstracts\n{baseline:.1%}", (baseline, n - 0.4), xytext=(4, 0), textcoords="offset points",
                va="bottom", ha="left", color=INK_2, fontsize=8)
    style(ax)
    ax.set_xlim(0, min(1, rows.ci_hi.max() * 1.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, n + 0.4)
    ax.set_xlabel("Talk rate (talks / accepted abstracts) · 95% Wilson CI", color=INK_2, labelpad=8)
    header(fig, title, subtitle, source, top=1 - 0.2 / fig.get_figheight())
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


ROLE = {"senior": "senior (last) author", "first": "first author"}


def largest_with_talks(df, n, basis):
    """The n largest rows for one credit basis, leaving out any with no talks (the public site never
    names an institution or metro for having none). Returns (rows sorted by talk rate, subtitle note)."""
    top = df[df.basis == basis].head(n)
    rows = top[top.talks > 0].sort_values(["talk_rate", "abstracts"], ascending=False)
    return rows, " Rows without a talk are not shown." if len(rows) < len(top) else ""


def institutions_figure(inst, summary, year, n_inst, path, basis="first"):
    pool = inst[inst.basis == basis]  # the largest institutions by volume (analysis.py keeps 50)
    rows = pool[pool.talks > 0].sort_values(["talk_rate", "abstracts"], ascending=False).head(n_inst)
    share_figure(
        rows, list(rows.institute), [f"{r.talks} / {r.abstracts}" for r in rows.itertuples()],
        summary.iloc[0].talk_rate,
        f"Institutions with the highest talk rate, ASHG {year}",
        f"Top {len(rows)} by talk rate among the {len(pool)} institutions with the most abstracts.\n"
        f"Institutions of the {ROLE[basis]}; an abstract counts for each of them.\n{CAVEAT}",
        f"Source: ASHG {year} abstracts. Institutions resolved with a hand-curated alias list; "
        "schools fold into their university (except HMS).", path, left=0.44)


def metros_figure(metros, summary, year, n_metros, path, basis="first", hosts=None):
    """hosts: {year: metro} of the meetings covered (one for a single year)."""
    rows, note = largest_with_talks(metros, n_metros, basis)
    hosts = hosts or {}
    host_of = {m: y for y, m in hosts.items()}
    pooled = len(hosts) > 1
    labels = [r.metro + (f" (host {host_of[r.metro]})" if pooled else " (host city)") if r.metro in host_of
              else r.metro for r in rows.itertuples()]
    held = (", ".join(f"{y} {m}" for y, m in hosts.items()) if pooled
            else f"ASHG {year} was held in {next(iter(hosts.values()), '')}")
    host_note = (f"Meetings: {held};\n" if pooled else f"{held}; ") + "local groups tend to bring extra posters to a home meeting.\n"
    share_figure(
        rows, labels, [f"{r.talks} / {r.abstracts}" for r in rows.itertuples()],
        summary.iloc[0].talk_rate,
        f"Talk rate in the {n_metros} largest metro areas, ASHG {year}",
        f"The {n_metros} metro areas with the most accepted abstracts, by the {ROLE[basis]}'s institutions.\n"
        f"An abstract counts once per metro.{note}\n"
        + (host_note if hosts else "") + CAVEAT,
        f"Source: ASHG {year} abstracts. Hand-made institution-to-metro map; multi-metro institutions "
        f"left out ({summary.iloc[0][f'{basis}_metro_coverage']:.0%} of abstracts placed).",
        path, left=0.3)


def regions_figure(regions, summary, year, path, basis="first"):
    rows = regions[regions.basis == basis].sort_values(["talk_rate", "abstracts"], ascending=False)
    coverage = summary.iloc[0][f"{basis}_country_coverage"]
    gap = (f"Only {coverage:.0%} of abstracts could be placed (this year's affiliations rarely give a location), and "
           "the misses lean non-US.\n" if coverage < 0.95 else "")
    share_figure(
        rows, list(rows.region), [f"{r.talks} / {r.abstracts}" for r in rows.itertuples()],
        summary.iloc[0].talk_rate,
        f"Talk rate by world region, ASHG {year}",
        f"Country of the {ROLE[basis]}'s affiliations, grouped into UN regions; an abstract counts once per "
        f"region.\n{gap}{CAVEAT}",
        f"Source: ASHG {year} abstracts. Countries from affiliation locations, else the resolved institution "
        "or place names, or a hand-checked list.", path, left=0.36)


def senior_authors_figure(authors, summary, year, path, pooled=False):
    authors = authors[authors.talks > 0]  # never name a person for having no talks
    labels = [f"{r.name}\n{r.institute}" if isinstance(r.institute, str) else r.name for r in authors.itertuples()]
    share_figure(
        authors, labels, [f"{r.talks} / {r.abstracts}" for r in authors.itertuples()],
        summary.iloc[0].talk_rate,
        f"Senior authors with the highest talk rate, ASHG {year}",
        f"Senior (last) authors with ≥{int(authors.min_abstracts.iloc[0])} abstracts, ranked by the lower end of "
        f"the 95% interval (so 5 of 6 beats 2 of 2).\n{CAVEAT}",
        f"Source: ASHG {year} abstracts. Same person = same "
        + ("first initial + last name across years\n(plus a shared institution for common surnames; different "
           "spelled-out first names kept apart)." if pooled else
           "first + last name (plus a shared institution for common surnames)."), path, row_h=0.62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", help="a year, or 'all' for every year pooled")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-inst", type=int, default=20)
    ap.add_argument("--n-metros", type=int, default=20)
    args = ap.parse_args()
    out, y = pathlib.Path(args.out), args.year
    pooled = y == "all"
    years = sorted(int(p.name[:4]) for p in out.glob("[0-9][0-9][0-9][0-9]_summary.csv"))
    label = f"{years[0]}–{years[-1]}" if pooled else y
    hosts = {v: HOST_METRO[v] for v in years if v in HOST_METRO} if pooled else \
        {int(y): HOST_METRO[int(y)]} if int(y) in HOST_METRO else {}
    summary = pd.read_csv(out / f"{y}_summary.csv")
    premium = pd.read_csv(out / f"{y}_premium.csv").rename(columns={"or": "or_"})
    headline(premium, summary, pd.read_csv(out / f"{y}_sample_size.csv"), label, args.k, out / f"{y}_headline.png",
             pooled)
    inst, metros = pd.read_csv(out / f"{y}_institutions.csv"), pd.read_csv(out / f"{y}_metros.csv")
    regions = pd.read_csv(out / f"{y}_regions.csv")
    for basis, suffix in (("first", ""), ("senior", "_senior")):
        institutions_figure(inst, summary, label, args.n_inst, out / f"{y}_institutions{suffix}.png", basis)
        metros_figure(metros, summary, label, args.n_metros, out / f"{y}_metros{suffix}.png", basis, hosts)
        regions_figure(regions, summary, label, out / f"{y}_regions{suffix}.png", basis)
    senior_authors_figure(pd.read_csv(out / f"{y}_senior_authors.csv"), summary, label,
                          out / f"{y}_senior_authors.png", pooled)
    print(f"wrote {y}_headline.png, {y}_{{institutions,metros,regions}}{{,_senior}}.png, "
          f"{y}_senior_authors.png to {out}/")


if __name__ == "__main__":
    main()
