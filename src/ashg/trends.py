"""Cross-year summary figures: selected phrases, the largest metro areas and world regions.

- outputs/trends_phrases.csv/.png  per phrase and year: share of talks and of posters mentioning
                                   it, plus the talk-vs-poster odds ratio (+0.5 smoothing, 95% CI)
- outputs/trends_metros.csv/.png   per metro and year: first-author abstracts and talk rate
                                   (Wilson 95% CI), from data/institutions + metros.csv
- outputs/trends_regions.csv/.png  per UN region and year: first-author abstracts and talk rate,
                                   from data/geography
- outputs/trends_drivers.json      per phrase and year: talks and posters mentioning it, and the
                                   first-author institutions and metros with the most abstracts
                                   mentioning it (for the site's explorer)

The figure shows the PHRASES below; the CSV and the explorer add the top EXPLORER_TOP phrases each
way from the pooled ranking (features.py all; related phrases grouped as in all_premium.csv).

A phrase spec is "label" or "label|alternative|..."; the last word of each may also take a
plural "s". Abstracts cut off in a booklet are left out of the phrase counts.

Usage: python src/ashg/trends.py [--years 2022 2023 2024 2025] [--phrases SPEC ...] [--n-metros 12]
                                [--explorer-top 50]
"""

import argparse
import json
import pathlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from analysis import ROOT, load_metros, premium_table
from features import mantel_haenszel, tokenize
from figures import GRID, HOST_METRO, INK, INK_2, POSTER, SURFACE, TALK, header, new_figure
from geography import REGION_OF

PHRASES = {
    "Consistently more common in talks": ["chromatin", "CRISPR", "cell type-specific", "largest",
                                          "MPRA|massively parallel reporter", "colocalization"],
    "Consistently more common in posters": ["patients", "next-generation sequencing", "panel", "likely pathogenic",
                                            "literature", "diagnosed"],
    "Topics to watch": ["long-read|long read", "single-cell|single cell", "machine learning",
                        "language model|large language model", "polygenic score|polygenic risk score",
                        "spatial transcriptomics|spatial transcriptomic"],
}
TOP_DRIVERS = 50  # rows kept per list; the site shows the top 10 with at least one talk
EXPLORER_TOP = 50  # pooled top phrases each way added to the explorer
SHORT = {"next-generation sequencing": "next-gen sequencing"}


def contains(tokens, phrase):
    n = len(phrase)
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i + n]
        if window[:-1] == phrase[:-1] and window[-1] in (phrase[-1], phrase[-1] + "s"):
            return True
    return False


def phrase_hits(abstracts, specs):
    """Complete abstracts (id, year, talk) with one boolean column per phrase label."""
    df = abstracts[~abstracts.truncated]
    docs = [tokenize(t) for t in df.title.fillna("") + " . " + df.body.fillna("")]
    words = [set(d) for d in docs]  # cheap pre-check before scanning a document
    out = pd.DataFrame({"abstract_id": df.abstract_id.to_numpy(), "year": df.year.to_numpy(),
                        "talk": (df.pres_type != "poster").to_numpy()})
    hits = {}
    for spec in specs:
        alts = [tokenize(a) for a in spec.split("|")]
        hits[spec.split("|")[0]] = [any((a[0] in w or a[0] + "s" in w) and contains(d, a) for a in alts)
                                    for d, w in zip(docs, words)]
    return pd.concat([out, pd.DataFrame(hits)], axis=1)


def explorer_specs(data, n, taken):
    """{group: [spec]} for the pooled top-n phrases each way, skipping any already in `taken` (specs).
    "<num>" n-grams get a label with "N" and match any number."""
    seen = {tuple(tokenize(a)) for spec in taken for a in spec.split("|")}
    top = premium_table(data, "all", n)
    out = {}
    for direction, group in (("talk", "Top in talks, all years"), ("poster", "Top in posters, all years")):
        out[group] = []
        for g in top[top.direction == direction].ngram:
            spec = g.replace("<num>", "N") + "|" + g.replace("<num>", "0") if "<num>" in g else g
            if tuple(tokenize(spec.split("|")[-1])) not in seen:
                out[group].append(spec)
    return out


def phrase_table(hits, phrases):
    rows = []
    for year, g in hits.groupby("year"):
        talk = g.talk.to_numpy()
        for p in phrases:
            hit = g[p].to_numpy()
            a, c = (hit & talk).sum(), (hit & ~talk).sum()
            b, d = talk.sum() - a, (~talk).sum() - c
            log_or = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
            se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
            rows.append({"year": year, "phrase": p, "abstracts": len(g), "docs": int(hit.sum()), "share": hit.mean(),
                         "talks": int(talk.sum()), "posters": int((~talk).sum()),
                         "talk_docs": int(a), "poster_docs": int(c),
                         "talk_share": a / talk.sum(), "poster_share": c / (~talk).sum(), "or": np.exp(log_or),
                         "or_lo": np.exp(log_or - 1.96 * se), "or_hi": np.exp(log_or + 1.96 * se)})
    return pd.DataFrame(rows)


def phrase_entry(phrase, group, rows):
    """Explorer entry: per-year [talks, posters] mentioning the phrase, and the year-stratified
    (Mantel-Haenszel) odds ratio with 95% CI over all years."""
    a, c, talks, posters = (rows[k].to_numpy() for k in ("talk_docs", "poster_docs", "talks", "posters"))
    log_or, se = mantel_haenszel(*(x[:, None] for x in (a, talks - a, c, posters - c)))
    ci = np.exp(log_or[0] + np.array([0, -1.96, 1.96]) * se[0])
    return {"phrase": phrase, "group": group, "counts": np.c_[a, c].tolist(), "or": [round(float(v), 3) for v in ci]}


def wilson(k, n, z=1.96):
    p = k / n
    centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def first_credit(data, years, metro_of):
    """One row per (abstract, first-author institute), with its metro (NaN if none)."""
    insts = pd.concat([pd.read_parquet(data / f"institutions/{y}.parquet") for y in years], ignore_index=True)
    insts["talk"] = insts.pres_type != "poster"
    credit = insts.explode("first_insts").rename(columns={"first_insts": "institute"})
    credit = credit.assign(metro=credit.institute.map(metro_of))
    return insts, credit


def metro_table(insts, credit, years, n_metros):
    """First-author credit per metro and year, every metro (not just each year's top 30)."""
    c = credit.dropna(subset=["metro"]).drop_duplicates(["abstract_id", "metro"])
    m = c.groupby(["year", "metro"]).agg(abstracts=("abstract_id", "size"), talks=("talk", "sum")).reset_index()
    m["baseline"] = m.year.map(insts.groupby("year").talk.mean())
    m["talk_rate"] = m.talks / m.abstracts
    m["ci_lo"], m["ci_hi"] = wilson(m.talks, m.abstracts)
    in_all = m.groupby("metro").year.nunique() == len(years)
    top = m[m.metro.isin(in_all[in_all].index)].groupby("metro").abstracts.sum().nlargest(n_metros).index
    hosts = [HOST_METRO[y] for y in years if HOST_METRO.get(y) in set(m.metro)]
    keep = list(dict.fromkeys([*top, *hosts]))
    return m[m.metro.isin(keep)][["year", "metro", "abstracts", "talks", "talk_rate", "ci_lo", "ci_hi", "baseline"]]


def drivers(hits, credit, phrases, years):
    """{phrase: {year or "all": {talks, posters, unmatched, institutes: [...], metros: [...]}}}.

    Rows are [name, talks, posters, entity_abstracts]: abstracts mentioning the phrase with a
    first author at that institute/metro, split by talk/poster, and that entity's complete
    abstracts in the same period (so the explorer can show its own mention rate).
    """
    complete = credit[credit.abstract_id.isin(hits.abstract_id)]
    out = {}
    for p in phrases:
        out[p] = {}
        for period in ["all", *years]:
            h = hits if period == "all" else hits[hits.year == period]
            mention = h.loc[h[p], ["abstract_id", "talk"]]
            scope = complete if period == "all" else complete[complete.year == period]
            entry = {"talks": int(mention.talk.sum()), "posters": int((~mention.talk).sum())}
            matched = set(scope.dropna(subset=["institute"]).abstract_id)
            entry["unmatched"] = int((~mention.abstract_id.isin(matched)).sum())
            for kind in ("institute", "metro"):
                s = scope.dropna(subset=[kind]).drop_duplicates(["abstract_id", kind])
                totals = s.groupby(kind).size()
                hit = s[s.abstract_id.isin(mention.abstract_id)]
                agg = hit.groupby(kind).talk.agg(talks="sum", n="size")
                agg = agg.assign(posters=agg.n - agg.talks).sort_values(["n", "talks"], ascending=False)
                entry[kind + "s"] = [[name, int(r.talks), int(r.posters), int(totals[name])]
                                     for name, r in agg.head(TOP_DRIVERS).iterrows()]
            out[p][str(period)] = entry
    return out


# --- figures: years run left to right, one small panel per phrase / metro ------------------

def panel_style(ax, years):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_xlim(-0.35, len(years) - 0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10], min_n_ticks=3))  # whole-% ticks, no "2.5%"


def series(ax, xs, ys, color, ring=None):
    ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(xs, ys, s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)


def panel_title(ax, text, width=22):
    ax.set_title(textwrap.fill(text, width), loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)


def phrases_figure(t, years, path):
    groups = [(g, [p.split("|")[0] for p in specs]) for g, specs in PHRASES.items()]
    groups = [(g, [p for p in ps if p in set(t.phrase)]) for g, ps in groups if ps]
    ncols = max(len(ps) for _, ps in groups)
    w, h = max(10, 2 * ncols), 1.6 + 2.45 * len(groups)
    fig, gs = new_figure((w, h), nrows=len(groups), ncols=ncols, hspace=1.05, wspace=0.32,
                         left=0.06, right=0.98, top=1 - 1.75 / h, bottom=0.55 / h)
    xs = np.arange(len(years))
    for row, (g, ps) in enumerate(groups):
        for col, p in enumerate(ps):
            ax = fig.add_subplot(gs[row, col])
            r = t[t.phrase == p].set_index("year").loc[years]
            panel_style(ax, years)
            series(ax, xs, r.poster_share, POSTER)
            series(ax, xs, r.talk_share, TALK)
            ax.set_ylim(0, max(r.talk_share.max(), r.poster_share.max()) * 1.25)
            panel_title(ax, SHORT.get(p, p), width=19)  # wrapped titles must stay clear of the group label
            if col == 0:  # group label above the row, aligned with the figure margin
                ax.annotate(g, (0.06, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, 24),
                            textcoords="offset points", ha="left", va="bottom", color=INK_2, fontsize=9.5)
    legend = [Line2D([], [], color=TALK, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="share of talks"),
              Line2D([], [], color=POSTER, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="share of posters")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.02, 1 - 0.85 / h), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.8)
    header(fig, f"Selected phrases, ASHG {years[0]}–{years[-1]}",
           "Share of talks and of posters mentioning each phrase (title or abstract text). The gap between the lines "
           "is the talk premium.",
           "Source: ASHG abstract booklets (plurals count). Each panel has its own scale. Accepted abstracts only: "
           "association, not acceptance odds.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def share_panels(m, key, years, path, title, subtitle, source, ncols=4, hosts=None):
    """One talk-rate panel per m[key] value (largest first), with Wilson band and the yearly baseline;
    hosts = {year: value} rings the host city."""
    order = list(m.groupby(key).abstracts.sum().sort_values(ascending=False).index)
    nrows = -(-len(order) // ncols)
    h = 2.3 + 2.0 * nrows
    fig, gs = new_figure((10, h), nrows=nrows, ncols=ncols, hspace=0.95, wspace=0.18,
                         left=0.06, right=0.98, top=1 - 1.85 / h, bottom=0.95 / h)
    xs = np.arange(len(years))
    ymax = min(1, m.ci_hi.max() * 1.05)
    base = m.groupby("year").baseline.first().loc[years]
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        r = m[m[key] == name].set_index("year").reindex(years)
        panel_style(ax, years)
        ax.plot(xs, base, color=INK_2, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.fill_between(xs, r.ci_lo, r.ci_hi, color=TALK, alpha=0.14, lw=0, zorder=1)
        series(ax, xs, r.talk_rate, TALK)
        # abstracts per year sit under each year label, clear of the data
        ax.set_xticks(xs, [f"{y}\n{'' if pd.isna(n) else f'n={int(n)}'}" for y, n in zip(years, r.abstracts)])
        for x, y in zip(xs, years):
            if hosts and hosts.get(y) == name:
                ax.scatter(x, r.talk_rate[y], s=150, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        ax.set_ylim(0, ymax)
        if i % ncols:
            ax.tick_params(axis="y", labelleft=False)
        panel_title(ax, name)
    legend = [Line2D([], [], color=TALK, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="talk rate"),
              Patch(color=TALK, alpha=0.14, label="95% Wilson CI"),
              Line2D([], [], color=INK_2, lw=1, ls=(0, (2, 2)), label="all abstracts")]
    if hosts:
        legend.append(Line2D([], [], ls="", marker="o", ms=11, markerfacecolor="none", markeredgecolor=INK,
                             label="host city that year"))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.02, 1 - 1.05 / h), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.8)
    header(fig, title, subtitle, source, top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def metros_figure(m, years, path):
    meetings = ", ".join(f"{y} {HOST_METRO[y]}" for y in years if y in HOST_METRO)
    share_panels(
        m, "metro", years, path, f"Talk rate in the largest metro areas, ASHG {years[0]}–{years[-1]}",
        "First-author metro; the metros with the most accepted abstracts over all years, plus each host city; "
        f"n = abstracts that year.\nMeetings: {meetings}. Local groups tend to bring extra posters to a home meeting.",
        "Source: ASHG abstract booklets. Hand-made institution-to-metro map; an abstract counts once per metro. "
        "Accepted abstracts only: association, not acceptance odds.", hosts=HOST_METRO)


def region_table(data, years):
    """First-author credit per UN region and year from data/geography (an abstract counts once per
    region), plus each year's share of abstracts with a known country (`coverage`; first authors
    survive a cut-off abstract, so this is over every abstract)."""
    geo = pd.concat([pd.read_parquet(data / f"geography/{y}.parquet") for y in years], ignore_index=True)
    geo["talk"] = geo.pres_type != "poster"
    credit = geo.explode("first_countries").dropna(subset=["first_countries"])
    credit = credit.assign(region=credit.first_countries.map(REGION_OF)).drop_duplicates(["abstract_id", "region"])
    m = credit.groupby(["year", "region"]).agg(abstracts=("abstract_id", "size"), talks=("talk", "sum")).reset_index()
    m["baseline"] = m.year.map(geo.groupby("year").talk.mean())
    m["coverage"] = m.year.map(geo.groupby("year").first_countries.apply(lambda s: (s.str.len() > 0).mean()))
    m["talk_rate"] = m.talks / m.abstracts
    m["ci_lo"], m["ci_hi"] = wilson(m.talks, m.abstracts)
    m["ci_lo"] = m.ci_lo.clip(lower=0)
    return m[["year", "region", "abstracts", "talks", "talk_rate", "ci_lo", "ci_hi", "baseline", "coverage"]]


def regions_figure(m, years, path):
    cov = m.groupby("year").coverage.first()
    good = cov[cov >= 0.95]
    low = [f"{y} ({cov[y]:.0%})" for y in years if cov[y] < 0.95]
    found = f"{good.min():.1%}" if good.min() == good.max() else f"{good.min():.1%}–{good.max():.1%}"
    gap = f"Countries found for {found} of abstracts" + (
        ", except " + ", ".join(low) + ": affiliations there rarely give a location, and the misses lean non-US."
        if low else ".")
    share_panels(
        m, "region", years, path, f"Talk rate by world region, ASHG {years[0]}–{years[-1]}",
        f"Country of the first author, grouped into UN regions; n = abstracts that year.\n{gap}",
        "Source: ASHG abstract booklets. Countries from affiliation locations, else the resolved institution, place "
        "names or a hand-checked list. Accepted abstracts only: association, not acceptance odds.", ncols=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--phrases", nargs="+", help="override the default phrase list (one group)")
    ap.add_argument("--n-metros", type=int, default=12)
    ap.add_argument("--explorer-top", type=int, default=EXPLORER_TOP)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--metros", default=ROOT / "metros.csv")
    args = ap.parse_args()
    data, out, years = pathlib.Path(args.data), pathlib.Path(args.out), sorted(args.years)
    if args.phrases:
        PHRASES.clear()
        PHRASES["Selected phrases"] = args.phrases
    groups = {**PHRASES, **explorer_specs(data, args.explorer_top, [p for ps in PHRASES.values() for p in ps])}
    specs = [p for ps in groups.values() for p in ps]
    labels = [s.split("|")[0] for s in specs]
    group_of = {s.split("|")[0]: g for g, ps in groups.items() for s in ps}
    abstracts = pd.concat([pd.read_parquet(p) for y in years for p in sorted((data / "abstracts").glob(f"{y}_*.parquet"))])
    hits = phrase_hits(abstracts, specs)
    t = phrase_table(hits, labels)
    t.insert(2, "group", t.phrase.map(group_of))
    t.to_csv(out / "trends_phrases.csv", index=False, float_format="%.4g")
    phrases_figure(t, years, out / "trends_phrases.png")

    insts, credit = first_credit(data, years, load_metros(args.metros).set_index("institute").metro)
    m = metro_table(insts, credit, years, args.n_metros)
    m.to_csv(out / "trends_metros.csv", index=False, float_format="%.4g")
    metros_figure(m, years, out / "trends_metros.png")

    regions = region_table(data, years)
    regions.to_csv(out / "trends_regions.csv", index=False, float_format="%.4g")
    regions_figure(regions, years, out / "trends_regions.png")

    totals = t.drop_duplicates("year").set_index("year").loc[years]
    payload = {"years": years, "totals": totals[["talks", "posters"]].values.tolist(),
               "phrases": [phrase_entry(p, group_of[p], t[t.phrase == p].set_index("year").loc[years]) for p in labels],
               "drivers": drivers(hits, credit, labels, years)}
    (out / "trends_drivers.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote trends_phrases.csv/.png ({len(labels)} phrases, {sum(map(len, PHRASES.values()))} in the figure), trends_metros.csv/.png "
          f"({m.metro.nunique()} metros), trends_regions.csv/.png, trends_drivers.json to {out}/")


if __name__ == "__main__":
    main()
