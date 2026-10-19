"""Cross-year summary figures: selected phrases, the largest metro areas and world regions.

- outputs/trends_phrases.csv/.png  per phrase and year: share of talks and of posters mentioning
                                   it, plus the talk-vs-poster odds ratio (+0.5 smoothing, 95% CI)
- outputs/trends_biobanks.png      per biobank and year: share of all abstracts mentioning it
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

Usage: python src/ashg/trends.py [--years 2022 2023 2024 2025 2026] [--phrases SPEC ...] [--n-metros 12]
                                [--explorer-top 50]
"""

import argparse
import json
import pathlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator, PercentFormatter

from analysis import ROOT, load_metros, premium_table
from features import mantel_haenszel, tokenize
from figures import GRID, HOST_METRO, INK, INK_2, POSTER, SURFACE, TALK, header, new_figure, short_number
from geography import REGION_OF

PHRASES = {
    "Consistently more common in talks": ["chromatin", "CRISPR", "cell type-specific", "largest",
                                          "MPRA|massively parallel reporter", "colocalization"],
    "Consistently more common in posters": ["panel", "literature", "diagnosed", "Sanger sequencing", "microarray",
                                            "symptoms"],
    # rising share of abstracts 2022-2026 (trend z >= 4); reviewed with 2026 in: machine learning and polygenic
    # score were flat and spatial transcriptomics had 5 talks in 2026, so they gave way to the fastest risers
    "Topics to watch": ["long-read|long read", "All of Us","language model|large language model",
                        "proteomics|proteomic", "multi-ancestry|multi-ancestral", "pangenome|pan-genome"],
}
# Clinical terms that leaned toward posters through 2025 but not in 2026 (trends_shift.png). The first group was
# in the posters group above before 2026; the second came from scanning every phrase for the largest 2026 shift,
# so its p-values are optimistic. No single shift survives an FDR correction over all phrases.
SHIFT = {
    "Poster-leaning through 2025 (pre-specified)": ["patients", "likely pathogenic", "diagnosis|diagnoses"],
    "Largest 2026 shifts among clinical terms (found by scanning every phrase)": [
        "idiopathic", "genetic disease", "therapy"],
}
# The most-mentioned biobanks 2022-2026 (trends_biobanks.png); short forms checked against the full names. Left out:
# eMERGE and deCODE (also ordinary words), TOPMed and gnomAD (not biobanks), and biobanks with under ~10 mentions a year.
BIOBANKS = {
    "Biobanks": ["UK Biobank|UKB|UKBB|UK-Biobank", "All of Us", "FinnGen",
                 "Million Veteran Program|Million Veterans Program|MVP",
                 "BioBank Japan|Biobank of Japan|BioBankJapan|BBJ", "Taiwan Biobank|TWB|TWBB", "BioVU",
                 "Penn Medicine Biobank|PMBB"],
}
# Topics for the overview's scrolling "what's hot" chart (share of all abstracts per year): changes in what the
# field uses, picked by hand from scanning the explorer's yearly counts. Not in any PNG.
STORIES = {
    "Field shifts": ["whole-genome sequencing|whole genome sequencing|WGS",
                     "exome sequencing|whole-exome sequencing|whole exome sequencing|exome-sequencing|WES",
                     "AlphaMissense", "gnomAD v4|gnomADv4", "COVID-19|SARS-CoV-2|COVID"],
}
TOP_DRIVERS = 50  # rows kept per list; the site shows the top 10 with at least one talk
EXPLORER_TOP = 50  # pooled top phrases each way added to the explorer


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


def more_phrases(data, years, taken):
    """Every pooled n-gram from features.py (in >= 5 talks over all years) not already in `taken`, as compact
    explorer rows [label, [talks, posters] per year flattened, or, or_lo, or_hi], most-mentioned first.
    Counts come from the pooled doc-term matrix, so unlike the curated phrases they are exact n-grams in
    the abstract text (no titles, no plurals); the OR is the same Mantel-Haenszel one as the phrase pages."""
    X = sparse.load_npz(data / "features/doc_term_all.npz").tocsc()
    vocab = pd.read_parquet(data / "features/vocab_all.parquet").ngram.to_numpy()
    # rows are features.py's: every data/abstracts/*.parquet in glob order, cut-off abstracts dropped
    meta = pd.concat([pd.read_parquet(p, columns=["year", "pres_type", "truncated"])
                      for p in sorted((data / "abstracts").glob("*.parquet"))], ignore_index=True)
    meta = meta[~meta.truncated]
    if len(meta) != X.shape[0]:
        raise ValueError("doc_term_all.npz is stale: rerun `make pooled`")
    talk, year = (meta.pres_type != "poster").to_numpy(), meta.year.to_numpy()
    counts = np.stack([np.asarray(X[m].sum(axis=0)).ravel() for y in years for m in (talk & (year == y),
                                                                                      ~talk & (year == y))], 1)
    prem = pd.read_parquet(data / "features/premium_all.parquet").set_index("ngram")
    seen = {" ".join(tokenize(s.split("|")[-1])) for s in taken}
    rows = []
    for j in np.argsort(-counts.sum(1), kind="stable"):
        g = vocab[j]
        label = g.replace("<num>", "N")
        if g in seen or label in seen or g.replace("<num>", "0") in seen or g not in prem.index:
            continue
        r = prem.loc[g]
        rows.append([label, counts[j].tolist(), *(round(float(r[k]), 3) for k in ("or", "or_lo", "or_hi"))])
    return rows


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

    Rows are [name, talks, posters, entity_abstracts, entity_talks]: abstracts mentioning the
    phrase with a first author at that institute/metro, split by talk/poster, and that entity's
    complete abstracts and talks in the same period (so the explorer can show its own mention
    rate and its odds of a talk with vs without the phrase).
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
                totals = s.groupby(kind).talk.agg(talks="sum", n="size")
                hit = s[s.abstract_id.isin(mention.abstract_id)]
                agg = hit.groupby(kind).talk.agg(talks="sum", n="size")
                agg = agg.assign(posters=agg.n - agg.talks).sort_values(["n", "talks"], ascending=False)
                entry[kind + "s"] = [[name, int(r.talks), int(r.posters), int(totals.n[name]), int(totals.talks[name])]
                                     for name, r in agg.head(TOP_DRIVERS).iterrows()]
            out[p][str(period)] = entry
    return out


# --- figures: years run left to right, one small panel per phrase / metro ------------------

def panel_style(ax, years, short_years=False):
    """short_years: "’22" labels for narrow panels, where five full years run together."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    ax.set_xticks(range(len(years)), [f"’{y % 100:02d}" if short_years else str(y) for y in years])
    ax.set_xlim(-0.35, len(years) - 0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10], min_n_ticks=3))  # whole-% ticks, no "2.5%"


def series(ax, xs, ys, color, ring=None):
    ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(xs, ys, s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)


def panel_title(ax, text, width=22):
    ax.set_title(textwrap.fill(text, width), loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)


def shift_table(t, phrases, years):
    """Last year vs the earlier years pooled, per phrase: Mantel-Haenszel log OR of a talk over the earlier
    years, the last year's log OR, and a two-sided z-test of the difference."""
    rows = []
    for p in phrases:
        r = t[t.phrase == p].set_index("year").loc[years]
        a, c, n1, n2 = (r[k].to_numpy() for k in ("talk_docs", "poster_docs", "talks", "posters"))
        cells = [x[:, None] for x in (a, n1 - a, c, n2 - c)]
        (lp,), (sp,) = mantel_haenszel(*(x[:-1] for x in cells))
        (ll,), (sl,) = mantel_haenszel(*(x[-1:] for x in cells))
        z = (ll - lp) / np.hypot(sl, sp)
        rows.append({"phrase": p, "or_before": np.exp(lp), "or_last": np.exp(ll), "z": z,
                     "p": 2 * norm.sf(abs(z)), "talks_last": int(a[-1]), "posters_last": int(c[-1])})
    return pd.DataFrame(rows)


def phrases_figure(t, years, path, groups=None, title=None, subtitle=None, notes=None):
    """One small panel per phrase, a row per group ({group: [spec]}; default PHRASES); notes = {phrase: text}
    shown under the panel's year axis."""
    groups = [(g, [p.split("|")[0] for p in specs]) for g, specs in (groups or PHRASES).items()]
    groups = [(g, [p for p in ps if p in set(t.phrase)]) for g, ps in groups if ps]
    ncols = max(len(ps) for _, ps in groups)
    extra = 0.4 if notes else 0  # room for the notes under each row
    w, h = max(10, 2 * ncols), 1.6 + (2.45 + extra) * len(groups)
    fig, gs = new_figure((w, h), nrows=len(groups), ncols=ncols, hspace=1.05 + extra, wspace=0.32,
                         left=0.06, right=0.98, top=1 - 1.75 / h, bottom=(0.55 + extra) / h)
    xs = np.arange(len(years))
    for row, (g, ps) in enumerate(groups):
        for col, p in enumerate(ps):
            ax = fig.add_subplot(gs[row, col])
            r = t[t.phrase == p].set_index("year").loc[years]
            panel_style(ax, years, short_years=True)
            series(ax, xs, r.poster_share, POSTER)
            series(ax, xs, r.talk_share, TALK)
            ax.set_ylim(0, max(r.talk_share.max(), r.poster_share.max()) * 1.25)
            panel_title(ax, p, width=19)  # wrapped titles must stay clear of the group label
            if notes and p in notes:
                ax.set_xlabel(notes[p], color=INK_2, fontsize=8.5, labelpad=4)
            if col == 0:  # group label above the row, aligned with the figure margin
                ax.annotate(g, (0.06, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, 24),
                            textcoords="offset points", ha="left", va="bottom", color=INK_2, fontsize=9.5)
    legend = [Line2D([], [], color=TALK, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="share of talks"),
              Line2D([], [], color=POSTER, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="share of posters")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.02, 1 - 0.85 / h), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.8)
    header(fig, title or f"Selected phrases, ASHG {years[0]}–{years[-1]}",
           subtitle or "Share of talks and of posters mentioning each phrase (title or abstract text). The gap "
           "between the lines is the talk premium.",
           "Source: ASHG abstracts (plurals count). Each panel has its own scale. Accepted abstracts only: "
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


def biobanks_figure(t, years, path, specs, ncols=4):
    """One panel per biobank: share of all accepted abstracts mentioning it, most-mentioned first."""
    names = [s.split("|")[0] for s in specs]
    names.sort(key=lambda p: -t[t.phrase == p].docs.sum())
    nrows = -(-len(names) // ncols)
    h = 2.0 + 2.0 * nrows
    fig, gs = new_figure((10, h), nrows=nrows, ncols=ncols, hspace=0.95, wspace=0.3,
                         left=0.06, right=0.98, top=1 - 1.35 / h, bottom=0.75 / h)
    xs = np.arange(len(years))
    for i, p in enumerate(names):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        r = t[t.phrase == p].set_index("year").loc[years]
        panel_style(ax, years, short_years=True)
        series(ax, xs, r.share, INK)
        # abstracts mentioning it, in a row under the years headed "n =" (n= on every tick runs together)
        for x, n in zip(xs, r.docs):
            ax.annotate(f"{n}", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -17),
                        textcoords="offset points", ha="center", va="top", color=INK_2, fontsize=8.5)
        ax.annotate("n =", (0, 0), xycoords="axes fraction", xytext=(-4, -17), textcoords="offset points",
                    ha="right", va="top", color=INK_2, fontsize=8.5)
        ax.set_ylim(0, r.share.max() * 1.25)
        if r.share.max() < 0.03:  # small shares need a decimal to tell the ticks apart
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 4, 5, 10], min_n_ticks=3))  # no "0.25%" → "0.2%"
        panel_title(ax, p, width=19)
    header(fig, f"Biobanks mentioned at ASHG {years[0]}–{years[-1]}",
           "Share of accepted abstracts mentioning each biobank (title or text, any spelling or acronym such as "
           "UKB, MVP, BBJ).\nBelow each year: abstracts mentioning it.",
           "Source: ASHG abstracts. Each panel has its own scale. Abstracts cut off in a booklet are left out.",
           top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def metros_figure(m, years, path):
    meetings = ", ".join(f"{y} {HOST_METRO[y]}" for y in years if y in HOST_METRO)
    share_panels(
        m, "metro", years, path, f"Talk rate in the largest metro areas, ASHG {years[0]}–{years[-1]}",
        "First-author metro; the metros with the most accepted abstracts over all years, plus each host city; "
        f"n = abstracts that year.\nMeetings: {meetings}. Local groups tend to bring extra posters to a home meeting.",
        "Source: ASHG abstracts. Hand-made institution-to-metro map; an abstract counts once per metro. "
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
        "Source: ASHG abstracts. Countries from affiliation locations, else the resolved institution, place "
        "names or a hand-checked list. Accepted abstracts only: association, not acceptance odds.", ncols=3)


def talk_rate_figure(summary, pooled_rate, years, path):
    """Talk rate (talks / accepted abstracts) per year, direct-labelled, with the pooled rate as a reference."""
    h = 3.9
    fig, gs = new_figure((10, h), left=0.07, right=0.98, top=1 - 1.15 / h, bottom=1.0 / h)
    ax = fig.add_subplot(gs[0])
    s = summary.set_index("year").loc[years]
    xs = np.arange(len(years))
    panel_style(ax, years)
    ax.axhline(pooled_rate, color=INK_2, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.annotate(f"all years {pooled_rate:.1%}", (-0.4, pooled_rate), xytext=(0, -4), textcoords="offset points",
                ha="left", va="top", color=INK_2, fontsize=8.5)
    series(ax, xs, s.talk_rate, TALK)
    for x, v in zip(xs, s.talk_rate):  # label on the side away from the reference line
        above = v >= pooled_rate
        ax.annotate(f"{v:.1%}", (x, v), xytext=(0, 9 if above else -9), textcoords="offset points", ha="center",
                    va="bottom" if above else "top", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(xs, [f"{y}{' · ' + HOST_METRO[y] if y in HOST_METRO else ''}\n{int(r.talks):,} talks / "
                       f"{int(r.abstracts):,} abstracts" for y, r in s.iterrows()])
    ax.set_ylim(0, s.talk_rate.max() * 1.35)
    ax.set_xlim(-0.45, len(years) - 0.55)
    header(fig, f"Talk rate, ASHG {years[0]}–{years[-1]}",
           "Share of accepted abstracts presented as a talk (platform or plenary) each year; the rest are posters.",
           "Source: ASHG abstracts. Accepted abstracts only, and authors can opt for a poster, so this is not an "
           "acceptance rate.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def sample_size_figure(ss, years, path):
    """Median (IQR band) of the largest stated sample size per abstract, talks vs posters, and the share of
    abstracts stating one; ss = the stacked per-year {year}_sample_size.csv tables."""
    h = 4.6
    fig, gs = new_figure((10, h), ncols=2, width_ratios=[1.6, 1], wspace=0.22,
                         left=0.07, right=0.98, top=1 - 1.45 / h, bottom=0.75 / h)
    xs = np.arange(len(years))
    ax_n, ax_cov = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for t, color, dx in (("poster", POSTER, 0.09), ("talk", TALK, -0.09)):  # IQR bars side by side, not overlapping
        r = ss[ss.type == t].set_index("year").reindex(years)
        ax_n.vlines(xs + dx, r.q25, r.q75, color=color, lw=2, alpha=0.35, zorder=1)
        series(ax_n, xs + dx, r.median_max_n, color)
        series(ax_cov, xs, r.coverage, color)
    panel_style(ax_n, years)
    ax_n.set_yscale("log")
    ax_n.yaxis.set_major_locator(LogLocator(base=10))
    ax_n.yaxis.set_minor_locator(NullLocator())
    ax_n.yaxis.set_major_formatter(FuncFormatter(lambda v, _: short_number(v)))
    ax_n.set_ylim(ss.q25.min() / 1.6, ss.q75.max() * 1.6)
    panel_title(ax_n, "Largest stated sample size per abstract", width=60)
    panel_style(ax_cov, years)
    ax_cov.set_ylim(0, 1)
    ax_cov.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    panel_title(ax_cov, "Abstracts stating a sample size", width=60)
    legend = [Line2D([], [], color=TALK, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="talks"),
              Line2D([], [], color=POSTER, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label="posters"),
              Line2D([], [], color=INK_2, lw=2, alpha=0.35, label="middle half (25th–75th percentile)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.02, 1 - 0.8 / h), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_2, handlelength=1.8)
    header(fig, f"Sample size in talks and posters, ASHG {years[0]}–{years[-1]}",
           "Median of the largest human-subject N stated in each abstract (\"12,000 individuals\", \"N = 450\"), "
           "among abstracts that state one.",
           "Source: ASHG abstracts. Log scale at left. Accepted abstracts only: association, not acceptance odds.",
           top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
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
    curated = {**PHRASES, **SHIFT, **BIOBANKS, **STORIES}
    groups = {**curated, **explorer_specs(data, args.explorer_top, [p for ps in curated.values() for p in ps])}
    specs = list(dict.fromkeys(p for ps in groups.values() for p in ps))  # a phrase in two groups is counted once
    labels = [s.split("|")[0] for s in specs]
    group_of = {}
    for g, ps in groups.items():
        for s in ps:
            group_of.setdefault(s.split("|")[0], g)  # its first group (All of Us stays a topic to watch)
    abstracts = pd.concat([pd.read_parquet(p) for y in years for p in sorted((data / "abstracts").glob(f"{y}_*.parquet"))])
    hits = phrase_hits(abstracts, specs)
    t = phrase_table(hits, labels)
    t.insert(2, "group", t.phrase.map(group_of))
    t.to_csv(out / "trends_phrases.csv", index=False, float_format="%.4g")
    phrases_figure(t, years, out / "trends_phrases.png")

    shift = shift_table(t, [s.split("|")[0] for ps in SHIFT.values() for s in ps], years)
    shift.to_csv(out / "trends_shift.csv", index=False, float_format="%.4g")
    before = f"{years[0]}–{str(years[-2])[2:]}"
    notes = {r.phrase: f"OR {before} {r.or_before:.2f} → {years[-1]} {r.or_last:.2f}\n"
                       + ("change p < 0.001" if r.p < 0.001 else f"change p = {r.p:.2g}")
             for r in shift.itertuples()}
    phrases_figure(t, years, out / "trends_shift.png", SHIFT, notes=notes,
                   title=f"What changed in {years[-1]}: clinical terms in talks",
                   subtitle="Share of talks and of posters mentioning each phrase. Below each panel: odds ratio of a "
                            "talk and p for the change.")
    biobanks_figure(t, years, out / "trends_biobanks.png", BIOBANKS["Biobanks"])

    insts, credit = first_credit(data, years, load_metros(args.metros).set_index("institute").metro)
    m = metro_table(insts, credit, years, args.n_metros)
    m.to_csv(out / "trends_metros.csv", index=False, float_format="%.4g")
    metros_figure(m, years, out / "trends_metros.png")

    regions = region_table(data, years)
    regions.to_csv(out / "trends_regions.csv", index=False, float_format="%.4g")
    regions_figure(regions, years, out / "trends_regions.png")

    ss = pd.concat([pd.read_csv(out / f"{y}_sample_size.csv") for y in years], ignore_index=True)
    ss.to_csv(out / "trends_sample_size.csv", index=False, float_format="%.4g")
    sample_size_figure(ss, years, out / "trends_sample_size.png")

    summary = pd.concat([pd.read_csv(out / f"{y}_summary.csv") for y in years], ignore_index=True)
    talk_rate_figure(summary, summary.talks.sum() / summary.abstracts.sum(), years, out / "trends_talk_rate.png")

    totals = t.drop_duplicates("year").set_index("year").loc[years]
    payload = {"years": years, "totals": totals[["talks", "posters"]].values.tolist(),
               "phrases": [phrase_entry(p, group_of[p], t[t.phrase == p].set_index("year").loc[years]) for p in labels],
               "drivers": drivers(hits, credit, labels, years),
               "more": more_phrases(data, years, specs)}
    (out / "trends_drivers.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote trends_phrases.csv/.png ({len(labels)} phrases, {sum(map(len, PHRASES.values()))} in the figure), trends_metros.csv/.png "
          f"({m.metro.nunique()} metros), trends_regions.csv/.png, trends_sample_size.csv/.png, trends_talk_rate.png, trends_drivers.json "
          f"to {out}/")


if __name__ == "__main__":
    main()
