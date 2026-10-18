"""Manhattan-style overviews of what abstracts mention: genes along the genome, and words A to Z.

- outputs/trends_genes.csv/.png  per year and gene: abstracts mentioning it (title or text), as a share of
                                 that year's abstracts; one row of the figure per year, genes at their hg38
                                 position, the most-mentioned labelled
- outputs/trends_words.csv       per word and year ("all" = every year pooled): share of abstracts using it
- outputs/{year}_words.png       (and all_words.png) words in alphabetical order, the most-used word under each
                                 letter labelled; the same words and scale every year
- outputs/trends_words_enriched.csv/.png  the same words by -log10 p for being used by more of the latest
                                 year's abstracts than of the earlier years' (one-sided Fisher), Bonferroni line
- outputs/manhattan.json         counts behind all of these, for the site's interactive versions

Genes are approved HGNC protein-coding symbols, matched case-sensitively as whole tokens (each part of a
hyphenated token also counts, so "EWSR1::ERG" and "HLA-DRB1" both match), except right before a "+" (a cell
population: "CD4+ T cells"). Symbols that in abstracts are
usually something else (MAF minor allele frequency, HR hazard ratio, TF transcription factor ...) are in
AMBIGUOUS; positions are each symbol's leftmost RefSeq transcript start on its usual chromosome. Words are
features.tokenize tokens, minus English stop words and numbers. Abstracts cut off in a booklet are left out.

Usage: python src/ashg/manhattan.py [--years 2022 ... 2026] [--genes data/raw/genes]
"""

import argparse
import json
import pathlib
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from scipy.stats import hypergeom
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from features import tokenize
from figures import GRID, INK, INK_2, SURFACE, TALK, header, new_figure

# Approved symbols checked against their contexts in the abstracts: mostly not the gene (or a mix)
AMBIGUOUS = {
    "MAF", "HR", "TF", "CAD", "GC", "CS", "PC", "MVP", "PIP", "TG", "AR", "SCD", "FH", "GAA", "BCR", "DBP", "CP",
    "PGC", "GPT", "GEM", "MB", "NHS", "FST", "ADA", "PAH", "F2", "C2", "C3", "C5", "HP", "PAM", "RPE", "SDS", "ACR",
    "CFI", "GDA", "VIP", "DCC", "FAP", "FEV", "CAT", "HRC", "ENG", "ARC", "GPI", "GCA", "SI", "TAT", "SCT", "KL",
    "MCC", "CCS", "MSC", "SMS", "WAS", "CPM", "HDC", "TRD", "SET", "ADM", "IDS", "OMG", "TRA", "GRN", "CRP", "POR",
    "CD4",  # "CD4 T cells"
}
CHROMS = [*map(str, range(1, 23)), "X", "Y"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
LIGHT = "#9fc0ea"  # lighter step of TALK's hue: every other chromosome / letter
N_GENE_LABELS, MIN_WORD_DOCS = 6, 10


def load_genes(gene_dir):
    """Approved protein-coding symbols with chromosome and position (hg38), from HGNC + UCSC refGene."""
    h = pd.read_csv(gene_dir / "hgnc_complete_set.txt", sep="\t", usecols=["symbol", "locus_group", "status"],
                    dtype=str)
    symbols = set(h[(h.locus_group == "protein-coding gene") & (h.status == "Approved")].symbol) - AMBIGUOUS
    ref = pd.read_csv(gene_dir / "refGene.txt.gz", sep="\t", header=None, usecols=[2, 4, 12],
                      names=["chrom", "start", "gene"])
    ref["chrom"] = ref.chrom.str.removeprefix("chr")
    ref = ref[ref.chrom.isin(CHROMS) & ref.gene.isin(symbols)]
    chrom = ref.groupby("gene").chrom.agg(lambda s: s.mode()[0])  # PAR genes sit on X and Y: keep the usual one
    ref = ref[ref.chrom == ref.gene.map(chrom)]
    return symbols, ref.groupby("gene").agg(chrom=("chrom", "first"), pos=("start", "min"))


def gene_mentions(text, symbols):
    found = set()
    for m in TOKEN_RE.finditer(text):
        t = m.group()
        if text[m.end():m.end() + 1] == "+":  # "CD4+ T cells", "VIP+ neurons": a cell population, not the gene
            continue
        if t in symbols:
            found.add(t)
        elif "-" in t:
            found.update(p for p in t.split("-") if p in symbols)
    return found


def gene_table(abstracts, symbols, where):
    df = abstracts[~abstracts.truncated]
    hits = pd.DataFrame({"year": df.year.to_numpy(), "talk": (df.pres_type != "poster").to_numpy(),
                         "gene": [gene_mentions(t, symbols) for t in df.title.fillna("") + " . " + df.body]})
    n = hits.groupby("year").size()
    e = hits.explode("gene").dropna(subset=["gene"])
    t = e.groupby(["year", "gene"]).agg(docs=("talk", "size"), talk_docs=("talk", "sum")).reset_index()
    t["share"] = t.docs / t.year.map(n)
    t = t.join(where, on="gene")
    missing = t.chrom.isna()
    if missing.any():
        print(f"  {t[missing].gene.nunique()} mentioned genes have no RefSeq position (left out of the figure)")
    return t[["year", "gene", "chrom", "pos", "docs", "talk_docs", "share"]].sort_values(["year", "docs"],
                                                                                        ascending=[True, False])


def word_table(abstracts):
    """Per word and year ("all" = pooled): abstracts using it and their share. Every year has the same words
    (those in MIN_WORD_DOCS+ abstracts pooled), so a word sits at the same place in every year's figure."""
    df = abstracts[~abstracts.truncated]
    used = pd.DataFrame({"year": df.year.to_numpy(),
                         "word": [list(set(tokenize(t))) for t in df.title.fillna("") + " . " + df.body]})
    used = used.explode("word")
    used = used[used.word.str[0].isin(list(string.ascii_lowercase)) & (used.word.str.len() > 1)
                & ~used.word.isin(ENGLISH_STOP_WORDS)]
    pooled = used.word.value_counts()
    words = sorted(pooled[pooled >= MIN_WORD_DOCS].index)
    n = df.groupby("year").size()
    per_year = used.groupby(["year", "word"]).size().unstack(0).reindex(words).fillna(0).astype(int)
    t = pd.concat([pd.DataFrame({"year": "all", "word": words, "docs": pooled.loc[words].to_numpy(),
                                 "share": pooled.loc[words].to_numpy() / len(df)})]
                  + [pd.DataFrame({"year": str(y), "word": words, "docs": per_year[y].to_numpy(),
                                   "share": per_year[y].to_numpy() / n[y]}) for y in per_year.columns],
                  ignore_index=True)
    return t


# --- figures ---------------------------------------------------------------------------------------

def manhattan_axes(ax, ymax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10], min_n_ticks=3))


def label_peaks(ax, xs, ys, labels, fontsize=8.5):
    """Label points tallest first, each just above its point, nudged sideways (then up) past any label already
    placed, with a leader line once it has moved."""
    renderer = ax.figure.canvas.get_renderer()
    placed = []
    slots = [(dx, dy) for dy in range(3, 60, 9) for dx in (0, -30, 30)]  # sideways before up: shorter leaders
    style = dict(textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=INK,
                 fontweight="bold", annotation_clip=False)
    for i in np.argsort(-np.asarray(ys), kind="stable"):
        for dx, dy in slots:
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), **style)
            box = t.get_window_extent(renderer).expanded(1.06, 1.1)  # the text alone: a leader's box would block
            t.remove()
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                leader = dict(arrowstyle="-", color=INK_2, lw=0.6, shrinkA=0, shrinkB=2) if (dx, dy) != slots[0] else None
                ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), arrowprops=leader, **style)
                break


def genome_layout(t):
    """Genes with a position on one x axis: chromosomes end to end, each as long as its last mentioned gene (enough
    for the layout), 4 Mb apart. Returns t with x and band (every other chromosome) and {chrom: (start, end)}."""
    t = t.dropna(subset=["chrom"])
    ends = t.groupby("chrom").pos.max().reindex(CHROMS).dropna()
    offset = dict(zip(ends.index, np.r_[0, np.cumsum(ends.to_numpy() + 4e6)[:-1]]))
    t = t.assign(x=t.pos + t.chrom.map(offset), band=t.chrom.map({c: i % 2 for i, c in enumerate(ends.index)}))
    return t, {c: (offset[c], offset[c] + ends[c]) for c in ends.index}


GENE_TICKS = {*map(str, range(1, 13)), "14", "16", "18", "20", "22", "X"}  # the rest are too narrow to label


def genes_figure(t, years, path):
    t, spans = genome_layout(t)
    h = 1.6 + 1.75 * len(years)
    fig, gs = new_figure((10, h), nrows=len(years), hspace=0.55, left=0.07, right=0.98, top=1 - 1.25 / h,
                         bottom=0.7 / h)
    ymax = t.share.max() * 1.3
    centers = [(a + b) / 2 for a, b in spans.values()]
    for row, y in enumerate(years):
        ax = fig.add_subplot(gs[row])
        r = t[t.year == y]
        manhattan_axes(ax, ymax)
        for band, color in ((0, TALK), (1, LIGHT)):
            s = r[r.band == band]
            ax.scatter(s.x, s.share, s=9, color=color, lw=0, zorder=2)
        top = r.nlargest(N_GENE_LABELS, "docs")
        label_peaks(ax, top.x.to_numpy(), top.share.to_numpy(), list(top.gene))
        ax.set_xlim(-2e7, max(b for _, b in spans.values()) + 2e7)
        ax.set_xticks(centers, [c if c in GENE_TICKS else "" for c in spans] if row == len(years) - 1 else [])
        ax.annotate(str(y), (0, 1), xycoords="axes fraction", xytext=(-46, 2), textcoords="offset points",
                    ha="left", va="bottom", color=INK, fontsize=10.5, fontweight="bold")
    header(fig, f"Genes mentioned at ASHG {years[0]}–{years[-1]}",
           "Share of each year's accepted abstracts naming each protein-coding gene (title or text), placed along "
           f"the genome\n(chromosomes 1–22, X, Y); the {N_GENE_LABELS} most-mentioned each year labelled.",
           "Source: ASHG abstracts. HGNC symbols, case-sensitive; symbols that are usually something else in "
           "abstracts (MAF, HR, TF, CAD ...) left out. Positions hg38.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def alphabetical(ax, t, y):
    """Plot t[y] for words in alphabetical order (t sorted by word), every other letter lighter, a tick per letter
    (letters too narrow to label left blank). Returns t with its x positions and letters."""
    t = t.reset_index(drop=True)
    t = t.assign(x=np.arange(len(t)), letter=t.word.str[0])
    letters = list(dict.fromkeys(t.letter))
    band = t.letter.map({l: i % 2 for i, l in enumerate(letters)})
    for b, color in ((0, TALK), (1, LIGHT)):
        s = t[band == b]
        ax.scatter(s.x, s[y], s=7, color=color, lw=0, zorder=2)
    g = t.groupby("letter").x.agg(["min", "max"]).loc[letters]
    ax.set_xticks((g["min"] + g["max"]) / 2, [l if r["max"] - r["min"] > len(t) / 120 else ""
                                              for l, r in g.iterrows()])
    ax.set_xlim(-len(t) * 0.01, len(t) * 1.01)
    return t


def words_figure(t, label, n_abstracts, ymax, path):
    """One year's (or the pooled) word shares, words A to Z; ymax shared so the years compare."""
    h = 4.6
    fig, gs = new_figure((10, h), left=0.07, right=0.98, top=1 - 1.2 / h, bottom=0.75 / h)
    ax = fig.add_subplot(gs[0])
    manhattan_axes(ax, ymax)
    t = alphabetical(ax, t, "share")
    top = t.loc[t.groupby("letter").share.idxmax()]
    top = top[top.share >= 0.12]  # the letters with a clear peak; the rest stay unlabelled
    label_peaks(ax, top.x.to_numpy(), top.share.to_numpy(), list(top.word))
    header(fig, f"Words used at ASHG {label}, A to Z",
           f"Share of the {n_abstracts:,} accepted abstracts using each word (title or text), alphabetically; "
           f"the {len(t):,} words in {MIN_WORD_DOCS}+ abstracts over all years.",
           "Source: ASHG abstracts. Common English words (the, of, and ...) and numbers left out; plurals are "
           "separate words. Same scale every year.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def enrichment_table(words, complete, year):
    """Per word: abstracts using it in `year` vs every earlier year pooled, the odds ratio (+0.5 in every cell)
    and a one-sided hypergeometric (Fisher) p for being more common in `year`."""
    earlier = [y for y in complete.index if y < year]
    docs = words[words.year != "all"].pivot(index="word", columns="year", values="docs")
    k, m = docs[str(year)], docs[[str(y) for y in earlier]].sum(axis=1)
    n1, n0 = complete[year], complete[earlier].sum()
    t = pd.DataFrame({"word": docs.index, "docs": k, "share": k / n1, "docs_before": m, "share_before": m / n0,
                      "or": (k + .5) * (n0 - m + .5) / ((n1 - k + .5) * (m + .5)),
                      "p": hypergeom.sf(k - 1, n1 + n0, k + m, n1)})
    return t.reset_index(drop=True), f"{earlier[0]}–{earlier[-1]}" if len(earlier) > 1 else str(earlier[0])


def enriched_figure(t, year, before, n1, n0, path):
    """Words A to Z by -log10 p for being more common in `year` than before, Bonferroni line, the strongest word
    under each letter labelled."""
    t = t.assign(logp=-np.log10(t.p.clip(lower=1e-300)))
    line = -np.log10(0.05 / len(t))
    h = 4.6
    fig, gs = new_figure((10, h), left=0.07, right=0.98, top=1 - 1.2 / h, bottom=0.75 / h)
    ax = fig.add_subplot(gs[0])
    manhattan_axes(ax, t.logp.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    t = alphabetical(ax, t, "logp")
    ax.axhline(line, color=INK_2, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.annotate("Bonferroni 5%", (1, line), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom", color=INK_2, fontsize=8.5, zorder=4,
                bbox=dict(fc=SURFACE, ec="none", pad=1))
    top = t.loc[t.groupby("letter").logp.idxmax()]
    top = top[top.logp >= 4 * line]  # the strongest word under each letter with a clear peak
    label_peaks(ax, top.x.to_numpy(), top.logp.to_numpy(), list(top.word))
    ax.set_ylabel("−log₁₀ p", color=INK_2, fontsize=9)
    header(fig, f"Words more common at ASHG {year} than {before}, A to Z",
           f"Overrepresentation of each word in the {n1:,} {year} abstracts vs the {n0:,} from {before};\n"
           f"{(t.p < 0.05 / len(t)).sum()} of {len(t):,} words significant after Bonferroni correction (dashed line).",
           "Source: ASHG abstracts (title or text). One-sided Fisher's exact test per word; Bonferroni correction "
           "over every word at 5%.", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def site_json(words, enriched, genes, complete, latest, before, path):
    """Data for the site's interactive versions: counts (shares are computed in the page), aligned to one sorted
    word list and one gene list. words: per year and "all"; enriched: -log10 p and the Bonferroni line; genes: x
    on the genome_layout axis, chromosome index, counts per year and "all"."""
    vocab = sorted(words.word.unique())
    by_year = words.pivot(index="word", columns="year", values="docs").reindex(vocab)
    n = {str(y): int(v) for y, v in complete.items()} | {"all": int(complete.sum())}
    logp = -np.log10(enriched.set_index("word").p.reindex(vocab).clip(lower=1e-300))
    g, spans = genome_layout(genes)
    chroms = list(spans)
    pos = g.drop_duplicates("gene").set_index("gene")[["x", "chrom"]].sort_values("x")
    per = g.pivot(index="gene", columns="year", values="docs").reindex(pos.index).fillna(0).astype(int)
    payload = {
        "n": n,
        "words": {"words": vocab, "counts": {y: by_year[y].astype(int).tolist() for y in by_year.columns},
                  "enriched": {"year": int(latest), "before": before, "line": round(-np.log10(0.05 / len(vocab)), 2),
                               "logp": [round(float(v), 1) for v in logp]}},
        "genes": {"genes": list(pos.index), "x": [round(v / 1e6, 2) for v in pos.x],  # Mb
                  "chrom": [chroms.index(c) for c in pos.chrom],
                  "chroms": [[c, round(a / 1e6, 1), round(b / 1e6, 1), c in GENE_TICKS] for c, (a, b) in spans.items()],
                  "counts": {str(y): per[y].tolist() for y in per.columns} | {"all": per.sum(axis=1).tolist()}},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--genes", default="data/raw/genes")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    data, out, years = pathlib.Path(args.data), pathlib.Path(args.out), sorted(args.years)
    abstracts = pd.concat([pd.read_parquet(p) for y in years
                           for p in sorted((data / "abstracts").glob(f"{y}_*.parquet"))], ignore_index=True)

    symbols, where = load_genes(pathlib.Path(args.genes))
    genes = gene_table(abstracts, symbols, where)
    genes.to_csv(out / "trends_genes.csv", index=False, float_format="%.4g")
    genes_figure(genes, years, out / "trends_genes.png")

    words = word_table(abstracts)
    words.to_csv(out / "trends_words.csv", index=False, float_format="%.4g")
    complete = abstracts[~abstracts.truncated].groupby("year").size()
    ymax = words.share.max() * 1.15
    for y in ["all", *map(str, years)]:
        label = f"{years[0]}–{years[-1]}" if y == "all" else y
        n = int(complete.sum() if y == "all" else complete[int(y)])
        words_figure(words[words.year == y], label, n, ymax, out / f"{y}_words.png")

    latest = years[-1]
    enriched, before = enrichment_table(words, complete, latest)
    enriched.sort_values("p").to_csv(out / "trends_words_enriched.csv", index=False, float_format="%.4g")
    enriched_figure(enriched, latest, before, int(complete[latest]), int(complete[complete.index < latest].sum()),
                    out / "trends_words_enriched.png")
    site_json(words, enriched, genes, complete, latest, before, out / "manhattan.json")
    print(f"wrote trends_genes.csv/.png ({genes.gene.nunique():,} genes), trends_words.csv and all_words.png + "
          f"{{year}}_words.png ({words.word.nunique():,} words), trends_words_enriched.csv/.png "
          f"({(enriched.p < 0.05 / len(enriched)).sum()} words enriched in {latest}) to {out}/")


if __name__ == "__main__":
    main()
