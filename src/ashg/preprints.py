"""Which ASHG abstracts were already preprinted on bioRxiv / medRxiv? (Analysis only; not on the site.)

PubMed was tried first and found nothing (title search limited to preprint records, all 2,965 2026 titles): it
indexes few bioRxiv/medRxiv preprints. Europe PMC indexes all of them, so both passes use it.

  search:  pass 1: exact-title query of every abstract            -> data/preprints/pass1.csv
           pass 2: every other abstract, by first-author surname + initial among preprints posted from
                   2025-06; keeps candidates with some title overlap, then scores each by authors shared with
                   the abstract and tf-idf cosine of the abstract text against the preprint's
                                                                  -> data/preprints/pass2.csv
  analyze: tiers, talk vs poster rate (Fisher), posting dates (Mann-Whitney), weekly figure
           -> data/preprints/matches.csv, outputs/<year>_preprints.csv, outputs/<year>_preprints.png

Match tiers (a title that changed between abstract and preprint is common, so title alone undercounts):
  exact  title similarity > 0.7 in pass 1
  A      pass 2: title similarity >= 0.7 and >= 2 authors shared
  B      pass 2: weaker: similarity >= 0.5, cosine >= 0.3, >= 3 shared; or cosine >= 0.45 and >= 5 shared
The figure and headline use exact + A; B is reported as a looser bound. B has not been hand-checked.
Matching needs a parseable first author, so a few abstracts are skipped. Results are lower bounds.

Usage: python src/ashg/preprints.py search  [--year 2026] [--limit N]
       python src/ashg/preprints.py analyze [--year 2026]
"""
import argparse
import difflib
import json
import pathlib
import re
import time
import unicodedata
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from figures import POSTER, TALK

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
PUBLISHERS = '(PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv")'
SINCE = "2025-06-01"  # earliest preprint date searched
STOP = set("a an the of and or in on for with to by from using use via is are as at into across between based among "
           "its their analysis study".split())
# conference milestones for the figure (2026)
DEADLINE, CONFERENCE = pd.Timestamp("2026-05-18"), pd.Timestamp("2026-10-20")


def ascii_(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", ascii_(s).lower()).strip()


def words(s):
    return {w for w in norm(s).split() if w not in STOP and len(w) > 2}


def query(q, size=100, core=False):
    for i in range(4):
        try:
            params = {"query": q, "format": "json", "resultType": "core" if core else "lite", "pageSize": size}
            with urllib.request.urlopen(API + urllib.parse.urlencode(params), timeout=40) as r:
                return json.load(r)["resultList"]["result"]
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def first_author(raw):
    """(initial, surname) of the first author in 'Given Surname (Affiliation), ...', or None."""
    f = re.split(r"\s*[\(,;]", raw.strip())[0]
    f = re.sub(r"[^A-Za-z\-' .]", "", ascii_(re.sub(r"[\d#*†‡]+", "", f))).strip().split()
    return (f[0][0], f[-1]) if len(f) >= 2 else None


def abstract_surnames(raw):
    out = set()
    for p in re.split(r"[,;]", re.sub(r"\([^)]*\)", "", raw)):
        w = re.sub(r"[^a-z\- ]", "", ascii_(p).lower()).split()
        if len(w) >= 2:
            out.add(w[-1])
    return out


def preprint_surnames(authors):
    return {re.sub(r"[^a-z\-]", "", ascii_(x.strip().split(" ")[0]).lower())
            for x in re.sub(r"<[^>]+>", "", authors or "").split(",") if x.strip()}


def load(data, year):
    return pd.concat([pd.read_parquet(p) for p in sorted((data / "abstracts").glob(f"{year}_*.parquet"))],
                     ignore_index=True)


def search(data, year, limit=None):
    out = data / "preprints"
    out.mkdir(exist_ok=True)
    df = load(data, year)
    if limit:
        df = df.head(limit)

    rows = []
    for i, (aid, title, raw) in enumerate(zip(df.abstract_id, df.title, df.authors_raw)):
        t = re.sub(r'["\\]', " ", title).strip()
        res = query(f'TITLE:"{t}" AND SRC:PPR AND {PUBLISHERS} AND FIRST_PDATE:[{SINCE} TO {year}-12-31]', 5)
        for x in res or []:
            rows.append({"abstract_id": aid, "title": title, "pre_title": x["title"], "doi": x.get("doi"),
                         "date": x.get("firstPublicationDate"), "authors": x.get("authorString"),
                         "sim": difflib.SequenceMatcher(None, norm(title), norm(x["title"])).ratio()})
        if i % 200 == 0:
            print(f"pass 1: {i}/{len(df)}", flush=True)
        time.sleep(0.15)
    pass1 = pd.DataFrame(rows, columns=["abstract_id", "title", "pre_title", "doi", "date", "authors", "sim"])
    pass1.to_csv(out / "pass1.csv", index=False)

    todo = df[~df.abstract_id.isin(pass1[pass1.sim > 0.7].abstract_id)]
    rows = []
    for i, (aid, title, raw) in enumerate(zip(todo.abstract_id, todo.title, todo.authors_raw)):
        fa = first_author(raw)
        res = query(f'AUTH:"{fa[1]} {fa[0]}" AND SRC:PPR AND {PUBLISHERS} AND FIRST_PDATE:[{SINCE} TO {year}-12-31]') if fa else None
        tw, shared_with = words(title), abstract_surnames(raw)
        for x in res or []:
            pw = words(x["title"])
            jac = len(tw & pw) / max(1, len(tw | pw))
            cont = len(tw & pw) / max(1, min(len(tw), len(pw)))
            sim = difflib.SequenceMatcher(None, norm(title), norm(x["title"])).ratio()
            if max(jac, sim) >= 0.4 or cont >= 0.7:
                rows.append({"abstract_id": aid, "title": title, "pre_title": x["title"], "doi": x.get("doi"),
                             "date": x.get("firstPublicationDate"), "authors": x.get("authorString"),
                             "s": max(jac, sim),
                             "shared": len(shared_with & preprint_surnames(x.get("authorString")))})
        if i % 200 == 0:
            print(f"pass 2: {i}/{len(todo)}", flush=True)
        time.sleep(0.15)
    cand = pd.DataFrame(rows, columns=["abstract_id", "title", "pre_title", "doi", "date", "authors", "s", "shared"])

    # cosine of abstract text vs the preprint's abstract, only where authors overlap
    cand = cand[cand.shared >= 2].copy()
    body = df.set_index("abstract_id").body
    pre_abs = {}
    for doi in cand.doi.dropna().unique():
        res = query(f'DOI:"{doi}"', 1, core=True)
        pre_abs[doi] = re.sub(r"<[^>]+>", " ", res[0].get("abstractText", "")) if res else ""
        time.sleep(0.15)
    cand["pre_abs"] = cand.doi.map(pre_abs).fillna("")
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    vec.fit(pd.concat([body, cand.pre_abs]))
    X, Y = vec.transform(cand.abstract_id.map(body).fillna("")), vec.transform(cand.pre_abs)
    cand["abs_cos"] = [cosine_similarity(X[i], Y[i])[0, 0] if len(cand.pre_abs.iloc[i]) > 50 else 0.0
                       for i in range(len(cand))]
    cand.drop(columns="pre_abs").to_csv(out / "pass2.csv", index=False)
    print(f"-> {out}/pass1.csv, pass2.csv")


def tiers(pass1, pass2):
    """One row per matched abstract, with its tier (best first)."""
    exact = pass1[pass1.sim > 0.7].sort_values("sim", ascending=False).drop_duplicates("abstract_id")
    exact = exact[["abstract_id", "pre_title", "doi", "date"]].assign(tier="exact")
    d = pass2.copy()
    d["tier"] = None
    d.loc[(d.s >= 0.7) & (d.shared >= 2), "tier"] = "A"
    d.loc[d.tier.isna() & (d.s >= 0.5) & (d.abs_cos >= 0.3) & (d.shared >= 3), "tier"] = "B"
    d.loc[d.tier.isna() & (d.abs_cos >= 0.45) & (d.shared >= 5), "tier"] = "B"
    d = d[d.tier.notna() & ~d.abstract_id.isin(exact.abstract_id)]
    d = d.sort_values(["abstract_id", "tier", "s"], ascending=[True, True, False]).drop_duplicates("abstract_id")
    return pd.concat([exact, d[["abstract_id", "pre_title", "doi", "date", "tier"]]], ignore_index=True)


def analyze(data, out, year):
    df = load(data, year).assign(talk=lambda d: d.pres_type != "poster")
    pre = data / "preprints"
    m = tiers(pd.read_csv(pre / "pass1.csv"), pd.read_csv(pre / "pass2.csv"))
    m.to_csv(pre / "matches.csv", index=False)

    rows = []
    for name, keep in [("exact + A", ["exact", "A"]), ("exact + A + B", ["exact", "A", "B"])]:
        hit = m[m.tier.isin(keep)]
        df["pre"] = df.abstract_id.isin(hit.abstract_id)
        g = df.groupby("talk").pre.agg(["sum", "count"])
        (ts, tn), (ps, pn) = g.loc[True], g.loc[False]
        fisher = fisher_exact([[ts, tn - ts], [ps, pn - ps]])
        h = hit.assign(date=pd.to_datetime(hit.date), talk=hit.abstract_id.map(df.set_index("abstract_id").talk))
        mw = mannwhitneyu(h[h.talk].date.astype("int64"), h[~h.talk].date.astype("int64"), alternative="less")
        rows.append({"rule": name, "talks_preprinted": ts, "talks": tn, "posters_preprinted": ps, "posters": pn,
                     "talk_rate": round(ts / tn, 4), "poster_rate": round(ps / pn, 4),
                     "odds_ratio": round(fisher.statistic, 2), "fisher_p": round(fisher.pvalue, 3),
                     "median_date_talks": h[h.talk].date.median().date(), "median_date_posters": h[~h.talk].date.median().date(),
                     "mwu_talks_earlier_p": round(mw.pvalue, 3)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"{year}_preprints.csv", index=False)
    print(summary.T.to_string(header=False))

    # weekly figure: exact + A, talks stacked on posters
    h = m[m.tier.isin(["exact", "A"])].assign(date=lambda d: pd.to_datetime(d.date))
    h["week"] = h.date.dt.to_period("W-SUN").dt.start_time
    h["group"] = h.abstract_id.map(df.set_index("abstract_id").talk).map({True: "Talks", False: "Posters"})
    wk = h.groupby(["week", "group"]).size().unstack(fill_value=0)
    wk = wk.reindex(pd.date_range(wk.index.min(), pd.Timestamp.today().normalize(), freq="W-MON"), fill_value=0)
    first = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(wk.index, wk.Posters, width=6, align="edge", color=POSTER,
           label=f"Posters ({first.posters_preprinted} / {first.posters:,}, {100 * first.poster_rate:.1f}%)")
    ax.bar(wk.index, wk.Talks, width=6, align="edge", bottom=wk.Posters, color=TALK,
           label=f"Talks ({first.talks_preprinted} / {first.talks}, {100 * first.talk_rate:.1f}%)")
    top = wk.sum(axis=1).max()
    ax.set_ylim(0, top * 1.2)
    ax.set_xlim(wk.index.min() - pd.Timedelta(days=7), CONFERENCE + pd.Timedelta(days=6))
    for d, label in [(DEADLINE, "Abstract deadline (May 18)"), (CONFERENCE, "Conference starts (Oct 20)")]:
        ax.axvline(d, color="#d62728", ls="--", lw=1.5)
        ax.text(d - pd.Timedelta(days=3), top * 1.17, label, color="#d62728", fontsize=9, ha="right", va="top")
    ax.legend(title="Preprinted", loc="upper left", frameon=False, fontsize=9, title_fontsize=9)
    ax.set(ylabel="Preprints per week", title=f"Preprint posting week (ASHG {year} abstracts)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / f"{year}_preprints.png", dpi=150)
    print(f"-> {out}/{year}_preprints.csv, .png; {pre}/matches.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "analyze"])
    ap.add_argument("--year", default="2026")
    ap.add_argument("--limit", type=int, help="search only the first N abstracts (smoke test)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    search(data, a.year, a.limit) if a.stage == "search" else analyze(data, out, a.year)


if __name__ == "__main__":
    main()
