"""Phase 4: aggregate outputs for one year (committable CSVs in outputs/).

Reads data/{institutions,people,features}/ and writes:
  outputs/{year}_institutions.csv   top-50 institutes, senior- and first-author credit
  outputs/{year}_senior_authors.csv top-10 senior authors by Wilson lower bound
  outputs/{year}_premium.csv        top talk- and poster-enriched n-grams (subsumption-filtered)
  outputs/{year}_sample_size.csv    sample-size summary by presentation type
  outputs/{year}_summary.csv        counts and baseline talk share

Usage: python src/ashg/analysis.py YEAR [--data data] [--out outputs] [--top 30]
"""

import argparse
import pathlib
import re

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse, stats

from features import drop_subsumed

SQL = pathlib.Path(__file__).resolve().parents[2] / "sql" / "leaderboards.sql"


def named_queries(path):
    """Split a SQL file into setup statements and '-- name: x' queries."""
    parts = re.split(r"^-- name: (\w+)\s*$", path.read_text(), flags=re.M)
    return parts[0], dict(zip(parts[1::2], parts[2::2]))


def premium_table(data, year, top):
    s = pd.read_parquet(data / f"features/premium_{year}.parquet")
    X = sparse.load_npz(data / f"features/doc_term_{year}.npz")
    vocab = pd.read_parquet(data / f"features/vocab_{year}.parquet").ngram.to_numpy()
    talk = drop_subsumed(s.sort_values("z", ascending=False).head(4 * top), X, vocab).head(top)
    poster = drop_subsumed(s.sort_values("z").head(4 * top), X, vocab).head(top)
    out = pd.concat([talk.assign(direction="talk"), poster.assign(direction="poster")])
    out.insert(0, "rank", out.groupby("direction").cumcount() + 1)
    return out


def sample_size_table(data, year):
    ss = pd.read_parquet(data / f"features/sample_size_{year}.parquet")
    ss["type"] = np.where(ss.pres_type == "poster", "poster", "talk")
    rows = []
    for t, g in ss.groupby("type"):
        n = g.max_n.dropna()
        rows.append({"year": year, "type": t, "abstracts": len(g), "with_n": len(n), "coverage": len(n) / len(g),
                     "median_max_n": n.median(), "q25": n.quantile(.25), "q75": n.quantile(.75)})
    out = pd.DataFrame(rows)
    talk, poster = (np.log10(ss.max_n[(ss.type == t) & ss.max_n.notna()]) for t in ("talk", "poster"))
    out["mannwhitney_p"] = stats.mannwhitneyu(talk, poster).pvalue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    data, out, y = pathlib.Path(args.data), pathlib.Path(args.out), args.year
    out.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW insts AS SELECT *, pres_type <> 'poster' AS talk FROM '{data}/institutions/{y}.parquet';
        CREATE VIEW inst_credit AS
            SELECT abstract_id, talk, unnest(senior_insts) AS institute, 'senior' AS basis FROM insts
            UNION ALL
            SELECT abstract_id, talk, unnest(first_insts) AS institute, 'first' AS basis FROM insts;
        CREATE VIEW people AS SELECT *, pres_type <> 'poster' AS talk FROM '{data}/people/{y}.parquet';
        CREATE VIEW baseline AS SELECT avg(talk::int) AS talk_share FROM insts;
    """)
    setup, queries = named_queries(SQL)
    con.execute(setup)
    results = {name: con.execute(q).df().assign(year=y) for name, q in queries.items()}

    summary = con.execute("""
        SELECT count(*) AS abstracts, sum(talk::int) AS talks, avg(talk::int) AS talk_share,
               avg((len(senior_insts) > 0)::int) AS senior_inst_coverage
        FROM insts""").df().assign(year=y)
    results["summary"] = summary
    results["premium"] = premium_table(data, y, args.top).assign(year=y)
    results["sample_size"] = sample_size_table(data, y)

    for name, df in results.items():
        df.to_csv(out / f"{y}_{name}.csv", index=False, float_format="%.4g")
    print(f"wrote {', '.join(f'{y}_{n}.csv' for n in results)} to {out}/")


if __name__ == "__main__":
    main()
