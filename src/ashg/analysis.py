"""Phase 4: aggregate outputs for one year (committable CSVs in outputs/).

Reads data/{institutions,people,features,geography}/ and writes:
  outputs/{year}_institutions.csv   top-50 institutes, senior- and first-author credit
  outputs/{year}_metros.csv         top-30 metro areas (institutes grouped via metros.csv)
  outputs/{year}_regions.csv        UN world regions (author countries from data/geography)
  outputs/{year}_countries.csv      top-25 countries
  outputs/{year}_senior_diversity.csv distinct senior authors per abstract, all / talks / posters
  outputs/{year}_senior_authors.csv top-10 senior authors by Wilson lower bound
  outputs/{year}_premium.csv        top talk- and poster-enriched n-grams (related phrases grouped)
  outputs/{year}_sample_size.csv    sample-size summary by presentation type
  outputs/{year}_summary.csv        counts and baseline talk rate

YEAR "all" pools every year in data/institutions (outputs/all_*.csv; no senior_diversity): people
come from data/people/all.parquet (matched across years) and phrases from features.py all.

Usage: python src/ashg/analysis.py YEAR|all [--data data] [--out outputs] [--top 30]
"""

import argparse
import pathlib
import re

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse, special, stats

from features import group_related
from geography import REGION_OF

ROOT = pathlib.Path(__file__).resolve().parents[2]
SQL = ROOT / "sql" / "leaderboards.sql"


def load_metros(path):
    """institute -> metro; blank metro = multi-metro institute, left out of metro totals."""
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    return df[df.metro.str.strip() != ""]


def named_queries(path):
    """Split a SQL file into setup statements and '-- name: x' queries."""
    parts = re.split(r"^-- name: (\w+)\s*$", path.read_text(), flags=re.M)
    return parts[0], dict(zip(parts[1::2], parts[2::2]))


def premium_table(data, year, top):
    s = pd.read_parquet(data / f"features/premium_{year}.parquet")
    X = sparse.load_npz(data / f"features/doc_term_{year}.npz")
    vocab = pd.read_parquet(data / f"features/vocab_{year}.parquet").ngram.to_numpy()
    talk = group_related(s.sort_values("z", ascending=False).head(4 * top), X, vocab).head(top)
    poster = group_related(s.sort_values("z").head(4 * top), X, vocab).head(top)
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


RAREFY_N = 200  # below the smallest talk count in any year (247)


def expected_distinct(sizes, n):
    """Expected distinct people among n abstracts drawn without replacement, where
    person i has sizes[i] abstracts: sum_i 1 - C(N - k_i, n) / C(N, n)."""
    sizes = np.asarray(sizes)
    N = sizes.sum()
    if n > N:
        return np.nan
    log_miss = special.gammaln(N - sizes + 1) - special.gammaln(N - sizes - n + 1) \
        - special.gammaln(N + 1) + special.gammaln(N - n + 1)
    return float(np.sum(1 - np.exp(np.where(N - sizes >= n, log_miss, -np.inf))))


def senior_diversity(people, year):
    """Distinct senior authors per abstract. `person` uses the resolved identities; `initials`
    keys everyone by first initial + surname, the only form 2023-2024 booklets give, so it is
    the one to compare across years. Ratios fall as abstracts grow, so `per_200` is the expected
    number of distinct people in a random 200 abstracts."""
    people = people.assign(initials=people.name_key.str[0] + " " + people.name_key.str.split().str[-1],
                           type=np.where(people.pres_type == "poster", "poster", "talk"))
    rows = []
    for t, g in [("all", people), *people.groupby("type")]:
        row = {"year": year, "type": t, "abstracts": len(g)}
        for key in ("person_id", "initials"):
            name = "person" if key == "person_id" else "initials"
            sizes = g.groupby(key).size()
            row[f"{name}_distinct"] = len(sizes)
            row[f"{name}_ratio"] = len(sizes) / len(g)
            row[f"{name}_per_{RAREFY_N}"] = expected_distinct(sizes, RAREFY_N)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", help="a year, or 'all' to pool every year")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--metros", default=ROOT / "metros.csv")
    args = ap.parse_args()
    data, out, y = pathlib.Path(args.data), pathlib.Path(args.out), args.year
    out.mkdir(exist_ok=True)
    years = sorted(p.stem for p in (data / "institutions").glob("[0-9]*.parquet")) if y == "all" else [y]

    def source(kind):
        return "read_parquet([" + ", ".join(f"'{data}/{kind}/{v}.parquet'" for v in years) + "])"

    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW insts AS SELECT *, pres_type <> 'poster' AS talk FROM {source('institutions')};
        CREATE VIEW inst_credit AS
            SELECT abstract_id, talk, unnest(senior_insts) AS institute, 'senior' AS basis FROM insts
            UNION ALL
            SELECT abstract_id, talk, unnest(first_insts) AS institute, 'first' AS basis FROM insts;
        CREATE VIEW people AS SELECT *, pres_type <> 'poster' AS talk FROM '{data}/people/{y}.parquet';
        CREATE VIEW baseline AS SELECT avg(talk::int) AS talk_rate FROM insts;
    """)
    all_metro_rows = pd.read_csv(args.metros, comment="#", dtype=str, keep_default_na=False)
    missing = set(con.execute("SELECT DISTINCT institute FROM inst_credit").df().institute) - set(all_metro_rows.institute)
    if missing:
        raise ValueError(f"institutes missing from {args.metros} (add a row, blank metro if multi-site): {sorted(missing)}")
    metros = load_metros(args.metros)
    regions = pd.DataFrame(list(REGION_OF.items()), columns=["country", "region"])
    con.execute(f"""
        CREATE VIEW metro_credit AS SELECT c.*, m.metro FROM inst_credit c JOIN metros m USING (institute);
        CREATE VIEW geo AS SELECT *, pres_type <> 'poster' AS talk FROM {source('geography')};
        CREATE VIEW country_credit AS
            SELECT g.*, r.region FROM (
                SELECT abstract_id, talk, unnest(senior_countries) AS country, 'senior' AS basis FROM geo
                UNION ALL
                SELECT abstract_id, talk, unnest(first_countries) AS country, 'first' AS basis FROM geo
            ) g JOIN regions r USING (country)""")
    setup, queries = named_queries(SQL)
    con.execute(setup)
    results = {name: con.execute(q).df().assign(year=y) for name, q in queries.items()}

    # totals and baseline count every abstract, including those cut off in the booklet;
    # text analyses (phrases, sample sizes) use only the complete ones
    summary = con.execute("""
        SELECT count(*) AS abstracts, sum(talk::int) AS talks, avg(talk::int) AS talk_rate,
               sum(truncated::int) AS truncated, sum((truncated AND talk)::int) AS truncated_talks,
               count(*) FILTER (WHERE NOT truncated) AS text_abstracts,
               count(*) FILTER (WHERE NOT truncated AND talk) AS text_talks,
               avg((len(senior_insts) > 0)::int) FILTER (WHERE NOT truncated) AS senior_inst_coverage,
               (SELECT count(DISTINCT abstract_id) FROM metro_credit WHERE basis = 'senior')
                   / count(*) FILTER (WHERE NOT truncated) AS senior_metro_coverage,
               (SELECT count(DISTINCT abstract_id) FROM country_credit WHERE basis = 'senior')
                   / count(*) FILTER (WHERE NOT truncated) AS senior_country_coverage,
               -- first authors survive a cut-off abstract, so their coverage is over every abstract
               avg((len(first_insts) > 0)::int) AS first_inst_coverage,
               (SELECT count(DISTINCT abstract_id) FROM metro_credit WHERE basis = 'first')
                   / count(*) AS first_metro_coverage,
               (SELECT count(DISTINCT abstract_id) FROM country_credit WHERE basis = 'first')
                   / count(*) AS first_country_coverage
        FROM insts""").df().assign(year=y)
    results["summary"] = summary
    if y != "all":
        results["senior_diversity"] = senior_diversity(pd.read_parquet(data / f"people/{y}.parquet"), y)
    results["premium"] = premium_table(data, y, args.top).assign(year=y)
    results["sample_size"] = sample_size_table(data, y)

    for name, df in results.items():
        df.to_csv(out / f"{y}_{name}.csv", index=False, float_format="%.4g")
    print(f"wrote {', '.join(f'{y}_{n}.csv' for n in results)} to {out}/")


if __name__ == "__main__":
    main()
