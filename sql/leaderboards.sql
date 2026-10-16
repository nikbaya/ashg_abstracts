-- Leaderboards over views created by analysis.py:
--   inst_credit(abstract_id, talk, institute, basis)   one row per credited institute
--   people(abstract_id, talk, person_id, display_name, institutes)
--   metro_credit(abstract_id, talk, institute, basis, metro)  inst_credit rows with a metro
--   country_credit(abstract_id, talk, institute, basis, country, region)  inst_credit rows with a country
--   baseline(talk_rate)

CREATE OR REPLACE MACRO wilson_lo(k, n) AS greatest(0,
    ((k / n + 1.96 ^ 2 / (2 * n)) - 1.96 * sqrt(k / n * (1 - k / n) / n + 1.96 ^ 2 / (4 * n ^ 2))) / (1 + 1.96 ^ 2 / n));  -- 0 talks: float noise below 0
CREATE OR REPLACE MACRO wilson_hi(k, n) AS
    ((k / n + 1.96 ^ 2 / (2 * n)) + 1.96 * sqrt(k / n * (1 - k / n) / n + 1.96 ^ 2 / (4 * n ^ 2))) / (1 + 1.96 ^ 2 / n);

-- name: institutions
-- Top 50 institutes by volume per credit basis; talk rate with 95% Wilson CI.
WITH agg AS (
    SELECT basis, institute, count(DISTINCT abstract_id) AS abstracts,
           count(DISTINCT abstract_id) FILTER (WHERE talk) AS talks
    FROM inst_credit GROUP BY basis, institute
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY basis ORDER BY abstracts DESC, talks DESC, institute) AS rank
    FROM agg
)
SELECT basis, rank, institute, abstracts, talks,
       talks / abstracts AS talk_rate,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_rate FROM baseline) AS baseline
FROM ranked WHERE rank <= 50 ORDER BY basis, rank;

-- name: senior_authors
-- Top 10 senior authors by Wilson lower bound among those with >= min_n abstracts
-- (min_n = 5, relaxed to 4 if fewer than 10 people qualify).
WITH agg AS (
    SELECT person_id, any_value(display_name) AS name, count(*) AS abstracts,
           count(*) FILTER (WHERE talk) AS talks,
           mode(institutes[1]) AS institute
    FROM people GROUP BY person_id
), threshold AS (
    SELECT CASE WHEN (SELECT count(*) FROM agg WHERE abstracts >= 5) >= 10 THEN 5 ELSE 4 END AS min_n
)
SELECT row_number() OVER (ORDER BY wilson_lo(talks, abstracts) DESC, talks DESC, name) AS rank,
       name, institute, abstracts, talks, talks / abstracts AS talk_rate,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_rate FROM baseline) AS baseline, (SELECT min_n FROM threshold) AS min_abstracts
FROM agg WHERE abstracts >= (SELECT min_n FROM threshold)
ORDER BY rank LIMIT 10;

-- name: metros
-- Top 30 metro areas by volume per credit basis (metro_credit: inst_credit + metro). An abstract counts
-- once per metro, however many of its institutes are there; `institutes` lists the 3 largest contributors.
WITH agg AS (
    SELECT basis, metro, count(DISTINCT abstract_id) AS abstracts,
           count(DISTINCT abstract_id) FILTER (WHERE talk) AS talks
    FROM metro_credit GROUP BY basis, metro
), per_inst AS (
    SELECT basis, metro, institute,
           row_number() OVER (PARTITION BY basis, metro ORDER BY count(DISTINCT abstract_id) DESC, institute) AS k
    FROM metro_credit GROUP BY basis, metro, institute
), top_insts AS (
    SELECT basis, metro, string_agg(institute, '; ' ORDER BY k) AS institutes
    FROM per_inst WHERE k <= 3 GROUP BY basis, metro
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY basis ORDER BY abstracts DESC, talks DESC, metro) AS rank
    FROM agg
)
SELECT basis, rank, metro, abstracts, talks,
       talks / abstracts AS talk_rate,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_rate FROM baseline) AS baseline, institutes
FROM ranked JOIN top_insts USING (basis, metro) WHERE rank <= 30 ORDER BY basis, rank;

-- name: regions
-- Every world region per credit basis (country_credit: inst_credit + country + region). An abstract
-- counts once per region; `countries` lists the 3 largest contributors.
WITH agg AS (
    SELECT basis, region, count(DISTINCT abstract_id) AS abstracts,
           count(DISTINCT abstract_id) FILTER (WHERE talk) AS talks
    FROM country_credit GROUP BY basis, region
), per_country AS (
    SELECT basis, region, country,
           row_number() OVER (PARTITION BY basis, region ORDER BY count(DISTINCT abstract_id) DESC, country) AS k
    FROM country_credit GROUP BY basis, region, country
), top_countries AS (
    SELECT basis, region, string_agg(country, '; ' ORDER BY k) AS countries
    FROM per_country WHERE k <= 3 GROUP BY basis, region
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY basis ORDER BY abstracts DESC, talks DESC, region) AS rank
    FROM agg
)
SELECT basis, rank, region, abstracts, talks,
       talks / abstracts AS talk_rate,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_rate FROM baseline) AS baseline, countries
FROM ranked JOIN top_countries USING (basis, region) ORDER BY basis, rank;

-- name: countries
-- Top 25 countries by volume per credit basis; an abstract counts once per country.
WITH agg AS (
    SELECT basis, country, any_value(region) AS region, count(DISTINCT abstract_id) AS abstracts,
           count(DISTINCT abstract_id) FILTER (WHERE talk) AS talks
    FROM country_credit GROUP BY basis, country
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY basis ORDER BY abstracts DESC, talks DESC, country) AS rank
    FROM agg
)
SELECT basis, rank, country, region, abstracts, talks,
       talks / abstracts AS talk_rate,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_rate FROM baseline) AS baseline
FROM ranked WHERE rank <= 25 ORDER BY basis, rank;
