-- Leaderboards over views created by analysis.py:
--   inst_credit(abstract_id, talk, institute, basis)   one row per credited institute
--   people(abstract_id, talk, person_id, display_name, institutes)
--   baseline(talk_share)

CREATE OR REPLACE MACRO wilson_lo(k, n) AS
    ((k / n + 1.96 ^ 2 / (2 * n)) - 1.96 * sqrt(k / n * (1 - k / n) / n + 1.96 ^ 2 / (4 * n ^ 2))) / (1 + 1.96 ^ 2 / n);
CREATE OR REPLACE MACRO wilson_hi(k, n) AS
    ((k / n + 1.96 ^ 2 / (2 * n)) + 1.96 * sqrt(k / n * (1 - k / n) / n + 1.96 ^ 2 / (4 * n ^ 2))) / (1 + 1.96 ^ 2 / n);

-- name: institutions
-- Top 50 institutes by volume per credit basis; talk share with 95% Wilson CI.
WITH agg AS (
    SELECT basis, institute, count(DISTINCT abstract_id) AS abstracts,
           count(DISTINCT abstract_id) FILTER (WHERE talk) AS talks
    FROM inst_credit GROUP BY basis, institute
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY basis ORDER BY abstracts DESC, talks DESC, institute) AS rank
    FROM agg
)
SELECT basis, rank, institute, abstracts, talks,
       talks / abstracts AS talk_share,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_share FROM baseline) AS baseline
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
       name, institute, abstracts, talks, talks / abstracts AS talk_share,
       wilson_lo(talks, abstracts) AS ci_lo, wilson_hi(talks, abstracts) AS ci_hi,
       (SELECT talk_share FROM baseline) AS baseline, (SELECT min_n FROM threshold) AS min_abstracts
FROM agg WHERE abstracts >= (SELECT min_n FROM threshold)
ORDER BY rank LIMIT 10;
