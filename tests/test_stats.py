"""Statistics against hand-computed values."""

import math
import pathlib

import duckdb
import numpy as np
import pytest

from analysis import named_queries
from features import ngrams, premium, sample_size, tokenize

SQL = pathlib.Path(__file__).resolve().parents[1] / "sql" / "leaderboards.sql"


def test_log_odds_ratio_matches_hand_computation():
    # 3 talks, 5 posters; "enhancer" in 2 talks and 1 poster -> a=2, b=1, c=1, d=4
    docs = ["enhancer loci"] * 2 + ["loci"] + ["enhancer"] + ["loci"] * 4
    talk = [1, 1, 1, 0, 0, 0, 0, 0]
    stats, _, _ = premium(docs, talk, min_talk_docs=1)
    r = stats.set_index("ngram").loc["enhancer"]
    a, b, c, d = 2.5, 1.5, 1.5, 4.5
    assert r.log_or == pytest.approx(math.log(a * d / (b * c)))
    assert r.se == pytest.approx(math.sqrt(1 / a + 1 / b + 1 / c + 1 / d))


def test_wilson_interval():
    con = duckdb.connect()
    con.execute(named_queries(SQL)[0])
    lo, hi = con.execute("SELECT wilson_lo(5.0, 11.0), wilson_hi(5.0, 11.0)").fetchone()
    # reference: statsmodels proportion_confint(5, 11, method="wilson")
    assert lo == pytest.approx(0.21270, abs=1e-4)
    assert hi == pytest.approx(0.71997, abs=1e-4)


def test_tokenizer_keeps_genomics_terms_and_stopword_ngrams():
    assert tokenize("scRNA-seq of CRISPR-Cas9 in 1,078 cells") == ["scrna-seq", "of", "crispr-cas9", "in", "<num>", "cells"]
    grams = ngrams("loss of function variants")
    assert "loss of function" in grams and "of function" not in grams and "loss of" not in grams


@pytest.mark.parametrize("text, n, noun", [
    ("We analysed 12,345 individuals of European ancestry.", 12345, "individuals"),
    ("a cohort of 1.2 million participants", 1.2e6, "participants"),
    ("whole genomes from 500 Finnish donors and 3 mice", 500, "donors"),
    ("Among patients (N = 4,381), we found", 4381, "patients"),
])
def test_sample_size(text, n, noun):
    assert sample_size(text) == (n, noun)


def test_sample_size_absent():
    v, noun = sample_size("We used n = 3 replicates per condition.")
    assert np.isnan(v) and noun is None
