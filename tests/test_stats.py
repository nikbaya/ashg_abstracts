"""Statistics against hand-computed values."""

import math
import pathlib

import duckdb
import numpy as np
import pandas as pd
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


# --- writing style, senior-author diversity ---------------------------------

from analysis import expected_distinct  # noqa: E402
from writing import find_typos, is_variant, mattr  # noqa: E402


def test_mattr_is_windowed_share_of_distinct_words():
    assert mattr(["a", "b", "a", "b"], window=2) == 1.0
    assert mattr(["a", "a", "a", "a"], window=2) == 0.5
    assert mattr(["a", "b", "c"], window=50) == 1.0  # shorter than the window: plain TTR


def test_expected_distinct_matches_hand_count():
    # people with 2 and 1 abstracts, draw 2 of 3: P(miss A) = 0, P(miss B) = C(2,2)/C(3,2) = 1/3
    assert expected_distinct([2, 1], 2) == pytest.approx(1 + 2 / 3)
    assert expected_distinct([1, 1, 1], 3) == pytest.approx(3)


def test_typos_use_corpus_as_dictionary():
    common = [["analysis"]] * 60 + [["downregulated"]] * 60 + [["genes"]] * 60
    docs = common + [["anlaysis"], ["downregulates"], ["pgenes"], ["Anlysis"]]
    bodies = docs[:-1]
    typos, _ = find_typos(docs, bodies, dictionary=set())
    assert typos == {"anlaysis": "analysis", "anlysis": "analysis"}  # inflection and glued letter skipped
    assert is_variant("tumour", "tumor") and is_variant("characterises", "characterised")


from scipy import sparse  # noqa: E402

from features import group_related, is_acronym, lexically_related  # noqa: E402


def test_related_phrases_group_by_acronym_or_shared_word_and_abstracts():
    assert is_acronym("qtls", "quantitative trait loci") and not is_acronym("eqtls", "quantitative trait loci")
    assert lexically_related("mpra", "massively parallel reporter")  # initials are a prefix of the acronym
    assert lexically_related("chromatin", "chromatin accessibility") and not lexically_related("africans", "asians")
    # rows: abstracts; "parallel reporter assay" links the two MPRA phrases, so all three are one group
    vocab = np.array(["reporter assay mpra", "parallel reporter", "parallel reporter assay", "chromatin",
                      "chromatin accessibility"])
    X = sparse.csc_matrix(np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]))
    ranked = pd.DataFrame({"ngram": vocab, "z": [5, 4, 3, 2, 1]})
    out = group_related(ranked, X, vocab)
    assert list(out.ngram) == ["reporter assay mpra", "chromatin", "chromatin accessibility"]
    assert out.related.iloc[0] == "parallel reporter; parallel reporter assay"
    assert out.related.iloc[1] == ""  # 1 of 4 shared abstracts: Jaccard 0.25 < 0.45


from features import mantel_haenszel  # noqa: E402


def test_mantel_haenszel_one_stratum_is_plain_log_or():
    log_or, se = mantel_haenszel([2], [1], [1], [4])
    a, b, c, d = 2.5, 1.5, 1.5, 4.5
    assert log_or[0] == pytest.approx(math.log(a * d / (b * c)))
    assert se[0] == pytest.approx(math.sqrt(1 / a + 1 / b + 1 / c + 1 / d))


def test_mantel_haenszel_removes_confounding_by_year():
    # within each year the phrase has no effect (OR 1 after smoothing is symmetric: a/b = c/d), but
    # it is common in the year with many talks, so pooling the tables would show a talk premium
    a, b, c, d = np.array([[49.5], [9.5]]), np.array([[49.5], [89.5]]), np.array([[9.5], [9.5]]), np.array([[9.5], [89.5]])
    log_or, _ = mantel_haenszel(a - .5, b - .5, c - .5, d - .5)
    assert log_or[0] == pytest.approx(0, abs=1e-12)
    pooled = math.log(a.sum() * d.sum() / (b.sum() * c.sum()))
    assert pooled > 0.5
