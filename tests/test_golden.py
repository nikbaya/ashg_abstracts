"""Golden set: abstracts hand-checked against the PDFs (the local review sheet from `make review`).

Abstract text is copyrighted by its authors, so the fixture keeps no text: the title's start and the
body's first and last characters are stored as [length, first 16 hex of SHA-256].
Needs the locally built data/abstracts/*.parquet (gitignored); skipped otherwise.
"""

import hashlib
import json
import pathlib

import pandas as pd
import pytest

from authors import key_authors, load_overrides, parse_authors

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/golden_2025.json").read_text())
FILES = sorted((ROOT / "data/abstracts").glob("2025_*.parquet"))


def matches(text, fingerprint):
    n, digest = fingerprint
    return len(text) == n and hashlib.sha256(text.encode()).hexdigest()[:16] == digest


pytestmark = pytest.mark.skipif(not FILES, reason="run extract_spans.py + segment.py for 2025 first")


@pytest.fixture(scope="module")
def abstracts():
    return pd.concat([pd.read_parquet(f) for f in FILES]).set_index("abstract_id")


@pytest.fixture(scope="module")
def overrides():
    return load_overrides(ROOT / "overrides.csv")


@pytest.mark.parametrize("g", GOLDEN, ids=[g["abstract_id"] for g in GOLDEN])
def test_golden(g, abstracts, overrides):
    r = abstracts.loc[g["abstract_id"]]
    assert r.page_start == g["page"]
    assert matches(r.title[: g["title_prefix"][0]], g["title_prefix"])
    assert matches(r.body[: g["body_start"][0]], g["body_start"])
    assert matches(r.body[-g["body_end"][0]:], g["body_end"])
    authors = parse_authors(r.authors_raw)
    if g["n_authors"] is not None:
        assert len(authors) == g["n_authors"]
    _, senior = key_authors(g["abstract_id"], authors, overrides)
    assert senior["name"].startswith(g["senior_name"])
