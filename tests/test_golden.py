"""Golden set: abstracts hand-checked against the PDFs (tests/fixtures/golden_2025_review.md).

Needs the locally built data/abstracts/*.parquet (gitignored); skipped otherwise.
"""

import json
import pathlib

import pandas as pd
import pytest

from authors import key_authors, load_overrides, parse_authors

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/golden_2025.json").read_text())
FILES = sorted((ROOT / "data/abstracts").glob("2025_*.parquet"))

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
    assert r.title.startswith(g["title_prefix"])
    assert r.body.startswith(g["body_start"]) and r.body.endswith(g["body_end"])
    authors = parse_authors(r.authors_raw)
    if g["n_authors"] is not None:
        assert len(authors) == g["n_authors"]
    _, senior = key_authors(g["abstract_id"], authors, overrides)
    assert senior["name"].startswith(g["senior_name"])
