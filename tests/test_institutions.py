"""Alias-table matching against aliases.csv."""

import pathlib

import pytest

from institutions import load_aliases, match_institutes

ALIASES = load_aliases(pathlib.Path(__file__).resolve().parents[1] / "aliases.csv")


def insts(*affils):
    return match_institutes({"name": "X", "affils": list(affils)}, ALIASES)


@pytest.mark.parametrize("affils, expected", [
    (["Broad Institute of MIT and Harvard"], ["Broad Institute"]),  # MIT/Harvard not double-counted
    (["Broad Inst"], ["Broad Institute"]),
    (["Massachusetts General Hospital/Broad Institute"], ["Broad Institute", "Massachusetts General Hospital"]),
    (["University of Washington"], ["University of Washington"]),
    (["Washington University School of Medicine"], ["Washington University in St. Louis"]),
    (["University of California", "San Diego"], ["UC San Diego"]),  # comma-split segments
    (["Perelman School of Medicine", "University of Pennsylvania"], ["University of Pennsylvania"]),
    (["Harvard Medical School", "Harvard T.H. Chan School of Public Health"],
     ["Harvard Medical School", "Harvard University"]),
    (["NHGRI", "National Institutes of Health"], ["NIH"]),
    (["The Chinese University of Hong Kong"], ["The Chinese University of Hong Kong"]),
    (["Boston Children’s Hospital"], ["Boston Children's Hospital"]),  # curly apostrophe
    (["Department of Genetics", "Some Small College"], []),
])
def test_match(affils, expected):
    assert insts(*affils) == expected
