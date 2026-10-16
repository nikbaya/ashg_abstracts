"""Author-string parsing, including submitter typos seen in the 2025 booklets."""

from authors import first_author, parse_authors, senior_author


def names(raw):
    return [a["name"] for a in parse_authors(raw)]


def test_basic_and_nested_parens():
    a = parse_authors("Ann Lee (Broad Institute, HMS), Bo Chen (Univ. of Barcelona (IBUB), Spain).")
    assert [x["name"] for x in a] == ["Ann Lee", "Bo Chen"]
    assert a[0]["affils"] == ["Broad Institute", "HMS"]
    assert a[1]["affils"] == ["Univ. of Barcelona (IBUB)", "Spain"]


def test_double_open_paren_typo():
    raw = "A One (X Univ), B Two ((Y Univ), C Three (Z Univ)"
    assert names(raw) == ["A One", "B Two", "C Three"]


def test_unclosed_paren_then_semicolon():
    raw = "A One (X Univ), B Two (Dept of Genetics, Y Univ; C Three (Z Univ), D Four (W Univ)"
    assert names(raw) == ["A One", "B Two", "C Three", "D Four"]
    assert senior_author(parse_authors(raw))["name"] == "D Four"


def test_unclosed_paren_does_not_split_at_institution():
    raw = "A One (Dept, Yale School of Medicine (YSM, New Haven), B Two (Z Univ)"
    assert senior_author(parse_authors(raw))["name"] == "B Two"


def test_superscript_digits_on_name():
    assert names("A One (X), Renato Polimanti1,3,18,19,20 (Yale)") == ["A One", "Renato Polimanti"]


def test_missing_comma_between_authors():
    assert names("A One (X Univ) B Two (Y Univ)") == ["A One", "B Two"]


def test_comma_before_paren():
    a = parse_authors("Sharon E. Plon, (Baylor College of Medicine), Jo Ng (Ambry)")
    assert a[0] == {"name": "Sharon E. Plon", "affils": ["Baylor College of Medicine"]}


def test_trailing_group_is_skipped_for_senior():
    for tail in [" for the InterLymph Consortium", " and ClinGen Expert Panel", " the Million Veteran Program"]:
        a = parse_authors("A One (X), Jin J. Zhou (UCLA, Phoenix VA)" + tail)
        assert senior_author(a)["name"] == "Jin J. Zhou"


def test_group_authors_skipped():
    a = parse_authors("A One (X), B Two (Y), Global Lipids Genetics Consortium")
    assert senior_author(a)["name"] == "B Two"
    a = parse_authors("TOPMed Group (Z), A One (X), NHLBI Trans-Omics for Precision Medicine (TOPMed)")
    assert senior_author(a)["name"] == "A One"
    assert first_author(a)["name"] == "A One"
