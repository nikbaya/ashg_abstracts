"""Author-string parsing, including submitter typos seen in the 2024-2025 booklets."""

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


# 2023-2024 numbered format


def test_numbered_affiliations():
    a = parse_authors("P. Kho1,2, S. Koyama3, T. Assimes1,3; 1Stanford Univ., Stanford, CA, "
                      "2VA Palo Alto, Palo Alto, CA, 3Broad Inst., Cambridge, MA")
    assert names("P. Kho1,2, S. Koyama3; 1X, 2Y, 3Z") == ["P. Kho", "S. Koyama"]
    assert a[0]["affils"] == ["Stanford Univ.", "Stanford", "CA", "VA Palo Alto", "Palo Alto", "CA"]
    assert senior_author(a)["affils"] == ["Stanford Univ.", "Stanford", "CA", "Broad Inst.", "Cambridge", "MA"]


def test_numbered_shared_affiliation():
    a = parse_authors("B. Liu, C. Tian, Y. Tong; Natl. Univ. of Singapore, Singapore, Singapore")
    assert all(x["affils"] == ["Natl. Univ. of Singapore", "Singapore", "Singapore"] for x in a)


def test_numbered_affiliation_starting_with_digit():
    a = parse_authors("S. Saini1, X. Wang2; 123andMe, Inc., Sunnyvale, CA, 223andMe, South San Francisco, CA")
    assert senior_author(a)["affils"] == ["23andMe", "South San Francisco", "CA"]


def test_numbered_postal_code_is_not_a_marker():
    a = parse_authors("A. One1, B. Two2; 1Fudan Univ., Shanghai, 200032, China, 2Yale Univ., New Haven, CT")
    assert a[0]["affils"] == ["Fudan Univ.", "Shanghai", "200032", "China"]


def test_numbered_groups_and_odd_names():
    raw = ("J-s. Lin1, Á. Carracedo2, M. AlThgafi (Abedalthagafi)1, V. G. Fowler, Jr.2, 23andMe Research Team, "
           "Japan COVID-19 Task Force; 1X Univ., Boston, MA, 2Y Inst., Madrid, Spain")
    a = parse_authors(raw)
    assert [x["name"] for x in a][:4] == ["J-s. Lin", "Á. Carracedo", "M. AlThgafi", "V. G. Fowler, Jr"]
    assert senior_author(a)["name"] == "V. G. Fowler, Jr"
    assert senior_author(a)["affils"] == ["Y Inst.", "Madrid", "Spain"]


def test_full_name_with_initial_is_not_numbered():
    a = parse_authors("F. Kumara Mastrorosa (Dept of Genome Sciences; Univ of Washington), B Two (Y)")
    assert names("F. Kumara Mastrorosa (Dept; Univ), B Two (Y)") == ["F. Kumara Mastrorosa", "B Two"]
    assert a[0]["affils"][0].startswith("Dept")


def test_authors_complete_at_page_break():
    from segment import authors_complete
    assert authors_complete("A. One1, B. Two2; 1X Univ., Boston, MA, 2Y Inst., Paris, France")
    assert not authors_complete("A. One1, B. Two2; 1X Univ., Boston, MA,")  # affiliation 2 still to come
    assert not authors_complete("A. One1, B. Two2, C. Three")  # names continue on the next page
