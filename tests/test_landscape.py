import re

from landscape import author_institutions, link_initials, person_key


def test_initial_name_with_middle_name_matches_without_it():
    expected = person_key("F. Satterstrom")
    assert person_key("F. Kyle Satterstrom") == expected
    assert person_key("F Kyle Satterstrom") == expected


def test_spelled_out_middle_name_remains_distinguishing():
    assert person_key("Mary Ann Lee") != person_key("Mary Lee")


def test_initials_stay_separate_when_abstract_affiliations_conflict():
    listed = [
        [("taylor lee", "Taylor Lee")],
        [("t lee", "T. Lee")],
    ]
    institutions = {
        "taylor lee": {"University of Oxford"},
        "t lee": {"Icahn School of Medicine at Mount Sinai"},
    }
    resolved, _, conflicts = link_initials(
        listed, institutions, ["oxford", "mount-sinai"],
        {"oxford": {"taylor lee": {"University of Oxford"}},
         "mount-sinai": {"t lee": {"Icahn School of Medicine at Mount Sinai"}}},
    )
    assert resolved[1] == [("t lee", "T. Lee")]
    assert conflicts == [(1, "t lee", "T. Lee", "taylor lee")]


def test_initials_merge_when_abstract_affiliations_agree():
    listed = [
        [("taylor lee", "Taylor Lee")],
        [("t lee", "T. Lee")],
    ]
    institutions = {
        "taylor lee": {"University of Oxford"},
        "t lee": {"University of Oxford"},
    }
    resolved, _, conflicts = link_initials(
        listed, institutions, ["full", "initial"],
        {"full": {"taylor lee": {"University of Oxford"}},
         "initial": {"t lee": {"University of Oxford"}}},
    )
    assert resolved[1] == [("taylor lee", "T. Lee")]
    assert not conflicts


def test_unique_initial_match_is_preserved_without_affiliation_evidence():
    listed = [[("taylor lee", "Taylor Lee")], [("t lee", "T. Lee")]]
    institutions = {"taylor lee": {"University of Oxford"}}
    resolved, _, conflicts = link_initials(listed, institutions, ["full", "unknown"], {})
    assert resolved[1] == [("taylor lee", "T. Lee")]
    assert not conflicts


def test_hyphenated_initials_match_hyphenated_given_name():
    listed = [[(person_key("Po-Ru Loh"), "Po-Ru Loh")],
              [(person_key("P-R. Loh"), "P-R. Loh")]]
    resolved, linked, conflicts = link_initials(listed, {}, ["full", "initial"], {})
    assert resolved[1] == [("poru loh", "P-R. Loh")]
    assert linked == 1
    assert not conflicts


def test_curated_author_alias_overrides_affiliation_conflict():
    listed = [
        [("emilie wigdor", "Emilie M. Wigdor")],
        [("e wigdor", "E. Wigdor")],
    ]
    institutions = {
        "emilie wigdor": {"University of Oxford"},
        "e wigdor": {"Wellcome Sanger Institute"},
    }
    resolved, _, conflicts = link_initials(
        listed, institutions, ["full", "initial"],
        {"full": {"emilie wigdor": {"University of Oxford"}},
         "initial": {"e wigdor": {"Wellcome Sanger Institute"}}},
    )
    assert resolved[1] == [("emilie wigdor", "E. Wigdor")]
    assert not conflicts


def test_author_institutions_follow_numbered_affiliation_references():
    aliases = [
        ("University of Oxford", re.compile(r"University of Oxford", re.I)),
        ("Icahn School of Medicine at Mount Sinai", re.compile(r"Icahn School of Medicine", re.I)),
    ]
    raw = "J. Tyrmi1, S. S. Venkatesh14,15; 1University of Helsinki, 14Big Data Institute, University of Oxford, 15Wellcome Centre, University of Oxford"
    found = author_institutions(raw, aliases)
    assert found["s venkatesh"] == {"University of Oxford"}


def test_author_institutions_read_parenthetical_affiliations():
    aliases = [("University of Oxford", re.compile(r"University of Oxford", re.I))]
    found = author_institutions("Taylor Lee (University of Oxford), T. Lee (University of Pennsylvania)", aliases)
    assert found == {"taylor lee": {"University of Oxford"}, "t lee": set()}


def test_initials_stay_separate_when_only_table_affiliation_conflicts():
    # 2024 poster 1104F: "A. Price, ...; Univ. of Colorado" has no numbered affiliations, so the table's first-author
    # institution is the only evidence for "A. Price"
    listed = [
        [("alkes price", "Alkes L. Price")],
        [("a price", "A. Price")],
    ]
    institutions = {"alkes price": {"Harvard University"}}
    resolved, _, conflicts = link_initials(
        listed, institutions, ["harvard", "colorado"],
        {"harvard": {"alkes price": {"Harvard University"}},
         "colorado": {"a price": {"University of Colorado"}}},
    )
    assert resolved[1] == [("a price", "A. Price")]
    assert conflicts == [(1, "a price", "A. Price", "alkes price")]
