"""Senior-author identity resolution, within a year and across years (initials)."""

import pandas as pd

from people import compatible, display_name, resolve_people


def occ(rows):
    return pd.DataFrame([{"name_key": k, "first": f, "tokens": set(t)} for k, f, t in rows])


def groups(df):
    return sorted(sorted(g.index) for _, g in df.groupby("person_id"))


def test_initials_join_the_spelled_out_name_across_years():
    df = resolve_people(occ([("a antonellis", None, ["UM"]), ("a antonellis", None, []),
                             ("a antonellis", "anthony", ["UM"])]), common=set())
    assert groups(df) == [[0, 1, 2]]


def test_different_first_names_stay_apart_and_initials_follow_institutions():
    # distinctive surname, but "J." covers Jennifer and John: fall back to shared institutions
    df = resolve_people(occ([("j doe", "jennifer", ["Broad"]), ("j doe", "john", ["Yale"]),
                             ("j doe", None, ["Yale"]), ("j doe", None, [])]), common=set())
    assert groups(df) == [[0], [1, 2], [3]]  # the affiliation-less "J. Doe" is ambiguous


def test_common_surname_needs_a_shared_institution():
    df = resolve_people(occ([("j li", None, ["Broad"]), ("j li", None, ["Stanford"]), ("j li", None, ["Broad"])]),
                        common={"li"})
    assert groups(df) == [[0, 2], [1]]


def test_short_forms_are_compatible_and_preferred_spelling_is_spelled_out():
    assert compatible({"alex", "alexander"}) and not compatible({"alex", "andrew"})
    assert display_name(pd.Series(["J. K. Pritchard", "J. K. Pritchard", "Jonathan Pritchard"])) == "Jonathan Pritchard"
