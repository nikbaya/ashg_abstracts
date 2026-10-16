"""Author countries from affiliation text."""

from geography import REGION_OF, segment_countries, text_countries


def test_location_segments():
    assert segment_countries(["Univ. of Copenhagen", "Copenhagen", "Denmark"]) == ["Denmark"]
    assert segment_countries(["Johns Hopkins Med.", "Baltimore", "MD"]) == ["United States"]
    assert segment_countries(["Concordia Univ.", "Montreal", "QC", "Canada"]) == ["Canada"]
    assert segment_countries(["Asan Med. Ctr.", "Seoul", "Korea", "Republic of"]) == ["South Korea"]
    assert segment_countries(["Univ. of Tehran", "Tehran", "Iran", "Islamic Republic of"]) == ["Iran"]
    assert segment_countries(["IRCCS San Raffaele", "Milan", "Italy; Univ. of Oxford", "Oxford", "United Kingdom"]) \
        == ["Italy", "United Kingdom"]
    assert segment_countries(["Stanford University", "Department of Genetics"]) == []


def test_text_names_most_specific_first():
    assert text_countries(["Seoul National University"]) == ["South Korea"]
    assert text_countries(["University of New Mexico"]) == ["United States"]  # not Mexico
    assert text_countries(["Indiana University"]) == ["United States"]  # not India
    assert text_countries(["University of British Columbia"]) == ["Canada"]
    assert text_countries(["5 Prime Sciences"]) == []


def test_every_country_has_one_region():
    assert REGION_OF["Mexico"] == "Latin America and the Caribbean" and REGION_OF["Canada"] == "Northern America"
