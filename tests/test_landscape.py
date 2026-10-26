from landscape import person_key


def test_initial_name_with_middle_name_matches_without_it():
    expected = person_key("F. Satterstrom")
    assert person_key("F. Kyle Satterstrom") == expected
    assert person_key("F Kyle Satterstrom") == expected


def test_spelled_out_middle_name_remains_distinguishing():
    assert person_key("Mary Ann Lee") != person_key("Mary Lee")
