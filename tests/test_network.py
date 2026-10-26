from network import author_key


def test_author_key_ignores_middle_names_and_initial_periods():
    expected = author_key("F. Satterstrom")
    assert author_key("F. Kyle Satterstrom") == expected
    assert author_key("F Kyle Satterstrom") == expected
