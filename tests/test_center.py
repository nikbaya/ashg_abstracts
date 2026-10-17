"""Center of mass: spherical midpoints, placing abstracts, clipping the site's basemap."""

import numpy as np
import pandas as pd
import pytest

from center import abstract_vectors, center, clip_ring, great_circle_km, unit


def test_midpoint_on_the_sphere():
    lat, lon = center(unit([0, 0], [-10, 10]))
    assert lat == pytest.approx(0) and lon == pytest.approx(0, abs=1e-9)
    lat, lon = center(unit([0, 0], [170, -170]))  # across the antimeridian, not through Greenwich
    assert abs(lon) == pytest.approx(180)
    lat, _ = center(unit([60, 60], [0, 180]))  # over the pole
    assert lat == pytest.approx(90)
    assert great_circle_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.01)


COORDS = pd.DataFrame({"country": ["United States", "Canada", "United Kingdom"],
                       "lat": [42.33, 45.50, 51.50], "lon": [-71.07, -73.59, -0.12]},
                      index=pd.Index(["Boston", "Montreal", "London"], name="metro"))
LABELS = {"France": (46.5, 2.5)}


def abstracts(*rows):
    return pd.DataFrame([{"metros": m, "first_countries": c, "talk": t} for m, c, t in rows])


def test_na_view_keeps_only_us_and_canadian_metros():
    d = abstracts((["Boston", "London"], ["United States", "United Kingdom"], True), (["London"], ["United Kingdom"], False))
    X, talk = abstract_vectors(d, COORDS, "na", LABELS)
    assert len(X) == 1 and talk.tolist() == [True]
    assert center(X) == pytest.approx((42.33, -71.07))  # its London half doesn't count here


def test_world_view_places_country_only_abstracts():
    d = abstracts((["London"], ["United Kingdom"], False), ([], ["United Kingdom"], False), ([], ["France"], True),
                  ([], [], True))  # no country: left out
    X, talk = abstract_vectors(d, COORDS, "world", LABELS)
    assert len(X) == 3
    assert center(X[1:2]) == pytest.approx((51.50, -0.12))  # the UK's placed abstracts
    assert center(X[2:3]) == pytest.approx(LABELS["France"])  # no placed abstract: the label point


def test_world_without_north_america():
    d = abstracts((["Montreal"], ["Canada"], True), (["Boston", "London"], ["United States", "United Kingdom"], True),
                  ([], ["United States"], False), (["London"], ["United Kingdom"], False))
    X, talk = abstract_vectors(d, COORDS, "world_no_na", LABELS)
    assert len(X) == 1 and talk.tolist() == [False]  # any North American first author drops the abstract


def test_clip_ring_to_box():
    square = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], float)
    out = clip_ring(square, 0, 0, 10, 10)
    assert sorted(map(tuple, out.round(9))) == [(0, 0), (0, 5), (5, 0), (5, 5)]
    assert len(clip_ring(square + 100, 0, 0, 10, 10)) == 0
