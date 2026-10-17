"""Geographic center of mass of first authors, per year: all accepted abstracts and talks, with and
without first authors from that year's host city, in North America (US and Canada), worldwide, and worldwide
without North America.

- outputs/trends_center.csv        one row per year x view (na/world/world_no_na) x host (with/without)
                                   x group (abstracts/talks): center lat/lon, abstracts, distance to the
                                   host city, and the bootstrap SD of the center east-west and north-south (km)
- outputs/trends_center_na.png     North America view, all accepted abstracts (host city included): a map and a
                                   zoom on the centers
- outputs/trends_center_world.png  world view, the same
- outputs/trends_center.json       every variant, projected for the site's interactive map, with each
                                   view's basemap as SVG paths

An abstract is placed at its first author's metro areas (metros.csv, coordinates in metro_coords.csv),
each of k metros weighted 1/k. The center is the mean of those points as unit vectors in 3D,
projected back to the surface (the geographic midpoint), so it is well defined worldwide.
- na: abstracts with a first-author metro in Northern America (US, Canada), counting only those metros.
- world: abstracts with a first-author metro; an abstract with only a country (data/geography) sits
  at the center of that country's placed abstracts that year (the Natural Earth label point if
  none), so every country keeps its full weight.
- world_no_na: world without abstracts that have a first author in Northern America (US, Canada).
"Without the host city" drops abstracts with any first-author metro equal to the host metro. Local
groups at institutions without a metro can't be identified, so they stay in. Every host so far is in
North America, so world_no_na is the same either way.

Basemaps are Natural Earth (public domain), downloaded by `make naturalearth` to data/raw/naturalearth.

Usage: python src/ashg/center.py [--years 2022 ...] [--boot 2000]
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from analysis import ROOT, load_metros
from figures import GRID, HOST_METRO, INK, INK_2, SURFACE, header, new_figure
from geography import REGION_OF

EARTH_KM = 6371.0
LAND, WATER, BORDER = "#efeee9", SURFACE, "#d6d4cd"
NE_COUNTRY = {"United States": "United States of America", "DR Congo": "Democratic Republic of the Congo"}
SHORT_HOST = {"Los Angeles": "LA", "Washington, DC": "DC"}
NORTH_AMERICA = "Northern America"  # geography.REGIONS: US, Canada, Bermuda (Mexico is Latin America)


# --- spherical geometry ------------------------------------------------------

def unit(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_latlon(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def great_circle_km(lat1, lon1, lat2, lon2):
    return EARTH_KM * np.arccos(np.clip((unit(lat1, lon1) * unit(lat2, lon2)).sum(-1), -1, 1))


def tangent_basis(lat, lon):
    """East and north unit vectors at (lat, lon)."""
    la, lo = np.radians(lat), np.radians(lon)
    return (np.array([-np.sin(lo), np.cos(lo), 0]),
            np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]))


def center(X, w=None):
    """Geographic midpoint of the rows of X (unit vectors, or per-abstract means of them)."""
    return to_latlon(X.sum(0) if w is None else w @ X)


def bootstrap_spread(X, lat, lon, boot, rng):
    """Resample abstracts: SD of the center east-west and north-south (km), in the tangent plane at the center."""
    W = rng.multinomial(len(X), np.full(len(X), 1 / len(X)), size=boot).astype(float)
    V = W @ X
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    e, n = tangent_basis(lat, lon)
    xy = EARTH_KM * np.c_[V @ e, V @ n]
    sd = xy.std(0, ddof=1)
    return sd[0], sd[1]


# --- placing abstracts -------------------------------------------------------

def load_years(data, years, metros_csv, coords_csv):
    """One row per abstract: year, talk, first-author metros (deduplicated) and countries."""
    metro_of = load_metros(metros_csv).set_index("institute").metro
    coords = pd.read_csv(coords_csv, comment="#").set_index("metro")
    missing = set(metro_of) - set(coords.index)
    if missing:
        raise ValueError(f"metros missing from {coords_csv}: {sorted(missing)}")
    frames = []
    for y in years:
        insts = pd.read_parquet(data / f"institutions/{y}.parquet")
        geo = pd.read_parquet(data / f"geography/{y}.parquet", columns=["abstract_id", "first_countries"])
        frames.append(insts.merge(geo, on="abstract_id", validate="1:1"))
    d = pd.concat(frames, ignore_index=True)
    d["talk"] = d.pres_type != "poster"
    d["metros"] = d.first_insts.apply(lambda l: list(dict.fromkeys(metro_of[i] for i in l if i in metro_of.index)))
    d["first_countries"] = d.first_countries.apply(list)
    return d[["abstract_id", "year", "talk", "metros", "first_countries"]], coords


def label_points(path):
    """Natural Earth country label points {name: (lat, lon)}, for countries with no placed abstract."""
    props = [f["properties"] for f in json.loads(pathlib.Path(path).read_text())["features"]]
    by_name = {p[k]: (p["LABEL_Y"], p["LABEL_X"]) for k in ("NAME_LONG", "NAME", "ADMIN") for p in props}
    out = {c: by_name.get(NE_COUNTRY.get(c, c)) for c in REGION_OF}
    missing = [c for c, v in out.items() if v is None]
    if missing:
        raise ValueError(f"countries not in {path} (add to NE_COUNTRY): {missing}")
    return out


def abstract_vectors(d, coords, view, labels=None):
    """(vectors, talk) for one year's abstracts: each abstract's mean unit vector (weight 1 in total).
    view "na": US and Canadian metros only; "world": metros, else the country's center (see module docstring);
    "world_no_na": the same without abstracts that have a first author in Northern America (US, Canada)."""
    xyz = dict(zip(coords.index, unit(coords.lat, coords.lon)))
    country = coords.country.to_dict()
    if view == "na":
        ms = d.metros.apply(lambda m: [x for x in m if REGION_OF[country[x]] == NORTH_AMERICA])
        keep = ms.str.len() > 0
        return np.array([np.mean([xyz[x] for x in m], 0) for m in ms[keep]]), d.talk[keep].to_numpy()
    if view == "world_no_na":
        na = d.first_countries.apply(lambda cs: any(REGION_OF[c] == NORTH_AMERICA for c in cs)) | \
            d.metros.apply(lambda ms: any(REGION_OF[country[x]] == NORTH_AMERICA for x in ms))
        d = d[~na]
    placed = d[d.metros.str.len() > 0]
    sums = {}
    for m in placed.metros:
        for x in m:
            sums[country[x]] = sums.get(country[x], 0) + xyz[x] / len(m)
    country_xyz = {c: v / np.linalg.norm(v) for c, v in sums.items()}

    def where(c):
        return country_xyz[c] if c in country_xyz else unit(*labels[c])

    vecs, talk = [], []
    for r in d.itertuples(index=False):
        if r.metros:
            vecs.append(np.mean([xyz[x] for x in r.metros], 0))
        elif r.first_countries:
            vecs.append(np.mean([where(c) for c in r.first_countries], 0))
        else:
            continue
        talk.append(r.talk)
    return np.array(vecs), np.array(talk)


def center_table(d, coords, labels, years, boot, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        host = HOST_METRO[y]
        hlat, hlon = coords.loc[host, ["lat", "lon"]]
        year = d[d.year == y]
        for hosted in ("with", "without"):
            sub = year if hosted == "with" else year[~year.metros.apply(lambda m: host in m)]
            for view in ("na", "world", "world_no_na"):
                X, talk = abstract_vectors(sub, coords, view, labels)
                for group, mask in (("abstracts", np.ones(len(X), bool)), ("talks", talk)):
                    lat, lon = center(X[mask])
                    sd_e, sd_n = bootstrap_spread(X[mask], lat, lon, boot, rng)
                    rows.append({"year": y, "view": view, "host_city": hosted, "group": group, "host": host,
                                 "lat": lat, "lon": lon, "abstracts": int(mask.sum()),
                                 "host_km": great_circle_km(lat, lon, hlat, hlon),
                                 "sd_east_km": sd_e, "sd_north_km": sd_n})
    return pd.DataFrame(rows)


# --- map projections (spherical; x, y in km) ----------------------------------

def albers(lat0, lon0, p1, p2):
    """Albers equal-area conic."""
    p1, p2, phi0 = np.radians([p1, p2, lat0])
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(c - 2 * n * np.sin(phi0)) / n

    def project(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        rho = np.sqrt(c - 2 * n * np.sin(np.radians(lat))) / n
        theta = n * np.radians((lon - lon0 + 180) % 360 - 180)
        return EARTH_KM * rho * np.sin(theta), EARTH_KM * (rho0 - rho * np.cos(theta))
    return project


def azimuthal(lat0, lon0):
    """Lambert azimuthal equal-area, centered on (lat0, lon0)."""
    p0, l0 = np.radians([lat0, lon0])

    def project(lat, lon):
        p, dl = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)) - l0
        k = np.sqrt(2 / np.maximum(1 + np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(dl), 1e-9))
        return (EARTH_KM * k * np.cos(p) * np.sin(dl),
                EARTH_KM * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(dl)))
    return project


def equal_earth(lat, lon):
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(np.radians(np.asarray(lat, float))))
    lam = np.radians(np.asarray(lon, float))
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * a4 * th ** 8 + 7 * a3 * th ** 6 + 3 * a2 * th ** 2 + a1))
    return EARTH_KM * x, EARTH_KM * (a4 * th ** 9 + a3 * th ** 7 + a2 * th ** 3 + a1 * th)


def rings(path, keep=lambda p: True):
    """Exterior and hole rings (lon/lat arrays) of every polygon in a GeoJSON file whose properties pass keep."""
    out = []
    for f in json.loads(pathlib.Path(path).read_text())["features"]:
        if not keep(f["properties"]):
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        out += [np.asarray(r) for p in polys for r in p]
    return out


def draw_land(ax, project, land, borders, holes=()):
    """land/borders/holes: lists of lon/lat rings (holes are lakes, painted as water)."""
    def coll(rs, **kw):
        return PolyCollection([np.c_[project(r[:, 1], r[:, 0])] for r in rs], **kw)
    ax.add_collection(coll(land, facecolor=LAND, edgecolor="none", zorder=0))
    ax.add_collection(coll(holes, facecolor=WATER, edgecolor="none", zorder=0.1))
    ax.add_collection(coll(borders, facecolor="none", edgecolor=BORDER, linewidth=0.6, zorder=0.2))


# --- figures -----------------------------------------------------------------

# reference places that aren't in metro_coords.csv
REF_PLACES = {"Winnipeg": (49.90, -97.14), "Moosonee": (51.27, -80.64), "Thunder Bay": (48.38, -89.25),
              "Sudbury": (46.49, -80.99), "Kuujjuarapik": (55.28, -77.75), "Chibougamau": (49.91, -74.37),
              "Oslo": (59.91, 10.75), "Tromsø": (69.65, 18.96), "Murmansk": (68.97, 33.08),
              "Arkhangelsk": (64.54, 40.54), "Vorkuta": (67.50, 64.05), "Moscow": (55.76, 37.62),
              "Kazan": (55.79, 49.12), "Yekaterinburg": (56.84, 60.60), "Omsk": (54.99, 73.37),
              "Novosibirsk": (55.03, 82.92), "Astana": (51.17, 71.43), "Perm": (58.01, 56.25), "Kyiv": (50.45, 30.52)}
# per view: projection, extents (lat range, lon range), reference places and water labels. "site" is the
# site's map, which holds every variant of the view (talks, host city dropped); "zoom" and "overview" are
# the PNG's (na and world only; abstracts with the host city). Places outside a map are skipped.
VIEWS = {
    "na": dict(proj=albers(37.5, -96, 29.5, 45.5), overview_proj=albers(37.5, -96, 29.5, 45.5),
               overview=((24.5, 50), (-124.5, -64)), zoom=((38.9, 42.0), (-91.2, -84.4)),
               site=((37.8, 42.6), (-92.4, -83.6)), basemap="north_america",
               refs=["Chicago", "Indianapolis", "St. Louis", "Cincinnati", "Iowa City", "Madison"],
               water={"Lake Michigan": (42.3, -87.2)}),
    "world": dict(proj=albers(52, -80, 45, 60), overview_proj=equal_earth,
                  overview=((-40, 75), (-170, 180)), zoom=((50.2, 57.2), (-88, -72)),
                  site=((47, 58.2), (-91, -71)), basemap="north_america",
                  refs=["Moosonee", "Thunder Bay", "Sudbury", "Kuujjuarapik", "Chibougamau", "Winnipeg"],
                  water={"Hudson Bay": (57.6, -84.5)}),
    "world_no_na": dict(proj=azimuthal(56, 50), site=((45.5, 64.5), (18, 84)), basemap="eurasia",
                        refs=["Moscow", "Kazan", "Novosibirsk", "Astana", "Perm",
                              "Helsinki", "Arkhangelsk", "Kyiv"],
                        water={"Caspian Sea": (46.8, 50.5), "Black Sea": (46.3, 34.5)}),
}
# (view, year) -> (dx, dy, ha) of the PNG's year label, set after looking at the figure
LABEL_OFFSETS = {("na", 2022): (-8, 0, "right"), ("na", 2023): (-6, -10, "right"), ("na", 2024): (8, -3, "left"),
                 ("na", 2025): (8, 6, "left"), ("na", 2026): (-8, 7, "right"),
                 ("world", 2022): (8, -4, "left"), ("world", 2023): (8, 4, "left"),
                 ("world", 2024): (-8, -6, "right"), ("world", 2025): (8, 7, "left"),
                 ("world", 2026): (8, 4, "left")}


def extent(project, lat_range, lon_range):
    xs, ys = project(*np.meshgrid(np.linspace(*lat_range, 30), np.linspace(*lon_range, 30)))
    return xs.min(), xs.max(), ys.min(), ys.max()


def map_axes(ax, project, lat_range, lon_range):
    x0, x1, y0, y1 = extent(project, lat_range, lon_range)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor(WATER)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)


def basemap(ne, scope):
    """(land, borders, lakes) lon/lat rings. "world": coarse countries; "north_america": US, Canada and
    Mexico with states, provinces and large lakes; "eurasia": northern-hemisphere countries and large lakes."""
    lakes = rings(ne / "ne_50m_lakes.geojson", lambda p: p.get("scalerank", 9) <= 1)
    if scope == "world":
        land = rings(ne / "ne_110m_admin_0_countries.geojson")
        return land, land, []
    if scope == "eurasia":  # the far side of the globe would wrap around an azimuthal map
        land = [r for r in rings(ne / "ne_50m_admin_0_countries.geojson") if r[:, 1].min() > 0]
        return land, land, lakes
    land = rings(ne / "ne_50m_admin_0_countries.geojson",
                 lambda p: p["ADMIN"] in ("United States of America", "Canada", "Mexico"))
    states = rings(ne / "ne_50m_admin_1_states_provinces_lakes.geojson", lambda p: p["adm0_a3"] in ("USA", "CAN"))
    return land, land + states, lakes


def place(coords, name):
    return tuple(coords.loc[name, ["lat", "lon"]]) if name in coords.index else REF_PLACES[name]


def center_figure(t, coords, ne, years, view, path):
    """All accepted abstracts, host city included: overview map (centers, host cities) and a zoom."""
    v = VIEWS[view]
    g = t[(t.view == view) & (t.group == "abstracts") & (t.host_city == "with")].sort_values("year")
    zoom_map = basemap(ne, v["basemap"])
    overview_map = zoom_map if view == "na" else basemap(ne, "world")
    (zlat, zlon), (olat, olon) = v["zoom"], v["overview"]
    zx0, zx1, zy0, zy1 = extent(v["proj"], zlat, zlon)
    ox0, ox1, oy0, oy1 = extent(v["overview_proj"], olat, olon)
    # equal heights; widths follow each map's aspect, filling the 9.6in between the margins
    w_ratio = [(ox1 - ox0) / (oy1 - oy0), (zx1 - zx0) / (zy1 - zy0)]
    wspace = 0.04
    map_h = 9.6 / (sum(w_ratio) * (1 + wspace / 2))
    h = 1.45 + map_h + 0.45
    fig, gs = new_figure((10, h), ncols=2, width_ratios=w_ratio, wspace=wspace,
                         left=0.02, right=0.98, top=1 - 1.45 / h, bottom=0.45 / h)

    ax = fig.add_subplot(gs[0])
    op = v["overview_proj"]
    map_axes(ax, op, olat, olon)
    draw_land(ax, op, *overview_map)
    for y in years:
        ax.scatter(*op(*place(coords, HOST_METRO[y])), s=60, facecolor="none", edgecolor=INK, linewidth=1.1,
                   zorder=5)
    if view == "na":  # name the hosts here; on the world map they'd pile up
        offsets = {"Los Angeles": (6, -7), "Washington, DC": (6, -6), "Boston": (6, -4), "Montreal": (-6, 7),
                   "Denver": (6, 0)}
        for y in years:
            name = HOST_METRO[y]
            dx, dy = offsets.get(name, (6, 0))
            ax.annotate(f"{y} {SHORT_HOST.get(name, name)}", op(*place(coords, name)), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8, color=INK, va="center",
                        ha="left" if dx > 0 else "right", zorder=6)
    ax.scatter(*op(g.lat.to_numpy(), g.lon.to_numpy()), s=14, color=INK, zorder=4)
    zx, zy = op(*np.meshgrid(np.linspace(*zlat, 20), np.linspace(*zlon, 20)))
    ax.add_patch(Rectangle((zx.min(), zy.min()), np.ptp(zx), np.ptp(zy), fill=False, ec=INK_2, lw=0.9,
                           ls=(0, (3, 2)), zorder=4))

    zax = fig.add_subplot(gs[1])
    zp = v["proj"]
    map_axes(zax, zp, zlat, zlon)
    draw_land(zax, zp, *zoom_map)
    for name in v["refs"]:
        lat, lon = place(coords, name)
        if not (zlat[0] < lat < zlat[1] and zlon[0] < lon < zlon[1] - 0.15 * np.ptp(zlon)):  # room for the label
            continue
        rx, ry = zp(lat, lon)
        zax.scatter(rx, ry, s=6, color=INK_2, zorder=1)
        zax.annotate(name, (rx, ry), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK_2,
                     va="center", zorder=1)
    for name, (lat, lon) in v["water"].items():
        if zlat[0] < lat < zlat[1] and zlon[0] < lon < zlon[1]:
            zax.annotate(name, zp(lat, lon), fontsize=7.5, color=INK_2, style="italic", ha="center",
                         va="center", zorder=1)
    x, y = zp(g.lat.to_numpy(), g.lon.to_numpy())
    zax.plot(x, y, color=INK, lw=1.2, alpha=0.45, zorder=2)
    zax.scatter(x, y, s=48, color=INK, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    for xi, yi, r in zip(x, y, g.itertuples()):
        dx, dy, ha = LABEL_OFFSETS.get((view, r.year), (7, 6, "left"))
        zax.annotate(f"{r.year} · {SHORT_HOST.get(r.host, r.host)}", (xi, yi), xytext=(dx, dy),
                     textcoords="offset points", fontsize=8.5, fontweight="bold", color=INK, zorder=6,
                     ha=ha, va="center")

    handles = [Line2D([], [], color=INK, marker="o", ms=7, lw=1.2, markeredgecolor=SURFACE,
                      label="center of mass, each year (labelled with that year's host city)"),
               Line2D([], [], ls="", marker="o", ms=8, markerfacecolor="none", markeredgecolor=INK,
                      label="host city")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1 - 1.05 / h), ncol=2, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=1.8)
    n = g.abstracts
    if view == "na":
        title = f"The center of North American abstracts, ASHG {years[0]}–{years[-1]}"
        sub = (f"Center of mass of first authors at US and Canadian metro areas, all accepted abstracts "
               f"({n.min():,}–{n.max():,} a year; about four in five\nUS and Canadian abstracts have a mapped metro). "
               "The center follows the meeting: west for Los Angeles, east for Boston, north for Montreal.")
        note = ""
    else:
        title = f"The center of ASHG abstracts worldwide, {years[0]}–{years[-1]}"
        sub = (f"Center of mass of first authors worldwide, all accepted abstracts ({n.min():,}–{n.max():,} a year). "
               "On the globe, a US-heavy field pulled toward\nEurope and East Asia centers in northern Canada. It "
               "sits farther north in years with a smaller share from North America (2023, 2026).")
        note = " 6–7% of 2025–26 abstracts have no country (mostly non-US)."
    header(fig, title, sub, "Source: ASHG abstracts; maps Natural Earth. Center = geographic midpoint of first-author "
           f"metros.{note}", top=1 - 0.2 / h)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- site data ---------------------------------------------------------------

SVG_WIDTH = 760


def clip_ring(pts, x0, y0, x1, y1):
    """Sutherland-Hodgman: a polygon (n x 2) clipped to the box."""
    for axis, bound, keep_below in ((0, x0, False), (0, x1, True), (1, y0, False), (1, y1, True)):
        if not len(pts):
            break
        inside = pts[:, axis] <= bound if keep_below else pts[:, axis] >= bound
        out = []
        for i in range(len(pts)):
            a, b, ia, ib = pts[i - 1], pts[i], inside[i - 1], inside[i]
            if ia != ib:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
            if ib:
                out.append(b)
        pts = np.array(out)
    return pts


def svg_path(pts, min_step=0.6):
    """Polygon as an SVG path, dropping points within min_step units of the last one kept."""
    kept = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - kept[-1][0]) + abs(p[1] - kept[-1][1]) >= min_step:
            kept.append(p)
    return "M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in kept) + "Z" if len(kept) > 2 else ""


def site_payload(t, coords, ne, view):
    """One view's map in SVG units (y down, clipped to the map) and every variant's centers."""
    v = VIEWS[view]
    project = v["proj"]
    x0, x1, y0, y1 = extent(project, *v["site"])
    s = SVG_WIDTH / (x1 - x0)
    w, h = SVG_WIDTH, (y1 - y0) * s

    def to_svg(lat, lon):
        x, y = project(lat, lon)
        return (np.asarray(x) - x0) * s, (y1 - np.asarray(y)) * s

    def layer(rs):
        out = []
        for r in rs:
            pts = clip_ring(np.c_[to_svg(r[:, 1], r[:, 0])], -4, -4, w + 4, h + 4)  # edges land off-map
            if len(pts) > 2:
                out.append(svg_path(pts))
        return "".join(out)

    def labels(items):
        out = []
        for name, (lat, lon) in items:
            x, y = to_svg(lat, lon)
            if 0 < x < w - 70 and 10 < y < h - 10:  # room for the label
                out.append([name, round(float(x), 1), round(float(y), 1)])
        return out

    land, borders, lakes = basemap(ne, v["basemap"])
    variants = {}
    tv = t[t.view == view]
    for (group, hosted), g in tv.groupby(["group", "host_city"]):
        other = tv[(tv.group == group) & (tv.host_city != hosted)].set_index("year")
        rows = []
        for r in g.sort_values("year").itertuples():
            x, y = to_svg(r.lat, r.lon)
            o = other.loc[r.year]
            rows.append({"year": int(r.year), "x": round(float(x), 1), "y": round(float(y), 1),
                         "lat": round(r.lat, 2), "lon": round(r.lon, 2), "n": int(r.abstracts),
                         "host_km": round(r.host_km), "sd_e": round(r.sd_east_km), "sd_n": round(r.sd_north_km),
                         "pull_km": round(float(great_circle_km(r.lat, r.lon, o.lat, o.lon)))})
        variants[f"{group}|{hosted}"] = rows
    # borders = None: stroke the land outlines (country-only maps)
    return {"width": w, "height": round(h), "land": layer(land),
            "borders": None if borders is land else layer(borders), "lakes": layer(lakes),
            "places": labels((n, place(coords, n)) for n in v["refs"]), "water": labels(v["water"].items()),
            "variants": variants}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--metros", default=ROOT / "metros.csv")
    ap.add_argument("--coords", default=ROOT / "metro_coords.csv")
    ap.add_argument("--naturalearth", default="data/raw/naturalearth")
    args = ap.parse_args()
    data, out, ne, years = pathlib.Path(args.data), pathlib.Path(args.out), pathlib.Path(args.naturalearth), sorted(args.years)
    d, coords = load_years(data, years, args.metros, args.coords)
    t = center_table(d, coords, label_points(ne / "ne_50m_admin_0_countries.geojson"), years, args.boot)
    t.to_csv(out / "trends_center.csv", index=False, float_format="%.5g")
    for view in ("na", "world"):
        center_figure(t, coords, ne, years, view, out / f"trends_center_{view}.png")
    payload = {"years": years, "hosts": {str(y): HOST_METRO[y] for y in years},
               "views": {view: site_payload(t, coords, ne, view) for view in VIEWS}}
    (out / "trends_center.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote trends_center.csv, trends_center_{{na,world}}.png, trends_center.json to {out}/")


if __name__ == "__main__":
    main()
