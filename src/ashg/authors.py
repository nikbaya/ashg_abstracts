"""Parse ASHG author strings and resolve senior-author identities.

2025 format: "Full Name (Affil, Affil), Name Two (Affil)." — affiliations are
inline parentheticals whose commas separate both affiliations and the parts of
one affiliation (dept, city), so they are kept as a flat list of segments.
2023-2024 format: numbered superscripts with an affiliation list after ";"
(see parse_numbered); its affiliations are split into the same flat segments.
"""

import csv
import re


def split_top_level(s, sep=",", split_after_paren=False):
    """Split on `sep` only where not nested inside parentheses.

    With split_after_paren, also split after a top-level ")" that is followed
    by a capitalized word (a missing comma between two authors).
    """
    parts, depth, buf = [], 0, []
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
        if split_after_paren and ch == ")" and depth == 0 and re.match(r"\s+[A-Z]", s[i + 1:]):
            parts.append("".join(buf))
            buf = []
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


INSTITUTION_WORDS = re.compile(
    r"univ|inst|dep|cent|hosp|school|college|lab|clinic|foundation|research|medic|national|health"
    r"|science|program|unit|division|faculty|genetic|genomic|biolog|consorti|ltd|inc\b|llc", re.I)
PARTICLES = {"de", "da", "del", "della", "der", "di", "du", "la", "le", "van", "von", "y", "bin", "al"}


def looks_like_person(text):
    words = text.split()
    return (2 <= len(words) <= 5 and not INSTITUTION_WORDS.search(text) and not re.search(r"\d", text)
            and all(w[0].isupper() or w.lower() in PARTICLES for w in words))


def repair_unclosed(raw):
    """Close an affiliation left open by the submitter, at the next "…, Person Name (".

    Only called when "(" outnumbers ")", where one unclosed parenthesis would
    otherwise swallow every later author.
    """
    out, depth, last_sep = [], 0, None
    for ch in raw:
        if ch in ",;":
            last_sep = len(out)
        elif ch == "(":
            if depth >= 1 and last_sep is not None and looks_like_person("".join(out[last_sep + 1:])):
                out[last_sep] = ","  # authors may have been separated by ";"
                out.insert(last_sep, ")")
                depth -= 1
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        out.append(ch)
    return "".join(out)


def repair_raw(raw):
    """Fix submitter typos in an author string that would derail parsing."""
    raw = raw.strip().rstrip(".")
    if raw.count("(") > raw.count(")"):
        raw = re.sub(r"\(\s*\(", "(", raw)  # typo "Name ((Affil)" swallows later authors
    if raw.count("(") > raw.count(")"):
        raw = repair_unclosed(raw)
    return re.sub(r"(?<=[a-z])\d+(?:\s*,\s*\d+)*(?=\s*\()", "", raw)  # "Polimanti1,3,18 (Affil)"


NUMBERED_RE = re.compile(r"^\s*[^\W\d_]{1,2}(?:-\s?[^\W\d_]{1,2})?\.[^,;(]*")  # "M. ", "K-H. ", "Á. "


def is_numbered(raw):
    """2023-2024 format: "A. Name1,2, B. Name2; 1Univ. of X, City, ST, 2Inst. Y, City, Country"."""
    first = raw.split(";")[0].split(",")[0]
    # a "(" in the first author is a 2025-style affiliation, unless it is "(Maiden name)1"
    return ";" in raw and bool(NUMBERED_RE.match(first)) and not re.search(r"\((?![^)]*\)\s*\d)", first)


def split_numbered_affils(text):
    """{number: affiliation}, or {0: text} for one shared, unnumbered affiliation.

    Markers are found in sequence (1, then 2, ...) so digits inside an
    affiliation ("2nd Floor") are not mistaken for the next marker. A marker may
    be followed by more digits if a word follows ("123andMe" = 1 + "23andMe"),
    but not by a bare number (", 200032, China").
    """
    text = text.strip()
    if not re.match(r"1\d*\s?[^\d\s,]", text):
        return {0: text}
    marks, pos, k = [], 0, 1
    while m := re.compile(rf"(?:^|,)\s*{k}(?=\d*\s?[^\d\s,])(?!(?:st|nd|rd|th)\b)").search(text, pos):
        marks.append((k, m.start(), m.end()))
        pos, k = m.end(), k + 1
    ends = [s for _, s, _ in marks[1:]] + [len(text)]
    return {k: text[e:end].strip(" ,") for (k, _, e), end in zip(marks, ends)}


def parse_numbered(raw):
    names, _, affil_text = raw.strip().rstrip(".").partition(";")
    affils = split_numbered_affils(affil_text)
    authors = []
    for tok in re.split(r",\s*", names):
        tok = tok.strip()
        if (stray := re.fullmatch(r"[A-Z]?\s*(\d+)", tok)) and authors:  # "Name1, 2" or "C. Brown, E2"
            authors[-1]["nums"].append(int(stray.group(1)))
            continue
        if (sfx := re.fullmatch(r"(Jr|Sr|II|III|IV|3rd)\.?(\d*)", tok)) and authors:
            authors[-1]["name"] += ", " + sfx.group(1)
            authors[-1]["nums"] += [int(sfx.group(2))] if sfx.group(2) else []
            continue
        m = re.match(r"^(.*?)\s*(\d+)?$", tok)
        name = clean_name(re.sub(r"\s*\([^)]*\)", "", m.group(1)))  # "M. AlThgafi (Abedalthagafi)1"
        if name:
            # people are always "Initial(s). Surname" here, so anything else is a group
            group = not NUMBERED_RE.match(name) or bool(re.search(r"\bon behalf of\b", name, re.I))
            authors.append({"name": name, "nums": [int(m.group(2))] if m.group(2) else [], "group": group})
    for a in authors:
        nums = a.pop("nums") or ([0] if 0 in affils else [])
        a["affils"] = [seg for n in nums if n in affils for seg in split_top_level(affils[n])]
    return authors


def parse_authors(raw):
    """Return [{'name': str, 'affils': [segment, ...]}, ...] in listed order."""
    if not raw:
        return []
    if is_numbered(raw):
        return parse_numbered(raw)
    raw = repair_raw(raw)
    authors = []
    # "... Last Author (Affil) for the X Consortium" / "... (Affil) the Y Program"
    trailing = re.search(r"\)\s+([^,()]+)$", raw)
    if trailing and not re.match(r"(and|for|on behalf of|the)\b", trailing.group(1), re.I) \
            and not GROUP_RE.search(trailing.group(1)):
        trailing = None  # a person with a missing comma; split_top_level handles it
    if trailing:
        raw = raw[: trailing.start() + 1]
    for chunk in split_top_level(raw, split_after_paren=True):
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", chunk, re.S)
        if m and not m.group(1) and authors and not authors[-1]["affils"]:
            authors[-1]["affils"] = split_top_level(m.group(2))  # "Name, (Affil)"
        elif m:
            authors.append({"name": clean_name(m.group(1)), "affils": split_top_level(m.group(2))})
        else:
            authors.append({"name": clean_name(chunk), "affils": []})
    if trailing:
        authors.append({"name": trailing.group(1).strip(), "affils": [], "group": True})
    return authors


GROUP_RE = re.compile(
    r"\b(consorti\w*|group|network|collaborat\w*|initiative|investigators|program(me)?|project"
    r"|study|team|biobank|alliance|registry)\b", re.I)


def clean_name(name):
    name = re.sub(r"\s+(for|on behalf of)\s+the\s+.*$", "", name.strip(), flags=re.I)
    name = re.sub(r"^and\s+", "", name)
    return re.sub(r"(?<=[a-z])\d[\d,\s]*$", "", name).strip()  # superscripts: "Xiang Shu2"


def is_group(author):
    name, affils = author["name"], author["affils"]
    # "NHLBI Trans-Omics for Precision Medicine (TOPMed)": the parenthetical is an acronym
    acronym_only = len(affils) == 1 and " " not in affils[0] and len(name.split()) >= 4
    return author.get("group", False) or acronym_only or bool(
        GROUP_RE.search(name) or re.match(r"on behalf of\b", name, re.I))


def senior_author(authors):
    """Last-listed author who is a person rather than a consortium/group."""
    for a in reversed(authors):
        if not is_group(a):
            return a
    return None


def first_author(authors):
    """First-listed author who is a person rather than a consortium/group."""
    return senior_author(authors[::-1])


def load_overrides(path="overrides.csv"):
    """Hand fixes for author lists too irregular to parse: {abstract_id: {role: author}}."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            author = {"name": r["name"], "affils": [a.strip() for a in r["affils"].split(";") if a.strip()]}
            out.setdefault(r["abstract_id"], {})[r["role"]] = author
    return out


def key_authors(abstract_id, authors, overrides):
    """(first, senior) author for an abstract, applying any hand overrides."""
    o = overrides.get(abstract_id, {})
    return o.get("first", first_author(authors)), o.get("senior", senior_author(authors))


def is_plausible_name(name):
    return bool(name) and len(name.split()) <= 5 and not re.search(r"[\d()]", name)
