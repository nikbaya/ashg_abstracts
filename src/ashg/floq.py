"""Pull ASHG abstracts from the online program (ashg.floq.live) when no PDF booklet exists.

The web app reads from a public AppSync GraphQL API using a key shipped in its JS bundle
(/static/js/main.*.js; looked up at run time by api_config). Talks are "timeslots" whose `info`
HTML holds the abstract, then "<b>Author Block</b>" and "Name (Affiliation), ...".

  fetch:     one request per session (~270 for 2026), one at a time -> one raw JSON dump
  build:     raw dump -> data/manual/<year>_<type>.yaml, one file per presentation type
  abstracts: those YAMLs -> data/abstracts/<year>_<type>.parquet, the table segment.py makes
             from a booklet. Lightning talks (new in 2026) are left out: every year-pooled step
             globs data/abstracts/ and counts anything not a poster as a talk.

The API sits behind AWS WAF: on 2026-10-17 a burst of ~270 requests 8-at-a-time got the
client IP 403'd (WAFForbiddenException), so fetch now goes sequentially with a pause.
"""
import argparse
import html
import json
import re
import time
import urllib.request

import pandas as pd
import yaml

APP = "https://ashg.floq.live"
SESSION_TYPE = "6a2a8b9eb1b77720033d5a6e"  # ASHG 2026 "session" timeslot type
FIELDS = "id name info start end parent deleted configuratorType classifications"
Q_SESSIONS = f"""query($type: String!, $limit: Int, $nextToken: String) {{
  getTimeslotsWithType(type: $type, limit: $limit, nextToken: $nextToken) {{ items {{ {FIELDS} }} nextToken }} }}"""
Q_CHILDREN = f"""query($parent: String!, $limit: Int, $nextToken: String) {{
  getChildrenOf(parent: $parent, limit: $limit, nextToken: $nextToken) {{ items {{ {FIELDS} }} nextToken }} }}"""
PAUSE = 1.0  # seconds between requests


def api_config():
    """(GraphQL endpoint, API key) read from the web app's JS bundle, as a browser gets them."""
    page = urllib.request.urlopen(f"{APP}/event/ashg-2026/abstracts", timeout=60).read().decode()
    bundle = urllib.request.urlopen(APP + re.search(r'src="(/static/js/main\.[^"]+\.js)"', page).group(1),
                                    timeout=120).read().decode()
    url, key = (re.search(rf'{name}:"([^"]+)"', bundle).group(1)
                for name in ("REACT_APP_AWS_APPSYNC_GRAPHQL_ENDPOINT", "REACT_APP_AWS_APPSYNC_API_KEY"))
    return url, key


API = None  # (endpoint, key), looked up on the first request


def gql(query, variables):
    global API
    API = API or api_config()
    time.sleep(PAUSE)
    req = urllib.request.Request(API[0], data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"x-api-key": API[1], "content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    if r.get("errors"):
        raise RuntimeError(r["errors"])
    return r["data"]


def paged(query, key, **variables):
    items, tok = [], None
    while True:
        d = gql(query, {**variables, "limit": 1000, "nextToken": tok})[key]
        items += d["items"]
        tok = d.get("nextToken")
        if not tok:
            return items


def fetch(out):
    sessions = [s for s in paged(Q_SESSIONS, "getTimeslotsWithType", type=SESSION_TYPE) if not s.get("deleted")]
    children = [paged(Q_CHILDREN, "getChildrenOf", parent=s["id"]) for s in sessions]
    json.dump({"sessions": sessions, "children": children}, open(out, "w"))
    print(f"{len(sessions)} sessions, {sum(map(len, children))} timeslots -> {out}")


def text(fragment):
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    paras = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in re.split(r"</p>\s*<p>|</?p>", fragment)]
    return "\n\n".join(re.sub(r"[ \t]+", " ", p) for p in paras if p)


def split_authors(raw):
    """'A (X), B (Y, Z)' -> [('A', 'X'), ('B', 'Y, Z')], splitting on top-level commas and newlines.
    Author blocks are free text typed by submitters (unbalanced parentheses, footnotes), so this is
    best-effort; authors_raw keeps the original."""
    out, buf, depth = [], "", 0
    for ch in raw + ",":
        if ch in ",\n" and depth == 0:
            if buf.strip():
                m = re.match(r"(.*?)\s*\((.*)\)$", buf.strip())
                out.append((m.group(1), m.group(2)) if m else (buf.strip(), None))
            buf = ""
            continue
        depth += (ch == "(") - (ch == ")")
        buf += ch
    return out


def session_type(session, n_talks):
    """Every 2026 session with abstracts is labelled in the program as a poster session (named
    "... Poster <Day> Session") or Plenary Sessions, Lightning Talks or Platform Sessions. The six
    Lightning Talks sessions are the only non-poster ones with 15 talks (platform sessions have 4-6)."""
    if "Poster" in session["name"]:
        return "poster"
    if "Plenary Abstract Session" in session["name"]:
        return "plenary"
    return "lightning" if n_talks > 10 else "platform"


def record(t, session_name):
    body_html, _, author_html = t["info"].partition("<b>Author Block</b>")
    # posters append "Poster Board Number\nBoard 8041"; the board is already in the title
    authors_raw = text(author_html).lstrip(": \n").split("Poster Board Number")[0].strip()
    pairs = split_authors(authors_raw)
    board, title = re.match(r"(?:Board (\S+):\s*)?(.*)", t["name"].strip(), re.S).groups()
    rec = {"title": title + "\n", "session": session_name, "floq_id": t["id"]}
    if board:
        rec["board"] = board
    rec["authors"] = [n for n, _ in pairs]
    rec["affiliations"] = list(dict.fromkeys(a for _, a in pairs if a))
    rec["authors_raw"] = authors_raw + "\n"
    rec["body"] = text(body_html) + "\n"
    return rec


class Dumper(yaml.SafeDumper):
    pass


Dumper.add_representer(str, lambda d, s: d.represent_scalar(
    "tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))

HEADER = """# ASHG {year} {pres_type} abstracts, from the public online program (ashg.floq.live).
# Built by src/ashg/floq.py from data/raw/; edit freely, but a rebuild overwrites edits.
# Lives under data/ (gitignored) because abstract text is copyrighted.
# authors_raw is the program's author block verbatim: "Name (Affiliation), ...".

"""


def build(raw_path, year, outdir):
    raw = json.load(open(raw_path))
    by_type = {}
    for s, kids in sorted(zip(raw["sessions"], raw["children"]), key=lambda sk: (sk[0]["start"] or "", sk[0]["name"])):
        talks = [t for t in kids if t["configuratorType"] == "timeslot:presentation" and not t.get("deleted")
                 and "Author Block" in (t["info"] or "")]
        if not talks:
            continue
        ptype = session_type(s, len(talks))
        by_type.setdefault(ptype, []).extend(record(t, s["name"]) for t in sorted(talks, key=lambda t: (t["start"], t["name"])))
    for ptype, recs in by_type.items():
        out = f"{outdir}/{year}_{ptype}.yaml"
        with open(out, "w") as f:
            f.write(HEADER.format(year=year, pres_type=ptype))
            yaml.dump({"abstracts": recs}, f, Dumper=Dumper, sort_keys=False, allow_unicode=True, width=10_000, indent=2)
        print(f"{ptype:>10}: {len(recs):>5} abstracts, {len({r['session'] for r in recs}):>3} sessions -> {out}")


def to_abstracts(manual, year, pres_type, out):
    recs = yaml.safe_load(open(manual))["abstracts"]
    ids = [r.get("board") or f"{i + 1:04d}" for i, r in enumerate(recs)]  # posters: board number
    if len(set(ids)) < len(ids):
        raise ValueError(f"{manual}: duplicate abstract ids")
    df = pd.DataFrame({
        "abstract_id": [f"{year}-{pres_type}-{i}" for i in ids],
        "year": year, "pres_type": pres_type,
        "session": [r.get("session") for r in recs],
        "title": [r["title"].strip() for r in recs],
        "authors_raw": [(r.get("authors_raw") or ", ".join(r.get("authors") or [])).strip() for r in recs],
        "body": [r["body"].strip() for r in recs],
        "truncated": False,  # the program has full text
    })
    df.to_parquet(out, index=False)
    print(f"{manual}: {len(df)} abstracts -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("out")
    b = sub.add_parser("build")
    b.add_argument("raw")
    b.add_argument("year", type=int)
    b.add_argument("outdir")
    a = sub.add_parser("abstracts")
    a.add_argument("manual")
    a.add_argument("year", type=int)
    a.add_argument("pres_type")
    a.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.out)
    elif args.cmd == "build":
        build(args.raw, args.year, args.outdir)
    else:
        to_abstracts(args.manual, args.year, args.pres_type, args.out)


if __name__ == "__main__":
    main()
