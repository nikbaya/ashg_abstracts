"""Countries and world regions of first and senior authors.

Per author, the first rule that finds anything wins:
  1. location: affiliation segments that are a country, a US state or a Canadian province code
     ("Univ. of Copenhagen, Copenhagen, Denmark"; "Johns Hopkins Med., Baltimore, MD"). The
     2022-2024 booklets end every affiliation this way; 2025 affiliations rarely carry one.
  2. institute: the author's resolved institutes (data/institutions) via countries.csv.
  3. text: country, demonym and major-city names anywhere in the affiliation ("Seoul National
     University", "Peking Union Medical College"), tried in order and blanked once matched, so
     "New Mexico" is not also Mexico.
  4. manual: affiliation_countries.csv, hand-checked countries for whole affiliation strings that
     rules 1-3 leave unplaced (mostly companies and smaller institutes in 2025).

Usage: python src/ashg/geography.py COUNTRIES.csv INSTITUTIONS.parquet OUT.parquet ABSTRACTS.parquet [...]
"""

import argparse
import pathlib
import re

import pandas as pd

from authors import key_authors, load_overrides, parse_authors

# UN M49 regions, with the Americas split into its two M49 sub-regions (the US would swamp one bucket)
REGIONS = {
    "Northern America": ["United States", "Canada", "Bermuda"],
    "Latin America and the Caribbean": ["Argentina", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba", "Ecuador",
                                        "Honduras", "Mexico", "Peru", "Puerto Rico", "Turks and Caicos Islands",
                                        "Uruguay", "Venezuela"],
    "Europe": ["Austria", "Belgium", "Bulgaria", "Croatia", "Czech Republic", "Denmark", "Estonia", "Finland", "France",
               "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Netherlands",
               "Norway", "Poland", "Portugal", "Romania", "Russia", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
               "Switzerland", "Ukraine", "United Kingdom"],
    "Asia": ["Bahrain", "Bangladesh", "China", "Cyprus", "Hong Kong", "India", "Indonesia", "Iran", "Israel",
             "Japan", "Jordan", "Kazakhstan", "Kuwait", "Lebanon", "Malaysia", "Nepal", "Oman", "Pakistan",
             "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Taiwan",
             "Thailand", "Turkey", "United Arab Emirates", "Uzbekistan", "Vietnam"],
    "Africa": ["Algeria", "Cameroon", "DR Congo", "Egypt", "Ethiopia", "Gambia", "Ghana", "Kenya", "Malawi", "Mali",
               "Morocco", "Nigeria", "Rwanda", "Senegal", "South Africa", "Sudan", "Tanzania", "Tunisia", "Uganda",
               "Zambia", "Zimbabwe"],
    "Oceania": ["Australia", "New Zealand"],
}
REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs}

US_STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC "
                "ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
CA_PROVINCES = set("AB BC MB NB NL NS ON PE QC SK".split())
# segment spellings that differ from the canonical name ("Korea, Republic of" splits into two segments)
SEGMENT_NAMES = {"USA": "United States", "U.S.A": "United States", "US": "United States", "UK": "United Kingdom",
                 "Korea": "South Korea", "Republic of Korea": "South Korea", "Russian Federation": "Russia",
                 "PR": "Puerto Rico", "Palestinian Territory": "Palestine", "Viet Nam": "Vietnam",
                 "England": "United Kingdom", "Scotland": "United Kingdom", "Wales": "United Kingdom",
                 "Northern Ireland": "United Kingdom", "Turkiye": "Turkey", "Türkiye": "Turkey"}
# second halves of split names: "Iran, Islamic Republic of" -> the segment before names the country
SPLIT_TAILS = {"Republic of", "Islamic Republic of", "Democratic Republic of the", "United Republic of"}
SPLIT_HEADS = {"Iran": "Iran", "Congo": "DR Congo", "Tanzania": "Tanzania", "Korea": "South Korea"}

# rule 3, in order; ambiguous names (Cambridge, Birmingham, London ON, Victoria, Perth, Georgia...) are left out
# or listed under the reading that dominates in genetics affiliations
TEXT_PATTERNS = [
    ("Australia", r"New South Wales|Australian?|Sydney|Melbourne|Brisbane|Adelaide|Queensland|Monash"),
    ("United States", r"New Mexico|New England|New Jersey|New York|New Hampshire|U\.?S\.?A\b|United States"),
    ("Canada", r"British Columbia|Canad(a|ian)|Toronto|Montr[eé]al|Vancouver|Ottawa|Calgary|Edmonton|Qu[eé]bec"
               r"|Winnipeg|McMaster|Alberta|Ontario|Manitoba|Saskatchewan|Nova Scotia|Halifax|Dalhousie|Sherbrooke"),
    ("United Kingdom", r"United Kingdom|\bUK\b|England|Scotland|Wales|London|Oxford|Edinburgh|Glasgow|Manchester"
                       r"|Leeds|Liverpool|Sheffield|Exeter|Cardiff|Dundee|Leicester|Nottingham|Southampton|NHS\b"),
    ("South Korea", r"Korea|Seoul|Yonsei|Busan|Daejeon|KAIST|Ulsan|Incheon|Daegu|Gwangju|Suwon|Sungkyunkwan"),
    ("Japan", r"Japan|Tokyo|Osaka|Kyoto|Nagoya|Keio|Tohoku|Kyushu|Hokkaido|Kobe|Yokohama|Chiba|Niigata|Okayama"
              r"|Hiroshima|Juntendo|RIKEN"),
    ("Hong Kong", r"Hong Kong"),
    ("Taiwan", r"Taiwan|Taipei|Kaohsiung|Taichung|Tainan|Chang Gung|Academia Sinica"),
    ("China", r"China|Chinese|Beijing|Peking|Shanghai|Tsinghua|Fudan|Wuhan|Guangzhou|Shenzhen|Nanjing|Zhejiang"
              r"|Sichuan|Sun Yat-sen|Hangzhou|Chengdu|Xiamen|Tianjin|Shandong|Jiangsu|Hunan|Xi'an|Changsha|Nanchang"),
    ("India", r"\bIndia\b|Delhi|Mumbai|Bangalore|Bengaluru|Chennai|Hyderabad|Kolkata|AIIMS"),
    ("Pakistan", r"Pakistan|Karachi|Lahore|Islamabad"),
    ("Singapore", r"Singapore"),
    ("Israel", r"Israel|Tel Aviv|Jerusalem|Weizmann|Technion|Haifa"),
    ("Saudi Arabia", r"Saudi|Riyadh|Jeddah|KAUST"),
    ("Qatar", r"Qatar|Doha|Sidra|Hamad"),
    ("United Arab Emirates", r"Emirates|Abu Dhabi|Dubai"),
    ("Turkey", r"Turkey|Turkish|T[uü]rkiye|Istanbul|Ankara|Izmir"),
    ("Germany", r"German|Berlin|Munich|M[uü]nchen|Heidelberg|Hamburg|T[uü]bingen|Frankfurt|Cologne|K[oö]ln|Bonn"
                r"|Leipzig|Dresden|G[oö]ttingen|Freiburg|Charit[eé]|Max Planck|Helmholtz|Kiel|L[uü]beck|Mainz"),
    ("France", r"France|French|Paris|Lyon|Marseille|Bordeaux|Toulouse|Lille|Strasbourg|Nantes|INSERM|Sorbonne"),
    ("Italy", r"Italy|Italian|Milan|Rome|Roma\b|Naples|Napoli|Turin|Torino|Bologna|Florence|Padova|Padua|Genoa"
              r"|Pavia|Siena|IRCCS"),
    ("Spain", r"Spain|Spanish|Madrid|Barcelona|Valencia|Sevilla|Seville"),
    ("Netherlands", r"Netherlands|Dutch|Amsterdam|Rotterdam|Utrecht|Leiden|Groningen|Nijmegen|Erasmus|Maastricht"),
    ("Belgium", r"Belgium|Leuven|Brussels|Ghent|Gent\b|Antwerp"),
    ("Switzerland", r"Switzerland|Swiss|Z[uü]rich|Geneva|Gen[eè]ve|Basel|Lausanne|\bBern\b|ETH\b"),
    ("Sweden", r"Sweden|Swedish|Stockholm|Uppsala|Gothenburg|G[oö]teborg|\bLund\b|Karolinska|Ume[aå]"),
    ("Denmark", r"Denmark|Danish|Copenhagen|Aarhus|Odense"),
    ("Norway", r"Norway|Norwegian|Oslo|Bergen|Trondheim"),
    ("Finland", r"Finland|Finnish|Helsinki|Turku|Oulu|Tampere"),
    ("Iceland", r"Iceland|Reykjav[ií]k|deCODE"),
    ("Estonia", r"Estonia|Tartu|Tallinn"),
    ("Ireland", r"Ireland|Irish|Dublin"),
    ("Austria", r"Austria|Vienna|\bWien\b|Innsbruck|Graz"),
    ("Poland", r"Poland|Polish|Warsaw|Krak[oó]w|Gda[nń]sk|Pozna[nń]|Wroc[lł]aw"),
    ("Czech Republic", r"Czech|Prague|Brno"),
    ("Portugal", r"Portugal|Lisbon|Porto\b"),
    ("Greece", r"Greece|Greek|Athens\b(?!,? GA)|Thessaloniki"),
    ("Russia", r"Russia|Moscow|St\.? Petersburg|Novosibirsk"),
    ("Brazil", r"Brazil|S[aã]o Paulo|Rio de Janeiro|Porto Alegre|Belo Horizonte"),
    ("Mexico", r"Mexico|Mexican|UNAM\b|Monterrey|Guadalajara"),
    ("Colombia", r"Colombia|Bogot[aá]|Medell[ií]n"),
    ("Argentina", r"Argentin|Buenos Aires"),
    ("Chile", r"\bChile\b"),
    ("Peru", r"\bPeru\b"),
    ("South Africa", r"South Africa|Cape Town|Johannesburg|Witwatersrand|Stellenbosch|Pretoria|Durban"),
    ("Nigeria", r"Nigeria|Lagos|Ibadan"),
    ("Ghana", r"Ghana|Accra"),
    ("Kenya", r"Kenya|Nairobi"),
    ("Uganda", r"Uganda|Kampala|Makerere"),
    ("Mali", r"\bMali\b|Bamako"),
    ("Egypt", r"Egypt|Cairo"),
    ("Tunisia", r"Tunisia|Tunis\b"),
    ("Morocco", r"Morocc|Rabat|Casablanca"),
    ("New Zealand", r"New Zealand|Auckland|Otago"),
    ("Thailand", r"Thailand|Bangkok|Mahidol"),
    ("Iran", r"\bIran|Tehran"),
    ("United States", r"\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia"
                      r"|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts"
                      r"|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|Carolina|Dakota|Ohio"
                      r"|Oklahoma|Oregon|Pennsylvania|Rhode Island|Tennessee|Texas|Utah|Vermont|Virginia|Washington"
                      r"|Wisconsin|Wyoming|Boston|Chicago|Philadelphia|Houston|Seattle|Baltimore|Pittsburgh|Atlanta"
                      r"|Los Angeles|San Francisco|San Diego|St\.? Louis|Nashville|Denver|Miami|Dallas|Cleveland"
                      r"|Cincinnati|Detroit|Minneapolis|Salt Lake|Phoenix|Bethesda|Rochester|Stanford|Yale|Harvard"
                      r"|Johns Hopkins|Mayo Clinic|Kaiser Permanente|Veterans Affairs)\b"),
]
TEXT_RES = [(c, re.compile(p)) for c, p in TEXT_PATTERNS]


def segment_countries(affils):
    """Rule 1: countries named by location segments."""
    segs = [s.strip().rstrip(".").strip() for a in affils for s in a.split(";")]
    found = []
    for i, s in enumerate(segs):
        if s in SPLIT_TAILS and i and segs[i - 1] in SPLIT_HEADS:
            c = SPLIT_HEADS[segs[i - 1]]
        elif s in US_STATES:
            c = "United States"
        elif s in CA_PROVINCES:
            c = "Canada"
        else:
            c = SEGMENT_NAMES.get(s, s if s in REGION_OF else None)
        if c and c not in found:
            found.append(c)
    return found


def text_countries(affils):
    """Rule 3: country, demonym and city names in the affiliation text."""
    text, found = ", ".join(affils), []
    for country, pat in TEXT_RES:
        if pat.search(text):
            text = pat.sub(lambda m: " " * len(m.group(0)), text)
            if country not in found:
                found.append(country)
    return found


def author_countries(author, insts, inst_country, manual=None):
    """(countries, rule) for one author; rule is location / institute / text / manual / None."""
    affils = author["affils"] if author else []
    if not affils:
        return [], None
    if found := segment_countries(affils):
        return found, "location"
    if found := list(dict.fromkeys(inst_country[i] for i in insts if inst_country.get(i))):
        return found, "institute"
    if found := text_countries(affils):
        return found, "text"
    if found := (manual or {}).get(", ".join(affils)):
        return found, "manual"
    return [], None


def load_inst_countries(path):
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    unknown = set(df.country[df.country != ""]) - set(REGION_OF)
    if unknown:
        raise ValueError(f"countries in {path} missing from geography.REGIONS: {sorted(unknown)}")
    return dict(zip(df.institute, df.country))


def load_manual(path):
    """Rule 4: hand-checked countries for whole affiliation strings no other rule places
    (affiliation,country,basis; several countries joined with "; ")."""
    if not pathlib.Path(path).exists():
        return {}
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    df = df[df.country != ""]
    manual = {a: c.split("; ") for a, c in zip(df.affiliation, df.country)}
    unknown = {c for cs in manual.values() for c in cs} - set(REGION_OF)
    if unknown:
        raise ValueError(f"countries in {path} missing from geography.REGIONS: {sorted(unknown)}")
    return manual


def resolve(df, insts, inst_country, overrides, manual=None):
    """One row per abstract, same author choice as institutions.py (no senior author if cut off)."""
    by_id = insts.set_index("abstract_id")
    rows = []
    for r in df.itertuples(index=False):
        first, senior = key_authors(r.abstract_id, parse_authors(r.authors_raw), overrides)
        if r.truncated:
            senior = None
            if first and "(" in first["name"]:
                first = None
        row = {"abstract_id": r.abstract_id, "year": r.year, "pres_type": r.pres_type, "truncated": r.truncated}
        for role, author in (("first", first), ("senior", senior)):
            countries, rule = author_countries(author, by_id.at[r.abstract_id, f"{role}_insts"], inst_country, manual)
            row[f"{role}_countries"], row[f"{role}_rule"] = countries, rule
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("countries")
    ap.add_argument("institutions")
    ap.add_argument("out")
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--overrides", default="overrides.csv")
    ap.add_argument("--manual", default="affiliation_countries.csv")
    args = ap.parse_args()
    inst_country = load_inst_countries(args.countries)
    insts = pd.read_parquet(args.institutions)
    missing = set(insts.senior_insts.explode().dropna()) | set(insts.first_insts.explode().dropna())
    missing -= set(inst_country)
    if missing:
        raise ValueError(f"institutes missing from {args.countries} (blank if multinational): {sorted(missing)}")
    overrides = load_overrides(args.overrides)
    manual = load_manual(args.manual)
    out = pd.concat([resolve(pd.read_parquet(p), insts, inst_country, overrides, manual) for p in args.abstracts],
                    ignore_index=True)
    out.to_parquet(args.out, index=False)
    s = out[~out.truncated]
    print(f"{len(out)} abstracts -> {args.out}; senior-author country found for "
          f"{(s.senior_countries.str.len() > 0).mean():.1%} of complete abstracts, by rule:")
    print(s.senior_rule.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
