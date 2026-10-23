"""Keywords with synonyms collapsed, for the Landscape page's paint-by-keyword.

Two sources of synonyms feed one grouping:

  The n-gram grouping from features.group_related (the one behind the phrase rows: "reporter assays", "parallel
  reporter" and "massively parallel" are one row): n-grams that share a content word and sit in mostly the same
  abstracts, or are an exact acronym of each other, are joined. It runs here over the BM25 terms that at least MIN_DF
  abstracts use.

  Abbreviations the authors define in place ("protein-protein interaction (PPI)"), which that grouping cannot see when
  the abbreviation shares no word with the phrase. A definition counts when the abbreviation's letters are the initials
  of the words before it. The abbreviation joins its long form's pieces (those used in mostly the same abstracts as the
  abbreviation or the long form), and the group is named "long form (ABBR)".

Two guards keep the definitions honest. An abbreviation only counts on its own when one long form accounts for at least
ABBR_SHARE of its definitions (IBD is both "inflammatory bowel disease" and "identity by descent", so it is matched by the
long form only), and a two-letter abbreviation also needs the long form in the same abstract (a bare "ES" is often
"ES cells").

An abstract's keywords are then its groups: the definition groups it uses, and the groups of its top BM25 terms.
"""


import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from features import group_related

STOP = {"of", "and", "the", "in", "for", "to", "a", "an", "on", "with"}
ABBR_SHARE = 0.8  # one long form must account for this share of an abbreviation's definitions
MIN_DEFS = 2      # an abbreviation needs this many definitions across the abstracts
MIN_DOCS = 2      # and a group this many abstracts that use it
MIN_DF = 5        # terms used by fewer abstracts are not grouped (nothing to connect)
JACCARD = 0.45    # as features.group_related: how much a long form's piece must overlap the abbreviation's abstracts
TERM = re.compile(r"[a-zA-Z][a-zA-Z0-9-]+")  # the BM25 tokenizer's words
PAREN = re.compile(r"\(\s*([A-Z][A-Za-z]{1,7})\s*\)")


def stem(word):
    """Plural stripped ("studies" -> "study", "variants" -> "variant"); only used to compare spellings."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def phrase_key(text):
    """Lowercase, hyphen- and possessive-insensitive, plurals merged: the form two spellings are compared on."""
    return " ".join(stem(w) for w in re.findall(r"[a-z0-9]+", text.lower().replace("'s", "")))


def initials_match(abbr, words):
    """Whether abbr's letters are the initials of words in order (stopwords may be skipped, none may be left over)."""
    a = abbr.lower()

    def go(ai, wi):
        if ai == len(a):
            return all(w.lower() in STOP for w in words[wi:])
        if wi == len(words):
            return False
        if words[wi][0].lower() == a[ai] and go(ai + 1, wi + 1):
            return True
        return words[wi].lower() in STOP and ai > 0 and go(ai, wi + 1)

    return go(0, 0)


def definitions(text):
    """(abbreviation, long form as written, lowercase) for each "long form (ABBR)" in the text."""
    out = []
    for m in PAREN.finditer(text):
        abbr = m.group(1)
        base = abbr[:-1] if abbr.endswith("s") and sum(c.isupper() for c in abbr) >= 2 else abbr
        if sum(c.isupper() for c in base) < 2:
            continue
        before = text[max(0, m.start() - 120):m.start()]
        parts = [(wm.start() + sm.start(), sm.group())  # words before "(", hyphenated words split into their parts
                 for wm in re.finditer(r"[A-Za-z][A-Za-z0-9'-]*", before)
                 for sm in re.finditer(r"[A-Za-z0-9']+", wm.group())]
        for k in range(len(base), min(len(base) + 4, len(parts)) + 1):
            if initials_match(base, [w for _, w in parts[-k:]]):
                out.append((base, before[parts[-k][0]:].strip().lower()))
                break
    return out


def mine(texts):
    """Definition groups: [{"label", "abbr", "written" long form, "docs": {abstract index: uses}}]."""
    defined = defaultdict(lambda: defaultdict(Counter))  # abbreviation (lowercase) -> long form key -> written form -> abstracts
    spelled = defaultdict(Counter)  # abbreviation (lowercase) -> its capitalizations ("LoF", "LOF")
    for t in texts:
        for abbr, form in set(definitions(t)):
            defined[abbr.lower()][phrase_key(form)][form] += 1
            spelled[abbr.lower()][abbr] += 1
    padded = [f" {phrase_key(t)} " for t in texts]
    groups = []
    for abbr, forms in defined.items():
        counts = {k: sum(c.values()) for k, c in forms.items()}
        total = sum(counts.values())
        if total < MIN_DEFS:
            continue
        key = max(counts, key=counts.get)
        alone = counts[key] / total >= ABBR_SHARE
        written = max(forms[key], key=forms[key].get)
        spellings = sorted(spelled[abbr], key=spelled[abbr].get, reverse=True)
        pat = re.compile(rf"\b(?:{'|'.join(map(re.escape, spellings))})s?\b")
        docs = {}
        for i, t in enumerate(texts):
            long = padded[i].count(f" {key} ")
            hits = len(pat.findall(t)) if alone and (long or (len(abbr) > 2 and any(sp in t for sp in spellings))) else 0
            if long + hits:
                docs[i] = long + hits
        if len(docs) >= MIN_DOCS:
            groups.append({"label": f"{written} ({spellings[0]})", "docs": docs, "abbr": abbr, "written": written})
    return groups


def pieces(written):
    """The BM25 terms a long form is made of: its words (stopwords dropped) and the pairs of neighbors."""
    words = [w for w in TERM.findall(written.lower()) if w not in ENGLISH_STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


class Keywords:
    """keywords.of(i): the groups of the abstract's top `limit` terms, best first, then any definition group it uses that
    those did not reach. keywords.synonyms maps each group's label to the other spellings in it ("; "-joined)."""

    def __init__(self, vec, vocab, texts, limit):
        vocab = np.asarray(vocab)
        X = (vec > 0).astype(np.int8).tocsc()
        df = np.asarray(X.sum(axis=0)).ravel()
        keep = np.flatnonzero(df >= MIN_DF)
        order = keep[np.lexsort((vocab[keep], -df[keep]))]  # most widespread first, so it heads its group
        ranked = pd.DataFrame({"ngram": vocab[order]})
        col = {t: i for i, t in enumerate(vocab)}

        self.defined = mine(list(texts))
        links, named = [], {}  # named: a term -> the label of the definition group it belongs to
        for g in self.defined:
            seen = set(g["docs"])
            members = [t for t in (g["abbr"].lower(), g["abbr"].lower() + "s") if t in col and df[col[t]] >= MIN_DF]
            for t in pieces(g["written"]):
                if t in col and df[col[t]] >= MIN_DF:
                    docs = set(X[:, col[t]].indices)
                    if len(docs & seen) >= JACCARD * len(docs | seen):
                        members.append(t)
            links += [(members[0], t) for t in members[1:]]
            named.update({t: g["label"] for t in members})

        grouped = group_related(ranked, X, vocab, links=links)
        size = {g["label"]: len(g["docs"]) for g in self.defined}
        self.label, self.synonyms, self.renamed = {}, {}, {}  # term -> group label; label -> other spellings; merged definition label -> group label
        for head, related in zip(grouped.ngram, grouped.related):
            members = [head] + (related.split("; ") if related else [])
            defs = {named[t] for t in members if t in named}
            label = max(defs, key=size.get) if defs else head  # a group that merged definitions is named after the most used
            self.label.update({t: label for t in members})
            self.renamed.update({d: label for d in defs})
            if len(members) > 1:
                self.synonyms[label] = "; ".join(t for t in members if t != label)

        self.by_doc = defaultdict(list)
        for g in self.defined:
            for i, n in g["docs"].items():
                self.by_doc[i].append((n, g["label"]))
        self.vec, self.vocab, self.limit = vec, vocab, limit

    def of(self, i):
        out, seen = [], set()
        row = self.vec.getrow(i)
        for j in np.argsort(-row.data):
            term = str(self.vocab[row.indices[j]])
            label = self.label.get(term, term)
            if label not in seen:
                seen.add(label)
                out.append(label)
                if len(out) >= self.limit:
                    break
        for _, label in sorted(self.by_doc[i], key=lambda p: -p[0]):
            label = self.renamed.get(label, label)
            if label not in seen:
                seen.add(label)
                out.append(label)
        return out
