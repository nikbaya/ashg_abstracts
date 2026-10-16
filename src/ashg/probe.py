"""Print a compact layout summary of an abstract booklet PDF.

Output is deliberately small: font-size histogram, lines that repeat across
pages (header/footer candidates), and a truncated span list for a few pages.

Usage: python src/ashg/probe.py PDF [--pages 1,2,50] [--sample 40] [--width 70]
"""

import argparse
import collections

import fitz


def iter_spans(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    yield span, text


def font_histogram(doc, page_idxs):
    hist = collections.Counter()
    for i in page_idxs:
        for span, text in iter_spans(doc[i]):
            bold = "B" if span["flags"] & 16 else " "
            italic = "I" if span["flags"] & 2 else " "
            sup = "^" if span["flags"] & 1 else " "
            hist[(round(span["size"], 1), bold + italic + sup, span["font"])] += len(text)
    return hist


def repeated_lines(doc, page_idxs, min_frac=0.3):
    counts = collections.Counter()
    for i in page_idxs:
        seen = set()
        for span, text in iter_spans(doc[i]):
            key = "".join("#" if c.isdigit() else c for c in text)[:60]
            seen.add(key)
        counts.update(seen)
    n = len(page_idxs)
    return [(k, c) for k, c in counts.most_common(15) if c >= min_frac * n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1,2,3", help="1-based pages to list spans for")
    ap.add_argument("--sample", type=int, default=40, help="pages sampled for stats")
    ap.add_argument("--width", type=int, default=70)
    ap.add_argument("--max-spans", type=int, default=60)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    n = doc.page_count
    step = max(1, n // args.sample)
    sample = list(range(0, n, step))[: args.sample]
    print(f"{args.pdf}: {n} pages, page size {doc[0].rect.width:.0f}x{doc[0].rect.height:.0f}")

    print("\n== font histogram (size, flags B/I/^, font) -> chars, sampled pages ==")
    for (size, flags, font), chars in font_histogram(doc, sample).most_common(15):
        print(f"  {size:5.1f} [{flags}] {font:30s} {chars}")

    print("\n== lines repeated on >=30% of sampled pages (digits -> #) ==")
    for text, c in repeated_lines(doc, sample):
        print(f"  {c:3d}/{len(sample)}  {text!r}")

    for p in [int(x) for x in args.pages.split(",")]:
        if not 1 <= p <= n:
            continue
        print(f"\n== page {p} spans (y, x, size, flags, text) ==")
        for k, (span, text) in enumerate(iter_spans(doc[p - 1])):
            if k >= args.max_spans:
                print("  ...")
                break
            x0, y0 = span["bbox"][:2]
            print(f"  {y0:6.1f} {x0:6.1f} {span['size']:5.1f} {span['flags']:3d} {text[: args.width]!r}")


if __name__ == "__main__":
    main()
