"""Pass 1: dump every text span of a booklet PDF to Parquet, in parallel.

Each worker opens the PDF itself and handles a contiguous page range, so no
document is ever shared across processes. No abstract logic lives here; pass 2
(segment.py) stitches spans into abstracts in page order.

Usage: python src/ashg/extract_spans.py PDF OUT.parquet [--chunk 50] [--workers N]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import fitz
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("page", pa.int32()),  # 1-based
        ("block", pa.int32()),
        ("line", pa.int32()),
        ("span", pa.int32()),
        ("x0", pa.float32()),
        ("y0", pa.float32()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("size", pa.float32()),
        ("flags", pa.int32()),
        ("font", pa.string()),
        ("text", pa.string()),
    ]
)


def extract_range(pdf_path, start, stop):
    doc = fitz.open(pdf_path)
    cols = {name: [] for name in SCHEMA.names}
    for i in range(start, stop):
        blocks = doc[i].get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for b, block in enumerate(blocks):
            for l, line in enumerate(block.get("lines", [])):
                for s, span in enumerate(line["spans"]):
                    if not span["text"].strip():
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    row = dict(page=i + 1, block=b, line=l, span=s, x0=x0, y0=y0, x1=x1, y1=y1,
                               size=span["size"], flags=span["flags"], font=span["font"], text=span["text"])
                    for k, v in row.items():
                        cols[k].append(v)
    return pa.table(cols, schema=SCHEMA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    n = fitz.open(args.pdf).page_count
    ranges = [(s, min(s + args.chunk, n)) for s in range(0, n, args.chunk)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex, pq.ParquetWriter(args.out, SCHEMA) as writer:
        # map() yields in submission order, so the file stays in page order
        for table in ex.map(extract_range, [args.pdf] * len(ranges), *zip(*ranges)):
            writer.write_table(table)
    print(f"{args.pdf}: {n} pages -> {pq.ParquetFile(args.out).metadata.num_rows} spans -> {args.out}")


if __name__ == "__main__":
    main()
