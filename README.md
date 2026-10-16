# ashg_abstracts

Analysis of ASHG annual meeting abstract booklets (2023–2025): which institutions and senior authors present most often, and which phrases are over-represented in talks vs posters.

**Caveat:** the booklets list only *accepted* abstracts, and authors can request poster-only presentation. "Talk share" here means the share of accepted abstracts presented as a talk (platform or plenary). It is not an acceptance or success rate.

## Setup

```sh
mamba env create -f environment.yml
mamba activate ashg
```

Put the ASHG booklet PDFs in the repo root. They are copyrighted, so they are gitignored and not redistributed, and the same goes for per-abstract derived data under `data/`. Only code and aggregate results are committed.

## License

The code is MIT-licensed (see `LICENSE`). It depends on [PyMuPDF](https://github.com/pymupdf/PyMuPDF), which is licensed under AGPL-3.0 and is installed separately as a dependency, not bundled with this repo.
