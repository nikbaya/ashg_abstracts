# ashg_abstracts

**Website:** https://nikbaya.github.io/ashg_abstracts/

Analysis of ASHG annual meeting abstract booklets (2023–2025): which institutions and senior authors present most often, and which phrases are over-represented in talks vs posters, and how topics and wording change over the years.

**Caveat:** the booklets list only *accepted* abstracts, and authors can request poster-only presentation. "Talk rate" here means the fraction of accepted abstracts presented as a talk (platform or plenary). It is not an acceptance or success rate.

## Setup

```sh
mamba env create -f environment.yml
mamba activate ashg
```

Put the ASHG booklet PDFs in the repo root. They are copyrighted, so they are gitignored and not redistributed, and the same goes for per-abstract derived data under `data/`. Only code and aggregate results are committed.

## License

The code is MIT-licensed (see `LICENSE`). It depends on [PyMuPDF](https://github.com/pymupdf/PyMuPDF), which is licensed under AGPL-3.0 and is installed separately as a dependency, not bundled with this repo.

## Site

`make site` renders the GitHub Pages site into `docs/` (one page per topic, a year toggle on each; everything pooled across years, the cross-year trends and a phrase lookup on the overview) from the aggregate CSVs in `outputs/`. Run `make pooled trends` after building every year so those exist. To publish, set the repo's **Settings → Pages** source to *Deploy from a branch*, `main`, `/docs`. The site leaves out any institution, metro area, country or person with no talks; the CSVs in `outputs/` keep them.
