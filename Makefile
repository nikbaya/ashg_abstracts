# Rebuild one year end to end:  make YEAR=2025
# Cross-year steps (after every year is built):        make pooled trends manhattan score writing
# Then render the site from every year in outputs/:  make site
YEAR ?= 2025
TYPES := platform plenary poster
PY := PYTHONPATH=src/ashg mamba run -n ashg python
ABSTRACTS = $(foreach t,$(TYPES),data/abstracts/$(YEAR)_$(t).parquet)

.PHONY: year spans abstracts validate institutions people geography features analysis figures pooled trends center naturalearth writing site test review genedata manhattan score preprints
year: validate institutions people geography features analysis figures

spans:
	@mkdir -p data/spans
	@for t in $(TYPES); do cfg=configs/$(YEAR)_$$t.yaml; \
	  pdf=$$(grep '^pdf:' $$cfg | awk '{print $$2}'); \
	  $(PY) src/ashg/extract_spans.py $$pdf data/spans/$(YEAR)_$$t.parquet; done

# 2026 has no booklet: abstracts come from the online program via data/manual/ (see src/ashg/floq.py)
ifeq ($(YEAR),2026)
abstracts:
	@mkdir -p data/abstracts
	@for t in $(TYPES); do $(PY) src/ashg/floq.py abstracts data/manual/$(YEAR)_$$t.yaml $(YEAR) $$t data/abstracts/$(YEAR)_$$t.parquet; done

validate: abstracts
else
abstracts: spans
	@mkdir -p data/abstracts
	@for t in $(TYPES); do $(PY) src/ashg/segment.py configs/$(YEAR)_$$t.yaml data/spans/$(YEAR)_$$t.parquet data/abstracts/$(YEAR)_$$t.parquet; done

validate: abstracts
	@for t in $(TYPES); do $(PY) src/ashg/validate.py configs/$(YEAR)_$$t.yaml data/spans/$(YEAR)_$$t.parquet data/abstracts/$(YEAR)_$$t.parquet --show 3 || exit 1; done
endif

institutions:
	@mkdir -p data/institutions
	$(PY) src/ashg/institutions.py aliases.csv data/institutions/$(YEAR).parquet $(ABSTRACTS) --top-unmatched 0

people: institutions
	@mkdir -p data/people
	$(PY) src/ashg/people.py data/institutions/$(YEAR).parquet data/people/$(YEAR).parquet $(ABSTRACTS)

geography: institutions
	@mkdir -p data/geography
	$(PY) src/ashg/geography.py countries.csv data/institutions/$(YEAR).parquet data/geography/$(YEAR).parquet $(ABSTRACTS)

features:
	@mkdir -p data/features
	$(PY) src/ashg/features.py $(YEAR) data/features $(ABSTRACTS)

analysis: people geography features
	$(PY) src/ashg/analysis.py $(YEAR)

figures: analysis
	$(PY) src/ashg/figures.py $(YEAR)

# every year pooled: outputs/all_*.csv/.png; people matched across years on initials
pooled:
	$(PY) src/ashg/features.py all data/features data/abstracts/*.parquet
	$(PY) src/ashg/people.py --initials data/institutions data/people/all.parquet data/abstracts/*.parquet
	$(PY) src/ashg/analysis.py all
	$(PY) src/ashg/figures.py all

trends: pooled center
	$(PY) src/ashg/trends.py

# center of mass of first authors per year (maps; Natural Earth basemaps, public domain)
NE_URL := https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
NE_FILES := ne_110m_admin_0_countries ne_50m_admin_0_countries ne_50m_admin_1_states_provinces_lakes ne_50m_lakes
naturalearth:
	@mkdir -p data/raw/naturalearth
	@for f in $(NE_FILES); do test -s data/raw/naturalearth/$$f.geojson || \
	  curl -sSfL -o data/raw/naturalearth/$$f.geojson $(NE_URL)/$$f.geojson || exit 1; done

center: naturalearth
	$(PY) src/ashg/center.py

# gene and word Manhattan plots (HGNC symbols + UCSC hg38 refGene positions, both public)
genedata:
	@mkdir -p data/raw/genes
	@test -s data/raw/genes/hgnc_complete_set.txt || curl -sSfL -o data/raw/genes/hgnc_complete_set.txt \
	  https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
	@test -s data/raw/genes/refGene.txt.gz || curl -sSfL -o data/raw/genes/refGene.txt.gz \
	  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz

manhattan: genedata
	$(PY) src/ashg/manhattan.py

# overrepresented-phrase score per abstract, for the site's score page (after `make pooled`)
score:
	$(PY) src/ashg/talk_score.py

# which abstracts were already preprinted (analysis only, not on the site); `search` takes ~40 min, `analyze` reuses it
preprints:
	$(PY) src/ashg/preprints.py analyze --year 2026

writing:
	$(PY) src/ashg/writing.py data/abstracts/*.parquet

site:
	$(PY) src/ashg/build_site.py

review:
	$(PY) src/ashg/review_sheet.py data/review/golden_$(YEAR)_review.md $(ABSTRACTS) --n 20

test:
	mamba run -n ashg python -m pytest -q tests
