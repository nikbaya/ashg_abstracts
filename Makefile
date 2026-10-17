# Rebuild one year end to end:  make YEAR=2025
# Cross-year steps (after every year is built):        make pooled trends writing
# Then render the site from every year in outputs/:  make site
YEAR ?= 2025
TYPES := platform plenary poster
PY := PYTHONPATH=src/ashg mamba run -n ashg python
ABSTRACTS = $(foreach t,$(TYPES),data/abstracts/$(YEAR)_$(t).parquet)

.PHONY: year spans abstracts validate institutions people geography features analysis figures pooled trends writing site test review
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

trends: pooled
	$(PY) src/ashg/trends.py

writing:
	$(PY) src/ashg/writing.py data/abstracts/*.parquet

site:
	$(PY) src/ashg/build_site.py

review:
	$(PY) src/ashg/review_sheet.py tests/fixtures/golden_$(YEAR)_review.md $(ABSTRACTS) --n 20

test:
	mamba run -n ashg python -m pytest -q tests
