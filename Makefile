# Rebuild one year end to end:  make YEAR=2025
# Then render the site from every year in outputs/:  make site
YEAR ?= 2025
TYPES := platform plenary poster
PY := PYTHONPATH=src/ashg mamba run -n ashg python
ABSTRACTS = $(foreach t,$(TYPES),data/abstracts/$(YEAR)_$(t).parquet)

.PHONY: year spans abstracts validate institutions people features analysis figures site test review
year: validate institutions people features analysis figures

spans:
	@mkdir -p data/spans
	@for t in $(TYPES); do cfg=configs/$(YEAR)_$$t.yaml; \
	  pdf=$$(grep '^pdf:' $$cfg | awk '{print $$2}'); \
	  $(PY) src/ashg/extract_spans.py $$pdf data/spans/$(YEAR)_$$t.parquet; done

abstracts: spans
	@mkdir -p data/abstracts
	@for t in $(TYPES); do $(PY) src/ashg/segment.py configs/$(YEAR)_$$t.yaml data/spans/$(YEAR)_$$t.parquet data/abstracts/$(YEAR)_$$t.parquet; done

validate: abstracts
	@for t in $(TYPES); do $(PY) src/ashg/validate.py configs/$(YEAR)_$$t.yaml data/spans/$(YEAR)_$$t.parquet data/abstracts/$(YEAR)_$$t.parquet --show 3 || exit 1; done

institutions:
	@mkdir -p data/institutions
	$(PY) src/ashg/institutions.py aliases.csv data/institutions/$(YEAR).parquet $(ABSTRACTS) --top-unmatched 0

people: institutions
	@mkdir -p data/people
	$(PY) src/ashg/people.py data/institutions/$(YEAR).parquet data/people/$(YEAR).parquet $(ABSTRACTS)

features:
	@mkdir -p data/features
	$(PY) src/ashg/features.py $(YEAR) data/features $(ABSTRACTS)

analysis: people features
	$(PY) src/ashg/analysis.py $(YEAR)

figures: analysis
	$(PY) src/ashg/figures.py $(YEAR)

site:
	$(PY) src/ashg/build_site.py

review:
	$(PY) src/ashg/review_sheet.py tests/fixtures/golden_$(YEAR)_review.md $(ABSTRACTS) --n 20

test:
	mamba run -n ashg python -m pytest -q tests
