# Technical Handover: ASHG Abstract Analysis Pipeline

## Project Objective
Parse multi-year ASHG conference abstract booklets (4000+ page PDFs per year) to extract, structure, and analyze abstract metadata and text. The goal is to generate high-engagement statistics regarding institutional success rates, senior author track records, and textual predictors of platform vs. poster presentation selections.

## Target Tech Stack
*   **PDF Parsing:** `PyMuPDF` (`fitz`) - *Strict requirement: Do not use `pdfplumber` due to memory/speed constraints on 4000-page documents.*
*   **Concurrency:** Python `concurrent.futures.ProcessPoolExecutor`
*   **Text/NLP:** `scikit-learn` (`TfidfVectorizer`, `CountVectorizer`), `spaCy` (for targeted regex/feature extraction only, not bulk processing).
*   **Data Storage & Querying:** Parquet format + `DuckDB`.

---

## Phase 1: Scalable PDF Extraction
**Goal:** Convert massive PDFs into structured chunks without blowing up RAM.
1.  **Chunking:** Use multiprocessing to split the PDF into 50-page chunks.
2.  **Parsing:** Use `PyMuPDF` to extract text. Rely on structural regex (font sizes, capitalization, session header boilerplate, abstract numbers like `101. `) to delineate abstract boundaries.
3.  **Extraction Targets:** For each abstract, isolate:
    *   Title
    *   Authors (isolate Senior Author: last name before affiliation block)
    *   Affiliations
    *   Abstract Body
    *   Presentation Type (Platform/Plenary vs. Poster) based on session headers.
4.  **Storage:** Append parsed results immediately to chunked `.jsonl` files or a local SQLite/DuckDB database. Do not hold the full document in memory.

---

## Phase 2: Entity Resolution (Solving the Affiliation Trap)
**Goal:** Deduplicate messy institutional names (e.g., "Broad Inst" vs "Broad Institute") across 15,000+ records.
*   **Constraint:** Do *not* use pairwise fuzzy matching (e.g., `TheFuzz` or `FuzzyWuzzy`), as $O(N^2)$ comparisons will bottleneck the pipeline.
*   **Solution:** 
    1. Clean strings (lowercase, strip standard terms like "University", "Dept").
    2. Vectorize using `TfidfVectorizer`.
    3. Cluster using `DBSCAN` or cosine similarity matrices to assign a unified `Institute_ID` to variations of the same institution.

---

## Phase 3: Text Feature Engineering
**Goal:** Extract sample sizes, verb tenses, and key n-grams from the abstract body text.
1.  **Sample Sizes:** Run targeted regex for patterns like $N = [0-9,]+$ or $n > [0-9]+$.
2.  **N-Grams (Buzzwords):** Use `scikit-learn`'s `CountVectorizer` with `ngram_range=(1, 3)` and `stop_words='english'` on the abstract bodies to build a matrix of phrase counts.

---

## Phase 4: Analysis & Querying (DuckDB)
**Goal:** Run performant aggregations locally on Parquet files.

**Query 1: Institutional & Author Leaderboards**
*   Calculate the total volume of submissions and the conversion rate: $\frac{\text{Platform Talks}}{\text{Total Submissions}}$ per Senior Author and per Clustered Institution.
*   Filter for top 50 by volume to avoid low-$N$ noise.

**Query 2: The "Platform Premium" (Smoothed Log-Odds Ratio)**
*   Calculate the relative overrepresentation of specific n-grams in Platforms vs. Posters. 
*   Filter out low-frequency phrases (e.g., must appear in at least 5 Platform abstracts per year).
*   Calculate the Odds Ratio using Laplace smoothing:

$$ \text{Platform Rate} = \frac{C_{platform} + 1}{N_{platform}} $$

$$ \text{Poster Rate} = \frac{C_{poster} + 1}{N_{poster}} $$

$$ \text{Odds Ratio} = \frac{\text{Platform Rate}}{\text{Poster Rate}} $$

*Where $C$ is the count of the specific n-gram, and $N$ is the total word count in that category's corpus.*

Sort descending by the resulting Odds Ratio to find the year's top "buzzwords."
## Open questions (later)
*   Are typo rates (`writing.py`) higher for senior authors from non-English-speaking countries? Join `data/geography` senior countries to the per-abstract typo counts; control for abstract length.
*   Is talk rate higher for abstracts written in US spelling than in UK spelling (color/colour, -ize/-ise, analyze/analyse, tumor/tumour)? Classify abstracts by spelling variant and compare talk rate, within country group, since spelling tracks author origin.
