# Follow-up memory

## Landscape author matching

- The requested `Po-Ru Loh` / `P-R. Loh` merge is supported by overlapping author-level affiliation evidence: Broad Institute, Harvard Medical School, and Brigham and Women's Hospital.
- The broader matching change in `src/ashg/landscape.py` also changes many unrelated author mappings. Comparing against the prior linker across available data found 36 occurrences across 16 names where both sides had affiliation evidence but none overlapped; many other changed mappings had insufficient evidence.
- Before regenerating landscape data or broadening this rule, narrow the matcher to handle hyphenated initials conservatively and add regression tests for ambiguity and affiliation conflicts. Do not assume all changed matches are valid.
- The current generated JSON files only have the `P-R. Loh` display label replaced; they do not reflect a full landscape rebuild with broader linker changes.
