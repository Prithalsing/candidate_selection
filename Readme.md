# Candidate Selection — Redrob Hackathon

Intelligent Candidate Discovery & Ranking for the **Senior AI Engineer** JD.
Ranks the top 100 candidates from a 100,000-candidate pool.

## Approach

A **gate-then-score hybrid**, built up one technique at a time:

```
Stage 0  HARD EXCLUSION   honeypot  -> score = 0
Stage 1  TITLE GATE       x title tier (1.0 ... 0.05)   (traps can't enter)
Stage 2  QUALITY SCORE    blend of structural rules + BM25 lexical + assessment
Stage 3  ENGAGEMENT MULT  x 0.50 ... 1.00 (applied last)

final = title_gate x quality x engagement   (0 if honeypot)
```

Three paradigms combined: **structural rules**, **lexical retrieval (BM25)**, and
**behavioral proof (skill-assessment scores + engagement)**.

## Structure

```
candidate-selection/
├── notebooks/              # built in order, each runs end-to-end
│   ├── 01_eda.ipynb                 exploratory data analysis
│   ├── 02_rule_based.ipynb          rule-based ranker (baseline)
│   ├── 03_honeypot_forensics.ipynb  verify the 53 honeypots, full forensics
│   ├── 04_bm25.ipynb                BM25 / TF-IDF lexical study
│   └── 05_hybrid.ipynb              FINAL ranker -> submission
├── outputs/                # generated artifacts (gitignored)
│   ├── submission_hybrid.csv        <- final submission (official 4-col format)
│   ├── honeypot_ids.csv             53 verified honeypots
│   ├── lexical_scores.parquet       bm25 + tfidf per candidate
│   └── scored_*_full.parquet        full scored tables (audit)
├── app.py                  # simple Streamlit upload/preview frontend
├── sample_candidates.json  # small sample for the app uploader
├── candidates.jsonl        # full 100K pool (gitignored)
├── validate_submission.py  # official challenge validator
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt

# run notebooks in order (01 -> 05); 05 writes outputs/submission_hybrid.csv
jupyter notebook

# validate the final submission with the official checker
python validate_submission.py outputs/submission_hybrid.csv
```

## Key design decisions

- **Title is a multiplicative gate**, not an additive score — a keyword-stuffing
  Tier-0 candidate (gate 0.05) cannot reach the top 100 no matter how many AI
  skills they list.
- **Honeypots are verified once** in `03` (53 candidates: date / YoE / skill-duration
  / expert-assessment impossibilities) and hard-excluded everywhere.
- **`skill_assessment_scores`** separates a *proven* expert from a *claimed* one —
  the differentiator that breaks score ties in the top 100.
- BM25 is length-normalized (robust to stuffers); TF-IDF cosine is not — so BM25 is
  the lexical input to the hybrid, never the decider.

## Notebooks (in order)

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Pool stats, titles, skills, behavioral signals, honeypot/stuffer scan |
| `02_rule_based.ipynb` | Structured rule scorer; valid baseline submission |
| `03_honeypot_forensics.ipynb` | Exhaustive honeypot verification; exports `honeypot_ids.csv` |
| `04_bm25.ipynb` | Okapi BM25 + TF-IDF; shows BM25 robust / TF-IDF fooled by stuffers |
| `05_hybrid.ipynb` | Final hybrid ranker; writes `outputs/submission_hybrid.csv` |
