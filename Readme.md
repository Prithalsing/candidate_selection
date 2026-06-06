# Candidate Selection — Redrob Hackathon

Intelligent Candidate Discovery & Ranking for the Senior AI Engineer JD.

## Structure

```
candidate-selection/
├── notebooks/          # EDA + scoring experiments (ipynb)
├── src/                # reusable helper functions
├── outputs/            # generated CSVs (gitignored)
├── candidates.jsonl    # full 100K pool (gitignored)
├── rank.py             # final submission script
├── validate_submission.py
├── requirements.txt
└── submission_metadata.yaml
```

## Reproduce

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./outputs/submission.csv
```

## Notebooks (in order)

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Full exploratory data analysis |
| `02_scoring_v1.ipynb` | Simple rule-based ranker |
| `03_scoring_v2.ipynb` | Improved scoring |
