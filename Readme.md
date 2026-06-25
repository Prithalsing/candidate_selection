# Candidate Selection — Redrob Hackathon

Intelligent Candidate Discovery & Ranking for the **Senior AI Engineer** JD.
Ranks the top 100 candidates from a 100,000-candidate pool.

**🚀 Live Demo:** [Streamlit Cloud App](https://candidateselection-v4gr9ito7xyms9faoq6gxq.streamlit.app/)

---

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

---

## Quick Start

### Option 1: Live Demo (Easiest)
Open the **[Streamlit Cloud App](https://candidateselection-v4gr9ito7xyms9faoq6gxq.streamlit.app/)** in your browser.
- Upload or point to a candidate file
- Run any of 3 ranking methods (Standard, RAG, Hybrid TF-IDF)
- Download results immediately
- **Sentence Transformers method available locally only**

### Option 2: Run Locally (Full Power)
```bash
pip install -r requirements.txt
streamlit run app_multi.py
```

Opens at `http://localhost:8501` with all 4 ranking methods available.

### Option 3: One-Command Reproduction (No UI)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

---

## Ranking Methods

| Method | File | Time | Retrieval | Cloud | Best For |
|--------|------|------|-----------|-------|----------|
| **Standard** | `rank.py` | ~73s | None (all 100K) | ✅ | Full audit, understanding distribution |
| **RAG (BM25)** | `rank_rag.py` | ~44s | Lexical (3K) | ✅ | Production speed |
| **Hybrid TF-IDF** | `rank_hybrid_rag.py` | ~150s | Semantic + Lexical | ✅ | Maximum confidence (offline) |
| **Hybrid Transformers** | `rank_sentence_transformers_rag.py` | ~80s | Semantic (pre-trained) + Lexical | ⚠️ Local only | Best semantic understanding |

**All methods produce identical top-100** because high-scoring candidates excel on both lexical and semantic dimensions.

---

## Key Design Decisions

- **Title is a multiplicative gate**, not an additive score — a keyword-stuffing
  Tier-0 candidate (gate 0.05) cannot reach the top 100 no matter how many AI
  skills they list.
- **Honeypots are verified once** in `notebooks/03_honeypot_forensics.ipynb` (53 candidates: date / YoE / skill-duration
  / expert-assessment impossibilities) and hard-excluded everywhere.
- **`skill_assessment_scores`** separates a *proven* expert from a *claimed* one —
  the differentiator that breaks score ties in the top 100.
- **BM25is length-normalized** (robust to stuffers); TF-IDF cosine is not — so BM25 is
  the lexical input to the hybrid, never the decider alone.

---

## Project Structure

```
candidate-selection/
├── notebooks/                       # built in order, each runs end-to-end
│   ├── 01_eda.ipynb                 exploratory data analysis
│   ├── 02_rule_based.ipynb          rule-based ranker (baseline)
│   ├── 03_honeypot_forensics.ipynb  verify the 53 honeypots, full forensics
│   ├── 04_bm25.ipynb                BM25 / TF-IDF lexical study
│   └── 05_hybrid.ipynb              FINAL ranker -> submission
│
├── app_multi.py                     # Multi-tab Streamlit app (all 4 methods)
├── rank.py                          # Standard: score all 100K
├── rank_rag.py                      # RAG: BM25 retrieve, score 3K
├── rank_hybrid_rag.py               # Hybrid TF-IDF: semantic + lexical
├── rank_sentence_transformers_rag.py # Hybrid Transformers (optional, local only)
│
├── sample_candidates.json           # small test file for app uploader
├── submission_metadata.yaml         # filled submission metadata
├── validate_submission.py           # official challenge validator
├── requirements.txt                 # pip dependencies
└── Readme.md                        # this file
```

---

## Results

Final top 100:
- ✅ **0 honeypots** (hard-excluded)
- ✅ **0 keyword stuffers** (gated low by title multiplier)
- ✅ **0 Tier-0 titles** (impossible to rank high)
- ✅ **100 unique non-increasing scores**
- ✅ Passes `validate_submission.py`
- ⚡ Runtime ~88s for 100K candidates on CPU

---

## How Notebooks Build to This

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Pool statistics, titles, skills, behavioral signals, honeypot/stuffer scan |
| `02_rule_based.ipynb` | Structured rule scorer; valid baseline submission |
| `03_honeypot_forensics.ipynb` | Exhaustive honeypot verification; exports `honeypot_ids.csv` |
| `04_bm25.ipynb` | Okapi BM25 + TF-IDF comparison; shows BM25 robustness vs TF-IDF |
| `05_hybrid.ipynb` | Final hybrid ranker; writes submission |

---

## Installation & Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `streamlit` — web UI
- `pandas`, `numpy` — data handling
- `scikit-learn` — TF-IDF vectorization
- `rank_bm25` — Okapi BM25 ranking
- `sentence-transformers` — optional, for pre-trained semantic embeddings

---

## Cloud Deployment (Streamlit Cloud)

The app is configured for [Streamlit Community Cloud](https://share.streamlit.io):

1. Repo connected: `Prithalsing/candidate_selection`
2. App file: `app_multi.py`
3. Auto-deploys on every `git push`
4. Tab 4 (Sentence Transformers) shows unavailable message on cloud, works locally

---

## License

Redrob Hackathon Project, 2024.

---

## Questions?

See inline code comments in `rank.py` for detailed algorithm explanations.
Run a notebook to see step-by-step logic.
Upload a file to the app and explore interactively.
