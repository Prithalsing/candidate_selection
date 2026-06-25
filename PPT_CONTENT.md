# Redrob Hackathon — Intelligent Candidate Discovery & Ranking
## Full Technical Write-Up (PPT Source Material)

---

## SLIDE 1 — Problem Statement

**Challenge:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Task:** Given a pool of **100,000 candidates**, rank the **top 100** most suitable for the role of **Senior AI Engineer**.

**What we're given:**
- `candidates.jsonl` — 100,000 candidate profiles in JSON-lines format (~487 MB)
- A Job Description for "Senior AI Engineer" (embeddings, retrieval, LLMs, PyTorch, BM25, RAG, vector DBs)
- A validator script (`validate_submission.py`) to check submission format

**What we must submit:**
- A CSV with exactly **4 columns**: `candidate_id, rank, score, reasoning`
- Exactly **100 rows** (ranks 1–100)
- Scores must be **non-increasing** (rank 1 has the highest score)
- All **100 candidate IDs must be unique**

**Evaluation metric:**
- **NDCG@10 (50% weight)** — the top 10 picks matter the most
- Precision@100 (remaining weight)

---

## SLIDE 2 — The Big Challenge: It's Not Just Ranking

Before we even think about scoring, the dataset contains **traps and manipulations**:

### 1. Honeypot Candidates (~53 verified)
Fake/impossible profiles deliberately injected to catch naive rankers:
- A candidate who "worked at a company from 2020–2022" but their profile says total career duration = 2 months
- A candidate claiming 3 years of experience but career history adds up to 7 years
- A candidate with "Expert" proficiency in 5 AI skills but 0 months of duration on all of them
- A candidate with "Expert" Python but scored **12/100** on the Python assessment

### 2. Keyword Stuffers (8,101 candidates)
Completely unrelated profiles (HR Managers, Accountants, Civil Engineers, etc.) who listed **4 or more AI keywords** in their skills section to game keyword-based rankers:
- An HR Manager listing: `python, pytorch, llms, embeddings, rag, fine-tuning`
- TF-IDF cosine similarity is **fooled by this** — it gave 69 stuffers a top-100 spot

### 3. Legitimate but Weak Candidates
~90,000+ candidates who are real but simply not the right fit for a Senior AI Engineer role.

**Our core insight:** A ranker that doesn't explicitly handle honeypots and stuffers will get garbage in the top 100.

---

## SLIDE 3 — Our Solution: Gate-Then-Score Hybrid

We built a **4-stage pipeline** that makes cheating mathematically impossible:

```
INPUT: 100,000 candidates
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 0 — HARD EXCLUSION (Honeypot Check)             │
│  Impossible profiles → score = 0, cannot rank          │
└────────────────────────┬────────────────────────────────┘
                         │ ~99,947 remain
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — TITLE GATE (Multiplicative)                 │
│  Tier-0 titles (HR, Accountant…) → multiplier = 0.05  │
│  Senior AI Engineer, ML Engineer → multiplier = 1.00  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — QUALITY BLEND (7-component weighted score)  │
│  skill(0.25) + BM25(0.20) + career-text(0.18) +        │
│  assessment(0.17) + career-hist(0.10) + YoE(0.06) +    │
│  location(0.04)                                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — ENGAGEMENT MULTIPLIER (0.50 → 1.00)         │
│  Last active date, response rate, notice period,        │
│  open-to-work flag                                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
      final = title_gate × quality × engagement
              (= 0 if honeypot)

OUTPUT: Top 100 candidates, CSV submission
```

**Key property:** A keyword stuffer with gate=0.05 can never reach the top 100 no matter how many AI skills they list. The gate is multiplicative, not additive — stuffing cannot overcome it.

---

## SLIDE 4 — How We Built It: The Iterative Journey

We built the system **notebook by notebook**, each one building on the last:

### Notebook 01 — Exploratory Data Analysis (`01_eda.ipynb`)
**Goal:** Understand the dataset before writing a single line of ranking logic.

**What we explored:**
- Candidate data structure (profile, skills, career_history, redrob_signals)
- Title distribution — most common titles in the pool
- Skills distribution — which AI skills appear most
- Years of experience distribution
- Geographic distribution (India, USA, UK, etc.)
- Engagement signals (last_active_date, notice_period_days, open_to_work_flag)
- Preliminary honeypot scan — date mismatches, YoE anomalies
- Keyword stuffer scan — Tier-0 titles with stuffed AI skills

**Key discoveries:**
- The skill field is `name` (not `skill_name`) — critical for correct extraction
- The engagement data lives under `redrob_signals` (not `engagement`)
- Reference date for the dataset is **2026-06-08** (not 2025-01-01 — this caused 99,995 false honeypot positives when we first used the wrong date)
- ~53 profiles have genuinely impossible combinations
- ~8,101 profiles are keyword stuffers

---

### Notebook 02 — Rule-Based Ranker (`02_rule_based.ipynb`)
**Goal:** Build a solid structured baseline — no ML, pure logic.

**Approach:**
- Map each candidate to a **title tier** (1.0 down to 0.05)
- Count **AI skill matches** from a curated list (PyTorch, LangChain, FAISS, etc.), weighted by proficiency and duration
- Score **years of experience** with a curve (peak at 5–8 years for "Senior")
- Add **career history signals** (company quality, title progression, recency)
- Add **location signal** (metro India, international tech hubs)
- Multiply by **engagement signal** (last active, notice period, response rate)

**Output:** `outputs/scored_v1_full.parquet` — all 100K scored, baseline submission

---

### Notebook 03 — Honeypot Forensics (`03_honeypot_forensics.ipynb`)
**Goal:** Exhaustively verify every possible honeypot signal across all 100,000 candidates.

**What we checked for every candidate:**

| Signal | How we detect it |
|---|---|
| **Date mismatch** | Compute actual months from start/end dates; compare to `duration_months` field; flag if delta > 2 months |
| **YoE mismatch** | Sum all `duration_months` across career; if total > declared YoE×12 + 18 months, flag it |
| **Skill duration anomaly** | If ≥4 skills are "Expert" or "Advanced" but have 0 months duration → impossible |
| **Expert + low assessment** | If skill proficiency = "Expert" but `skill_assessment_scores` shows < 40/100 → impossible |

**Signals we explored but ruled out (natural variance, not honeypots):**
- `skill_dur_gt_yoe` — 13,449 candidates; skills can predate full-time employment (hobbies, education)
- `signup_after_active` — 7,496 candidates; platform data ordering, not a real impossibility

**Result:** **53 confirmed HARD honeypots**
- date_mismatch: 35
- yoe_mismatch: 24
- skill_anomaly: 13
- expert_assess_impossible: 1

**Output:** `outputs/honeypot_ids.csv`, `outputs/soft_suspect_ids.csv`, `outputs/forensics_full.parquet`

---

### Notebook 04 — BM25 Lexical Study (`04_bm25.ipynb`)
**Goal:** Add job-description-specific language matching. Can keyword counting find candidates our rules miss?

**Why we needed this:**
- Our skill list is finite — BM25 can match job-specific wording we didn't hardcode
- Career descriptions and headlines contain rich signal we weren't using

**BM25 vs TF-IDF experiment:**

| Method | Stuffers in top 100 | Why |
|---|---|---|
| TF-IDF cosine similarity | **69 stuffers** | Cosine similarity rewards documents that share many query terms, regardless of document length. An HR Manager with 15 stuffed AI skills scores higher than a real ML Engineer with 5 genuine ones. |
| **Okapi BM25** | **0 stuffers** | BM25 applies **length normalization** — adding more terms to a long document gives diminishing returns. A 15-term stuffed list on an HR Manager profile gets penalized relative to a focused ML Engineer profile. |

**Why BM25 over TF-IDF:** BM25's k1 and b parameters (k1=1.5, b=0.75) control term frequency saturation and document length normalization. This is exactly what makes it stuffer-robust.

**BM25 implementation (from scratch — no external library):**
```
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
BM25(d,q) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1×(1 - b + b×|d|/avgdl)]
```
Scores normalized by 99.9th percentile, clipped to 1.0.

**Output:** `outputs/lexical_scores.parquet`

---

### Notebook 05 — Final Hybrid Ranker (`05_hybrid.ipynb`)
**Goal:** Combine everything into one clean, robust pipeline.

**Quality blend (7 components, weights sum to 1.0):**

| Component | Weight | What it measures |
|---|---|---|
| Skill depth | **0.25** | Core AI skills (PyTorch, FAISS, HuggingFace…) weighted by duration + endorsements |
| BM25 lexical | **0.20** | JD-text match; length-normalized; stuffer-robust |
| Career text | **0.18** | How many JD keywords appear in job descriptions and headlines |
| Skill assessment | **0.17** | Actual test scores from `skill_assessment_scores` — separates *proven* experts from *claimed* ones |
| Career history | **0.10** | AI-titled roles, career progression, non-consulting background |
| Years of experience | **0.06** | Curve peaking at 5–8 years for "Senior" level |
| Location | **0.04** | Metro India > international > other |

**Title gate multipliers:**

| Tier | Example titles | Gate |
|---|---|---|
| Tier 5 (perfect match) | Senior AI Engineer, ML Engineer, NLP Engineer | 1.00 |
| Applied Scientist tier | Applied Scientist, Research Scientist, ML Scientist | 1.00 |
| Tier 4 (adjacent) | Data Scientist, Computer Vision Engineer | 0.80 |
| Tier 3 (transferable) | Software Engineer, Backend Engineer | 0.40 |
| Tier 2 (peripheral) | Data Engineer, Data Analyst, DevOps | 0.20 |
| Tier 0 (unrelated) | HR Manager, Accountant, Civil Engineer | **0.05** |

**Engagement multiplier (0.50 → 1.00):**
- Last active ≤7 days: +0.30 | ≤30 days: +0.22 | ≤90 days: +0.14 | ≤180 days: +0.07
- Recruiter response rate × 0.25
- Notice period ≤15 days: +0.25 | ≤30: +0.20 | ≤60: +0.12 | >60: +0.04
- Open to work flag: +0.20

**Final formula:**
```
final_score = title_gate × quality_blend × engagement_multiplier
            = 0  (if honeypot)
```

**Result:**
- 0 honeypots in top 100
- 0 keyword stuffers in top 100
- 0 Tier-0 titles in top 100
- 100 unique, non-increasing scores
- Passes `validate_submission.py` ✅

**Output:** `outputs/submission_hybrid.csv` — the official submission

---

## SLIDE 5 — Key Technical Decisions (With Reasoning)

### Decision 1: Title as a Multiplicative Gate (Not Additive Score)
**Why:** If title were additive, a keyword stuffer could compensate with enough skill points. As a multiplier, a 0.05 gate can never be overcome — `0.05 × anything ≤ 1.0` is always less than `1.0 × anything ≤ 1.0`. The gate is a hard ceiling.

### Decision 2: BM25 Over TF-IDF
**Why:** We ran the experiment. TF-IDF put 69 stuffers in the top 100. BM25 put 0. The difference is length normalization — BM25 penalizes overly long documents (i.e., stuffed skill lists). TF-IDF cosine does not.

### Decision 3: skill_assessment_scores as the Tie-Breaker
**Why:** Two candidates can both claim "Expert Python + Expert PyTorch." One scored 82/100 on assessments. The other scored 31/100. Only `skill_assessment_scores` can distinguish them. Without this signal, score ties would be broken by candidate_id alphabetically — a meaningless tie-break. With it, assessed quality wins.

### Decision 4: CURRENT_DATE = 2026-06-08 (Not 2025-01-01)
**Why:** The dataset was synthetically generated with this reference date. Using 2025-01-01 caused the date-mismatch honeypot check to flag 99,995 false positives (almost everyone). The correct date was discovered by examining the gap between stored dates and computed dates — all mismatches showed a Δ of exactly −17 months (the difference between 2025-01-01 and 2026-06-08).

### Decision 5: Skip Embeddings, Use BM25
**Why:** Embedding models (sentence-transformers, OpenAI embeddings) require GPU or significant CPU time on 100,000 documents. BM25 is algebraic (no model inference), runs in ~88 seconds for 100K on CPU, and is already stuffer-robust. The incremental gain from embeddings did not justify the complexity and compute cost.

### Decision 6: Verify Honeypots Forensically First
**Why:** If we had just added a honeypot gate without verifying, we'd either over-filter (removing real candidates) or under-filter (missing honeypots). Notebook 03 runs all 4 checks on all 100K and confirms exactly 53 impossibles. This gives us confidence in the gate.

---

## SLIDE 6 — Architecture Overview

```
candidates.jsonl (100,000 profiles, 487 MB)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                        rank.py                                  │
│                                                                  │
│  load_candidates()  ──  auto-detects .json array vs .jsonl     │
│                                                                  │
│  compute_bm25()     ──  Okapi BM25 (k1=1.5, b=0.75)           │
│                         Tokenize all 100K docs vs JD query     │
│                         Normalize by 99.9th percentile         │
│                                                                  │
│  For each candidate:                                             │
│    is_honeypot()    ──  4 checks (date/YoE/skill/assess)       │
│    title_gate()     ──  5-tier multiplier (1.0 → 0.05)         │
│    skill_depth()    ──  Tier C/B/A skills × duration           │
│    career_text()    ──  JD keywords in job descriptions        │
│    career_hist()    ──  Progression, seniority, AI titles      │
│    yoe_score()      ──  Curve: peak at 5–8 yrs                 │
│    location_score() ──  Metro India > international            │
│    assessment_score()── skill_assessment_scores mean           │
│    engagement_mult()──  last_active + response + notice        │
│                                                                  │
│    quality = Σ(weight × component)                              │
│    final = title_gate × quality × engagement  (0 if honeypot)  │
│                                                                  │
│  Sort by score desc, candidate_id asc (tie-break)              │
│  Take top 100                                                    │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
submission.csv  (candidate_id, rank, score, reasoning)
```

**Runtime:** ~88 seconds for 100,000 candidates on a standard CPU (no GPU needed).

---

## SLIDE 7 — Results

### Final Top-100 Statistics

| Metric | Value |
|---|---|
| Honeypots in top 100 | **0** |
| Keyword stuffers in top 100 | **0** |
| Tier-0 titles in top 100 | **0** |
| Unique scores | **100** (no ties) |
| Scores non-increasing | **✅** |
| Passes `validate_submission.py` | **✅** |

### What the top candidates look like
- Titles: Senior AI Engineer, ML Engineer, Applied Scientist, Senior NLP Engineer, Research Scientist
- YoE: Typically 5–10 years
- Skills: PyTorch, HuggingFace, FAISS, LangChain, sentence-transformers, RAG, fine-tuning
- Career text: Heavy use of "embedding," "retrieval," "semantic search," "fine-tuning," "vector"
- Assessment scores: Mean ~71/100 for expert-claimed skills (vs ~52 for advanced, ~30 for intermediate)
- Engagement: Open to work, recent activity, short notice periods

### BM25 vs TF-IDF experiment result

| | BM25 | TF-IDF Cosine |
|---|---|---|
| Stuffers in top 100 | **0** | 69 |
| Honeypots in top 100 | 0 | 0 (excluded by gate) |
| Approach chosen | ✅ Yes | ❌ No |

---

## SLIDE 8 — Notebook Pipeline (Visual)

```
01_eda.ipynb
  └─ Understand the data, find patterns, spot honeypots/stuffers

      ▼

02_rule_based.ipynb
  └─ First working ranker (structural rules only)
  └─ Produces: scored_v1_full.parquet

      ▼

03_honeypot_forensics.ipynb
  └─ Exhaustive verification of all 100K for impossible profiles
  └─ Produces: honeypot_ids.csv (53), soft_suspect_ids.csv, forensics_full.parquet

      ▼

04_bm25.ipynb
  └─ BM25 from scratch, comparison with TF-IDF
  └─ Proves BM25 is stuffer-robust, TF-IDF is not
  └─ Produces: lexical_scores.parquet

      ▼

05_hybrid.ipynb
  └─ Combines everything: rules + BM25 + assessments + engagement
  └─ Final gate-then-score pipeline
  └─ Produces: submission_hybrid.csv ← THE SUBMISSION
```

All notebooks run end-to-end sequentially. `rank.py` consolidates all logic into a single reproducible script.

---

## SLIDE 9 — Deliverables

| Deliverable | Description |
|---|---|
| `outputs/submission_hybrid.csv` | Official 4-column submission (100 rows) |
| `rank.py` | Single-command reproduction script (~88s, CPU only) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `notebooks/02_rule_based.ipynb` | Rule-based baseline |
| `notebooks/03_honeypot_forensics.ipynb` | Honeypot verification |
| `notebooks/04_bm25.ipynb` | BM25 lexical study |
| `notebooks/05_hybrid.ipynb` | Final hybrid ranker |
| `app.py` | Streamlit app — upload candidates, get ranked CSV |
| `submission_metadata.yaml` | Team info, approach summary, AI declarations |
| GitHub | https://github.com/Prithalsing/candidate_selection |
| Live Demo | https://candidateselection-apt69vvwqg5vda5vbxez4b.streamlit.app/ |

**Reproduce the submission in one command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## SLIDE 10 — Tools & Stack

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.12 | All logic |
| Data | pandas, pyarrow | Loading, processing, parquet I/O |
| ML/Math | numpy, scikit-learn | Numerics, TF-IDF (study only) |
| BM25 | Built from scratch | Okapi BM25 (k1=1.5, b=0.75) — no library used |
| Frontend | Streamlit | Upload + rank demo app |
| Notebooks | Jupyter | EDA and iterative development |
| Versioning | Git + GitHub | Full history at Prithalsing/candidate_selection |
| Cloud | Streamlit Community Cloud | Live demo |
| AI assistance | Claude, Gemini, Antigravity | Brainstorming, code writing/editing |

**No GPU required. No internet during ranking. Fully offline, CPU-only.**

---

## SLIDE 11 — Why Our Approach Wins

1. **We don't just rank — we verify.** Exhaustive honeypot forensics (Notebook 03) means we know exactly which 53 profiles are impossible, not just "suspicious."

2. **Stuffers cannot game our system.** The title gate is multiplicative — an HR Manager with 100 AI skills still scores the same as one with 0 AI skills in the top-100 race.

3. **BM25 finds candidates our keyword list misses.** If a candidate wrote "dense passage retrieval" instead of our keyword "semantic search," BM25 catches it. Hardcoded keyword lists miss synonyms and phrasings.

4. **Assessment scores separate claimed from proven.** Anyone can write "Expert PyTorch." Only someone who actually scored 82/100 on the assessment proves it. This is the signal that breaks ties at the top.

5. **Single reproducible command.** `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` — end to end, any machine, ~88 seconds.

---

## APPENDIX — Data Structure Reference

Each candidate in `candidates.jsonl` looks like:

```json
{
  "candidate_id": "CAND_0012345",
  "profile": {
    "current_title": "Senior ML Engineer",
    "years_of_experience": 7,
    "country": "India",
    "location": "Bangalore",
    "headline": "ML Engineer specializing in LLMs and RAG systems",
    "summary": "..."
  },
  "skills": [
    {
      "name": "pytorch",
      "proficiency": "expert",
      "duration_months": 36,
      "endorsements": 12
    }
  ],
  "career_history": [
    {
      "title": "ML Engineer",
      "company": "Startup XYZ",
      "start_date": "2021-06-01",
      "end_date": "2024-01-01",
      "duration_months": 31,
      "description": "Built embedding-based retrieval systems using FAISS..."
    }
  ],
  "redrob_signals": {
    "last_active_date": "2026-06-01",
    "recruiter_response_rate": 0.85,
    "notice_period_days": 30,
    "open_to_work_flag": true,
    "skill_assessment_scores": {
      "pytorch": 88,
      "python": 91
    }
  }
}
```

**Critical field names (discovered during EDA):**
- Skills: `s["name"]` — NOT `s["skill_name"]`
- Engagement: `c["redrob_signals"]` — NOT `c["engagement"]`
- Assessment: `c["redrob_signals"]["skill_assessment_scores"]` — dictionary of `{skill_name: score}`
