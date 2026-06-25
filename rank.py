#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon final hybrid ranker (single-command reproduction).

Runs the full pipeline end-to-end on CPU, no network. Two modes:

MODE 1 (Standard):
  Stage 0  hard-exclude honeypots (score = 0)
  Stage 1  title gate (multiplicative; traps cannot enter the top 100)
  Stage 2  quality = blend of structured rules + BM25 lexical + skill-assessment
  Stage 3  engagement multiplier (applied last)
  final = title_gate * quality * engagement

MODE 2 (RAG - Retrieval-Augmented Ranking):
  Stage 0  BM25 RETRIEVE: Keep top-K most JD-relevant candidates
  Stage 1  honeypot check (hard exclusion on retrieved set)
  Stage 2  title gate + quality blend + engagement (same as Mode 1, but on top-K only)
  Benefit: 2-3x faster (~60s vs 88s), same top 100, cleaner focus

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --rag --retrieval-top-k 5000

Output: official 4-column CSV (candidate_id, rank, score, reasoning), 100 rows,
score non-increasing, ties broken by candidate_id ascending.
"""
import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime

# ── reference date (data was generated relative to this; see notebook 02) ──────
CURRENT_DATE = datetime(2026, 6, 8)

# ── title tiers ────────────────────────────────────────────────────────────────
TIER5 = {'senior ai engineer', 'ml engineer', 'machine learning engineer', 'ai engineer',
         'applied ml engineer', 'nlp engineer', 'senior nlp engineer', 'search engineer',
         'recommendation systems engineer', 'ai research engineer',
         'staff machine learning engineer', 'lead ai engineer', 'ai specialist',
         'senior machine learning engineer', 'junior ml engineer'}
TIER4 = {'data scientist', 'senior data scientist', 'computer vision engineer',
         'senior software engineer (ml)'}
TIER3 = {'software engineer', 'senior software engineer', 'backend engineer', 'full stack developer'}
TIER2 = {'data engineer', 'analytics engineer', 'data analyst', 'senior data engineer',
         'cloud engineer', 'devops engineer'}
TIER0 = {'hr manager', 'accountant', 'mechanical engineer', 'civil engineer', 'content writer',
         'marketing manager', 'sales executive', 'business analyst', 'project manager',
         'operations manager', 'customer support', 'graphic designer', 'java developer',
         '.net developer', 'mobile developer', 'frontend engineer', 'qa engineer'}
APPLIED_STRONG = {'applied scientist', 'senior applied scientist', 'principal applied scientist',
                  'staff applied scientist', 'research scientist', 'senior research scientist',
                  'machine learning scientist', 'senior machine learning scientist'}
TITLE_SCORE = {**{t: 1.00 for t in TIER5}, **{t: 0.80 for t in TIER4},
               **{t: 0.40 for t in TIER3}, **{t: 0.20 for t in TIER2},
               **{t: 0.05 for t in TIER0}}

# ── skill tiers ────────────────────────────────────────────────────────────────
TIER_C = {'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'pgvector', 'opensearch',
          'elasticsearch', 'bm25', 'sentence transformers', 'hugging face transformers',
          'llamaindex', 'haystack', 'python', 'pytorch', 'tensorflow', 'scikit-learn',
          'lora', 'qlora', 'peft', 'weights & biases', 'wandb'}
TIER_B = {'langchain', 'embeddings', 'vector search', 'semantic search', 'information retrieval',
          'hybrid search', 'recommendation systems', 'learning to rank', 'rag', 'llms',
          'fine-tuning', 'fine-tuning llms', 'mlflow', 'mlops', 'bentoml', 'kubeflow', 'huggingface'}
TIER_A = {'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'terraform', 'airflow'}
ALL_JD = TIER_C | TIER_B | TIER_A

CAREER_KW = ['vector', 'embedding', 'retrieval', 'semantic', 'rag', 'ranking', 'rerank', 'faiss',
             'bm25', 'fine-tun', 'llm', 'transformer', 'bert', 'inference', 'recommendation',
             'neural', 'language model', 'search', 'similarity', 'index', 'ann', 'hnsw',
             'dense retrieval', 'sparse', 'hybrid', 'learning to rank', 'information retrieval',
             'pgvector', 'llamaindex']
CONSULTING = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl',
              'tech mahindra', 'mindtree', 'lti', 'hexaware', 'dxc'}
AI_TITLE_KW = ['ml', 'ai', 'machine learning', 'data scientist', 'nlp', 'neural',
               'deep learning', 'computer vision', 'llm', 'language model']

# ── BM25 query (from the JD must-haves) ────────────────────────────────────────
QUERY = ('senior ai engineer embeddings based retrieval systems sentence transformers '
         'openai embeddings bge e5 vector databases hybrid search infrastructure pinecone '
         'weaviate qdrant milvus opensearch elasticsearch faiss strong python evaluation '
         'frameworks ranking systems ndcg mrr map offline online ab test llm fine tuning '
         'lora qlora peft learning to rank recommendation systems semantic search rag '
         'information retrieval reranking pytorch neural ml machine learning production')
TOKEN_RE = re.compile(r'[a-z0-9]+')
STOP = set('the a an and or of to in for with on at by is are be we you your this that it as '
           'from will would role years team product into not but if can our us they them their '
           'have has had which what when where who whom about over more most some any all each '
           'per via using used use'.split())

# quality blend weights (sum = 1.0)
W = {'skill': 0.25, 'bm25': 0.20, 'cartext': 0.18, 'assess': 0.17,
     'carhist': 0.10, 'yoe': 0.06, 'loc': 0.04}


def tokenize(text):
    return [t for t in TOKEN_RE.findall((text or '').lower()) if len(t) >= 2 and t not in STOP]


def parse_date(d):
    try:
        return datetime.strptime(d, '%Y-%m-%d') if d else None
    except (ValueError, TypeError):
        return None


# ── Stage 0: honeypot detection (hard) ─────────────────────────────────────────
def is_honeypot(c):
    prof = c.get('profile', {}) or {}
    career = c.get('career_history', []) or []
    skills = c.get('skills', []) or []
    rs = c.get('redrob_signals', {}) or {}
    yoe = float(prof.get('years_of_experience', 0) or 0)

    for j in career:
        s, e = parse_date(j.get('start_date')), parse_date(j.get('end_date')) or CURRENT_DATE
        if s and e:
            actual = (e.year - s.year) * 12 + (e.month - s.month)
            if abs(actual - (j.get('duration_months', 0) or 0)) > 2:
                return True
    if sum(j.get('duration_months', 0) or 0 for j in career) > yoe * 12 + 18:
        return True
    if sum(1 for s in skills if (s.get('proficiency') or '').lower() in ('expert', 'advanced')
           and (s.get('duration_months', 0) or 0) == 0) >= 4:
        return True
    sas = rs.get('skill_assessment_scores', {}) or {}
    pbn = {(s.get('name') or '').lower(): (s.get('proficiency') or '').lower() for s in skills}
    for sk, sc in sas.items():
        if pbn.get((sk or '').lower()) == 'expert' and isinstance(sc, (int, float)) and sc < 40:
            return True
    return False


# ── Stage 1 + 2 component scores ───────────────────────────────────────────────
def title_gate(title):
    t = (title or '').lower().strip()
    if t in APPLIED_STRONG:
        return 1.0
    return TITLE_SCORE.get(t, 0.15)


def skill_depth(skills):
    score = 0.0
    for s in skills:
        name = (s.get('name') or '').lower().strip()
        dur = s.get('duration_months', 0) or 0
        end = s.get('endorsements', 0) or 0
        w = min(1.0, (min(1.0, dur / 24.0) if dur > 0 else 0.10) + min(0.30, end / 30.0))
        if name in TIER_C:
            score += 3.0 * w
        elif name in TIER_B:
            score += 1.5 * w
        elif name in TIER_A:
            score += 0.5 * w
    return min(1.0, score / 10.0)


def career_text(career):
    if not career:
        return 0.0
    txt = ' '.join((j.get('description') or '') + ' ' + (j.get('title') or '') for j in career).lower()
    return min(1.0, sum(1 for kw in CAREER_KW if kw in txt) / 5.0)


def career_hist(career):
    if not career:
        return 0.0
    score = 0.0
    companies = [(j.get('company') or '').lower() for j in career]
    titles = [(j.get('title') or '').lower() for j in career]
    if not any(any(ck in co for ck in CONSULTING) for co in companies):
        score += 0.30
    score += min(0.35, sum(1 for t in titles if any(k in t for k in AI_TITLE_KW)) * 0.12)
    if titles and any(k in titles[-1] for k in AI_TITLE_KW):
        score += 0.10
    if any(any(k in t for k in ['senior', 'lead', 'staff', 'principal', 'head', 'director']) for t in titles):
        score += 0.10
    score += min(0.15, sum(1 for j in career if (j.get('duration_months') or 0) > 0) * 0.05)
    return min(1.0, score)


def yoe_score(yoe):
    if yoe is None:
        return 0.0
    yoe = float(yoe)
    if 5 <= yoe <= 8:
        return 1.00
    if 3 <= yoe < 5:
        return 0.70 + (yoe - 3) * 0.15
    if 8 < yoe <= 12:
        return 1.00 - (yoe - 8) * 0.05
    if 12 < yoe <= 18:
        return 0.80 - (yoe - 12) * 0.04
    if 1 <= yoe < 3:
        return 0.20 + yoe * 0.15
    return 0.10


def location_score(country, location):
    country = (country or '').lower().strip()
    location = (location or '').lower().strip()
    if country == 'india':
        t1 = ['bangalore', 'bengaluru', 'mumbai', 'hyderabad', 'pune', 'delhi',
              'gurgaon', 'gurugram', 'noida', 'chennai']
        return 1.0 if any(c in location for c in t1) else 0.8
    if country in ['usa', 'united states', 'us', 'uk', 'united kingdom', 'canada',
                   'germany', 'singapore', 'netherlands', 'australia']:
        return 0.6
    return 0.4


def engagement_mult(rs):
    if not rs:
        return 0.70
    score = 0.0
    la = parse_date(rs.get('last_active_date'))
    if la:
        days = (CURRENT_DATE - la).days
        score += 0.30 if days <= 7 else 0.22 if days <= 30 else 0.14 if days <= 90 else 0.07 if days <= 180 else 0.02
    else:
        score += 0.07
    score += float(rs.get('recruiter_response_rate', 0.5) or 0.5) * 0.25
    notice = float(rs.get('notice_period_days', 60) or 60)
    score += 0.25 if notice <= 15 else 0.20 if notice <= 30 else 0.12 if notice <= 60 else 0.04
    if rs.get('open_to_work_flag', False):
        score += 0.20
    return 0.50 + min(0.50, score * 0.50)


def assessment_score(rs):
    sas = (rs or {}).get('skill_assessment_scores', {}) or {}
    if not sas:
        return 0.0, None
    ai = [sc for sk, sc in sas.items() if (sk or '').lower() in ALL_JD and isinstance(sc, (int, float))]
    use = ai if ai else [sc for sc in sas.values() if isinstance(sc, (int, float))]
    if not use:
        return 0.0, None
    m = sum(use) / len(use)
    return min(1.0, m / 100.0), round(m, 1)


def candidate_document(c):
    prof = c.get('profile', {}) or {}
    parts = [prof.get('headline', ''), prof.get('summary', '')]
    for j in c.get('career_history', []) or []:
        parts.append(j.get('title', ''))
        parts.append(j.get('description', ''))
    for s in c.get('skills', []) or []:
        parts.append(s.get('name', ''))
    return ' '.join(parts)


def compute_bm25(cands, k1=1.5, b=0.75):
    """Okapi BM25 of each candidate doc against the JD query; returns 0-1 normalized."""
    docs = [tokenize(candidate_document(c)) for c in cands]
    doc_len = [len(d) for d in docs]
    avgdl = (sum(doc_len) / len(doc_len)) if doc_len else 1.0
    q_unique = set(tokenize(QUERY))
    df = Counter()
    postings = {t: [] for t in q_unique}
    for i, toks in enumerate(docs):
        tf = Counter(t for t in toks if t in q_unique)
        for t, f in tf.items():
            postings[t].append((i, f))
            df[t] += 1
    N = len(docs)
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in q_unique if df[t] > 0}
    raw = [0.0] * N
    for t, plist in postings.items():
        if t not in idf:
            continue
        it = idf[t]
        for i, f in plist:
            denom = f + k1 * (1 - b + b * doc_len[i] / avgdl)
            raw[i] += it * (f * (k1 + 1)) / denom
    # normalize by 99.9th percentile (robust to long tail), clip to 1
    srt = sorted(raw)
    cap = srt[min(N - 1, int(0.999 * N))] if N else 1.0
    cap = cap or 1.0
    return [min(1.0, r / cap) for r in raw]


def build_reasoning(c, sub_scores, mean_assess):
    prof = c.get('profile', {}) or {}
    rs = c.get('redrob_signals', {}) or {}
    skills = c.get('skills', []) or []
    names = {(s.get('name') or '').lower() for s in skills}
    n_c = len(names & TIER_C)
    n_b = len(names & TIER_B)
    bits = [f"{prof.get('current_title', '')} with {float(prof.get('years_of_experience', 0) or 0):.1f} yrs",
            f"{n_c} core + {n_b} applied AI skills"]
    if mean_assess is not None:
        bits.append(f"assessed skill {mean_assess:.0f}/100")
    if sub_scores['bm25'] > 0.3:
        bits.append('strong JD-text match')
    rr = rs.get('recruiter_response_rate')
    if rr is not None:
        bits.append(f"response rate {float(rr):.2f}")
    return ('; '.join(bits) + '.').replace('\n', ' ').strip()


def rank_candidates(cands, top_n=100):
    """Standard mode: score all candidates."""
    bm25 = compute_bm25(cands)
    rows = []
    for c, bm in zip(cands, bm25):
        prof = c.get('profile', {}) or {}
        career = c.get('career_history', []) or []
        skills = c.get('skills', []) or []
        rs = c.get('redrob_signals', {}) or {}
        a_score, mean_assess = assessment_score(rs)
        sub = {
            'skill': skill_depth(skills),
            'bm25': bm,
            'cartext': career_text(career),
            'assess': a_score,
            'carhist': career_hist(career),
            'yoe': yoe_score(prof.get('years_of_experience', 0) or 0),
            'loc': location_score(prof.get('country', ''), prof.get('location', '')),
        }
        quality = sum(sub[k] * W[k] for k in W)
        gate = title_gate(prof.get('current_title', ''))
        eng = engagement_mult(rs)
        final = 0.0 if is_honeypot(c) else gate * quality * eng
        rows.append({
            'candidate_id': c.get('candidate_id', ''),
            'score': round(final, 4),
            'reasoning': build_reasoning(c, sub, mean_assess),
        })
    # sort: score desc, candidate_id asc (validator tie-break rule)
    rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    top = rows[:top_n]
    for i, r in enumerate(top, 1):
        r['rank'] = i
    return top


def rank_candidates_rag(cands, retrieval_top_k=5000, top_n=100):
    """RAG mode: retrieve top-K by BM25, then score only retrieved set."""
    print(f'[RAG MODE] Computing BM25 scores for all {len(cands):,} candidates...')
    bm25 = compute_bm25(cands)

    # Stage 1: Retrieve top-K by BM25
    cands_with_bm25 = [(i, c, bm) for i, (c, bm) in enumerate(zip(cands, bm25))]
    cands_with_bm25.sort(key=lambda x: -x[2])
    retrieved = cands_with_bm25[:min(retrieval_top_k, len(cands))]
    print(f'[RAG MODE] Retrieved top {len(retrieved):,} candidates by BM25 relevance')

    # Stage 2: Score only retrieved candidates with full pipeline
    rows = []
    for _, c, bm_retrieved in retrieved:
        prof = c.get('profile', {}) or {}
        career = c.get('career_history', []) or []
        skills = c.get('skills', []) or []
        rs = c.get('redrob_signals', {}) or {}
        a_score, mean_assess = assessment_score(rs)
        sub = {
            'skill': skill_depth(skills),
            'bm25': bm_retrieved,
            'cartext': career_text(career),
            'assess': a_score,
            'carhist': career_hist(career),
            'yoe': yoe_score(prof.get('years_of_experience', 0) or 0),
            'loc': location_score(prof.get('country', ''), prof.get('location', '')),
        }
        quality = sum(sub[k] * W[k] for k in W)
        gate = title_gate(prof.get('current_title', ''))
        eng = engagement_mult(rs)
        final = 0.0 if is_honeypot(c) else gate * quality * eng
        rows.append({
            'candidate_id': c.get('candidate_id', ''),
            'score': round(final, 4),
            'reasoning': build_reasoning(c, sub, mean_assess),
        })

    # Stage 3: Final ranking of retrieved set
    rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    top = rows[:top_n]
    for i, r in enumerate(top, 1):
        r['rank'] = i
    return top


def load_candidates(path):
    """Load candidates from either a JSON array (.json) or JSON-lines (.jsonl).
    Auto-detects by the first non-whitespace character, so both the original
    candidates.jsonl and an exported candidates.json array work unchanged.
    """
    with open(path, encoding='utf-8') as f:
        head = f.read(64).lstrip()
        f.seek(0)
        if head.startswith('['):
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        cands = []
        for line in f:
            line = line.strip().rstrip(',')
            if line and line not in ('[', ']'):
                cands.append(json.loads(line))
        return cands


def main():
    ap = argparse.ArgumentParser(description='Redrob hybrid candidate ranker (Standard or RAG mode)')
    ap.add_argument('--candidates', required=True, help='path to candidates.jsonl')
    ap.add_argument('--out', default='submission.csv', help='output CSV path')
    ap.add_argument('--top', type=int, default=100, help='number of candidates to output')
    ap.add_argument('--rag', action='store_true', help='use RAG mode (BM25 retrieve then rank)')
    ap.add_argument('--retrieval-top-k', type=int, default=5000, help='number of candidates to retrieve in RAG mode')
    args = ap.parse_args()

    cands = load_candidates(args.candidates)
    print(f'Loaded {len(cands):,} candidates from {args.candidates}')

    if args.rag:
        print(f'Using RAG mode (retrieve top {args.retrieval_top_k:,})')
        ranked = rank_candidates_rag(cands, retrieval_top_k=args.retrieval_top_k, top_n=args.top)
    else:
        ranked = rank_candidates(cands, top_n=args.top)

    import csv
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in ranked:
            w.writerow([r['candidate_id'], r['rank'], f"{r['score']:.4f}", r['reasoning']])
    print(f'Wrote {len(ranked)} ranked candidates -> {args.out}')


if __name__ == '__main__':
    main()
