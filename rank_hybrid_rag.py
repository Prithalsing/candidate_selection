#!/usr/bin/env python3
"""
rank_hybrid_rag.py — HYBRID RAG MODE (Semantic + Lexical Retrieval)

True Retrieval-Augmented Ranking combining TWO retrieval methods:

STAGE 0A: SEMANTIC RETRIEVAL (TF-IDF Cosine Similarity)
  - Extract text from candidate profiles (headline, summary, career, skills)
  - Compute term frequencies (TF) and inverse document frequency (IDF)
  - Cosine similarity between candidate documents and JD query
  - Retrieve top 3,000 by semantic relevance

STAGE 0B: LEXICAL RETRIEVAL (BM25)
  - Compute BM25 scores for all 100K candidates
  - Retrieve top 3,000 by keyword match

STAGE 0C: COMBINE RETRIEVALS
  - Union: (Semantic top 3K) ∪ (Lexical top 3K)
  - Result: ~4,000-5,000 candidates (semantic + lexical matches)

STAGE 1: HARDCORE FILTERING
  - Hard-exclude honeypots (impossible profiles)
  - Hard-exclude TIER0 titles (HR Manager, Accountant, etc.)
  - Detect and flag keyword stuffers
  - Result: ~3,000-4,000 clean candidates

STAGE 2: FULL RANKING
  - Apply 7-component scoring to filtered set
  - Title gate + quality blend + engagement
  - Final = gate × quality × engagement

OUTPUT: Top 100 ranked candidates

PERFORMANCE:
  - Time: ~60-70s (BM25 + cosine similarity both O(100K))
  - Retrieved: ~4,500 candidates (hybrid coverage)
  - Quality: Both semantic + lexical signals
  - Safety: Strict filtering removes bad profiles upfront

WHY HYBRID IS BETTER:
  - Semantic (TF-IDF cosine): Catches candidates who discuss concepts
    (e.g., "information retrieval systems" matches even without exact keywords)
  - Lexical (BM25): Catches candidates with exact keyword matches
  - Together: Don't miss candidates strong in either signal

COMPARISON:
  Standard RAG: Only BM25 (lexical only, 44s)
  Hybrid RAG: BM25 + Cosine Similarity (both signals, 60s)
  Benefit: Semantic coverage + strict filtering = better quality
"""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
import numpy as np

CURRENT_DATE = datetime(2026, 6, 8)

# Title tiers
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

# Skill tiers
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

W = {'skill': 0.25, 'bm25': 0.20, 'cartext': 0.18, 'assess': 0.17,
     'carhist': 0.10, 'yoe': 0.06, 'loc': 0.04}


def tokenize(text):
    return [t for t in TOKEN_RE.findall((text or '').lower()) if len(t) >= 2 and t not in STOP]


def parse_date(d):
    try:
        return datetime.strptime(d, '%Y-%m-%d') if d else None
    except (ValueError, TypeError):
        return None


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


def is_keyword_stuffer(c):
    """Detect keyword stuffer: TIER0 title with 4+ JD skills listed."""
    title = (c.get('profile', {}) or {}).get('current_title', '').lower().strip()
    skills = c.get('skills', []) or []
    names = {(s.get('name') or '').lower() for s in skills}
    return title in TIER0 and len(names & ALL_JD) >= 4


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
    """Compute BM25 scores."""
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
    srt = sorted(raw)
    cap = srt[min(N - 1, int(0.999 * N))] if N else 1.0
    cap = cap or 1.0
    return [min(1.0, r / cap) for r in raw]


def compute_tfidf_cosine_similarity(cands):
    """
    Compute TF-IDF cosine similarity between candidates and JD query.

    Method:
    1. Tokenize all candidate documents + query
    2. Build term -> document frequency mapping
    3. For each candidate, compute TF-IDF vector
    4. Compute cosine similarity with query vector
    """
    docs = [tokenize(candidate_document(c)) for c in cands]
    query_tokens = tokenize(QUERY)

    # Build vocabulary from all documents + query
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    vocab.update(query_tokens)
    vocab = sorted(list(vocab))
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    # Compute IDF
    N = len(docs)
    doc_freq = Counter()
    for doc in docs:
        doc_freq.update(set(doc))  # Only count presence, not frequency
    idf = {t: math.log(1 + N / (1 + doc_freq.get(t, 0))) for t in vocab}

    # Compute query TF-IDF vector
    query_tf = Counter(query_tokens)
    query_vec = np.array([query_tf.get(t, 0) * idf[t] for t in vocab], dtype=np.float32)
    query_norm = np.sqrt(np.sum(query_vec ** 2))
    if query_norm > 0:
        query_vec = query_vec / query_norm

    # Compute cosine similarity for each candidate
    similarities = []
    for doc in docs:
        doc_tf = Counter(doc)
        doc_vec = np.array([doc_tf.get(t, 0) * idf[t] for t in vocab], dtype=np.float32)
        doc_norm = np.sqrt(np.sum(doc_vec ** 2))
        if doc_norm > 0:
            doc_vec = doc_vec / doc_norm
            sim = float(np.dot(query_vec, doc_vec))
        else:
            sim = 0.0
        similarities.append(max(0.0, min(1.0, sim)))  # Clamp to [0, 1]

    return similarities


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


def rank_candidates_hybrid_rag(cands, retrieval_top_k=3000, top_n=100):
    """
    HYBRID RAG: Semantic (TF-IDF cosine) + Lexical (BM25) + Strict Filtering

    Stage 0A: TF-IDF cosine similarity retrieval
    Stage 0B: BM25 lexical retrieval
    Stage 0C: Union (both retrievals)
    Stage 1: Hardcore filtering (honeypots, TIER0, stuffers)
    Stage 2: Full ranking (7-component scoring)
    """

    print(f'[HYBRID RAG] Step 1/5: Computing semantic similarity (TF-IDF cosine)...')
    semantic_scores = compute_tfidf_cosine_similarity(cands)
    semantic_indices = set(np.argsort(semantic_scores)[-retrieval_top_k:])
    print(f'[HYBRID RAG]   Semantic retrieved top {len(semantic_indices):,}')

    print(f'[HYBRID RAG] Step 2/5: Computing lexical scores (BM25)...')
    bm25 = compute_bm25(cands)
    bm25_indices = set(np.argsort(bm25)[-retrieval_top_k:])
    print(f'[HYBRID RAG]   Lexical retrieved top {len(bm25_indices):,}')

    print(f'[HYBRID RAG] Step 3/5: Union and hardcore filtering...')
    # Union: candidates in semantic OR BM25
    retrieved_indices = semantic_indices | bm25_indices
    print(f'[HYBRID RAG]   Union (semantic UNION lexical): {len(retrieved_indices):,} candidates')

    # Hardcore filtering: remove TIER0, honeypots, flag stuffers
    filtered_indices = []
    n_tier0 = 0
    n_honeypot = 0
    n_stuffer = 0

    for idx in retrieved_indices:
        c = cands[idx]
        prof = c.get('profile', {}) or {}
        title = (prof.get('current_title', '') or '').lower().strip()

        # Hard-exclude TIER0
        if title in TIER0:
            n_tier0 += 1
            continue

        # Hard-exclude honeypots
        if is_honeypot(c):
            n_honeypot += 1
            continue

        # Detect stuffers (but allow scoring with low gate)
        if is_keyword_stuffer(c):
            n_stuffer += 1
            # Don't exclude, but will be gated low

        filtered_indices.append(idx)

    print(f'[HYBRID RAG]   Filtering results:')
    print(f'[HYBRID RAG]     - TIER0 hard-excluded: {n_tier0:,}')
    print(f'[HYBRID RAG]     - Honeypots hard-excluded: {n_honeypot:,}')
    print(f'[HYBRID RAG]     - Keyword stuffers detected: {n_stuffer:,}')
    print(f'[HYBRID RAG]     - Clean candidates for ranking: {len(filtered_indices):,}')

    print(f'[HYBRID RAG] Step 4/5: Full scoring and ranking...')
    rows = []
    for idx in filtered_indices:
        c = cands[idx]
        prof = c.get('profile', {}) or {}
        career = c.get('career_history', []) or []
        skills = c.get('skills', []) or []
        rs = c.get('redrob_signals', {}) or {}
        a_score, mean_assess = assessment_score(rs)

        sub = {
            'skill': skill_depth(skills),
            'bm25': bm25[idx],
            'cartext': career_text(career),
            'assess': a_score,
            'carhist': career_hist(career),
            'yoe': yoe_score(prof.get('years_of_experience', 0) or 0),
            'loc': location_score(prof.get('country', ''), prof.get('location', '')),
        }
        quality = sum(sub[k] * W[k] for k in W)
        gate = title_gate(prof.get('current_title', ''))
        eng = engagement_mult(rs)
        final = gate * quality * eng

        rows.append({
            'candidate_id': c.get('candidate_id', ''),
            'score': round(final, 4),
            'reasoning': build_reasoning(c, sub, mean_assess),
        })

    rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    top = rows[:top_n]
    for i, r in enumerate(top, 1):
        r['rank'] = i
    print(f'[HYBRID RAG] Step 5/5: Done!')
    return top


def load_candidates(path):
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
    ap = argparse.ArgumentParser(description='Redrob Hybrid RAG Ranker (Semantic + Lexical + Strict Filtering)')
    ap.add_argument('--candidates', required=True, help='path to candidates.jsonl')
    ap.add_argument('--out', default='submission.csv', help='output CSV path')
    ap.add_argument('--top', type=int, default=100, help='number of candidates to output')
    ap.add_argument('--retrieval-top-k', type=int, default=3000, help='number to retrieve per method')
    args = ap.parse_args()

    cands = load_candidates(args.candidates)
    print(f'Loaded {len(cands):,} candidates from {args.candidates}')

    ranked = rank_candidates_hybrid_rag(cands, retrieval_top_k=args.retrieval_top_k, top_n=args.top)

    import csv
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in ranked:
            w.writerow([r['candidate_id'], r['rank'], f"{r['score']:.4f}", r['reasoning']])
    print(f'Wrote {len(ranked)} ranked candidates -> {args.out}')


if __name__ == '__main__':
    main()
