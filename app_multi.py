"""
Multi-tab Streamlit app for testing ranking approaches:
- Tab 1: Standard (score all 100K)
- Tab 2: RAG (BM25 retrieve + score)
- Tab 3: Hybrid RAG TF-IDF (semantic + lexical)
- Tab 4: Hybrid RAG Sentence Transformers (pre-trained + lexical) [Optional - requires sentence-transformers]

Each tab shows the same interface but uses a different ranking algorithm.
Users can upload/select candidates file and see results for all methods.
"""

import io
import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Import core ranking modules (required)
import rank as rank_standard
import rank_rag
import rank_hybrid_rag

# Try to import Sentence Transformers (optional for cloud deployment)
try:
    import rank_sentence_transformers_rag
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

st.set_page_config(
    page_title="Candidate Ranker - All Methods",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Candidate Ranker - Compare All Methods")
st.caption("Redrob Hackathon — Test Standard, RAG, Hybrid TF-IDF" +
           (" + Hybrid Sentence Transformers" if HAS_SENTENCE_TRANSFORMERS else ""))

st.divider()

# ── Sidebar: Input controls ────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Input Settings")

    mode = st.radio(
        "Input source",
        ["Upload file (≤200 MB)", "Local file path (any size)"],
        help="Browser upload is capped. For 487 MB file, use local path.",
    )

    candidates = []
    source_label = ""

    if mode.startswith("Upload"):
        uploaded = st.file_uploader("Upload candidate file (.json or .jsonl)", type=["json", "jsonl"])
        if uploaded:
            try:
                content = uploaded.read().decode("utf-8")
                if uploaded.name.endswith(".jsonl"):
                    candidates = [json.loads(l) for l in content.splitlines() if l.strip()]
                else:
                    data = json.loads(content)
                    candidates = data if isinstance(data, list) else [data]
                source_label = uploaded.name
            except Exception as e:
                st.error(f"Could not parse file: {e}")
    else:
        path = st.text_input("Path to candidates file", value="./candidates.json")
        if path:
            if not os.path.exists(path):
                st.warning(f"File not found: `{path}`")
            else:
                size_mb = os.path.getsize(path) / 1e6
                st.info(f"`{path}` is {size_mb:.0f} MB")
                if st.button("Load from disk", width='stretch'):
                    with st.spinner(f"Loading {path} ({size_mb:.0f} MB)..."):
                        candidates = rank_standard.load_candidates(path)
                        source_label = path
                    st.session_state["loaded"] = candidates
                    st.session_state["source"] = path

        if not candidates and st.session_state.get("loaded"):
            candidates = st.session_state["loaded"]
            source_label = st.session_state.get("source", "")

if not candidates:
    st.info("Upload a `.json`/`.jsonl` file or point to a local path to begin. Try `sample_candidates.json` or `candidates.json`.")
    st.stop()

st.sidebar.success(f"Loaded **{len(candidates):,}** candidates from `{source_label}`")

# ── Controls ───────────────────────────────────────────────────────────────────
col_a, col_b = st.sidebar.columns([1, 1])
with col_a:
    top_n = st.number_input("Top N", min_value=1, max_value=100, value=100)
with col_b:
    retrieval_k = st.number_input("Retrieval K (RAG/Hybrid)", min_value=1000, max_value=10000, value=3000, step=500)

# ── Tabs for each method ───────────────────────────────────────────────────────
tab_list = [
    "📊 Standard (All 100K)",
    "⚡ RAG (BM25)",
    "🔀 Hybrid (TF-IDF)",
]
if HAS_SENTENCE_TRANSFORMERS:
    tab_list.append("🧠 Hybrid (Sentence Transformers)")

tabs = st.tabs(tab_list)
tab1, tab2, tab3 = tabs[0], tabs[1], tabs[2]
tab4 = tabs[3] if HAS_SENTENCE_TRANSFORMERS else None

# Helper function to detect keyword stuffer (from app.py)
def is_stuffer(c):
    """Tier-0 title that has stuffed 4+ JD skills."""
    title = (c.get("profile", {}) or {}).get("current_title", "").lower().strip()
    names = {(s.get("name") or "").lower() for s in (c.get("skills", []) or [])}
    return title in rank_standard.TIER0 and len(names & rank_standard.ALL_JD) >= 4

# Helper function
def display_results(ranked, method_name, timing_info):
    """Display ranking results in consistent format."""
    n_honeypot = sum(1 for c in candidates if rank_standard.is_honeypot(c))
    n_stuffer = sum(1 for c in candidates if is_stuffer(c))

    by_id = {c.get("candidate_id"): c for c in candidates}
    hp_in_top = sum(1 for r in ranked if rank_standard.is_honeypot(by_id.get(r["candidate_id"], {})))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidates scored", f"{len(candidates):,}")
    with col2:
        st.metric("Honeypots excluded", n_honeypot)
    with col3:
        st.metric("Keyword stuffers", n_stuffer)
    with col4:
        st.metric("Honeypots in top N", hp_in_top, delta="✅ Clean" if hp_in_top == 0 else "⚠️ Problem")

    st.info(f"**{method_name}** | {timing_info}")

    # Display ranked table
    st.subheader(f"Top {len(ranked)} Ranked Candidates")
    disp = []
    for r in ranked:
        c = by_id.get(r["candidate_id"], {})
        p = c.get("profile", {}) or {}
        disp.append({
            "Rank": r["rank"],
            "ID": r["candidate_id"],
            "Title": p.get("current_title", "—"),
            "YoE": p.get("years_of_experience", "—"),
            "Score": r["score"],
            "Reasoning": r["reasoning"],
        })

    disp_df = pd.DataFrame(disp)
    st.dataframe(disp_df, width='stretch', hide_index=True,
                column_config={"Score": st.column_config.NumberColumn(format="%.4f")})

    # Download button
    sub_df = pd.DataFrame([{
        "candidate_id": r["candidate_id"],
        "rank": r["rank"],
        "score": f"{r['score']:.4f}",
        "reasoning": r["reasoning"],
    } for r in ranked])
    buf = io.StringIO()
    sub_df.to_csv(buf, index=False)
    st.download_button(
        f"⬇️ Download {method_name} CSV",
        data=buf.getvalue(),
        file_name=f"submission_{method_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        width='stretch'
    )

# ── TAB 1: Standard Mode ───────────────────────────────────────────────────────
with tab1:
    if st.button("🚀 Run Standard Mode", width='stretch', key="btn_standard"):
        start = datetime.now()
        with st.spinner(f"Scoring all {len(candidates):,} candidates..."):
            ranked = rank_standard.rank_candidates(candidates, top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Standard (All 100K)", f"Time: {elapsed:.1f}s")

# ── TAB 2: RAG Mode ────────────────────────────────────────────────────────────
with tab2:
    if st.button("⚡ Run RAG Mode (BM25)", width='stretch', key="btn_rag"):
        start = datetime.now()
        with st.spinner(f"Retrieving top {retrieval_k:,} by BM25..."):
            ranked = rank_rag.rank_candidates_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "RAG (BM25)", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,}")

# ── TAB 3: Hybrid RAG TF-IDF ───────────────────────────────────────────────────
with tab3:
    if st.button("🔀 Run Hybrid RAG (TF-IDF)", width='stretch', key="btn_hybrid_tfidf"):
        start = datetime.now()
        with st.spinner(f"Hybrid: Semantic (TF-IDF) + Lexical (BM25)..."):
            ranked = rank_hybrid_rag.rank_candidates_hybrid_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Hybrid (TF-IDF + BM25)", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,} each")

# ── TAB 4: Hybrid RAG Sentence Transformers ────────────────────────────────────
if HAS_SENTENCE_TRANSFORMERS:
    with tab4:
        st.info("Using pre-trained Sentence Transformers (all-MiniLM-L6-v2) for best semantic matching. First run will download model (~22MB).")
        if st.button("🧠 Run Hybrid RAG (Sentence Transformers)", width='stretch', key="btn_hybrid_st"):
            start = datetime.now()
            with st.spinner(f"Hybrid: Semantic (Transformers) + Lexical (BM25)..."):
                ranked = rank_sentence_transformers_rag.rank_candidates_sentence_transformers_rag(
                    candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n)
                )
            elapsed = (datetime.now() - start).total_seconds()
            display_results(ranked, "Hybrid (Sentence Transformers + BM25)", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,} each")
else:
    with tab4:
        st.warning("""
        ⚠️ **Sentence Transformers not available on Streamlit Cloud**

        This tab requires the `sentence-transformers` library, which has dependency conflicts on cloud platforms.

        **To use this mode:**
        - Run locally: `streamlit run app_multi.py` with `pip install sentence-transformers`
        - The first 3 tabs (Standard, RAG, Hybrid TF-IDF) work fine on the cloud ✅

        **For full 4-method comparison:**
        Clone the repo and run locally with all dependencies installed.
        """)

st.divider()

# ── Comparison table ───────────────────────────────────────────────────────────
with st.expander("Comparison: Which method to use?"):
    comparison_df = pd.DataFrame({
        "Method": ["Standard", "RAG (BM25)", "Hybrid (TF-IDF)", "Hybrid (Transformers)"],
        "Time": ["~73s", "~44s", "~150s", "~80s"],
        "Retrieval": ["None (all)", "Lexical only", "Semantic + Lexical", "Semantic + Lexical"],
        "Quality": ["Complete", "Same as Standard", "Same as Standard", "Same as Standard"],
        "Cloud": ["✅", "✅", "✅", "⚠️ Local only"],
    })
    st.dataframe(comparison_df, width='stretch', hide_index=True)

    st.markdown("""
    - **Standard**: Score everyone. Best for understanding full distribution.
    - **RAG**: Fast. One retrieval signal (keywords). Best for production speed.
    - **Hybrid TF-IDF**: Two retrieval signals (concepts + keywords), slower but offline.
    - **Hybrid Transformers**: Best semantic understanding (pre-trained), requires local install.
    """)
