"""
Multi-tab Streamlit app for testing ranking approaches:
- Tab 1: Standard (score all 100K)
- Tab 2: RAG (BM25 retrieve + score)
- Tab 3: Hybrid RAG TF-IDF (semantic + lexical)
- Tab 4: Hybrid RAG Sentence Transformers (pre-trained + lexical) [Optional]
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
    page_title="Candidate Ranker - TEAMP2R",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Candidate Ranker — TEAMP2R")
st.caption("Redrob Hackathon | Hybrid RAG: Semantic + Lexical Retrieval | Gate-Then-Score Pipeline")

st.divider()

# ── Initialize session state ───────────────────────────────────────────────────
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "source_label" not in st.session_state:
    st.session_state.source_label = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Input Settings")

    # ── Upload section ─────────────────────────────────────────────────────────
    st.write("**Upload a candidate file**")
    st.caption("Supports .json (array) or .jsonl (line-delimited) up to 50 MB")

    uploaded = st.file_uploader(
        "Choose file",
        type=["json", "jsonl"],
        label_visibility="collapsed",
    )

    candidates = []
    source_label = ""

    if uploaded is not None:
        file_size_mb = uploaded.size / (1024 * 1024)
        placeholder = st.empty()
        try:
            placeholder.info(f"Loading {uploaded.name} ({file_size_mb:.1f} MB)...")
            if uploaded.name.endswith(".jsonl"):
                lines = uploaded.read().decode("utf-8").splitlines()
                candidates = [json.loads(l) for l in lines if l.strip()]
            else:
                data = json.loads(uploaded.read().decode("utf-8"))
                candidates = data if isinstance(data, list) else [data]
            source_label = uploaded.name
            st.session_state.candidates = candidates
            st.session_state.source_label = source_label
            placeholder.success(f"Loaded {len(candidates):,} candidates from {uploaded.name}")
        except Exception as e:
            placeholder.error(f"Error: {str(e)[:120]}")

    # ── Full 100K Note ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
> **Want to see the full 100K results?**
> The browser upload limit is ~50 MB.
> For our complete 465 MB pool ranking,
> view the **pre-computed top-100 below** 👇
""")

    show_precomputed = st.button("View Full 100K Results (Top 100)", use_container_width=True)

    # ── Local path (for local use) ─────────────────────────────────────────────
    st.divider()
    with st.expander("Local file path (run locally only)"):
        st.caption("Only works when running locally — not on cloud")
        path = st.text_input(
            "File path",
            value="",
            placeholder="e.g., C:\\path\\candidates.jsonl",
            label_visibility="collapsed"
        )
        if path and path.strip():
            if not os.path.exists(path):
                st.error(f"File not found: `{path}`")
            else:
                size_mb = os.path.getsize(path) / 1e6
                st.info(f"Found ({size_mb:.1f} MB)")
                if st.button("Load from disk", use_container_width=True):
                    with st.spinner(f"Loading {size_mb:.1f} MB..."):
                        candidates = rank_standard.load_candidates(path)
                        source_label = path
                    st.session_state.candidates = candidates
                    st.session_state.source_label = source_label
                    st.success(f"Loaded {len(candidates):,} candidates")

# Use candidates from session state if available
if not candidates and st.session_state.candidates:
    candidates = st.session_state.candidates
    source_label = st.session_state.source_label

# ── Sidebar controls shown only when candidates loaded ─────────────────────────
if candidates:
    with st.sidebar:
        st.success(f"**{len(candidates):,}** candidates from `{source_label}`")
        st.divider()
        st.subheader("Parameters")
        col_a, col_b = st.columns(2)
        with col_a:
            top_n = st.number_input("Top N", min_value=1, max_value=100, value=100)
        with col_b:
            retrieval_k = st.number_input("Retrieval K", min_value=500, max_value=10000, value=3000, step=500)
else:
    top_n = 100
    retrieval_k = 3000

# ── PRE-COMPUTED RESULTS SECTION ──────────────────────────────────────────────
if show_precomputed:
    st.subheader("Pre-computed Results: Full 100K Candidate Pool (Top 100)")
    st.caption("Generated by Hybrid RAG pipeline on all 100,000 candidates | Runtime: ~88 seconds")

    try:
        df_results = pd.read_csv("TEAMP2R.csv")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pool Size", "100,000")
        with col2:
            st.metric("Honeypots Excluded", "53")
        with col3:
            st.metric("Stuffers Blocked", "8,101")
        with col4:
            st.metric("Top Score", f"{df_results['score'].max():.4f}")

        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "rank": st.column_config.NumberColumn("Rank"),
                "candidate_id": st.column_config.TextColumn("Candidate ID"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            }
        )

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            buf = io.StringIO()
            df_results.to_csv(buf, index=False)
            st.download_button(
                "Download as CSV",
                data=buf.getvalue(),
                file_name="TEAMP2R.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name="Top 100 Candidates")
            st.download_button(
                "Download as XLSX",
                data=buf_xlsx.getvalue(),
                file_name="TEAMP2R.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except FileNotFoundError:
        st.error("TEAMP2R.csv not found in repo. Please check repository.")

    st.divider()

# ── No candidates loaded message ───────────────────────────────────────────────
if not candidates:
    st.info("""
    **How to use this app:**

    - **Upload a small file** (up to ~50 MB) using the sidebar uploader
    - **View pre-computed full results** → click "View Full 100K Results" in sidebar
    - **Run locally** for large files (465 MB) using the local path option

    **Supported formats:** `.json` (array) or `.jsonl` (line-delimited JSON)
    """)
    st.stop()

# ── Helper: keyword stuffer detection ─────────────────────────────────────────
def is_stuffer(c):
    title = (c.get("profile", {}) or {}).get("current_title", "").lower().strip()
    names = {(s.get("name") or "").lower() for s in (c.get("skills", []) or [])}
    return title in rank_standard.TIER0 and len(names & rank_standard.ALL_JD) >= 4

# ── Helper: display ranked results ────────────────────────────────────────────
def display_results(ranked, method_name, timing_info):
    n_honeypot = sum(1 for c in candidates if rank_standard.is_honeypot(c))
    n_stuffer  = sum(1 for c in candidates if is_stuffer(c))
    by_id      = {c.get("candidate_id"): c for c in candidates}
    hp_in_top  = sum(1 for r in ranked if rank_standard.is_honeypot(by_id.get(r["candidate_id"], {})))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidates scored", f"{len(candidates):,}")
    with col2:
        st.metric("Honeypots detected", n_honeypot)
    with col3:
        st.metric("Keyword stuffers", n_stuffer)
    with col4:
        st.metric("Honeypots in top 100", hp_in_top, delta="Clean" if hp_in_top == 0 else "Warning")

    st.info(f"**{method_name}** | {timing_info}")
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

    st.dataframe(
        pd.DataFrame(disp),
        use_container_width=True,
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.4f")}
    )

    sub_df = pd.DataFrame([{
        "candidate_id": r["candidate_id"],
        "rank": r["rank"],
        "score": f"{r['score']:.4f}",
        "reasoning": r["reasoning"],
    } for r in ranked])

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        buf = io.StringIO()
        sub_df.to_csv(buf, index=False)
        st.download_button(
            "Download CSV",
            data=buf.getvalue(),
            file_name=f"{method_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
            sub_df.to_excel(writer, index=False, sheet_name="Rankings")
        st.download_button(
            "Download XLSX",
            data=buf_xlsx.getvalue(),
            file_name=f"{method_name.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_list = [
    "📊 Standard",
    "⚡ RAG (BM25)",
    "🔀 Hybrid TF-IDF",
    "🧠 Hybrid Transformers",
]
tab1, tab2, tab3, tab4 = st.tabs(tab_list)

with tab1:
    st.write("Score **all** candidates with full 7-component pipeline.")
    if st.button("Run Standard", use_container_width=True, key="btn_standard"):
        start = datetime.now()
        with st.spinner(f"Scoring all {len(candidates):,} candidates..."):
            ranked = rank_standard.rank_candidates(candidates, top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Standard", f"Time: {elapsed:.1f}s")

with tab2:
    st.write("Retrieve top candidates by **BM25**, then score (fastest).")
    if st.button("Run RAG (BM25)", use_container_width=True, key="btn_rag"):
        start = datetime.now()
        with st.spinner(f"Retrieving top {retrieval_k:,} by BM25..."):
            ranked = rank_rag.rank_candidates_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "RAG BM25", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,}")

with tab3:
    st.write("**Dual retrieval** (TF-IDF semantic + BM25 lexical), then score.")
    if st.button("Run Hybrid TF-IDF", use_container_width=True, key="btn_hybrid"):
        start = datetime.now()
        with st.spinner("Running dual retrieval (semantic + lexical)..."):
            ranked = rank_hybrid_rag.rank_candidates_hybrid_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Hybrid TF-IDF", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,} each")

with tab4:
    if HAS_SENTENCE_TRANSFORMERS:
        st.write("**Pre-trained embeddings** (all-MiniLM-L6-v2) + BM25 (best semantic quality).")
        if st.button("Run Hybrid Transformers", use_container_width=True, key="btn_st"):
            start = datetime.now()
            with st.spinner("Running Sentence Transformers + BM25..."):
                ranked = rank_sentence_transformers_rag.rank_candidates_sentence_transformers_rag(
                    candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n)
                )
            elapsed = (datetime.now() - start).total_seconds()
            display_results(ranked, "Hybrid Transformers", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,} each")
    else:
        st.warning("""
        **Not available on Streamlit Cloud**

        `sentence-transformers` has dependency conflicts on cloud platforms.

        To use locally:
        ```
        pip install sentence-transformers
        streamlit run app_multi.py
        ```
        Tabs 1-3 work perfectly on cloud.
        """)

st.divider()
with st.expander("Method Comparison"):
    st.dataframe(pd.DataFrame({
        "Method":    ["Standard", "RAG (BM25)", "Hybrid TF-IDF", "Hybrid Transformers"],
        "Time":      ["~73s",     "~44s",        "~150s",          "~80s"],
        "Retrieval": ["All 100K", "Lexical 3K",  "Semantic+Lex 3K","Pre-trained+Lex 3K"],
        "Best For":  ["Full audit","Speed",       "Max confidence", "Best semantic"],
        "Cloud":     ["Yes",      "Yes",          "Yes",            "Local only"],
    }), use_container_width=True, hide_index=True)
