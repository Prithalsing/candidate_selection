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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Candidate Ranker - Compare All Methods")
st.caption("Redrob Hackathon — Test Standard, RAG, Hybrid TF-IDF" +
           (" + Hybrid Sentence Transformers" if HAS_SENTENCE_TRANSFORMERS else ""))

st.divider()

# ── Initialize session state ───────────────────────────────────────────────────
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "source_label" not in st.session_state:
    st.session_state.source_label = ""

# ── Sidebar: Input controls ────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Input Settings")

    mode = st.radio(
        "Input source",
        ["Upload file (≤500 MB)", "Local file path (any size)"],
        help="Upload: Browser handles file. Local: Must exist on server.",
        index=0
    )

    candidates = []
    source_label = ""

    # UPLOAD MODE: Handle file upload with streaming
    if mode.startswith("Upload"):
        st.write("**Upload a candidate file** (.json or .jsonl)")
        uploaded = st.file_uploader(
            "Choose file",
            type=["json", "jsonl"],
            label_visibility="collapsed",
            help="JSON array or newline-delimited JSON"
        )

        if uploaded is not None:
            file_size_mb = uploaded.size / (1024 * 1024)
            progress_placeholder = st.empty()

            try:
                progress_placeholder.info(f"Processing {uploaded.name} ({file_size_mb:.1f} MB)...")

                # Read file in chunks to avoid memory issues
                if uploaded.name.endswith(".jsonl"):
                    # JSON-lines: stream line by line
                    lines = uploaded.read().decode("utf-8").splitlines()
                    candidates = []
                    for i, line in enumerate(lines):
                        if line.strip():
                            try:
                                candidates.append(json.loads(line))
                            except json.JSONDecodeError:
                                st.warning(f"Could not parse line {i+1}, skipping")
                                continue
                else:
                    # JSON array: read entire and parse
                    content = uploaded.read().decode("utf-8")
                    data = json.loads(content)
                    candidates = data if isinstance(data, list) else [data]

                source_label = uploaded.name

                # Save to session state
                st.session_state.candidates = candidates
                st.session_state.source_label = source_label

                progress_placeholder.success(f"Loaded {len(candidates):,} candidates from {uploaded.name}")

            except json.JSONDecodeError as e:
                progress_placeholder.error(f"Invalid JSON format: {str(e)[:100]}")
                st.stop()
            except UnicodeDecodeError:
                progress_placeholder.error("File encoding error. Please use UTF-8 encoded file.")
                st.stop()
            except Exception as e:
                progress_placeholder.error(f"Error processing file: {str(e)[:100]}")
                st.stop()

    # LOCAL FILE MODE: Load from disk path
    else:
        st.write("**Enter path to candidates file**")
        path = st.text_input(
            "File path",
            value="",
            placeholder="e.g., /path/to/candidates.json or C:\\path\\candidates.jsonl",
            label_visibility="collapsed"
        )

        if path and path.strip():
            if not os.path.exists(path):
                st.error(f"File not found: `{path}`")
            else:
                try:
                    size_mb = os.path.getsize(path) / 1e6
                    st.info(f"Found: `{path}` ({size_mb:.1f} MB)")

                    if st.button("Load from disk", width='stretch', use_container_width=True):
                        with st.spinner(f"Loading {path} ({size_mb:.1f} MB)..."):
                            candidates = rank_standard.load_candidates(path)
                            source_label = path

                        # Save to session state
                        st.session_state.candidates = candidates
                        st.session_state.source_label = source_label
                        st.success(f"Loaded {len(candidates):,} candidates")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)[:100]}")

# Use candidates from session state if available
if not candidates and st.session_state.candidates:
    candidates = st.session_state.candidates
    source_label = st.session_state.source_label

# Stop if no candidates loaded
if not candidates:
    st.info("""
    **How to get started:**

    1. Choose upload mode: "Upload file" or "Local file path"
    2. Upload a `.json` (array) or `.jsonl` (line-delimited) file
    3. File can be up to 500MB
    4. Click "Run" button on any tab
    5. See rankings and download results

    **Example files:**
    - `sample_candidates.json` - Small test file
    - `candidates.json` - Full 100K pool (500MB)
    """)
    st.stop()

# Display loaded candidates
with st.sidebar:
    st.success(f"**{len(candidates):,}** candidates loaded from `{source_label}`")

# ── Controls ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Ranking Parameters")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        top_n = st.number_input("Top N", min_value=1, max_value=100, value=100)
    with col_b:
        retrieval_k = st.number_input("Retrieval K", min_value=1000, max_value=10000, value=3000, step=500)

# ── Tabs for each method ───────────────────────────────────────────────────────
tab_list = [
    "📊 Standard (All 100K)",
    "⚡ RAG (BM25)",
    "🔀 Hybrid (TF-IDF)",
    "🧠 Hybrid (Sentence Transformers)",
]

tabs = st.tabs(tab_list)
tab1, tab2, tab3, tab4 = tabs

# Helper function to detect keyword stuffer
def is_stuffer(c):
    """Tier-0 title that has stuffed 4+ JD skills."""
    title = (c.get("profile", {}) or {}).get("current_title", "").lower().strip()
    names = {(s.get("name") or "").lower() for s in (c.get("skills", []) or [])}
    return title in rank_standard.TIER0 and len(names & rank_standard.ALL_JD) >= 4

# Helper function to display results
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
        st.metric("Honeypots detected", n_honeypot)
    with col3:
        st.metric("Keyword stuffers", n_stuffer)
    with col4:
        status = "Clean" if hp_in_top == 0 else "Warning"
        st.metric("Honeypots in top 100", hp_in_top, delta=status)

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
        f"Download {method_name} CSV",
        data=buf.getvalue(),
        file_name=f"submission_{method_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        width='stretch'
    )

# ── TAB 1: Standard Mode ───────────────────────────────────────────────────────
with tab1:
    st.write("Score all candidates with full 7-component pipeline (comprehensive but slower)")
    if st.button("Run Standard Mode", width='stretch', key="btn_standard", use_container_width=True):
        from datetime import datetime
        start = datetime.now()
        with st.spinner(f"Scoring all {len(candidates):,} candidates..."):
            ranked = rank_standard.rank_candidates(candidates, top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Standard (All 100K)", f"Time: {elapsed:.1f}s")

# ── TAB 2: RAG Mode ────────────────────────────────────────────────────────────
with tab2:
    st.write("Retrieve top candidates by BM25, then score (faster)")
    if st.button("Run RAG Mode (BM25)", width='stretch', key="btn_rag", use_container_width=True):
        from datetime import datetime
        start = datetime.now()
        with st.spinner(f"Retrieving top {retrieval_k:,} by BM25..."):
            ranked = rank_rag.rank_candidates_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "RAG (BM25)", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,}")

# ── TAB 3: Hybrid RAG TF-IDF ───────────────────────────────────────────────────
with tab3:
    st.write("Dual retrieval (semantic + lexical) then score (most thorough)")
    if st.button("Run Hybrid RAG (TF-IDF)", width='stretch', key="btn_hybrid_tfidf", use_container_width=True):
        from datetime import datetime
        start = datetime.now()
        with st.spinner(f"Hybrid: Semantic (TF-IDF) + Lexical (BM25)..."):
            ranked = rank_hybrid_rag.rank_candidates_hybrid_rag(candidates, retrieval_top_k=int(retrieval_k), top_n=int(top_n))
        elapsed = (datetime.now() - start).total_seconds()
        display_results(ranked, "Hybrid (TF-IDF + BM25)", f"Time: {elapsed:.1f}s | Retrieved: {retrieval_k:,} each")

# ── TAB 4: Hybrid RAG Sentence Transformers ────────────────────────────────────
if HAS_SENTENCE_TRANSFORMERS:
    with tab4:
        st.write("Pre-trained semantic embeddings + BM25 (best semantic understanding)")
        if st.button("Run Hybrid RAG (Sentence Transformers)", width='stretch', key="btn_hybrid_st", use_container_width=True):
            from datetime import datetime
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
        **Sentence Transformers not available on Streamlit Cloud**

        This method requires the `sentence-transformers` library, which has dependency conflicts on cloud.

        **Local installation:** Run `pip install sentence-transformers` then use locally.

        **For cloud:** Use Tab 1-3 (all work perfectly on cloud).
        """)

st.divider()

# ── Comparison table ───────────────────────────────────────────────────────────
with st.expander("Method Comparison"):
    comparison_df = pd.DataFrame({
        "Method": ["Standard", "RAG (BM25)", "Hybrid (TF-IDF)", "Hybrid (Transformers)"],
        "Time": ["~73s", "~44s", "~150s", "~80s"],
        "Retrieval": ["None (all)", "Lexical only", "Semantic + Lexical", "Pre-trained + Lexical"],
        "Best For": ["Complete audit", "Speed", "Max confidence", "Best semantic"],
        "Cloud": ["Yes", "Yes", "Yes", "Local only"],
    })
    st.dataframe(comparison_df, width='stretch', hide_index=True, use_container_width=True)
