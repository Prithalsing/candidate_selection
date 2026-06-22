import io
import json
import os

import pandas as pd
import streamlit as st

import rank  # the hybrid ranker (same logic as the CLI / notebooks)

st.set_page_config(page_title="Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Candidate Ranker")
st.caption("Redrob Hackathon — Senior AI Engineer JD  •  hybrid: rules + BM25 + assessment")

st.divider()


# ── helpers (reuse rank.py logic, no duplication) ──────────────────────────────
def is_stuffer(c):
    """Tier-0 title that has stuffed 4+ JD skills."""
    title = (c.get("profile", {}) or {}).get("current_title", "").lower().strip()
    names = {(s.get("name") or "").lower() for s in (c.get("skills", []) or [])}
    return title in rank.TIER0 and len(names & rank.ALL_JD) >= 4


def parse_upload(file):
    content = file.read().decode("utf-8")
    if file.name.endswith(".jsonl"):
        return [json.loads(l) for l in content.splitlines() if l.strip()]
    data = json.loads(content)
    return data if isinstance(data, list) else [data]


# ── input: upload (small files) OR local path (large files) ────────────────────
mode = st.radio(
    "Input source",
    ["Upload file (≤200 MB)", "Local file path (any size)"],
    horizontal=True,
    help="Browser upload is capped and memory-heavy. For the full 487 MB file, "
         "use a local path — it streams from disk and skips the browser entirely.",
)

candidates = []
source_label = ""

if mode.startswith("Upload"):
    uploaded = st.file_uploader("Upload candidate file (.json or .jsonl)", type=["json", "jsonl"])
    if uploaded:
        try:
            candidates = parse_upload(uploaded)
            source_label = uploaded.name
        except Exception as e:
            st.error(f"Could not parse file: {e}")
else:
    path = st.text_input("Path to candidates file on this machine",
                         value="./candidates.json",
                         help="e.g. ./candidates.json or ./candidates.jsonl (array or JSON-lines)")
    if path:
        if not os.path.exists(path):
            st.warning(f"File not found: `{path}`")
        else:
            size_mb = os.path.getsize(path) / 1e6
            if size_mb > 250:
                st.info(f"`{path}` is {size_mb:.0f} MB — large files need plenty of RAM "
                        f"(~4 GB for the full 100K). This works locally; not on low-RAM hosts.")
            if st.button("📂 Load from disk", use_container_width=True):
                with st.spinner(f"Loading {path} ({size_mb:.0f} MB)…"):
                    candidates = rank.load_candidates(path)   # auto-detects array vs jsonl
                    source_label = path
                st.session_state["loaded"] = candidates
                st.session_state["source"] = path
    # keep loaded data across reruns
    if not candidates and st.session_state.get("loaded"):
        candidates = st.session_state["loaded"]
        source_label = st.session_state.get("source", "")

if not candidates:
    st.info("Upload a `.json`/`.jsonl` file, or point to one on disk. "
            "Try `sample_candidates.json` (small) or `candidates.json` (full, via path mode).")
    st.stop()

st.success(f"Loaded **{len(candidates):,}** candidates from `{source_label}`")


# ── controls ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns([1, 3])
with col_a:
    top_n = st.number_input("Top N to return", min_value=1, max_value=100,
                            value=min(100, len(candidates)))
run = st.button("🚀 Rank candidates", type="primary", use_container_width=True)

if not run:
    st.divider()
    st.subheader("Preview")
    prev = [{
        "ID": c.get("candidate_id", "—"),
        "Title": (c.get("profile", {}) or {}).get("current_title", "—"),
        "YoE": (c.get("profile", {}) or {}).get("years_of_experience", "—"),
        "Country": (c.get("profile", {}) or {}).get("country", "—"),
    } for c in candidates[:100]]
    st.dataframe(pd.DataFrame(prev), use_container_width=True, hide_index=True)
    st.stop()


# ── run the hybrid pipeline ─────────────────────────────────────────────────────
with st.spinner(f"Running hybrid ranker on {len(candidates):,} candidates…"):
    ranked = rank.rank_candidates(candidates, top_n=int(top_n))
    n_honeypot = sum(1 for c in candidates if rank.is_honeypot(c))
    n_stuffer = sum(1 for c in candidates if is_stuffer(c))

by_id = {c.get("candidate_id"): c for c in candidates}

st.divider()

# ── safety metrics ──────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidates scored", f"{len(candidates):,}")
m2.metric("Honeypots excluded", n_honeypot, help="Impossible profiles forced to score 0")
m3.metric("Keyword stuffers", n_stuffer, help="Tier-0 titles with 4+ stuffed AI skills (gated out)")
hp_in_top = sum(1 for r in ranked if rank.is_honeypot(by_id.get(r["candidate_id"], {})))
m4.metric("Honeypots in top N", hp_in_top, help="Must be 0")

if hp_in_top == 0:
    st.success("✅ Top N is clean — no honeypots; keyword stuffers gated out by the title multiplier.")

# ── ranked table (enriched for display) ─────────────────────────────────────────
st.subheader(f"Top {len(ranked)} ranked candidates")
disp = []
for r in ranked:
    c = by_id.get(r["candidate_id"], {})
    p = c.get("profile", {}) or {}
    disp.append({
        "Rank": r["rank"],
        "Candidate ID": r["candidate_id"],
        "Title": p.get("current_title", "—"),
        "YoE": p.get("years_of_experience", "—"),
        "Country": p.get("country", "—"),
        "Score": r["score"],
        "Why": r["reasoning"],
    })
disp_df = pd.DataFrame(disp)
st.dataframe(disp_df, use_container_width=True, hide_index=True,
            column_config={"Score": st.column_config.NumberColumn(format="%.4f")})

# ── download in official 4-column submission format ─────────────────────────────
sub_df = pd.DataFrame([{
    "candidate_id": r["candidate_id"],
    "rank": r["rank"],
    "score": f"{r['score']:.4f}",
    "reasoning": r["reasoning"],
} for r in ranked])
buf = io.StringIO()
sub_df.to_csv(buf, index=False)
st.download_button("⬇️ Download submission.csv (official format)",
                   data=buf.getvalue(), file_name="submission.csv",
                   mime="text/csv", use_container_width=True)

with st.expander("How the score is computed"):
    st.markdown(
        "**`final = title_gate × quality × engagement`** (0 if honeypot)\n\n"
        "- **Stage 0 — honeypot exclusion:** date / YoE / skill-duration / expert-assessment impossibilities → score 0\n"
        "- **Stage 1 — title gate (×1.0…0.05):** keyword-stuffer Tier-0 profiles can't reach the top\n"
        "- **Stage 2 — quality blend:** skill 0.25, BM25 0.20, career-text 0.18, assessment 0.17, "
        "career-history 0.10, YoE 0.06, location 0.04\n"
        "- **Stage 3 — engagement (×0.50…1.00):** response rate, recency, notice period, open-to-work"
    )
