import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Candidate Ranker")
st.caption("Redrob Hackathon — Senior AI Engineer JD")

st.divider()

uploaded = st.file_uploader(
    "Upload candidate file (.json or .jsonl)",
    type=["json", "jsonl"],
)

candidates = []

if uploaded:
    try:
        content = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(l) for l in content.splitlines() if l.strip()]
        else:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]

        if len(candidates) > 100:
            st.warning(f"Showing first 100 of {len(candidates)} candidates.")
            candidates = candidates[:100]

        st.success(f"Loaded **{len(candidates)}** candidates from `{uploaded.name}`")

    except Exception as e:
        st.error(f"Could not parse file: {e}")

if candidates:
    st.divider()
    st.subheader("Candidates Preview")

    rows = []
    for c in candidates:
        p = c.get("profile", {})
        rows.append({
            "ID":       c.get("candidate_id", "—"),
            "Name":     p.get("anonymized_name", "—"),
            "Title":    p.get("current_title", "—"),
            "YoE":      p.get("years_of_experience", "—"),
            "Location": p.get("location", "—"),
            "Country":  p.get("country", "—"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.info("Upload a .json or .jsonl file to preview candidates.")
