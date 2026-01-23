from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="P8 Monitoring", layout="wide")
st.title("Monitoring drift — Projet 8")

default_path = Path("outputs/monitoring/drift_report.json")
path_str = st.text_input("Chemin du drift_report.json", value=str(default_path))

path = Path(path_str)
if not path.exists():
    st.warning("Fichier introuvable. Lance d'abord `monitoring/run_drift.py`.")
    st.stop()

rep = load_report(path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prod rows", rep.get("n_prod_rows", 0))
col2.metric("Ref rows", rep.get("n_ref_rows", 0))
col3.metric("Features", rep.get("features_total", 0))
col4.metric("Generated (UTC)", rep.get("generated_at_utc", ""))

st.divider()

all_df = pd.DataFrame(rep.get("all_features", []))

min_psi = st.slider("Filtre PSI minimal", 0.0, 1.0, 0.1, 0.01)
only_high_missing = st.checkbox("Afficher seulement delta missing >= 5%", value=False)

df_view = all_df.copy()
df_view = df_view[df_view["psi"] >= min_psi]
if only_high_missing:
    df_view = df_view[df_view["missing_rate_delta"].abs() >= 0.05]

df_view = df_view.sort_values("psi", ascending=False)

left, right = st.columns(2)

with left:
    st.subheader("Top PSI")
    st.dataframe(
        pd.DataFrame(rep.get("top_psi", [])),
        width="stretch",
        hide_index=True,
    )

with right:
    st.subheader("Top delta missing")
    st.dataframe(
        pd.DataFrame(rep.get("top_missing_delta", [])),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.subheader("Toutes les features (filtrées)")
st.dataframe(df_view, width="stretch", hide_index=True)
