from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="P8 Monitoring", layout="wide")
st.title("Monitoring drift — Projet 8")

# ✅ Priorité à l'env var (Docker/HF), fallback local
default_drift = os.getenv("P8_DRIFT_REPORT", "outputs/monitoring/drift_report.json")
path_str = st.text_input("Chemin du drift_report.json", value=default_drift)

path = Path(path_str)

# Petit debug utile (évite la confusion host vs container)
st.caption(f"Resolved path: {path.resolve()} | exists={path.exists()}")

if not path.exists():
    st.warning("Fichier introuvable. Lance d'abord `monitoring/run_drift.py` (ou monte le volume outputs en Docker).")
    st.stop()

rep = load_report(path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prod rows", rep.get("n_prod_rows", 0))
col2.metric("Ref rows", rep.get("n_ref_rows", 0))
col3.metric("Features", rep.get("features_total", 0))
col4.metric("Generated (UTC)", rep.get("generated_at_utc", ""))

st.divider()

# ✅ Tables top (doivent exister dans ton JSON)
left, right = st.columns(2)

with left:
    st.subheader("Top PSI")
    top_psi = pd.DataFrame(rep.get("top_psi", []))
    st.dataframe(top_psi, use_container_width=True, hide_index=True)

with right:
    st.subheader("Top delta missing")
    top_miss = pd.DataFrame(rep.get("top_missing_delta", []))
    st.dataframe(top_miss, use_container_width=True, hide_index=True)

# ✅ Table complète si présente
all_features = rep.get("all_features", [])
if isinstance(all_features, list) and len(all_features) > 0:
    st.divider()
    st.subheader("Toutes les features (filtrées)")
    all_df = pd.DataFrame(all_features)

    # garde-fous colonnes
    if "psi" in all_df.columns:
        min_psi = st.slider("Filtre PSI minimal", 0.0, 1.0, 0.1, 0.01)
        df_view = all_df[all_df["psi"] >= min_psi].copy()
    else:
        st.warning("Le JSON ne contient pas la colonne `psi` dans `all_features`.")
        df_view = all_df.copy()

    if "missing_rate_delta" in df_view.columns:
        only_high_missing = st.checkbox("Afficher seulement delta missing >= 5%", value=False)
        if only_high_missing:
            df_view = df_view[df_view["missing_rate_delta"].abs() >= 0.05]

    if "psi" in df_view.columns:
        df_view = df_view.sort_values("psi", ascending=False)

    st.dataframe(df_view, use_container_width=True, hide_index=True)

st.divider()

# ✅ Bench perf
bench_default = os.getenv("P8_BENCH_REPORT", "outputs/perf/bench_baseline.json")
bench_path_str = st.text_input("Chemin du bench_baseline.json", value=bench_default)
bench_path = Path(bench_path_str)
st.caption(f"Resolved bench path: {bench_path.resolve()} | exists={bench_path.exists()}")

st.subheader("Bench performances")
if not bench_path.exists():
    st.warning("Bench introuvable. Génère `outputs/perf/bench_baseline.json` puis relance.")
else:
    bench = load_report(bench_path)

    c1, c2, c3 = st.columns(3)
    pm = bench.get("predict_ms", {})
    bm = bench.get("batch_per_row_ms", {})
    bt = bench.get("batch_total_ms", {})

    c1.metric("Predict mean (ms)", f"{pm.get('mean', 0):.2f}")
    c1.metric("Predict p95 (ms)", f"{pm.get('p95', 0):.2f}")

    c2.metric("Batch/row mean (ms)", f"{bm.get('mean', 0):.2f}")
    c2.metric("Batch/row p95 (ms)", f"{bm.get('p95', 0):.2f}")

    c3.metric("Batch total mean (ms)", f"{bt.get('mean', 0):.2f}")
    c3.metric("Batch total p95 (ms)", f"{bt.get('p95', 0):.2f}")

    with st.expander("Voir le JSON brut"):
        st.json(bench)
