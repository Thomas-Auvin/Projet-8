from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def features_df_from_report(rep: dict) -> pd.DataFrame:
    """
    Retourne un DataFrame 'plat' avec au moins:
    feature, psi, missing_rate_delta, ref_missing_rate, prod_missing_rate, type, ref_n, prod_n
    en supportant plusieurs schémas possibles.
    """
    obj = (
        rep.get("all_features")
        or rep.get("features")
        or rep.get("per_feature")
        or rep.get("feature_stats")
    )

    # 1) Si dict: {feature: {stats...}}
    if isinstance(obj, dict):
        rows = []
        for feat, stats in obj.items():
            if isinstance(stats, dict):
                rows.append({"feature": feat, **stats})
        df = pd.DataFrame(rows)
    # 2) Si liste: [{feature:..., psi:...}, ...]
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame()

    # Fallback minimal: si df vide, on essaie d’agréger top_psi + top_missing_delta
    if df.empty:
        top_psi = rep.get("top_psi", [])
        top_miss = rep.get("top_missing_delta", []) or rep.get("top_missing", [])
        df = pd.DataFrame(top_psi + top_miss).drop_duplicates(subset=["feature"], keep="first")

    # Harmoniser quelques noms de colonnes possibles
    rename_map = {
        "missing_delta": "missing_rate_delta",
        "delta_missing": "missing_rate_delta",
        "ref_missing": "ref_missing_rate",
        "prod_missing": "prod_missing_rate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


st.set_page_config(page_title="P8 Monitoring", layout="wide")
st.title("Monitoring — Projet 8")

# Drift report (prend l'env en priorité)
default_drift = os.getenv("P8_DRIFT_REPORT", "outputs/monitoring/drift_report.json")
path_str = st.text_input("Chemin du drift_report.json", value=str(default_drift))

path = Path(path_str)
if not path.is_absolute():
    # dans Docker, on est en général dans /home/user/app
    path = (Path.cwd() / path).resolve()

if not path.exists():
    st.warning(f"Fichier introuvable: {path}\n\nLance d'abord `monitoring/run_drift.py`.")
    st.stop()

rep = load_report(path)

# Debug utile (tu peux le laisser, ou le virer ensuite)
with st.expander("Debug drift_report.json", expanded=False):
    st.write("Path:", str(path))
    st.write("Keys:", list(rep.keys()))
    st.write("Type all_features:", type(rep.get("all_features")).__name__)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prod rows", rep.get("n_prod_rows", rep.get("prod_rows", 0)))
col2.metric("Ref rows", rep.get("n_ref_rows", rep.get("ref_rows", 0)))
col3.metric("Features", rep.get("features_total", rep.get("n_features", 0)))
col4.metric("Generated (UTC)", rep.get("generated_at_utc", ""))

st.divider()

all_df = features_df_from_report(rep)

# Si toujours pas de colonne psi, on s’arrête proprement avec un message clair
if "psi" not in all_df.columns:
    st.error(
        "Le drift_report.json ne contient pas de colonne 'psi' exploitable dans 'all_features'.\n"
        "Vérifie le format de sortie de `compute_drift()` (idéal: all_features = liste de dicts)."
    )
    with st.expander("Aperçu du report (extrait)", expanded=False):
        st.json(rep)
    st.stop()

# Assurer missing_rate_delta si absent
if "missing_rate_delta" not in all_df.columns:
    all_df["missing_rate_delta"] = 0.0

min_psi = st.slider("Filtre PSI minimal", 0.0, 50.0, 0.1, 0.05)
only_high_missing = st.checkbox("Afficher seulement delta missing >= 5%", value=False)

df_view = all_df.copy()
df_view = df_view[df_view["psi"] >= float(min_psi)]
if only_high_missing:
    df_view = df_view[df_view["missing_rate_delta"].abs() >= 0.05]

df_view = df_view.sort_values("psi", ascending=False)

left, right = st.columns(2)

with left:
    st.subheader("Top PSI")
    st.dataframe(pd.DataFrame(rep.get("top_psi", [])), use_container_width=True, hide_index=True)

with right:
    st.subheader("Top delta missing")
    st.dataframe(
        pd.DataFrame(rep.get("top_missing_delta", []) or rep.get("top_missing", [])),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Toutes les features (filtrées)")
st.dataframe(df_view, use_container_width=True, hide_index=True)

# Bench performances (env en priorité)
bench_default = os.getenv("P8_BENCH_REPORT", "outputs/perf/bench_baseline.json")
bench_path = Path(bench_default)
if not bench_path.is_absolute():
    bench_path = (Path.cwd() / bench_path).resolve()

st.divider()
st.subheader("Bench performances")
st.caption(f"Chemin: {bench_path}")

if not bench_path.exists():
    st.warning("Fichier bench introuvable. Vérifie le volume mount et/ou le chemin.")
else:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    st.json(bench)
