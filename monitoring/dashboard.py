from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# -------- utils --------
def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def show_df(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    # compat Streamlit: certains ont "width", d'autres encore "use_container_width"
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def features_df_from_report(rep: dict[str, Any]) -> pd.DataFrame:
    """
    Supporte plusieurs schémas possibles:
    - all_features: list[dict]
    - all_features: dict[feature -> dict]
    - features/per_feature/feature_stats: idem
    """
    obj = (
        rep.get("all_features")
        or rep.get("features")
        or rep.get("per_feature")
        or rep.get("feature_stats")
    )

    if isinstance(obj, dict):
        rows: list[dict[str, Any]] = []
        for feat, stats in obj.items():
            if isinstance(stats, dict):
                rows.append({"feature": feat, **stats})
        df = pd.DataFrame(rows)
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame()

    # Harmoniser colonnes (si variations)
    rename_map = {
        "missing_delta": "missing_rate_delta",
        "delta_missing": "missing_rate_delta",
        "ref_missing": "ref_missing_rate",
        "prod_missing": "prod_missing_rate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


def bench_metrics(bench: dict[str, Any]) -> dict[str, float]:
    pm = bench.get("predict_ms", {}) or {}
    bt = bench.get("batch_total_ms", {}) or {}
    br = bench.get("batch_per_row_ms", {}) or {}

    def g(d: dict[str, Any], k: str) -> float:
        try:
            return float(d.get(k, 0.0) or 0.0)
        except Exception:
            return 0.0

    return {
        "predict_mean": g(pm, "mean"),
        "predict_p95": g(pm, "p95"),
        "batch_total_mean": g(bt, "mean"),
        "batch_total_p95": g(bt, "p95"),
        "batch_row_mean": g(br, "mean"),
        "batch_row_p95": g(br, "p95"),
    }


def pct_delta(cur: float, base: float) -> float | None:
    if base <= 0:
        return None
    return (cur - base) / base * 100.0


# -------- UI --------
st.set_page_config(page_title="P8 Monitoring", layout="wide")
st.title("Monitoring — Projet 8")

tab_drift, tab_bench = st.tabs(["Drift", "Bench"])


# =========================
# Drift tab
# =========================
with tab_drift:
    st.subheader("Drift report")

    default_drift = os.getenv("P8_DRIFT_REPORT", "outputs/monitoring/drift_report.json")
    drift_str = st.text_input(
        "Chemin drift_report.json", value=default_drift, key="drift_path"
    )
    drift_path = resolve_path(drift_str)

    st.caption(f"Resolved: {drift_path} | exists={drift_path.exists()}")

    if not drift_path.exists():
        st.warning(
            "Drift introuvable. Lance d'abord `python -m monitoring.run_drift` "
            "ou monte correctement le volume `outputs` en Docker."
        )
    else:
        rep = load_json(drift_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prod rows", rep.get("n_prod_rows", rep.get("prod_rows", 0)))
        c2.metric("Ref rows", rep.get("n_ref_rows", rep.get("ref_rows", 0)))
        c3.metric("Features", rep.get("features_total", rep.get("n_features", 0)))
        c4.metric("Generated (UTC)", rep.get("generated_at_utc", ""))

        st.divider()

        left, right = st.columns(2)
        with left:
            st.subheader("Top PSI")
            show_df(pd.DataFrame(rep.get("top_psi", [])))

        with right:
            st.subheader("Top delta missing")
            show_df(
                pd.DataFrame(
                    rep.get("top_missing_delta", []) or rep.get("top_missing", [])
                )
            )

        st.divider()

        all_df = features_df_from_report(rep)
        if all_df.empty:
            st.info("Aucune table 'all_features' exploitable dans ce report.")
        else:
            st.subheader("Toutes les features (filtrées)")

            if "psi" in all_df.columns:
                min_psi = st.slider("Filtre PSI minimal", 0.0, 50.0, 0.1, 0.05)
                df_view = all_df[all_df["psi"] >= float(min_psi)].copy()
            else:
                st.warning("Colonne 'psi' absente → pas de filtre PSI possible.")
                df_view = all_df.copy()

            if "missing_rate_delta" in df_view.columns:
                only_high_missing = st.checkbox(
                    "Afficher seulement delta missing >= 5%", value=False
                )
                if only_high_missing:
                    df_view = df_view[df_view["missing_rate_delta"].abs() >= 0.05]

            if "psi" in df_view.columns:
                df_view = df_view.sort_values("psi", ascending=False)

            show_df(df_view)


# =========================
# Bench tab
# =========================
with tab_bench:
    st.subheader("Bench performances")

    default_base = os.getenv(
        "P8_BENCH_BASELINE",
        os.getenv("P8_BENCH_REPORT", "outputs/perf/bench_baseline.json"),
    )
    default_cur = os.getenv("P8_BENCH_CURRENT", "outputs/perf/bench_current.json")

    base_str = st.text_input(
        "Chemin bench baseline", value=default_base, key="bench_base"
    )
    cur_str = st.text_input("Chemin bench current", value=default_cur, key="bench_cur")

    base_path = resolve_path(base_str)
    cur_path = resolve_path(cur_str)

    st.caption(f"Baseline: {base_path} | exists={base_path.exists()}")
    st.caption(f"Current : {cur_path} | exists={cur_path.exists()}")

    base = load_json(base_path) if base_path.exists() else None
    cur = load_json(cur_path) if cur_path.exists() else None

    if base is None and cur is None:
        st.warning(
            "Aucun bench trouvé. Génère au moins un fichier JSON de bench dans `outputs/perf/`."
        )
    else:
        # Affichage en colonnes + deltas si possible
        m_base = bench_metrics(base) if base else None
        m_cur = bench_metrics(cur) if cur else None

        colA, colB, colC = st.columns(3)
        colA.markdown("### Baseline")
        colB.markdown("### Current")
        colC.markdown("### Delta vs baseline")

        keys = [
            ("predict_mean", "Predict mean (ms)"),
            ("predict_p95", "Predict p95 (ms)"),
            ("batch_row_mean", "Batch/row mean (ms)"),
            ("batch_row_p95", "Batch/row p95 (ms)"),
            ("batch_total_mean", "Batch total mean (ms)"),
            ("batch_total_p95", "Batch total p95 (ms)"),
        ]

        for k, label in keys:
            b = (m_base or {}).get(k, 0.0)
            c = (m_cur or {}).get(k, 0.0)

            colA.metric(label, f"{b:.2f}" if base else "—")
            colB.metric(label, f"{c:.2f}" if cur else "—")

            if base and cur:
                d = pct_delta(c, b)
                colC.metric(label, f"{d:+.1f}%" if d is not None else "—")
            else:
                colC.metric(label, "—")

        st.divider()

        with st.expander("Voir baseline JSON brut"):
            if base is None:
                st.write("Baseline introuvable.")
            else:
                st.json(base)

        with st.expander("Voir current JSON brut"):
            if cur is None:
                st.write("Current introuvable.")
            else:
                st.json(cur)
