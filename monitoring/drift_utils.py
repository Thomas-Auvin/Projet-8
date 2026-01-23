from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


EPS = 1e-6


def _safe_log(x: float) -> float:
    return math.log(max(x, EPS))


def _psi_from_distributions(p_ref: pd.Series, p_prod: pd.Series) -> float:
    """
    PSI = sum((p_ref - p_prod) * ln(p_ref/p_prod))
    p_ref / p_prod : distributions indexées sur les mêmes bins/catégories.
    """
    # align
    idx = p_ref.index.union(p_prod.index)
    p_ref = p_ref.reindex(idx, fill_value=0.0)
    p_prod = p_prod.reindex(idx, fill_value=0.0)

    # clip eps
    p_ref = p_ref.clip(lower=EPS)
    p_prod = p_prod.clip(lower=EPS)

    return float(((p_ref - p_prod) * np.log(p_ref / p_prod)).sum())


def _is_numeric_series(s: pd.Series, min_numeric_ratio: float = 0.95) -> Tuple[bool, pd.Series]:
    """
    Essaie de convertir en numérique. Retourne (is_numeric, numeric_series).
    """
    numeric = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    if non_null == 0:
        return True, numeric  # tout missing -> on traite comme numérique mais PSI=0
    ratio = numeric.notna().sum() / max(non_null, 1)
    return ratio >= min_numeric_ratio, numeric


def psi_numeric(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """
    PSI numérique basé sur des bins définis sur la référence (quantiles).
    On gère un bin "MISSING".
    """
    ref_num = pd.to_numeric(ref, errors="coerce")
    prod_num = pd.to_numeric(prod, errors="coerce")

    # bin missing séparément
    ref_missing = ref_num.isna()
    prod_missing = prod_num.isna()

    ref_vals = ref_num[~ref_missing]
    prod_vals = prod_num[~prod_missing]

    if ref_vals.empty and prod_vals.empty:
        return 0.0

    # si ref quasi-constante ou trop peu de valeurs uniques
    if ref_vals.nunique(dropna=True) <= 1:
        # distrib = {constant} + missing
        # PSI vient surtout des missing diffs
        p_ref = pd.Series(
            {
                "CONST": float((~ref_missing).mean()),
                "MISSING": float(ref_missing.mean()),
            }
        )
        p_prod = pd.Series(
            {
                "CONST": float((~prod_missing).mean()),
                "MISSING": float(prod_missing.mean()),
            }
        )
        return _psi_from_distributions(p_ref, p_prod)

    # bins quantiles sur ref
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref_vals.to_numpy(dtype=float), qs))
    if len(edges) < 3:
        # fallback linspace
        mn, mx = float(ref_vals.min()), float(ref_vals.max())
        if mn == mx:
            edges = np.array([mn - 1, mn, mn + 1], dtype=float)
        else:
            edges = np.linspace(mn, mx, bins + 1)

    # étendre pour capturer prod hors range
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref_vals, bins=edges, include_lowest=True)
    prod_bins = pd.cut(prod_vals, bins=edges, include_lowest=True)

    p_ref = ref_bins.value_counts(normalize=True).sort_index()
    p_prod = prod_bins.value_counts(normalize=True).sort_index()

    # ajouter missing
    p_ref.loc["MISSING"] = float(ref_missing.mean())
    p_prod.loc["MISSING"] = float(prod_missing.mean())

    # renormaliser pour que somme=1 (au cas où)
    p_ref = p_ref / p_ref.sum()
    p_prod = p_prod / p_prod.sum()

    return _psi_from_distributions(p_ref, p_prod)


def psi_categorical(ref: pd.Series, prod: pd.Series, top_k: int = 30) -> float:
    """
    PSI catégoriel : on prend les top_k catégories de la ref, le reste -> OTHER.
    Missing -> MISSING.
    """
    ref_s = ref.astype("object")
    prod_s = prod.astype("object")

    ref_s = ref_s.where(ref_s.notna(), other="MISSING").astype(str)
    prod_s = prod_s.where(prod_s.notna(), other="MISSING").astype(str)

    top = ref_s.value_counts().head(top_k).index.tolist()

    def map_cat(x: str) -> str:
        if x == "MISSING":
            return "MISSING"
        return x if x in top else "OTHER"

    ref_m = ref_s.map(map_cat)
    prod_m = prod_s.map(map_cat)

    p_ref = ref_m.value_counts(normalize=True)
    p_prod = prod_m.value_counts(normalize=True)

    return _psi_from_distributions(p_ref, p_prod)


@dataclass
class FeatureDrift:
    feature: str
    feature_type: str  # "numeric" | "categorical"
    psi: float
    ref_missing_rate: float
    prod_missing_rate: float
    missing_rate_delta: float
    ref_n: int
    prod_n: int


def compute_drift(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    bins: int = 10,
    cat_top_k: int = 30,
) -> Dict[str, Any]:
    """
    Retourne un dict prêt à sérialiser en JSON.
    """
    # align colonnes : union pour ne rien perdre
    cols = sorted(set(df_ref.columns).union(df_prod.columns))
    df_ref = df_ref.reindex(columns=cols)
    df_prod = df_prod.reindex(columns=cols)

    out: list[FeatureDrift] = []

    for c in cols:
        s_ref = df_ref[c]
        s_prod = df_prod[c]

        ref_n = int(len(s_ref))
        prod_n = int(len(s_prod))

        ref_missing_rate = float(pd.isna(s_ref).mean())
        prod_missing_rate = float(pd.isna(s_prod).mean())

        is_num, s_ref_num = _is_numeric_series(s_ref)
        if is_num:
            s_prod_num = pd.to_numeric(s_prod, errors="coerce")
            psi = psi_numeric(s_ref_num, s_prod_num, bins=bins)
            ftype = "numeric"
        else:
            psi = psi_categorical(s_ref, s_prod, top_k=cat_top_k)
            ftype = "categorical"

        out.append(
            FeatureDrift(
                feature=c,
                feature_type=ftype,
                psi=float(psi),
                ref_missing_rate=ref_missing_rate,
                prod_missing_rate=prod_missing_rate,
                missing_rate_delta=float(prod_missing_rate - ref_missing_rate),
                ref_n=ref_n,
                prod_n=prod_n,
            )
        )

    # tables triées
    by_psi = sorted(out, key=lambda x: x.psi, reverse=True)
    by_missing_delta = sorted(out, key=lambda x: abs(x.missing_rate_delta), reverse=True)

    def fd_to_dict(fd: FeatureDrift) -> Dict[str, Any]:
        return {
            "feature": fd.feature,
            "type": fd.feature_type,
            "psi": fd.psi,
            "ref_missing_rate": fd.ref_missing_rate,
            "prod_missing_rate": fd.prod_missing_rate,
            "missing_rate_delta": fd.missing_rate_delta,
            "ref_n": fd.ref_n,
            "prod_n": fd.prod_n,
        }

    return {
        "n_ref_rows": int(len(df_ref)),
        "n_prod_rows": int(len(df_prod)),
        "features_total": int(len(cols)),
        "top_psi": [fd_to_dict(x) for x in by_psi[:30]],
        "top_missing_delta": [fd_to_dict(x) for x in by_missing_delta[:30]],
        "all_features": [fd_to_dict(x) for x in out],
        "thresholds": {
            "psi_low": 0.1,
            "psi_medium": 0.2,
            "psi_high": 0.3,
        },
    }
