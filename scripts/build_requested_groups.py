from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.input_adapter import InputAdapter


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_features_by_cum_importance(
    imp_df: pd.DataFrame, target: float
) -> list[str]:
    # attend colonnes: feature, importance
    if "feature" not in imp_df.columns or "importance" not in imp_df.columns:
        raise ValueError("feature_importances.csv doit contenir les colonnes: feature, importance")

    imp = imp_df[["feature", "importance"]].copy()
    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce").fillna(0.0)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    total = float(imp["importance"].sum())
    if total <= 0:
        # fallback: tout égal
        imp["importance_norm"] = 1.0 / max(len(imp), 1)
    else:
        imp["importance_norm"] = imp["importance"] / total

    imp["cum"] = imp["importance_norm"].cumsum()

    # inclure la ligne qui franchit le seuil
    idx = int((imp["cum"] >= target).idxmax()) if (imp["cum"] >= target).any() else len(imp) - 1
    selected = imp.loc[:idx, "feature"].astype(str).tolist()
    return selected


def _compute_min_rate(ratio: float) -> float:
    # Règle que tu as décrite
    # - si > 50% => on désactive la règle (0.0)
    # - si < 10% => 0.30
    # - si < 20% => 0.40
    # - sinon => 0.50
    if ratio > 0.50:
        return 0.0
    if ratio < 0.10:
        return 0.30
    if ratio < 0.20:
        return 0.40
    return 0.50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--importances", type=str, default="data/reference/feature_importances.csv")
    ap.add_argument("--model-meta", type=str, default="app/artifacts/model_meta.json")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--write-report", action="store_true", help="écrit aussi data/reference/requested_config.json")
    args = ap.parse_args()

    imp_path = Path(args.importances)
    meta_path = Path(args.model_meta)

    meta = _load_json(meta_path)
    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("model_meta.json ne contient pas feature_names (liste non vide).")

    adapter = InputAdapter.from_feature_names([str(x) for x in feature_names])

    imp_df = pd.read_csv(imp_path)
    selected_features = _pick_features_by_cum_importance(imp_df, float(args.target))

    # on ignore ce qui n’existe pas dans le modèle
    feature_set = set(map(str, feature_names))
    selected_features = [f for f in selected_features if f in feature_set]

    # mapping dummy_col -> group_name
    col_to_group: dict[str, str] = {}
    for gname, g in adapter.groups.items():
        for col in g.columns:
            col_to_group[col] = gname

    requested: set[str] = set()
    for feat in selected_features:
        if feat in col_to_group:
            requested.add(col_to_group[feat])  # groupe OHE => 1 clé compacte
        else:
            requested.add(feat)  # feature simple => elle-même

    requested_groups = sorted(requested)

    allowed = sorted(list(adapter.allowed_input_keys()))
    ratio = (len(requested_groups) / max(len(allowed), 1))
    min_rate = _compute_min_rate(ratio)

    # update meta (ce que ton API lit déjà)
    meta["requested_groups"] = requested_groups
    meta["min_filled_rate_requested"] = float(min_rate)
    meta["requested_groups_source"] = {
        "type": "feature_importances",
        "target_cum_importance": float(args.target),
        "selected_features_count": len(selected_features),
        "requested_groups_count": len(requested_groups),
        "allowed_input_keys_count": len(allowed),
        "requested_ratio": float(ratio),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _save_json(meta_path, meta)

    print("✅ Updated:", meta_path)
    print(" - selected_features_count:", len(selected_features))
    print(" - requested_groups_count:", len(requested_groups))
    print(" - allowed_input_keys_count:", len(allowed))
    print(" - requested_ratio:", round(ratio, 4))
    print(" - min_filled_rate_requested:", min_rate)
    print(" - first_30_requested_groups:", requested_groups[:30])

    if args.write_report:
        report_path = Path("data/reference/requested_config.json")
        report = {
            "requested_groups": requested_groups,
            "min_filled_rate_requested": float(min_rate),
            "target_cum_importance": float(args.target),
            "selected_features_count": len(selected_features),
            "allowed_input_keys": allowed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_json(report_path, report)
        print("✅ Report:", report_path)


if __name__ == "__main__":
    main()
