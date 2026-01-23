# app/model_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class LoadedModel:
    pipeline: Any
    threshold: float
    model_version: str
    feature_names: list[str]
    meta: dict[str, Any]


def _infer_feature_names(pipeline: Any, meta: dict[str, Any]) -> list[str]:
    # 1) meilleur cas : pipeline a feature_names_in_
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    # 2) sinon : prendre sur le 1er step (imputer) si présent
    if hasattr(pipeline, "named_steps") and "imputer" in pipeline.named_steps:
        imp = pipeline.named_steps["imputer"]
        if hasattr(imp, "feature_names_in_"):
            return list(imp.feature_names_in_)

    # 3) fallback : meta (moins fiable)
    if "feature_names" in meta and isinstance(meta["feature_names"], list):
        return list(meta["feature_names"])

    raise RuntimeError(
        "Impossible d'inférer les feature names (ni pipeline.feature_names_in_, ni imputer.feature_names_in_, ni meta)."
    )


def load_model(artifacts_dir: Path) -> LoadedModel:
    artifacts_dir = artifacts_dir.resolve()
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    threshold = float(meta.get("threshold", 0.5))
    model_version = str(meta.get("model_version") or meta.get("run_id") or meta.get("run_name") or "unknown")

    feature_names = _infer_feature_names(pipeline, meta)

    return LoadedModel(
        pipeline=pipeline,
        threshold=threshold,
        model_version=model_version,
        feature_names=feature_names,
        meta=meta,
    )
