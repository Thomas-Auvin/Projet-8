from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.input_adapter import InputAdapter
from app.model_loader import load_model


def pick_example_value_for_group(adapter: InputAdapter, key: str) -> Any:
    # Si c'est un groupe OHE, on choisit une modalité existante
    if key in adapter.groups:
        g = adapter.groups[key]
        if g.value_to_column:
            return sorted(g.value_to_column.keys())[0]
        return ""
    # Sinon feature numérique/engineered : valeur neutre
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="app/artifacts")
    parser.add_argument("--out-dir", default="app/examples")
    parser.add_argument("--json-name", default="example_input_compact.json")
    parser.add_argument("--csv-name", default="example_input_compact.csv")
    args = parser.parse_args()

    loaded = load_model(Path(args.artifacts_dir))
    adapter = InputAdapter.from_feature_names(loaded.feature_names)

    meta = loaded.meta or {}
    requested = meta.get("requested_groups") or meta.get("required_groups") or []
    if not isinstance(requested, list):
        requested = []

    requested = [str(x) for x in requested]

    # On remplit toutes les requested => l’exemple passe forcément la règle min_filled_rate
    features: dict[str, Any] = {k: pick_example_value_for_group(adapter, k) for k in requested}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"features": features}
    (out_dir / args.json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame([features])
    df.to_csv(out_dir / args.csv_name, index=False)

    print("✅ Wrote:", (out_dir / args.json_name))
    print("✅ Wrote:", (out_dir / args.csv_name), "| n_cols =", df.shape[1])


if __name__ == "__main__":
    main()
