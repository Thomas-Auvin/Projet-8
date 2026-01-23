from __future__ import annotations

from pathlib import Path
import pandas as pd
import httpx

try:
    from project_paths import DATA_DIR  # type: ignore
    sample_path = DATA_DIR / "reference" / "reference_sample.csv"
except Exception:
    sample_path = Path("data") / "reference" / "reference_sample.csv"

API_URL = "http://127.0.0.1:8000/predict"


def main() -> None:
    df = pd.read_csv(sample_path)
    row = df.iloc[0].to_dict()

    # au cas où TARGET traîne dans un fichier
    row.pop("TARGET", None)

    # JSON safe: NaN -> None
    features = {k: (None if pd.isna(v) else v) for k, v in row.items()}

    payload = {"features": features}

    r = httpx.post(API_URL, json=payload, timeout=60.0)
    print("Status:", r.status_code)
    print("Response:", r.text)


if __name__ == "__main__":
    main()
