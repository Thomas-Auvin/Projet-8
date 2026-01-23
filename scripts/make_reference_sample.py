import pandas as pd
from pathlib import Path

RAW = Path("data/raw_local/application_train_features.csv")  # adapte le nom si différent
OUT = Path("data/reference/reference_sample.csv")

df = pd.read_csv(RAW)
df.sample(n=min(1000, len(df)), random_state=42).to_csv(OUT, index=False)
print("Wrote:", OUT, "shape=", df.shape)
