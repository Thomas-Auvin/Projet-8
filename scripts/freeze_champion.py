from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


def pick_tracking_uri() -> str:
    # 1) si l'env est déjà set, on la respecte
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # 2) sinon on privilégie la DB si elle existe
    if Path("mlflow.db").exists():
        return "sqlite:///mlflow.db"

    # 3) fallback file store
    return "file:./mlruns"


def resolve_run_id(
    client: MlflowClient,
    experiment_name: str,
    run_id: Optional[str],
    run_name: Optional[str],
    model_name: Optional[str],
) -> str:
    if run_id:
        return run_id

    # Option A : depuis Model Registry (si tu as un model name)
    if model_name:
        # prend la version la plus récente du modèle
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        if not versions:
            raise RuntimeError(f"Aucune version trouvée pour le modèle '{model_name}'")
        # tri par version (string) -> int
        versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
        return versions_sorted[0].run_id

    # Option B : depuis le run name (tag mlflow.runName)
    if not run_name:
        raise RuntimeError("Il faut fournir --run-id ou --run-name ou --model-name.")

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' introuvable.")

    # run name = tag 'mlflow.runName'
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=5,
    )
    if not runs:
        raise RuntimeError(
            f"Aucun run trouvé avec run_name='{run_name}' dans experiment='{experiment_name}'."
        )

    return runs[0].info.run_id


def find_model_artifact_path(client: MlflowClient, run_id: str) -> str:
    """
    Cherche un dossier d'artefact qui ressemble à un MLflow Model
    (contient un fichier 'MLmodel').
    """
    # essai direct : la plupart du temps c'est "model"
    try_paths = ["model", "sklearn-model", "classifier", "pipeline"]
    for p in try_paths:
        try:
            tmp = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{p}/MLmodel")
            if Path(tmp).exists():
                return p
        except Exception:
            pass

    # recherche plus générale : on parcourt les artefacts à la racine
    root_items = client.list_artifacts(run_id, path="")
    for it in root_items:
        if it.is_dir:
            p = it.path
            try:
                tmp = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{p}/MLmodel")
                if Path(tmp).exists():
                    return p
            except Exception:
                pass

    raise RuntimeError(
        "Impossible de trouver un artefact de type 'MLflow Model' (fichier MLmodel) dans ce run."
    )


def load_feature_names() -> list[str]:
    # On préfère un petit fichier de référence (rapide)
    ref = Path("data/reference/reference_sample.csv")
    if ref.exists():
        import pandas as pd

        df = pd.read_csv(ref)
        cols = df.columns.tolist()
        # au cas où TARGET est présent
        cols = [c for c in cols if c != "TARGET"]
        return cols

    # fallback sur le gros fichier
    raw = Path("data/raw_local/application_train_features.csv")
    if raw.exists():
        import pandas as pd

        df = pd.read_csv(raw, nrows=5)  # juste pour les colonnes
        cols = df.columns.tolist()
        cols = [c for c in cols if c != "TARGET"]
        return cols

    raise RuntimeError(
        "Impossible de charger les feature_names : ni data/reference/reference_sample.csv ni data/raw_local/application_train_features.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="home-credit-benchmark")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    tracking_uri = pick_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    rid = resolve_run_id(
        client=client,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        run_name=args.run_name,
        model_name=args.model_name,
    )

    run = client.get_run(rid)

    model_art_path = find_model_artifact_path(client, rid)
    model_uri = f"runs:/{rid}/{model_art_path}"

    # charge le modèle (sklearn flavor)
    model = mlflow.sklearn.load_model(model_uri)

    artifacts_dir = Path("app/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # export joblib
    import joblib

    joblib.dump(model, artifacts_dir / "model.joblib")

    # meta
    feature_names = load_feature_names()
    threshold = run.data.params.get("threshold", "0.5")

    meta = {
        "run_id": rid,
        "experiment": args.experiment_name,
        "run_name": run.data.tags.get("mlflow.runName"),
        "model_artifact_path": model_art_path,
        "threshold": float(threshold),
        "feature_names": feature_names,
    }

    (artifacts_dir / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ Export terminé")
    print("Tracking URI:", tracking_uri)
    print("Run ID:", rid)
    print("Saved:", artifacts_dir / "model.joblib")
    print("Saved:", artifacts_dir / "model_meta.json")


if __name__ == "__main__":
    main()
