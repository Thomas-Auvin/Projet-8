# fonction_mlflow.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


def log_binary_clf_metrics(
    prefix: str,
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Loggue des métriques de classification binaire dans MLflow à partir de probas.
    Retourne aussi un dict de métriques.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    y_pred = (y_proba_arr >= threshold).astype(int)

    metrics = {
        f"{prefix}_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        f"{prefix}_ap": float(average_precision_score(y_true_arr, y_proba_arr)),
        f"{prefix}_precision": float(
            precision_score(y_true_arr, y_pred, zero_division=0)
        ),
        f"{prefix}_recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(y_true_arr, y_pred)),
        f"{prefix}_bal_acc": float(balanced_accuracy_score(y_true_arr, y_pred)),
        f"{prefix}_pred_pos_rate": float(np.mean(y_pred)),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics


def run_model(
    name: str,
    estimator,
    X_train,
    y_train,
    X_valid,
    y_valid,
    phase: str = "baseline",
    feature_set: str = "v1_all_tables",
    threshold: float = 0.5,
    scaler_with_mean: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne un pipeline simple (imputer -> scaler -> modèle),
    calcule des métriques train/valid, loggue dans MLflow, renvoie un dict avec le pipeline.

    Note:
    - si tu passes un modèle sans predict_proba, ça lèvera une erreur explicite.
    """
    pipe_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=scaler_with_mean)),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("estimator", estimator.__class__.__name__)
        mlflow.log_param("threshold", float(threshold))

        pipe_model.fit(X_train, y_train)

        if not hasattr(pipe_model, "predict_proba"):
            raise AttributeError(
                f"{estimator.__class__.__name__} ne supporte pas predict_proba. "
                "Active probability=True (si SVC), ou adapte la fonction pour utiliser decision_function."
            )

        y_train_proba = pipe_model.predict_proba(X_train)[:, 1]
        y_valid_proba = pipe_model.predict_proba(X_valid)[:, 1]

        train_metrics = log_binary_clf_metrics(
            "train", y_train, y_train_proba, threshold=threshold
        )
        valid_metrics = log_binary_clf_metrics(
            "valid", y_valid, y_valid_proba, threshold=threshold
        )

        print(
            f"[{name}] "
            f"AUC valid={valid_metrics['valid_auc']:.3f} | "
            f"AP valid={valid_metrics['valid_ap']:.3f} | "
            f"F1 valid={valid_metrics['valid_f1']:.3f} "
            f"(thr={threshold})"
        )

        return {
            "name": name,
            **train_metrics,
            **valid_metrics,
            "pipeline": pipe_model,
        }


def run_model_cv(
    name: str,
    estimator,
    X,
    y,
    model_name: str = "unknown",
    n_splits: int = 3,
    random_state: int = 42,
    use_sampling: bool = True,
    smote_ratio: float = 0.5,
    use_under: bool = True,
    under_ratio: float = 1.0,
    phase: str = "cv",
    feature_set: str = "v1_all_tables",
    threshold: float = 0.5,
    scaler_with_mean: bool = True,
) -> Dict[str, Any]:
    """
    CV stratifiée avec option :
      - SMOTE (over-sampling)
      - RandomUnderSampler (under-sampling)

    Loggue dans MLflow :
      - métriques par fold (auc/ap/f1) avec step=fold_idx
      - métriques OOF globales (auc/ap/precision/recall/f1/accuracy/bal_acc/pred_pos_rate)

    Retourne :
      - métriques OOF, moyennes folds, pipeline, probas OOF.
    """
    # Sécurise y
    y_arr = np.asarray(y)
    n = len(y_arr)
    oof_probas = np.zeros(n, dtype=float)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    aucs: list[float] = []
    aps: list[float] = []
    f1s: list[float] = []
    fold_metrics: list[dict[str, float]] = []

    # Pipeline (imputer -> scaler -> sampling -> model)
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=scaler_with_mean)),
    ]

    if use_sampling:
        steps.append(
            (
                "smote",
                SMOTE(sampling_strategy=smote_ratio, random_state=random_state),
            )
        )
        if use_under:
            steps.append(
                (
                    "under",
                    RandomUnderSampler(
                        sampling_strategy=under_ratio, random_state=random_state
                    ),
                )
            )

    steps.append(("model", estimator))
    pipe_cv = ImbPipeline(steps=steps)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag(
            "validation",
            "cv_smote_under"
            if (use_sampling and use_under)
            else "cv_smote"
            if use_sampling
            else "cv",
        )
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("estimator", estimator.__class__.__name__)

        mlflow.log_param("n_splits", int(n_splits))
        mlflow.log_param("random_state", int(random_state))
        mlflow.log_param("threshold", float(threshold))
        mlflow.log_param("use_sampling", bool(use_sampling))
        mlflow.log_param("use_under", bool(use_under))
        mlflow.log_param("smote_ratio", float(smote_ratio) if use_sampling else "None")
        mlflow.log_param(
            "under_ratio",
            float(under_ratio) if (use_sampling and use_under) else "None",
        )

        # CV
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y_arr), start=1):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            pipe_cv.fit(X_train_fold, y_train_fold)

            if not hasattr(pipe_cv, "predict_proba"):
                raise AttributeError(
                    f"{estimator.__class__.__name__} ne supporte pas predict_proba."
                )

            y_valid_proba = pipe_cv.predict_proba(X_valid_fold)[:, 1]
            oof_probas[valid_idx] = y_valid_proba

            y_valid_pred = (y_valid_proba >= threshold).astype(int)

            auc_fold = float(roc_auc_score(y_valid_fold, y_valid_proba))
            ap_fold = float(average_precision_score(y_valid_fold, y_valid_proba))
            f1_fold = float(f1_score(y_valid_fold, y_valid_pred, zero_division=0))

            aucs.append(auc_fold)
            aps.append(ap_fold)
            f1s.append(f1_fold)

            fold_metrics.append({"fold": float(fold_idx), "auc": auc_fold, "ap": ap_fold, "f1": f1_fold})

            mlflow.log_metric("auc_fold", auc_fold, step=fold_idx)
            mlflow.log_metric("ap_fold", ap_fold, step=fold_idx)
            mlflow.log_metric("f1_fold", f1_fold, step=fold_idx)

        # OOF global
        oof_metrics = log_binary_clf_metrics("oof", y_arr, oof_probas, threshold=threshold)

        mlflow.log_metric("auc_mean_cv", float(np.mean(aucs)))
        mlflow.log_metric("ap_mean_cv", float(np.mean(aps)))
        mlflow.log_metric("f1_mean_cv", float(np.mean(f1s)))

        print(
            f"[{name}] "
            f"AUC OOF={oof_metrics['oof_auc']:.3f} | "
            f"AP OOF={oof_metrics['oof_ap']:.3f} | "
            f"F1 OOF={oof_metrics['oof_f1']:.3f} "
            f"(thr={threshold}) | "
            f"mean folds AUC={np.mean(aucs):.3f}, AP={np.mean(aps):.3f}, F1={np.mean(f1s):.3f}"
        )

        return {
            "name": name,
            **oof_metrics,
            "auc_folds": aucs,
            "ap_folds": aps,
            "f1_folds": f1s,
            "fold_metrics": fold_metrics,
            "pipeline": pipe_cv,
            "oof_probas": oof_probas,
            "y_true": np.asarray(y).copy(),
        }
