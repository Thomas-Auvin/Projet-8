---
title: Projet-8 — Credit Scoring API (MLOps)
emoji: 🏦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Projet 8 — Credit Scoring API (MLOps)

API de scoring crédit industrialisée (FastAPI) avec :
- **chargement modèle + seuil** depuis des artifacts
- **prédiction unitaire et batch**
- **logging des entrées/sorties** dans SQLite (traçabilité)
- **monitoring de dérive** (PSI + missing rates) basé sur les logs
- **bench performance** + optimisation (logging batch en transaction unique)
- **CI/CD** : tests + build Docker + déploiement automatique vers Hugging Face Spaces

> Projet réalisé dans le cadre de la formation Data Scientist (OpenClassrooms) — dimension MLOps (Projet 8).

---

## Démo (Hugging Face)

Le Space expose une **API FastAPI** :
- `GET /docs` : Swagger UI
- `GET /health` : état de l’API
- `POST /predict` : scoring 1 ligne
- `POST /predict_batch` : scoring batch

**Endpoints utiles :**
- `/` (root) : redirige vers `/docs`
- `/docs` : interface de test
- `/health` : statut + version modèle

> Sur Hugging Face, la persistance est prévue via `/data` (si “Persistent Storage” est activé).

---

## Structure du dépôt

app/
main.py # API FastAPI (lifespan, endpoints)
model_loader.py # chargement pipeline + meta
schemas.py # schémas Pydantic (requests/responses)
storage.py # SqliteStore (logging single + batch)
artifacts/ # model.joblib + model_meta.json (LFS)
monitoring/
run_drift.py # génère outputs/monitoring/drift_report.json
drift_utils.py # PSI numeric/categorical + missing stats
dashboard.py # (si présent) Streamlit pour visualiser le drift
scripts/
bench_api.py # bench latence /predict + /predict_batch
tests/
conftest.py # env de test (DB temp, artifacts test)
test_batch.py # test predict_batch
.github/workflows/
ci.yml # CI (pytest + build Docker)
deploy_hf.yml # CD vers Hugging Face (master -> HF main)
data/
reference/reference_sample.csv
outputs/
monitoring/drift_report.json (généré)
perf/bench_latest.json (généré)


---

## Variables d’environnement

| Variable | Défaut | Rôle |
|---|---:|---|
| `P8_ARTIFACTS_DIR` | `app/artifacts/` | Dossier des artifacts (`model.joblib`, `model_meta.json`) |
| `P8_DB_PATH` | `data/prod/predictions.sqlite` (local) / `/data/predictions.sqlite` (HF conseillé) | Chemin SQLite pour le logging |
| `P8_STRICT_INPUT` | `1` | `1` : refuse si features manquantes ; `0` : mode relax (missing -> `null`) |

---

## Lancer en local (uv)

### Prérequis
- Python 3.13
- `uv` installé

### Installer les dépendances
```bash
uv sync --frozen --dev

Lancer l’API

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

Ouvrir :

    http://127.0.0.1:8000/docs

    http://127.0.0.1:8000/health

Utiliser l’API
Prédiction unitaire — /predict

Payload :

{
  "features": {
    "feature_1": 0.12,
    "feature_2": null
  }
}

Réponse (exemple) :

{
  "request_id": "…",
  "proba_default": 0.083,
  "threshold": 0.148,
  "decision": 0,
  "model_version": "…",
  "latency_ms": 12.4
}

Prédiction batch — /predict_batch

Payload :

{
  "rows": [
    {"feature_1": 0.12, "feature_2": null},
    {"feature_1": 0.34, "feature_2": 1.0}
  ]
}

Réponse (exemple) :

{
  "n_rows": 2,
  "items": [
    {
      "request_id": "…",
      "proba_default": 0.08,
      "threshold": 0.148,
      "decision": 0,
      "model_version": "…",
      "latency_ms": 3.1
    },
    {
      "request_id": "…",
      "proba_default": 0.23,
      "threshold": 0.148,
      "decision": 1,
      "model_version": "…",
      "latency_ms": 3.1
    }
  ]
}

Logging & base SQLite (traçabilité)

Chaque prédiction écrit une ligne dans la table predictions :

    ts_utc, request_id, model_version

    proba_default, threshold, decision, latency_ms

    input_json : features (JSON)

Objectif : auditabilité (on sait ce qui a été envoyé et ce qui a été prédit) + matière première pour le monitoring.
Monitoring drift (PSI + missing)

Le monitoring compare :

    référence : data/reference/reference_sample.csv

    prod : reconstruction des features depuis predictions.input_json

Générer un rapport :

uv run python -m monitoring.run_drift --limit 2000

Sortie :

    outputs/monitoring/drift_report.json

Le rapport contient :

    top_psi : features les plus dérivantes (PSI)

    top_missing_delta : variations de taux de missing

    thresholds : repères PSI (0.1 / 0.2 / 0.3)

Interprétation rapide :

    PSI < 0.1 : faible

    0.1–0.2 : modérée

    0.2–0.3 : importante

        0.3 : forte

Plan d’action (exemple) :

    PSI > 0.2 sur plusieurs features : investigation (sources de données, schéma, préprocessing)

    PSI > 0.3 : alerte, décision de recalibrage / ré-entraînement / rollback

Performance : bench + optimisation
Bench

Mesure la latence /predict et /predict_batch :

uv run python scripts/bench_api.py --base-url http://127.0.0.1:8000 --n 50 --batch-size 200

Sortie :

    outputs/perf/bench_latest.json

Optimisation implémentée

    Logging batch optimisé : insertion SQLite en 1 transaction via executemany (au lieu de N inserts/commits)

    Gain attendu : baisse de la latence totale /predict_batch et du coût par ligne.

Tests

uv run pytest -q

Les tests utilisent :

    DB temporaire via P8_DB_PATH

    artifacts de test (tests/assets/artifacts)

    P8_STRICT_INPUT=0 en test pour faciliter les payloads

Docker
Build

docker build -t projet8-api .

Run

docker run --rm -p 7860:7860 \
  -e P8_DB_PATH=/data/predictions.sqlite \
  -e P8_STRICT_INPUT=1 \
  -v "$(pwd)/data:/data" \
  projet8-api

CI/CD (GitHub → Hugging Face)

    CI : pytest + build Docker sur PR / push

    CD : sur master, déploiement automatique vers Hugging Face Space (branche main)

Convention retenue :

    GitHub = source de vérité (master)

    Hugging Face = cible de déploiement (main)

Notes Hugging Face

    model.joblib est versionné via Git LFS.

    Les artefacts locaux (*.db, *.sqlite, *.pdf, etc.) sont ignorés et ne doivent pas être poussés sur HF.

    Pour persister la DB en prod, utiliser /data/predictions.sqlite (si stockage persistant activé).

Licence

Projet pédagogique (OpenClassrooms). Usage/redistribution selon le cadre de la formation.