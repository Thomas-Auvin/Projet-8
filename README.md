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
- chargement du modèle + seuil depuis des artefacts (`model.joblib`, `model_meta.json`)
- prédiction unitaire, batch, et upload CSV
- validation d’entrée (mode strict/relax)
- logging des entrées/sorties + latence en **SQLite** (traçabilité)
- monitoring de dérive **PSI + missing rate** basé sur les logs
- bench perf + optimisation
- CI/CD : tests + build Docker + déploiement Hugging Face Spaces

> Projet réalisé dans le cadre de la formation Data Scientist (OpenClassrooms) — dimension MLOps (Projet 8).

---

## Endpoints (API)

- `GET /docs` : Swagger UI
- `GET /health` : statut + version du modèle + nombre de features
- `GET /model` : informations modèle (version, seuil, meta)
- `GET /features` : features attendues + clés autorisées + groupes OHE
- `GET /examples` : liste des fichiers d’exemples
- `GET /examples/example_input_compact.json` : exemple JSON “compact”
- `GET /examples/example_input_compact.csv` : exemple CSV “compact”
- `POST /predict` : prédiction sur 1 ligne
- `POST /predict_batch` : prédiction batch (JSON)
- `POST /predict_csv` : upload CSV → sortie JSON ou CSV

**Input “compact”**
L’API accepte des clés “compactes” (ex: `NAME_INCOME_TYPE="Working"`) et reconstruit les features attendues (dummies + numériques). Le reste est mis à `NaN` pour alignement avec le modèle. 
👉 Utiliser `GET /features` pour voir les clés/groupes disponibles.

---

## Variables d’environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `P8_ARTIFACTS_DIR` | `app/artifacts/` | Dossier des artefacts (`model.joblib`, `model_meta.json`) |
| `P8_DB_PATH` | `data/prod/predictions.sqlite` | Chemin SQLite (logging) |
| `P8_STRICT_INPUT` | `1` | `1` : refuse les clés inconnues + peut appliquer une règle de “taux de remplissage” ; `0` : mode plus permissif |
| `P8_DISABLE_DB_LOG` | `0` | `1` : désactive le logging en base |
| `P8_DB_ASYNC_WRITER` | `1` | Active/désactive le writer async (si implémenté) |

---

## Lancer en local (uv)

### Prérequis
- Python 3.13
- `uv` installé

### Installer

```bash
uv sync --frozen --dev
```

### Démarrer l'API 

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Docker

L'API étant dockerisé, voici les commandes pour la faire tourner en docker : 

```bash

# Built à l'installation et après changement du code
docker compose up --build

# lancement sans built
docker compose up

# Stop
docker compose down
```

Pour ouvrir aussi bien en local qu'en docker:

    http://127.0.0.1:8000/docs

  
### Utilisation de l'API 

3 possibilités pour la prédiction 


POST /predict
Il est question ici d'importer un individu en format json pour obtenir une réponse en format json
Payload :
```json
{
  "features": {
    "AMT_INCOME_TOTAL": 150000,
    "NAME_INCOME_TYPE": "Working"
  }
}
```
Réponse :
```json
{
  "request_id": "...",
  "proba_default": 0.083,
  "threshold": 0.148,
  "decision": 0,
  "model_version": "...",
  "latency_ms": 12.4
}
```

POST /predict_batch
Il est question ici d'importer un ensemble d'individus en format json pour obtenir une réponse en format json
```json
{
  "rows": [
    {"AMT_INCOME_TOTAL": 150000, "NAME_INCOME_TYPE": "Working"},
    {"AMT_INCOME_TOTAL": 90000, "NAME_INCOME_TYPE": "Pensioner"}
  ]
}

```

POST /predict_csv
Il est question ici d'importer un ensemble d'individus en format csv pour obtenir une réponse en format csv

- upload d’un .csv

- paramètre output=json|csv


### Stockage (SQLite) & traçabilité

Chaque prédiction écrit une ligne en DB (table predictions) :

ts_utc, request_id, model_version

proba_default, threshold, decision, latency_ms

input_json (features alignées en JSON)

Objectif : auditabilité + matière première pour le monitoring.

Screenshot disponible dans le repo: capture de la table + colonnes + exemple de lignes (DB Browser for SQLite). 

### Monitoring drift (PSI + missing)

Le monitoring compare :

référence : data/reference/reference_sample.csv

prod : reconstruction des features depuis predictions.input_json via run_drift

Pour générer un rapport :
```bash
uv run python -m monitoring.run_drift --limit 2000 --ref-csv data/reference/reference_sample.csv
```
puis 


La sortie est disponible ici :

outputs/monitoring/drift_report.json 

Le report contient notamment :

top_psi : features les plus dérivantes (PSI)

top_missing_delta : variations de taux de missing

alerts : status ok|warn|critical + items (seuils configurables) 

### Performance (bench)

La mesure d'un bench latence est faite via la production d'un bench latest :
```bash
uv run python scripts/bench_api.py --base-url http://127.0.0.1:8000 --n 50 --batch-size 200
```

La sortie est disponible ici :

outputs/perf/bench_latest.json


### Dashboard 

Pour lire les données de drift et de bench, il est possible d'utiliser le dashboard. 
pour le lancer en local :
```bash
uv run streamlit run monitoring/dashboard.py
```
Pour le lancer en docker:
```bash
docker compose up
```

### Test
Pour toute modification decode, il est possible de faire passer un ensemble de test avant de push 

```bash
# Tests
uv run pytest -q

# Tests + coverage (affichage console)
uv run pytest --cov=app --cov=monitoring --cov-report=term-missing

# (Optionnel) Générer un rapport HTML de coverage
uv run pytest --cov=app --cov=monitoring --cov-report=html
```
### CI/CD

Pour toutes modofications du code, il faut passer par une PR via une nouvelle branche car master est protégé

CI : lint/tests + build Docker sur push/PR

CD : déploiement automatique vers Hugging Face Spaces (master → HF main)

### Exemple Hugging Face

L'API est déployé sur l'application Hugging Face permettant de disposer d'un exemple directe à l'adresse suivante : 
https://huggingface.co/spaces/Thomas-Auvin/Projet-8

### Licence

Projet pédagogique (OpenClassrooms). Usage/redistribution selon le cadre de la formation.