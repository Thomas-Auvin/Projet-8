# project_paths.py (à la racine du projet)
from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Remonte les parents à partir de start (par défaut cwd) et retourne la racine du projet.

    Critère principal (très fiable) :
    - présence de pyproject.toml OU .git à la racine

    Critère de renfort :
    - présence d'au moins 1 dossier métier parmi: data/, notebooks/, app/, ml/, src/, scripts/
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()

    must_have_any = {"pyproject.toml", ".git"}
    helpful_dirs = {"data", "notebooks", "app", "ml", "src", "scripts", "tests"}

    for p in [start, *start.parents]:
        has_must = any((p / m).exists() for m in must_have_any)
        has_help = any((p / d).is_dir() for d in helpful_dirs)

        # racine probable si pyproject/.git + au moins un dossier métier
        if has_must and has_help:
            return p

    # fallback : si jamais pyproject/.git existent sans dossier métier détecté
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p

    raise RuntimeError(
        f"Impossible de trouver la racine du projet depuis: {start}\n"
        "Assure-toi que project_paths.py est bien à la racine, "
        "ou qu'il existe un pyproject.toml ou un dossier .git."
    )


ROOT = find_project_root()

# Dossiers data (tu peux adapter si ta structure réelle est différente)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_local"      # cohérent avec ta logique 'data complète locale'
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"  # utile pour échantillons committables

# Sorties
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# MLflow
MLRUNS_DIR = ROOT / "mlruns"

# Création des dossiers (safe)
for d in [
    DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REFERENCE_DIR,
    OUT_DIR, FIG_DIR, TAB_DIR,
    MLRUNS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
