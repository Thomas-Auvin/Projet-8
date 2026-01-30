# app/input_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import re

import numpy as np


class InputError(ValueError):
    """Erreur d'input utilisateur (422)."""


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return True
        if s.lower() in {"nan", "none", "null"}:
            return True
    return False


def _norm_str(v: Any) -> str:
    # normalisation douce pour matcher les catégories
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _to_float_or_nan(v: Any) -> float:
    if _is_missing(v):
        return float("nan")
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return float(v)
    if isinstance(v, str):
        # accepte "1,23" -> 1.23
        s = v.strip().replace(",", ".")
        return float(s)
    raise InputError(f"Valeur numérique invalide: {v!r}")


def _to_binary_or_nan(v: Any) -> float:
    """Mappe des valeurs usuelles en 0/1."""
    if _is_missing(v):
        return float("nan")
    if isinstance(v, (int, float)):
        fv = float(v)
        if fv in (0.0, 1.0):
            return fv
        raise InputError(f"Valeur binaire attendue (0/1), reçu: {v!r}")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "oui"}:
            return 1.0
        if s in {"0", "false", "f", "no", "n", "non"}:
            return 0.0
        raise InputError(f"Valeur binaire attendue (oui/non, y/n, true/false, 0/1), reçu: {v!r}")
    raise InputError(f"Valeur binaire invalide: {v!r}")


@dataclass(frozen=True)
class OneHotGroup:
    name: str
    columns: List[str]               # les dummies exacts attendus par le modèle
    value_to_column: Dict[str, str]  # valeur normalisée -> dummy col


@dataclass
class InputAdapter:
    feature_names: List[str]
    feature_set: Set[str]
    groups: Dict[str, OneHotGroup]  # group_name -> OneHotGroup

    @classmethod
    def from_feature_names(cls, feature_names: List[str]) -> "InputAdapter":
        """
        Construit les groupes OHE à partir d'une liste de préfixes Home Credit (catégorielles).
        On reste volontairement simple + robuste pour P8.
        """
        feature_set = set(feature_names)

        # Préfixes catégoriels typiques Home Credit (présents chez toi via feature_names)
        # Tu peux en ajouter si tu vois d'autres familles *_<category>
        ohe_prefixes = [
            "CODE_GENDER",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "WALLSMATERIAL_MODE",
            "EMERGENCYSTATE_MODE",
        ]

        groups: Dict[str, OneHotGroup] = {}

        for pref in ohe_prefixes:
            cols = [f for f in feature_names if f.startswith(pref + "_")]
            if len(cols) < 2:
                continue

            value_to_col: Dict[str, str] = {}
            for c in cols:
                # suffix = ce qu'il y a après "PREF_"
                suffix = c[len(pref) + 1 :]
                value_to_col[_norm_str(suffix)] = c
                # autoriser aussi l'utilisateur à donner directement le nom complet de la dummy
                value_to_col[_norm_str(c)] = c

            groups[pref] = OneHotGroup(name=pref, columns=cols, value_to_column=value_to_col)

        return cls(feature_names=feature_names, feature_set=feature_set, groups=groups)

    def allowed_input_keys(self) -> Set[str]:
        # L'utilisateur peut fournir soit des colonnes directes, soit des colonnes "groupe" (ex ORGANIZATION_TYPE)
        return set(self.feature_set) | set(self.groups.keys())

    def to_aligned_features(
        self,
        user_row: Dict[str, Any],
        forbid_unknown_keys: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Transforme une ligne "compacte" (avec éventuellement des groupes catégoriels)
        en dict aligné sur feature_names (valeurs float ou NaN).

        Retourne: (aligned_features, stats)
        """
        # 1) Unknown keys ?
        if forbid_unknown_keys:
            unknown = set(user_row.keys()) - self.allowed_input_keys()
            if unknown:
                unk = sorted(list(unknown))[:30]
                raise InputError(
                    f"Clés inconnues: {unk} (et {max(0, len(unknown)-len(unk))} autres). "
                    "Utilise GET /features pour voir les clés autorisées."
                )

        # 2) init full vector NaN
        aligned: Dict[str, float] = {f: float("nan") for f in self.feature_names}

        # 3) gérer d'abord les groupes OHE (si présents)
        # Si un groupe est fourni (ex ORGANIZATION_TYPE="Postal"), on force toutes ses dummies à 0 et une à 1.
        for gname, g in self.groups.items():
            if gname not in user_row:
                continue
            v = user_row.get(gname)
            if _is_missing(v):
                # on laisse NaN partout (imputation fera le reste)
                continue
            key = _norm_str(v)
            col = g.value_to_column.get(key)
            if col is None:
                # aide utilisateur : quelques valeurs attendues
                examples = sorted({k for k in g.value_to_column.keys() if k and "_" not in k})[:10]
                raise InputError(
                    f"Valeur invalide pour {gname}: {v!r}. "
                    f"Exemples possibles: {examples} ..."
                )
            # set all 0 then chosen 1
            for c in g.columns:
                aligned[c] = 0.0
            aligned[col] = 1.0

        # 4) features directes
        for k, v in user_row.items():
            if k in self.groups:
                # déjà traité (groupe OHE)
                continue
            if k not in self.feature_set:
                # unknown déjà géré si forbid_unknown_keys=True, sinon on ignore
                continue

            # Heuristique simple:
            # - si colonne ressemble à FLAG_* => binaire
            # - sinon float
            if k.startswith("FLAG_") or k in {"TARGET"}:
                aligned[k] = _to_binary_or_nan(v)
            else:
                aligned[k] = _to_float_or_nan(v)

        # 5) stats utiles
        n_total = len(self.feature_names)
        n_missing = sum(1 for f in self.feature_names if np.isnan(aligned[f]))
        stats = {
            "n_features": n_total,
            "n_missing": n_missing,
            "missing_rate": float(n_missing / n_total) if n_total else 0.0,
            "used_groups": [g for g in self.groups.keys() if g in user_row and not _is_missing(user_row.get(g))],
        }
        return aligned, stats
