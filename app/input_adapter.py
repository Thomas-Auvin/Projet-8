from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple
import re

import numpy as np


class InputError(ValueError):
    """Erreur d'input utilisateur (422)."""


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        # supporte float natif + numpy float (np.float64, etc.)
        return bool(np.isnan(float(v)))
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


def _is_nan(v: Any) -> bool:
    try:
        return bool(np.isnan(float(v)))
    except Exception:
        return True


def _to_float_or_nan(v: Any) -> float:
    if _is_missing(v):
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)) and not (
        isinstance(v, (float, np.floating)) and np.isnan(float(v))
    ):
        return float(v)
    if isinstance(v, str):
        # accepte "1,23" -> 1.23
        s = v.strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            # IMPORTANT: on remonte une InputError (422) plutôt qu'un ValueError brut (500)
            raise InputError(f"Valeur numérique invalide: {v!r}") from None
    raise InputError(f"Valeur numérique invalide: {v!r}")


def _to_binary_or_nan(v: Any) -> float:
    """Mappe des valeurs usuelles en 0/1."""
    if _is_missing(v):
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)):
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
        raise InputError(
            f"Valeur binaire attendue (oui/non, y/n, true/false, 0/1), reçu: {v!r}"
        )
    raise InputError(f"Valeur binaire invalide: {v!r}")


def _validate_plausibility(aligned: Dict[str, float]) -> None:
    """
    Contrôles simples de plausibilité métier (prod-hardening minimal).
    - DAYS_BIRTH (jours): négatif + âge estimé entre 18 et 120 ans
    - AMT_INCOME_TOTAL : strictement > 0 (0 ou négatif = incohérent)
    """

    # --- 1) Âge plausible (Home Credit: DAYS_BIRTH est en jours, souvent négatif)
    if "DAYS_BIRTH" in aligned:
        v = aligned.get("DAYS_BIRTH", float("nan"))
        if not _is_nan(v):
            days = float(v)
            if days > 0:
                raise InputError(
                    "DAYS_BIRTH doit être négatif (nombre de jours avant aujourd'hui)."
                )
            age_years = -days / 365.25
            if age_years < 18 or age_years > 120:
                raise InputError(
                    f"Âge incohérent (≈ {age_years:.1f} ans). Attendu: [18, 120]."
                )

    # --- 2) Revenu non nul
    if "AMT_INCOME_TOTAL" in aligned:
        inc = aligned.get("AMT_INCOME_TOTAL", float("nan"))
        if not _is_nan(inc):
            inc_f = float(inc)
            if inc_f <= 0:
                raise InputError(
                    "AMT_INCOME_TOTAL doit être strictement > 0 (revenu nul/négatif incohérent)."
                )


@dataclass(frozen=True)
class OneHotGroup:
    name: str
    columns: List[str]  # les dummies exacts attendus par le modèle
    value_to_column: Dict[str, str]  # valeur normalisée -> dummy col


@dataclass
class InputAdapter:
    feature_names: List[str]
    feature_set: Set[str]
    groups: Dict[str, OneHotGroup]  # group_name -> OneHotGroup

    # caches pour éviter de reconstruire des structures à chaque requête
    _allowed_keys: Set[str] = field(init=False, repr=False)
    _template: Dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._allowed_keys = set(self.feature_set) | set(self.groups.keys())
        self._template = {f: float("nan") for f in self.feature_names}

    @classmethod
    def from_feature_names(cls, feature_names: List[str]) -> "InputAdapter":
        """
        Construit les groupes OHE à partir d'une liste de préfixes Home Credit (catégorielles).
        On reste volontairement simple + robuste pour P8.
        """
        feature_set = set(feature_names)

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
            if len(cols) < 1:
                continue

            value_to_col: Dict[str, str] = {}
            for c in cols:
                suffix = c[len(pref) + 1 :]
                value_to_col[_norm_str(suffix)] = c
                value_to_col[_norm_str(c)] = c

            groups[pref] = OneHotGroup(
                name=pref, columns=cols, value_to_column=value_to_col
            )

        return cls(feature_names=feature_names, feature_set=feature_set, groups=groups)

    def allowed_input_keys(self) -> Set[str]:
        # copie pour éviter mutation externe
        return set(self._allowed_keys)

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
                    f"Clés inconnues: {unk} (et {max(0, len(unknown) - len(unk))} autres). "
                    "Utilise GET /features pour voir les clés autorisées."
                )

        # 2) init full vector NaN (copie rapide)
        aligned: Dict[str, float] = self._template.copy()

        # features effectivement renseignées (≠ NaN) pour calculer n_missing sans scanner feature_names
        filled: Set[str] = set()

        # 3) groupes OHE
        for gname, g in self.groups.items():
            if gname not in user_row:
                continue
            v = user_row.get(gname)
            if _is_missing(v):
                continue

            # Cas spécial: groupe avec UNE seule dummy
            if len(g.columns) == 1:
                c0 = g.columns[0]

                # accepte 0/1 numérique
                if isinstance(
                    v, (int, float, np.integer, np.floating)
                ) and not _is_missing(v):
                    fv = float(v)
                    if fv in (0.0, 1.0):
                        aligned[c0] = fv
                        filled.add(c0)
                        continue

                # accepte string (suffix ou dummy complète)
                key = _norm_str(v)
                col = g.value_to_column.get(key)
                if col is not None:
                    aligned[c0] = 1.0
                    filled.add(c0)
                    continue

                examples = sorted(
                    {k for k in g.value_to_column.keys() if k and "_" not in k}
                )[:10]
                raise InputError(
                    f"Valeur invalide pour {gname}: {v!r}. "
                    f"Exemples possibles: {examples} ... ou bien 0/1."
                )

            # Groupe multi-dummies
            key = _norm_str(v)
            col = g.value_to_column.get(key)
            if col is None:
                examples = sorted(
                    {k for k in g.value_to_column.keys() if k and "_" not in k}
                )[:10]
                raise InputError(
                    f"Valeur invalide pour {gname}: {v!r}. "
                    f"Exemples possibles: {examples} ..."
                )

            for c in g.columns:
                aligned[c] = 0.0
                filled.add(c)
            aligned[col] = 1.0
            filled.add(col)

        # 4) features directes
        for k, v in user_row.items():
            if k in self.groups:
                continue
            if k not in self.feature_set:
                continue
            if _is_missing(v):
                continue

            if k.startswith("FLAG_") or k in {"TARGET"}:
                val = _to_binary_or_nan(v)
            else:
                val = _to_float_or_nan(v)

            aligned[k] = val
            if not np.isnan(val):
                filled.add(k)

        # 4bis) plausibility checks (âge, revenu, etc.)
        _validate_plausibility(aligned)

        # 5) stats utiles (sans scan complet des features)
        n_total = len(self.feature_names)
        n_missing = n_total - len(filled)
        stats = {
            "n_features": n_total,
            "n_missing": n_missing,
            "missing_rate": float(n_missing / n_total) if n_total else 0.0,
            "used_groups": [
                g
                for g in self.groups.keys()
                if g in user_row and not _is_missing(user_row.get(g))
            ],
        }
        return aligned, stats
