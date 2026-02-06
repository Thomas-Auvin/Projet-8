# tests/test_input_adapter.py
from __future__ import annotations

import math
import pytest

from app.input_adapter import InputAdapter, InputError


def _make_adapter() -> InputAdapter:
    """
    Adapter minimal mais réaliste, avec :
    - numériques
    - flags binaires
    - OHE multi-dummies (CODE_GENDER)
    - OHE single-dummy (HOUSETYPE_MODE / EMERGENCYSTATE_MODE)
    """
    feature_names = [
        # numériques
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "DAYS_BIRTH",
        # binaires
        "FLAG_OWN_CAR",
        "FLAG_EMP_PHONE",
        # OHE multi dummies (2)
        "CODE_GENDER_F",
        "CODE_GENDER_M",
        # OHE single dummy (cas Home Credit pénible)
        "HOUSETYPE_MODE_block of flats",
        "EMERGENCYSTATE_MODE_No",
    ]
    return InputAdapter.from_feature_names(feature_names)


def test_allowed_input_keys_contains_groups_and_raw_features() -> None:
    ad = _make_adapter()
    allowed = ad.allowed_input_keys()

    # raw features présentes
    assert "AMT_CREDIT" in allowed
    assert "FLAG_OWN_CAR" in allowed

    # groupes (doivent être autorisés)
    assert "CODE_GENDER" in allowed
    assert "HOUSETYPE_MODE" in allowed
    assert "EMERGENCYSTATE_MODE" in allowed

    # dummies complètes autorisées aussi
    assert "CODE_GENDER_F" in allowed
    assert "HOUSETYPE_MODE_block of flats" in allowed


def test_unknown_keys_raise_input_error_by_default() -> None:
    ad = _make_adapter()
    with pytest.raises(InputError) as e:
        ad.to_aligned_features({"AMT_CREDIT": 1000, "UNKNOWN_COL": 1})

    msg = str(e.value)
    assert "Clés inconnues" in msg
    assert "UNKNOWN_COL" in msg


def test_unknown_keys_can_be_ignored_when_forbid_false() -> None:
    ad = _make_adapter()
    aligned, stats = ad.to_aligned_features(
        {"AMT_CREDIT": 1000, "UNKNOWN_COL": 1},
        forbid_unknown_keys=False,
    )
    assert math.isclose(aligned["AMT_CREDIT"], 1000.0)
    # unknown ignoré
    assert "UNKNOWN_COL" not in aligned
    assert stats["n_features"] == len(ad.feature_names)


def test_numeric_parsing_accepts_comma_decimal() -> None:
    ad = _make_adapter()
    aligned, _ = ad.to_aligned_features({"AMT_INCOME_TOTAL": "1,23"})
    assert math.isclose(aligned["AMT_INCOME_TOTAL"], 1.23)


def test_numeric_missing_values_become_nan() -> None:
    ad = _make_adapter()
    aligned, stats = ad.to_aligned_features(
        {
            "AMT_INCOME_TOTAL": None,
            "AMT_CREDIT": "",
            "DAYS_BIRTH": "NaN",
        }
    )
    assert math.isnan(aligned["AMT_INCOME_TOTAL"])
    assert math.isnan(aligned["AMT_CREDIT"])
    assert math.isnan(aligned["DAYS_BIRTH"])
    assert 0.0 <= stats["missing_rate"] <= 1.0


def test_binary_parsing_common_strings() -> None:
    ad = _make_adapter()
    aligned, _ = ad.to_aligned_features(
        {
            "FLAG_OWN_CAR": "oui",
            "FLAG_EMP_PHONE": "no",
        }
    )
    assert math.isclose(aligned["FLAG_OWN_CAR"], 1.0)
    assert math.isclose(aligned["FLAG_EMP_PHONE"], 0.0)


def test_binary_parsing_rejects_invalid_value() -> None:
    ad = _make_adapter()
    with pytest.raises(InputError) as e:
        ad.to_aligned_features({"FLAG_OWN_CAR": "maybe"})

    assert "Valeur binaire attendue" in str(e.value)


def test_ohe_group_sets_all_dummies_then_one_is_1() -> None:
    ad = _make_adapter()
    aligned, stats = ad.to_aligned_features({"CODE_GENDER": "f"})

    assert math.isclose(aligned["CODE_GENDER_F"], 1.0)
    assert math.isclose(aligned["CODE_GENDER_M"], 0.0)
    assert "CODE_GENDER" in stats["used_groups"]


def test_ohe_group_accepts_full_dummy_name_as_value() -> None:
    ad = _make_adapter()
    aligned, _ = ad.to_aligned_features({"CODE_GENDER": "CODE_GENDER_M"})
    assert math.isclose(aligned["CODE_GENDER_M"], 1.0)
    assert math.isclose(aligned["CODE_GENDER_F"], 0.0)


def test_ohe_group_invalid_value_gives_helpful_message() -> None:
    ad = _make_adapter()
    with pytest.raises(InputError) as e:
        ad.to_aligned_features({"CODE_GENDER": "alien"})

    msg = str(e.value)
    assert "Valeur invalide pour CODE_GENDER" in msg
    assert "Exemples possibles" in msg


def test_single_dummy_ohe_group_is_supported_when_user_sends_1_or_true() -> None:
    """
    Robustesse: certains utilisateurs envoient 0/1 pour un groupe single-dummy.
    Chez toi, un groupe single-dummy doit être considéré comme "présent" si 1/true,
    et laissé NaN si missing/None.
    """
    ad = _make_adapter()

    # 1 => activer la dummy
    aligned, stats = ad.to_aligned_features({"HOUSETYPE_MODE": 1})
    assert math.isclose(aligned["HOUSETYPE_MODE_block of flats"], 1.0)
    assert "HOUSETYPE_MODE" in stats["used_groups"]

    # 0 => activer la dummy à 0 (et pas lever d'erreur)
    aligned2, stats2 = ad.to_aligned_features({"HOUSETYPE_MODE": 0})
    assert math.isclose(aligned2["HOUSETYPE_MODE_block of flats"], 0.0)
    assert "HOUSETYPE_MODE" in stats2["used_groups"]


def test_single_dummy_ohe_group_missing_keeps_nan() -> None:
    ad = _make_adapter()
    aligned, stats = ad.to_aligned_features({"HOUSETYPE_MODE": None})
    assert math.isnan(aligned["HOUSETYPE_MODE_block of flats"])
    assert "HOUSETYPE_MODE" not in stats["used_groups"]


def test_single_dummy_ohe_group_rejects_unexpected_string_value() -> None:
    """
    Sur single-dummy, seules valeurs raisonnables attendues:
    - 0/1, true/false, oui/non
    - ou le suffix exact ('block of flats') ou la dummy entière
    Si l'utilisateur envoie une string étrange, on veut une erreur claire.
    """
    ad = _make_adapter()
    with pytest.raises(InputError) as e:
        ad.to_aligned_features(
            {"HOUSETYPE_MODE": "0.0"}
        )  # cas qui t'a posé problème sur HF

    msg = str(e.value)
    assert "Valeur invalide pour HOUSETYPE_MODE" in msg or "Valeur binaire" in msg


def test_single_dummy_ohe_group_accepts_suffix_value() -> None:
    ad = _make_adapter()
    aligned, _ = ad.to_aligned_features({"HOUSETYPE_MODE": "block of flats"})
    assert math.isclose(aligned["HOUSETYPE_MODE_block of flats"], 1.0)


def test_stats_missing_count_is_consistent() -> None:
    ad = _make_adapter()
    aligned, stats = ad.to_aligned_features({"AMT_CREDIT": 1000})
    # au moins 1 valeur non-NaN
    assert not math.isnan(aligned["AMT_CREDIT"])
    assert stats["n_features"] == len(ad.feature_names)
    assert 0 <= stats["n_missing"] <= stats["n_features"]
    assert math.isclose(stats["missing_rate"], stats["n_missing"] / stats["n_features"])
