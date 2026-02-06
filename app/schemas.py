# app/schemas.py
from __future__ import annotations

from typing import Dict, List, Union
from pydantic import BaseModel, Field

Number = Union[int, float]
Value = Union[Number, str, bool, None]  # <= IMPORTANT : pour les groupes OHE (strings)


class PredictRequest(BaseModel):
    features: Dict[str, Value] = Field(
        ...,
        description="Mapping clé -> valeur (numérique, string pour groupes OHE, bool, ou null)",
    )


class PredictResponse(BaseModel):
    request_id: str
    proba_default: float
    threshold: float
    decision: int
    model_version: str
    latency_ms: float


class PredictBatchRequest(BaseModel):
    rows: List[Dict[str, Value]] = Field(
        ...,
        description="Liste de lignes (valeurs numériques + strings possibles pour groupes OHE)",
    )


class PredictBatchItem(BaseModel):
    request_id: str
    proba_default: float
    threshold: float
    decision: int
    model_version: str
    latency_ms: float


class PredictBatchResponse(BaseModel):
    n_rows: int
    items: List[PredictBatchItem]
