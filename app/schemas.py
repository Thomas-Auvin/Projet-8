from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

Number = Union[int, float]


class PredictRequest(BaseModel):
    features: Dict[str, Optional[Number]] = Field(
        ..., description="Mapping feature_name -> numeric value (or null)"
    )


class PredictResponse(BaseModel):
    request_id: str
    proba_default: float
    threshold: float
    decision: int
    model_version: str
    latency_ms: float


class PredictBatchRequest(BaseModel):
    rows: List[Dict[str, Optional[Number]]] = Field(
        ..., description="List of feature dicts (one per row)"
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
