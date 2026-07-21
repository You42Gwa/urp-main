"""Validated records shared by collection, gating, annotation, and KG stages."""

from pydantic import BaseModel, Field


class GateDecision(BaseModel):
    representative: bool
    knowledge_contribution: bool
    image_type: str
    supported_claim_ids: list[str] = Field(default_factory=list)
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class EvidenceEdge(BaseModel):
    source: str
    relation: str
    target: str
    source_document: str
    section: str | None = None
    evidence: str
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
