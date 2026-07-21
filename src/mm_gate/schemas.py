"""Validated records shared by collection, gating, annotation, and KG stages."""

from pydantic import BaseModel, Field, HttpUrl


class ImageRecord(BaseModel):
    image_id: str
    file_title: str | None = None
    source_url: HttpUrl
    file_page_url: HttpUrl | None = None
    caption: str | None = None
    section: str | None = None
    mime_type: str | None = None
    license_name: str | None = None
    license_url: HttpUrl | None = None


class ArticleRecord(BaseModel):
    article_id: int
    title: str
    source_url: HttpUrl
    lead_summary: str
    revision_id: int | None = None
    revision_timestamp: str | None = None
    sections: list[str] = Field(default_factory=list)
    images: list[ImageRecord] = Field(default_factory=list)


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
