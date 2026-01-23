from typing import Optional, List, Literal
from pydantic import BaseModel, Field

RuleConfidence = Literal["must", "should", "may", "unspecified"]

class RuleMetadataModel(BaseModel):
    competition: str
    year: int
    section: str
    domain: str
    source: str

class AnalyzedRule(BaseModel):
    text: str
    confidence: RuleConfidence
    metadata: RuleMetadataModel

class EngineeringSnippet(BaseModel):
    text: str
    domain: str
    topic: str
    source: str

class CitationModel(BaseModel):
    competition: Optional[str]
    year: Optional[int]
    section: Optional[str]
    source: str
    confidence: RuleConfidence

class AnswerPayload(BaseModel):
    answer: str
    citation: List[CitationModel]
    assumptions: List[str]=Field(default_factory=list)

class AuditFinding(BaseModel):
    claim: str
    status: Literal["COMPLIANT", "AMBIGUOUS", "LIKELY NON-COMPLIANT"]
    explanation: str
    citations:List[CitationModel]