from dataclasses import dataclass

@dataclass
class RuleMetadata:
    competition: str
    year: int
    section: str
    domain: str
    source: str


@dataclass
class EnggMetadata:
    domain: str
    topic: str
    source: str
    