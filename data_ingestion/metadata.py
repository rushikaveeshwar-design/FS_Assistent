from dataclasses import dataclass
from typing import Optional

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

@dataclass
class ImageMetadata:
    source: str
    domain: str
    competition: str
    year: int
    section: Optional[str]
    caption: Optional[str] = None
    
    