from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class SubsystemMemory(BaseModel):
    description: Optional[str] = None
    materials: Optional[str] = None
    geometry: Optional[str] = None
    compliance_notes: Optional[str] = None
    open_issues: List[str] = Field(default_factory=list)

class ProjectMemory(BaseModel):
    competition: Optional[str] = None
    year: Optional[int] = None
    vehicle_type: Optional[str] = None
    drivetrain_type: Optional[str] = None
    chassis_type: Optional[str] = None

    subsystems: Dict[str, SubsystemMemory] = Field(default_factory=dict)
    persistent_assumptions: List[str] = Field(default_factory=list)
    contradictions_detected: List[str] = Field(default_factory=list)
