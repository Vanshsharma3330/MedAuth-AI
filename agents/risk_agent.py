from typing import Optional, List
from pydantic import BaseModel



class RiskAssessment(BaseModel):
    risk_level: str
    risk_score: float
    fraud_indicators: List[str] = []
    requires_peer_review: bool