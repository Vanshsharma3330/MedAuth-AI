from typing import Optional, List
from pydantic import BaseModel


class PeerReviewResult(BaseModel):
    requires_peer_review: bool
    risk_level: str
    risk_Score: float
    fraud_indicators: List[str] = []
    narrative: str