from typing import Optional, List
from pydantic import BaseModel


class VerificationResult(BaseModel):
    plausible: bool
    confidence: float
    reasoning: str