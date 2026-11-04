from pydantic import BaseModel
from typing import Optional


class EligibilityResult(BaseModel):
    is_active: bool
    plan_name: str
    benefit_coverage_percent: float
    standard_cost: float
    patient_responsibility_estimate: float
    eligible: bool
    eligibility_reason: str