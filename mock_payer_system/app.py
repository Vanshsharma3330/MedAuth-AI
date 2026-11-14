from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


_DATA_DIR = Path(__file__).parent.parent / "data" / "payer_mock_data"


def _load(filename: str) -> dict:
    path = _DATA_DIR / filename
    if not path.exists():
        path = Path(__file__).parent / "data" / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_eligibility_db: dict[str, dict] = {}
_prior_claims_db: dict[str, dict] = {}
_provider_db: dict[str, dict] = {}
_procedure_cost_db: dict[str, dict] = {}


def _init_db() -> None:
    global _eligibility_db, _prior_claims_db, _provider_db, _procedure_cost_db
    try:
        elig_data = _load("eligibility.json")
        _eligibility_db = {e["member_id"]: e for e in elig_data.get("eligibility", [])}

        claims_data = _load("prior_claims.json")
        _prior_claims_db = {c["patient_id"]: c for c in claims_data.get("prior_claims", [])}

        provider_data = _load("provider_verification.json")
        _provider_db = {p["npi"]: p for p in provider_data.get("providers", [])}

        cost_data = _load("procedure_costs.json")
        _procedure_cost_db = {c["procedure_code"]: c for c in cost_data.get("procedure_costs", [])}
    except Exception as exc:
        print(f"[PayerSystem] Warning – could not load data files: {exc}")


_init_db()



app = FastAPI(
    title="Mock Payer System API",
    description="Simulated payer backend for MedAuth AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class EligibilityResponse(BaseModel):
    member_id: str
    is_active: bool
    plan_name: str
    effective_date: str
    benefit_coverage_percent: float


class PriorRequest(BaseModel):
    procedure_code: str
    request_date: str


class PriorClaimsResponse(BaseModel):
    patient_id: str
    prior_pa_requests: list[PriorRequest]


class ProviderVerificationResponse(BaseModel):
    npi: str
    provider_name: str
    verified: bool
    sanctioned: bool
    specialty: str


class ProcedureCostResponse(BaseModel):
    procedure_code: str
    description: str
    standard_cost: float




@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "eligibility_loaded": len(_eligibility_db),
        "prior_claims_loaded": len(_prior_claims_db),
        "providers_loaded": len(_provider_db),
        "procedures_loaded": len(_procedure_cost_db),
    }


@app.get("/eligibility/{member_id}", response_model=EligibilityResponse)
def get_eligibility(member_id: str):
    """Retrieve insurance eligibility details by member ID."""
    record = _eligibility_db.get(member_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Member '{member_id}' not found")
    return record


@app.get("/prior-claims/{patient_id}", response_model=PriorClaimsResponse)
def get_prior_claims(patient_id: str):
    """Retrieve prior prior-authorization request history by patient ID."""
    record = _prior_claims_db.get(patient_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return record

@app.get("/provider-verification/{npi}", response_model=ProviderVerificationResponse)
def get_provider_verification(npi: str):
    """Retrieve provider verification/sanction status by NPI."""
    record = _provider_db.get(npi)
    if not record:
        raise HTTPException(status_code=404, detail=f"Provider NPI '{npi}' not found")
    return record

@app.get("/procedure-cost/{procedure_code}", response_model=ProcedureCostResponse)
def get_procedure_cost(procedure_code: str):
    """Retrieve standard cost for a CPT procedure code."""
    record = _procedure_cost_db.get(procedure_code)
    if not record:
        raise HTTPException(status_code=404, detail=f"Procedure '{procedure_code}' not found")
    return record

@app.get("/eligibility", response_model=list[EligibilityResponse])
def list_eligibility(active_only: bool = False):
    """List all eligibility records, optionally filtering to active only."""
    records = list(_eligibility_db.values())
    if active_only:
        records = [r for r in records if r.get("is_active")]
    return records



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mock_payer_system.app:app", host="0.0.0.0", port=8020, reload=True)