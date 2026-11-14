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