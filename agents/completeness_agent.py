from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class PatientCase(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    member_id: Optional[str] = None
    diagnosis_code: Optional[str] = None
    procedure_code: Optional[str] = None
    requested_service: Optional[str] = None
    provider_name: Optional[str] = None
    provider_npi: Optional[str] = None
    urgency: Optional[str] = None
    clinical_justification: Optional[str] = None
    ordering_physician: Optional[str] = None
    plan_name: Optional[str] = None
    lab_test_name: Optional[str] = None
    lab_result_value: Optional[float] = None
    lab_reference_range_low: Optional[float] = None
    lab_reference_range_high: Optional[float] = None
    imaging_findings: Optional[str] = None
    referring_physician: Optional[str] = None
    detected_language: Optional[str] = None
    was_translated: Optional[bool] = None
    doc_types_found: List[str] = []



_RULES_FILE = Path(__file__).parent.parent / "configs" / "rules.yaml"


def _load_rules() -> dict:
    with open(_RULES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class CompletenessAgent:

    def __init__(self):
        self.rules = _load_rules()

    def merge_documents(self, records: list) -> PatientCase:
        merged_dict = {}
        doc_types_found = []

        for record in records:
            rec_dict = record.model_dump() if hasattr(record, "model_dump") else (record.dict() if hasattr(record, "dict") else dict(record))

            doc_type = rec_dict.get("doc_type")
            if doc_type and doc_type not in doc_types_found:
                doc_types_found.append(doc_type)

            for key, val in rec_dict.items():
                if val is not None and val != "":
                    merged_dict[key] = val

        if "doc_types_found" in PatientCase.model_fields:
            merged_dict["doc_types_found"] = doc_types_found


        return PatientCase(**{k: v for k, v in merged_dict.items() if k in PatientCase.model_fields})


    def check_completeness(self, doc_types_found: List[str]) -> dict:
        mandatory_docs = (
            self.rules.get("mandatory_documents")
            or self.rules.get("general", {}).get("mandatory_documents")
            or [
                "pa_request_form",
                "clinical_notes",
                "physician_order",
                "insurance_card",
                "lab_results",
                "imaging_report",
                "referral_letter",
                "photo_id"
            ]
        )

        found_set = set(doc_types_found or [])
        missing_documents = [doc for doc in mandatory_docs if doc not in found_set]
        is_complete: bool = len(missing_documents) == 0

        return{
            "complete":is_complete,
            "missing_documents": missing_documents,
            "decision": None if is_complete else "PENDED_ADDITIONAL_INFO",
        }


    def cross_verify(self, case: PatientCase) -> dict:
        required_fields = [
            ("patient_id", "Patient ID"),
            ("member_id", "Member ID"),
            ("diagnosis_code", "Diagnosis code"),
            ("procedure_code", "Procedure code"),
            ("requested_service", "Requested service"),

        ]

        data = {}
        if hasattr(case, 'model_dump'):
            try:
                data = case.model_dump()
            except Exception:
                data = {}
        elif hasattr(case, "dict"):
            try:
                data = case.dict()
            except Exception:
                data = {}
        elif isinstance(case, dict):
            data = case


        issues = []

        for field_key, field_name in required_fields:
            val = data.get(field_key) if data else None

            if val is None:
                val = getattr(case, field_key, None)
            if val is None and field_key == 'requested_service':
                val = getattr(case, "service_name", None) or getattr(case, "procedure_name", None) or (data.get("service_name") if data else None)

            if val is None or (isinstance(val, str) and str(val).strip() == ""):
                issues.append(f"{field_name} is missing from the submitted documents.")


        is_consistent = len(issues) == 0

        return {"consistent": is_consistent,
                "issues": issues,
                }


if __name__ == "__main__":
    agent = CompletenessAgent()

    print("=== check_completeness(): FALSE case (missing documents) ===")
    print(agent.check_completeness(["pa_request_form", "clinical_notes", "insurance_card"]))

    print("\n=== check_completeness(): TRUE case (all 8 mandatory docs present) ===")
    print(agent.check_completeness([
        "pa_request_form", "clinical_notes", "physician_order", "insurance_card",
        "lab_results", "imaging_report", "referral_letter", "photo_id",
    ]))

    print("\n=== cross_verify(): FALSE case (missing fields) ===")
    incomplete_case = PatientCase(patient_id="PAT-TEST", diagnosis_code=None, procedure_code=None)
    print(agent.cross_verify(incomplete_case))

    print("\n=== cross_verify(): TRUE case (all 5 required fields present) ===")
    complete_case = PatientCase(
        patient_id="PAT-TEST-2", member_id="MEM-TEST-2", diagnosis_code="M54.5",
        procedure_code="72148", requested_service="MRI Lumbar Spine without contrast",
    )
    print(agent.cross_verify(complete_case))