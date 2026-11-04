from typing import Optional
from pydantic import BaseModel


# medical fields extracted from document
class DocumentRecord(BaseModel):
    doc_type: Optional[str] = None
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
    raw_text: str = ""
    file_path: str = ""
    file_type: str = ""
    extraction_method: str = ""