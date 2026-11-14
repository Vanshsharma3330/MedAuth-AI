from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import litellm
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)
from pydantic import BaseModel

if TYPE_CHECKING:
    from agents.document_agent import ParsedDocument


_MODELS_FILE = Path(__file__).parent.parent / "configs" / "models.yaml"
_PERSONA_FILE = Path(__file__).parent.parent / "configs" / "persona.yaml"

_EXTRACTABLE_FIELDS = [
    "patient_id", "patient_name", "date_of_birth", "member_id",
    "diagnosis_code", "procedure_code", "requested_service",
    "provider_name", "provider_npi", "urgency", "clinical_justification",
    "ordering_physician", "plan_name", "imaging_findings", "referring_physician",
    "lab_test_name", "lab_result_value", "lab_reference_range_low", "lab_reference_range_high",
]

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



class InformationExtractionAgent:

    def __init__(self):
        with open(_MODELS_FILE, 'r', encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("information_extraction_agent", {})

        with open(_PERSONA_FILE, "r", encoding="utf-8") as f:
            personas = (yaml.safe_load(f) or {}).get("agents", {})
            self.persona = personas.get("information_extraction_agent", {}).get("persona", "")

        self.model = cfg.get("model", "us.amazon.nova-lite-v1:0")
        self.temperature = cfg.get("temperature", 0.0)

    def classify(self, raw_text: str) -> str:
        if not raw_text:
            return "unknown"

        text_upper = raw_text.upper()

        if "PRIOR AUTHORIZATION REQUEST FORM" in text_upper:
            return "pa_request_form"
        if "CLINICAL NOTES" in text_upper:
            return "clinical_notes"
        if "PHYSICIAN ORDER" in text_upper:
            return "physician_order"
        if "INSURANCE CARD" in text_upper:
            return "insurance_card"
        if "IMAGING REPORT" in text_upper:
            return "imaging_report"
        if "REFERRAL LETTER" in text_upper:
            return "referral_letter"
        if "PATIENT PHOTO ID" in text_upper:
            return "photo_id"

        return "unknown"


    def extract(self, parsed: "ParsedDocument") -> DocumentRecord:
        raw_text_content = parsed.translated_data or parsed.parsed_data or ""

        if parsed.file_type == "json":
            try:
                json_data = json.loads(parsed.translated_data)
            except Exception:
                json_data = {}
            doc_type = json_data.get("document_type", "lab_results")
            extraction_method = "json+llm"
        else:
            doc_type = self.classify(raw_text_content)
            extraction_method = f"{parsed.file_type}+llm"

        fields = self._extract_fields_llm(raw_text_content, doc_type)

        return DocumentRecord(
            doc_type=doc_type,
            patient_id=fields.get("patient_id"),
            patient_name=fields.get("patient_name"),
            date_of_birth=fields.get("date_of_birth"),
            member_id=fields.get("member_id"),
            diagnosis_code=fields.get("diagnosis_code"),
            procedure_code=fields.get("procedure_code"),
            requested_service=fields.get("requested_service"),
            provider_name=fields.get("provider_name"),
            provider_npi=fields.get("provider_npi"),
            urgency=fields.get("urgency"),
            clinical_justification=fields.get("clinical_justification"),
            ordering_physician=fields.get("ordering_physician"),
            plan_name=fields.get("plan_name"),
            lab_test_name=fields.get("lab_test_name"),
            lab_result_value=fields.get("lab_result_value"),
            lab_reference_range_low=fields.get("lab_reference_range_low"),
            lab_reference_range_high=fields.get("lab_reference_range_high"),
            imaging_findings=fields.get("imaging_findings"),
            referring_physician=fields.get("referring_physician"),
            detected_language=parsed.detected_language,
            was_translated=parsed.was_translated,
            raw_text=raw_text_content,
            file_path=parsed.file_path,
            file_type=parsed.file_type,
            extraction_method=extraction_method,
        )

    def _extract_fields_llm(self, raw_text: str, doc_type: str) -> dict:
        fallback_dict = {k: None for k in _EXTRACTABLE_FIELDS}

        prompt = (
            f"You are extracting medical information from a '{doc_type}' document.\n"
            f"Extract exactly the following fields verbatim (preserve codes, numbers, names exactly):\n"
            f"{json.dumps(_EXTRACTABLE_FIELDS)}\n\n"
            f"Rules:\n"
            f"- Use null for any field that is NOT present in the text.\n"
            f"- Never guess or invent values.\n"
            f"- Return ONLY a valid JSON object matching these keys.\n\n"
            f"Document text:\n{raw_text}"
        )

        try:
            response = litellm.completion(
                model="openai/" + self.model,
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_BASE_URL"),
            )
            raw_content = response.choices[0].message.content or ""
            cleaned_json_str = self._strip_json_fences(raw_content)
            data = json.loads(cleaned_json_str)
            return {k: data.get(k) for k in _EXTRACTABLE_FIELDS}
        except Exception:
            return fallback_dict

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """
        TODO: Some gateway models wrap JSON-mode output in ```json ... ```
        fences. Strip a leading ```[language]\\n and a trailing ``` from
        `text` (via re.sub, case covered: text may also have none, in which
        case return it unchanged) and return the stripped text.
        """
        if not text:
            return "{}"

        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from types import SimpleNamespace

    _sample = Path(__file__).parent.parent / "data" / "incoming" / "patient001_pa_request_form.txt"
    _text = _sample.read_text(encoding="utf-8")
    parsed_doc = SimpleNamespace(
        parsed_data=_text, translated_data=_text, file_path=str(_sample), file_type="txt",
        detected_language="en", was_translated=False,
    )
    record = InformationExtractionAgent().extract(parsed_doc)
    print(record.model_dump_json(indent=2, exclude={"raw_text"}))