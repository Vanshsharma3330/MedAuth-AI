from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from annotated_types import doc
import litellm
import pdfplumber
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)
from docx import Document
from langdetect import detect_langs, LangDetectException
import pytesseract
from PIL import Image
from pydantic import BaseModel


_MODELS_FILE = Path(__file__).parent.parent / "configs" / "models.yaml"
_PERSONA_FILE = Path(__file__).parent.parent / "configs" / "persona.yaml"


if not shutil.which("tesseract"):
    for _candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    ):
        if Path(_candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = _candidate
            break


# raw documents parsing and translation results.
class ParsedDocument(BaseModel):
    parsed_data: str = ""
    translated_data: str = ""
    file_path: str = ""
    file_type: str = ""
    detected_language: Optional[str] = None
    was_translated: Optional[bool] = None

class TranslationResult(BaseModel):
    detected_language: str
    confidence: float
    was_translated: bool
    translated_text: str



class TranslationAgent:
    def __init__(self):
        with open(_MODELS_FILE, encoding="utf-8") as f:
            self.config = (yaml.safe_load(f) or {}).get("translation_agent", {})
        with open(_PERSONA_FILE, encoding="utf-8") as f:
            self.persona = yaml.safe_load(f)['agents']["translation_agent"]["persona"]
        self.model = self.config.get("model", "us.amazon.nova-lite-v1:0")
        self.temperature = self.config.get("temperature", 0.0)

    def detect_language(self, raw_text: str) -> tuple[str, float]:
        stripped = raw_text.strip() if raw_text else ""
        if not stripped:
            return ("en", 0.0)
        
        try:
            langs = detect_langs(stripped)
            if langs:
                top = langs[0]
                return (str(top.lang), round(float(top.prob), 4))
            return ("en", 0.0)
        except (LangDetectException, Exception):
            return ("en", 0.0)

    def translate(self, raw_text: str) -> dict:
        lang, confidence = self.detect_language(raw_text)

        if lang == "en":
            result = TranslationResult(
                detected_language = lang,
                confidence = confidence,
                was_translated = False,
                translated_text = raw_text,
            )
            return result.model_dump()
        
        prompt = (
            "Translate the following medical document to English. "
            "Preserve every diagnosis code (ICD-10), procedure code (CPT), "
            "numeri value, date, and proper name Exactly as written. "
            "Return ONLY the translated text with no preamble or commentary.\n\n"
            f"Document:\n{raw_text}"
        )

        try: 
            response = litellm.completion(
                model = "openai/" + self.model, 
                messages = [
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": prompt},
                ],

                temperature = self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_BASE_URL"),
            )

            translated = response.choices[0].message.content.strip()

            result = TranslationResult(
                detected_language = lang,
                confidence = confidence,
                was_translated = True,
                translated_text = translated,
            )

            return result.model_dump()

        except Exception:
            result = TranslationResult(
                detected_language = lang,
                confidence = confidence,
                was_translated = False,
                translated_text = raw_text
            )
            return result.model_dump()


class DocumentAgent:
    def parse(self, file_path: str) -> ParsedDocument:
        p = Path(file_path)
        ext = p.suffix.lower()

        if ext == ".json":
            return self._parse_json(file_path)

        if ext == ".txt":
            raw_text = self._read_txt(file_path)
            file_type = "txt"
        elif ext == ".pdf":
            raw_text = self._read_pdf(file_path)
            file_type = "pdf"
        elif ext == ".docx":
            raw_text = self._read_docx(file_path)
            file_type = "docx"  
        elif ext in {".png", ".jpg", ".jpeg"}:
            raw_text = self._read_image(file_path)
            file_type = ext.lstrip(".")
        else:
            raise ValueError(f"Unsupported file type: {ext}")


        translation = TranslationAgent().translate(raw_text)

        return ParsedDocument(
            parsed_data = raw_text,
            translated_data = translation.get("translated_text", raw_text),
            file_path = str(p.resolve()),
            file_type = file_type,
            detected_language = translation.get("detected_language"),
            was_translated = translation.get("was_translated"),
        )


    def _parse_json(self, file_path: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rendered_text = json.dumps(data, indent = 2)

        return ParsedDocument(
            parsed_data = rendered_text,
            translated_data = rendered_text,
            file_path = str(Path(file_path).resolve()),
            file_type = "json",
            detected_language = None,
            was_translated = None
        )



    def _read_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _read_pdf(self, file_path: str) -> str:
        text_parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n".join(text_parts)
    
    def _read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    
    def _read_image(self, file_path: str) -> str:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    samples = [
        ("TXT",                  "data/incoming/patient001_pa_request_form.txt"),
        ("PDF",                  "data/incoming/patient006_pa_request_form.pdf"),
        ("DOCX",                 "data/incoming/patient006_clinical_notes.docx"),
        ("PNG (OCR)",            "data/incoming/patient006_physician_order.png"),
        ("JSON",                 "data/incoming/patient001_lab_results.json"),
        ("DOCX + translation",   "data/incoming/patient002_referral_letter.docx"),
    ]

    agent = DocumentAgent()
    for label, file_path in samples:
        print(f"\n========== {label}  ({file_path}) ==========")
        try:
            result = agent.parse(file_path)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")