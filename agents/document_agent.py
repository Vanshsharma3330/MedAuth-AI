from pydantic import BaseModel
from typing import Optional


# raw documents parsing and translation results.
class ParsedDocument(BaseModel):
    parsed_data: str = ""
    translated_data: str = ""
    file_path: str = ""
    file_type: str = ""
    detected_language: Optional[str] = None
    was_translated: Optional[bool] = None