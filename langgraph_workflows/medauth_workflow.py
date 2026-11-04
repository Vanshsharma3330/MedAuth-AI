from typing import TypedDict


class CaseState(TypedDict, total=False):
    patient_id: str
    file_paths: list

    documents: list
    doc_types_found: list

    patient_case: dict
    completeness_result: dict
    cross_verify_result: dict

    eligibility_result: dict

    verification_result: dict
    risk_assessment: dict
    peer_review_result: dict

    applicable_rules: list

    decision: dict

    reflection: dict

    report: dict
    report_json_path: str
    report_html_path: str

    reviewer_decision: str
    human_review_required: bool
    human_review_reason: str

    status: str
    errors: list

    chat_query: str
    chat_history: list
    chat_answer: str
    chat_tools_used: str
    chat_context_used: str
    chat_sources: list
    chat_reflection: dict