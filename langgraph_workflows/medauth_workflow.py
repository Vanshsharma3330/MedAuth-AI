from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)
import yaml
from langgraph.graph import END, StateGraph


_CONFIG_FILE = Path(__file__).parent.parent / "configs" / "workflow_dev.yaml"


def _load_config() -> dict:
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_config = _load_config()


def _stub(node: str) -> bool:
    return _config.get("stubs", {}).get(node, True)


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



def route_after_completeness(state: dict) -> str:
    if state.get("completeness_result", {}).get("complete", False):
        return "eligibility"
    return "reporting"


def needs_human_review(state: dict) -> tuple:
    decision = state.get("decision") or {}
    errors = state.get("errors") or []

    if not decision:
        if (
            not state.get("completeness_result", {}).get("complete", False)
            and not errors
        ):
            return (
                False,
                "Mandatory documents missing: auto-pended for additional information, no clinical determination to sign off.",
            )
        return (
            True,
            "No determination was produced (processing error): manual review required.",
        )

    routing = decision.get("routing") or {}
    if routing.get("escalate_to_human", True):
        return True, routing.get("reason", "Manual approval required.")

    reflection = state.get("reflection") or {}
    if reflection and not reflection.get("passed", True):
        return True, "Decision self-check failed: manual review required."
    if errors:
        return True, "Processing errors occurred: manual review required."

    return False, routing.get("reason", "No manual approval required.")


def _post_index_status(state: dict) -> str:
    return (
        "case_indexed" if state.get("human_review_required", True) else "auto_finalized"
    )


def _auto_finalize(state: dict, review_required: bool, review_reason: str) -> dict:
    if review_required:
        return {}
    outcome = {"human_review_required": False, "human_review_reason": review_reason}
    try:
        from agents.human_review_agent import HumanReviewAgent

        has_decision = bool(state.get("decision"))
        if has_decision:
            HumanReviewAgent().auto_finalize(
                state.get("patient_id", ""),
                {**state.get("report", {}), "decision": state["decision"]},
                review_reason,
            )
        else:
            HumanReviewAgent().log_event(
                state.get("patient_id", ""),
                "auto_pended_missing_documents",
                {"reason": review_reason},
            )
        outcome["reviewer_decision"] = (
            "auto_approved" if has_decision else "not_required"
        )
    except Exception as exc:
        logger.warning("Auto-finalize audit logging failed: %s", exc)
        outcome["reviewer_decision"] = "not_required"
    return outcome


def route_after_indexing(state: dict) -> str:
    return(
        "human_review" if state.get("human_review_required", True) else "auto_finalize"
    )


def build_case_graph():
    graph = StateGraph(CaseState)
    graph.add_node("document", document_node)
    graph.add_node("completeness", completeness_node)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("risk", risk_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("decision", decision_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("reporting", reporting_node)
    graph.add_node("indexing", indexing_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("document")

    graph.add_edge("document", "completeness")
    graph.add_conditional_edges(
        "completeness",
        route_after_completeness,
        {"eligibility": "eligibility", "reporting": "reporting"},
    )
    graph.add_edge("eligibility", "risk")
    graph.add_edge("risk", "retrieval")
    graph.add_edge("retrieval", "decision")
    graph.add_edge("decision", "reflection")
    graph.add_edge("reflection", "reporting")
    graph.add_edge("reporting", "indexing")
    graph.add_conditional_edges(
        "indexing",
        route_after_indexing,
        {"human_review": "human_review", "auto_finalize": END},
    )
    graph.add_edge("human_review", END)

    return graph.compile()



def run_case(patient_id: str, file_paths: list) -> dict:
    graph = build_case_graph()
    initial_state = {
        "patient_id": patient_id,
        "file_paths": file_paths,
        "status": "started",
        "errors": [],
    }
    return graph.invoke(initial_state)


def _chat_stub(state: dict) -> dict:
    logger.info("[Stub:Chat] returning canned chat answer")
    return {
        **state,
        "chat_answer": "Stub: this case was approved; risk level is low.",
        "chat_tools_used": ["get_decision", "get_risk_assessment"],
        "chat_context_used": "Stub context.",
        "chat_sources": [],
        "chat_reflection": {
            "answer_relevance": 0.9,
            "groundedness": 0.9,
            "context_relevance": 0.9,
            "passed": True,
            "feedback": "Stub: grounded.",
        },
        "status": "chat_complete",
    }


def _chat_real(state: dict) -> dict:
    from agents.rag_agents.augmentation_agent import AugmentationAgent
    from agents.rag_agents.generation_agent import GenerationAgent
    from agents.rag_agents.reflection_agent import ReflectionAgent

    patient_id = state.get("patient_id", "")
    query = state.get("chat_query", "")
    history = state.get("chat_history") or []

    try:
        augmented = AugmentationAgent().augment(query, history=history)
        result = GenerationAgent().answer(patient_id, augmented, history=history)
        reflection = ReflectionAgent().evaluate(
            result.get("query", query),
            result.get("answer", ""),
            result.get("context_used", ""),
        )
        return {
            **state,
            "chat_answer": result.get("answer", ""),
            "chat_tools_used": result.get("tools_used", []),
            "chat_context_used": result.get("context_used", ""),
            "chat_sources": result.get("sources", []),
            "chat_reflection": reflection,
            "status": "chat_complete",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Case chat error: {exc}")
        return {
            **state,
            "chat_answer": f"I ran into an error answering that: {exc}",
            "chat_tools_used": [],
            "chat_context_used": "",
            "chat_sources": [],
            "chat_reflection": {},
            "status": "chat_complete",
            "errors": errors,
        }


def chat_node(state: dict) -> dict:
    return _chat_stub(state) if _stub("chat") else _chat_real(state)


def build_chat_graph():
    graph = StateGraph(CaseState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()

def run_case_chat(patient_id: str, query: str, history: list | None = None) -> dict:
    graph = build_chat_graph()
    initial_state = {
        "patient_id": patient_id,
        "chat_query": query,
        "chat_history": history or [],
        "status": "chat_started",
        "errors": [],
    }
    return graph.invoke(initial_state)


def _document_stub(state: dict) -> dict:
    logger.info("[Stub:Document] returning hardcoded extracted documents")
    return {
        **state,
        "documents": [
            {
                "doc_type": "pa_request_form",
                "patient_id": state.get("patient_id", "PAT-STUB-001"),
                "patient_name": "Stub Patient",
                "member_id": "MEM-STUB",
                "diagnosis_code": "M54.5",
                "procedure_code": "72148",
                "requested_service": "MRI Lumbar Spine without contrast",
                "urgency": "routine",
            }
        ],
        "doc_types_found": [
            "pa_request_form",
            "clinical_notes",
            "physician_order",
            "insurance_card",
        ],
        "status": "documents_extracted",
    }


def _completeness_stub(state: dict) -> dict:
    logger.info("[Stub:Completeness] returning complete + consistent")
    return {
        **state,
        "patient_case": {
            "patient_id": state.get("patient_id", "PAT-STUB-001"),
            "patient_name": "Stub Patient",
            "member_id": "MEM-STUB",
            "diagnosis_code": "M54.5",
            "procedure_code": "72148",
            "requested_service": "MRI Lumbar Spine without contrast",
        },
        "completeness_result": {
            "complete": True,
            "missing_documents": [],
            "decision": None,
        },
        "cross_verify_result": {"consistent": True, "issues": []},
        "status": "completeness_checked",
    }


def _eligibility_stub(state: dict) -> dict:
    logger.info("[Stub:Eligibility] returning stub active coverage")
    return {
        **state,
        "eligibility_result": {
            "is_active": True,
            "plan_name": "MedCore PPO Gold",
            "benefit_coverage_percent": 80.0,
            "standard_cost": 1200.0,
            "patient_responsibility_estimate": 240.0,
            "eligible": True,
            "eligibility_reason": "Stub: active coverage.",
        },
        "status": "eligibility_checked",
    }


def _risk_stub(state: dict) -> dict:
    logger.info("[Stub:Risk] returning low-risk stub")
    risk_assessment = {
        "risk_level": "low",
        "risk_score": 0.20,
        "fraud_indicators": [],
        "requires_peer_review": False,
    }
    return {
        **state,
        "verification_result": {
            "plausible": True,
            "confidence": 0.9,
            "reasoning": "Stub: clinical justification plausibly supports the request.",
        },
        "risk_assessment": risk_assessment,
        "peer_review_result": {
            **risk_assessment,
            "narrative": "Stub: low risk, no fraud indicators, peer review not required.",
        },
        "status": "risk_assessed",
    }


def _retrieval_stub(state: dict) -> dict:
    logger.info("[Stub:Retrieval] returning stub applicable rules")
    return {
        **state,
        "applicable_rules": [
            {
                "rule_id": "MRI-1",
                "rule_text": "Requires active insurance eligibility (is_active=True) at time of service.",
            },
        ],
        "status": "policy_retrieved",
    }


def _decision_stub(state: dict) -> dict:
    logger.info("[Stub:Decision] returning stub approval determination")
    return {
        **state,
        "decision": {
            "decision": "APPROVED",
            "confidence_score": 0.85,
            "key_findings": ["Stub: all rules passed."],
            "explanation": "Stub explanation.",
            "next_steps": ["No manual approval required - finalized automatically."],
            "routing": {
                "escalate_to_human": False,
                "reason": "Stub: clean high-confidence approval, no manual review needed.",
            },
            "rule_results": [
                {
                    "rule_id": "MRI-1",
                    "rule_text": "Stub rule",
                    "passed": True,
                    "reason": "Stub",
                }
            ],
        },
        "status": "decision_synthesized",
    }


def _reflection_stub(state: dict) -> dict:
    logger.info("[Stub:Reflection] returning stub pass")
    return {
        **state,
        "reflection": {
            "groundedness": 0.9,
            "relevance": 0.9,
            "confidence_alignment": 0.9,
            "passed": True,
            "feedback": "Stub: determination is grounded.",
        },
        "status": "reflected",
    },


def _reporting_stub(state: dict) -> dict:
    logger.info("[Stub:Reporting] building minimal report")
    case = state.get("patient_case", {})
    review_required, review_reason = needs_human_review(state)
    return {
        **state,
        "human_review_required": review_required,
        "human_review_reason": review_reason,
        **(
            {}
            if review_required
            else {
                "reviewer_decision": (
                    "auto_approved" if state.get("decision") else "not_required"
                )
            }
        ),
        "report": {
            "report_id": f"RPT-{state.get('patient_id', 'PAT-STUB-001')}-stub",
            "patient_id": state.get("patient_id", "PAT-STUB-001"),
            "patient_name": case.get("patient_name", "Stub Patient"),
            "decision": state.get("decision", {}),
            "eligibility_result": state.get("eligibility_result", {}),
            "risk_assessment": state.get("risk_assessment", {}),
            "human_review_required": review_required,
        },
        "report_json_path": "",
        "report_html_path": "",
        "status": "reported",
    }



def _indexing_stub(state: dict) -> dict:
    logger.info("[Stub:Indexing] pretending to index the case report")
    return {**state, "status": _post_index_status(state)}


def _human_review_stub(state: dict) -> dict:
    logger.info(
        "[Stub:HumanReview] manual approval required - pausing for clinical reviewer decision"
    )
    return {**state, "reviewer_decision": "pending", "status": "awaiting_human_review"}


def _document_real(state: dict) -> dict:
    from agents.document_agent import DocumentAgent
    from agents.information_extraction_agent import InformationExtractionAgent

    documents, doc_types = [], []
    errors = list(state.get("errors") or [])

    try:
        parser = DocumentAgent()
        extractor = InformationExtractionAgent()
    except Exception as exc:
        errors.append(f"Document/Information Extraction agent init error: {exc}")
        return {
            **state,
            "documents": documents,
            "doc_types_found": doc_types,
            "status": "documents_extracted",
            "errors": errors,
        }

    for file_path in state.get("file_paths", []):
        try:
            parsed = parser.parse(file_path)
            record = extractor.extract(parsed)
            documents.append(record.model_dump())
            doc_types.append(record.doc_type)
        except Exception as exc:
            errors.append(f"Document extraction error ({file_path}): {exc}")
    return {
        **state,
        "documents": documents,
        "doc_types_found": doc_types,
        "status": "documents_extracted",
        "errors": errors,
    }


def _completeness_real(state: dict) -> dict:
    from agents.completeness_agent import CompletenessAgent

    try:
        agent = CompletenessAgent()
        case = agent.merge_documents(state.get("documents", []))
        completeness_result = agent.check_completeness(state.get("doc_types_found", []))
        cross_verify_result = agent.cross_verify(case)
        return {
            **state,
            "patient_case": case.model_dump(),
            "completeness_result": completeness_result,
            "cross_verify_result": cross_verify_result,
            "status": "completeness_checked",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Completeness check error: {exc}")
        return {**state, "status": "completeness_checked", "errors": errors}


def _eligibility_real(state: dict) -> dict:
    from agents.eligibility_agent import EligibilityAgent

    case = state.get("patient_case", {})
    try:
        agent = EligibilityAgent()
        result = agent.check_eligibility(
            member_id=case.get("member_id", ""),
            procedure_code=case.get("procedure_code", ""),
        )
        return {
            **state,
            "eligibility_result": result.model_dump(),
            "status": "eligibility_checked",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Eligibility check error: {exc}")
        return {**state, "status": "eligibility_checked", "errors": errors}


def _risk_real(state: dict) -> dict:
    from agents.verification_agent import VerificationAgent
    from agents.risk_agent import RiskAgent
    from agents.peer_review_agent import PeerReviewAgent

    case = state.get("patient_case", {})
    errors = list(state.get("errors") or [])

    try:
        verification = (
            VerificationAgent()
            .verify_clinical_justification(
                diagnosis_code=case.get("diagnosis_code", ""),
                procedure_code=case.get("procedure_code", ""),
                clinical_justification=case.get("clinical_justification"),
            )
            .model_dump()
        )
    except Exception as exc:
        errors.append(f"Verification error: {exc}")
        verification = {
            "plausible": True,
            "confidence": 0.0,
            "reasoning": f"Verification error: {exc}",
        }

    try:
        risk_result = (
            RiskAgent()
            .assess(
                patient_id=case.get("patient_id", ""),
                procedure_code=case.get("procedure_code", ""),
                diagnosis_code=case.get("diagnosis_code", ""),
                provider_npi=case.get("provider_npi", ""),
                lab_result_value=case.get("lab_result_value"),
                lab_reference_range_low=case.get("lab_reference_range_low"),
                lab_reference_range_high=case.get("lab_reference_range_high"),
            )
            .model_dump()
        )
    except Exception as exc:
        errors.append(f"Risk assessment error: {exc}")
        risk_result = {
            "risk_level": "unknown",
            "risk_score": 0.0,
            "fraud_indicators": [],
            "requires_peer_review": True,
        }

    try:
        peer_review = PeerReviewAgent().synthesize(risk_result).model_dump()
    except Exception as exc:
        errors.append(f"Peer review synthesis error: {exc}")
        peer_review = {
            **risk_result,
            "narrative": f"Peer review narrative unavailable: {exc}",
        }

    return {
        **state,
        "verification_result": verification,
        "risk_assessment": risk_result,
        "peer_review_result": peer_review,
        "status": "risk_assessed",
        "errors": errors,
    }


def _retrieval_real(state: dict) -> dict:
    from agents.decision_agent import DecisionAgent

    case = state.get("patient_case", {})
    try:
        agent = DecisionAgent()
        rules = agent.get_applicable_rules(case.get("procedure_code", ""))
        return {**state, "applicable_rules": rules, "status": "policy_retrieved"}
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Policy rule lookup error: {exc}")
        return {
            **state,
            "applicable_rules": [],
            "status": "policy_retrieved",
            "errors": errors,
        }


def _decision_real(state: dict) -> dict:
    from agents.decision_agent import DecisionAgent

    try:
        agent = DecisionAgent()
        rule_results = agent.apply_rules(
            state.get("applicable_rules", []),
            state.get("eligibility_result", {}),
            state.get("risk_assessment", {}),
        )
        decision = agent.synthesize(
            rule_results,
            state.get("eligibility_result", {}),
            state.get("risk_assessment", {}),
        )
        return {
            **state,
            "decision": decision.model_dump(),
            "status": "decision_synthesized",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Decision synthesis error: {exc}")
        return {**state, "status": "decision_synthesized", "errors": errors}


def _reflection_real(state: dict) -> dict:
    from agents.decision_agent import Decision, DecisionAgent

    try:
        agent = DecisionAgent()
        decision = Decision(**state.get("decision", {}))
        result = agent.reflect(decision)
        return {**state, "reflection": result, "status": "reflected"}
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Reflection error: {exc}")
        return {**state, "status": "reflected", "errors": errors}

def _reporting_real(state: dict) -> dict:
    from agents.reporting_agent import ReportingAgent

    review_required, review_reason = needs_human_review(state)
    try:
        agent = ReportingAgent()
        report = agent.generate(
            patient_case=state.get("patient_case", {}),
            eligibility_result=state.get("eligibility_result", {}),
            risk_assessment=state.get("risk_assessment", {}),
            decision=state.get("decision", {}),
            reflection=state.get("reflection", {}),
            human_review_required=review_required,
        )
        auto = _auto_finalize(
            {**state, "report": report}, review_required, review_reason
        )
        return {
            **state,
            "report": report,
            "human_review_required": review_required,
            "human_review_reason": review_reason,
            **auto,
            "report_json_path": report.get("report_json_path", ""),
            "report_html_path": report.get("report_html_path", ""),
            "status": "reported",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Reporting error: {exc}")
        return {
            **state,
            "status": "reported",
            "errors": errors,
            "human_review_required": True,
            "human_review_reason": f"Reporting error: {exc}",
        }


def _indexing_real(state: dict) -> dict:
    from agents.rag_agents.indexing_agent import IndexingAgent

    report = state.get("report", {})
    patient_id = state.get("patient_id", "")
    try:
        if report:
            IndexingAgent().index_case(report, patient_id)
            return {**state, "status": _post_index_status(state)}
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Case indexing error: {exc}")
        return {**state, "status": _post_index_status(state), "errors": errors}


def _human_review_real(state: dict) -> dict:
    from agents.human_review_agent import HumanReviewAgent

    report = state.get("report", {})
    try:
        agent = HumanReviewAgent()
        agent.request_review(state.get("patient_id", ""), report)
        return {
            **state,
            "reviewer_decision": "pending",
            "status": "awaiting_human_review",
        }
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"Human review gate error: {exc}")
        return {
            **state,
            "reviewer_decision": "pending",
            "status": "awaiting_human_review",
            "errors": errors,
        }





def document_node(state: dict) -> dict:
    return _document_stub(state) if _stub("document") else _document_real(state)


def completeness_node(state: dict) -> dict:
    return (
        _completeness_stub(state)
        if _stub("completeness")
        else _completeness_real(state)
    )


def eligibility_node(state: dict) -> dict:
    return (
        _eligibility_stub(state) if _stub("eligibility") else _eligibility_real(state)
    )

def risk_node(state: dict) -> dict:
    return _risk_stub(state) if _stub("risk") else _risk_real(state)


def retrieval_node(state: dict) -> dict:
    return _retrieval_stub(state) if _stub("retrieval") else _retrieval_real(state)


def decision_node(state: dict) -> dict:
    return _decision_stub(state) if _stub("decision") else _decision_real(state)

def reflection_node(state: dict) -> dict:
    return _reflection_stub(state) if _stub("reflection") else _reflection_real(state)


def reporting_node(state: dict) -> dict:
    return _reporting_stub(state) if _stub("reporting") else _reporting_real(state)


def indexing_node(state: dict) -> dict:
    return _indexing_stub(state) if _stub("indexing") else _indexing_real(state)


def human_review_node(state: dict) -> dict:
    return (
        _human_review_stub(state)
        if _stub("human_review")
        else _human_review_real(state)
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s"
    )
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "PAT-2024-001"
    base = Path(__file__).parent.parent / "data" / "incoming"
    files = sorted(str(p) for p in base.glob("patient001_*"))
    state = run_case(patient_id, files)
    print(f"Status   : {state.get('status')}")
    print(f"Decision : {state.get('decision', {}).get('decision')}")