from __future__ import annotations
import re
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
load_dotenv(override=True)

import chromadb
from langchain_openai import OpenAIEmbeddings
import os


_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "models.yaml"


def _load_models_config() -> dict:
    with open(_MODELS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class IndexingAgent:
    def __init__(self):
        config_data = _load_models_config() or {}
        self.config = config_data.get("rag_indexing_agent",{})
        model_name = self.config.get("embedding_model","text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        kwargs = {
            "model": model_name,
            "openai_api_key":api_key,
        }
        if base_url:
            kwargs["openai_api_base"] = base_url
        
        self.embeddings = OpenAIEmbeddings(**kwargs)
        self.client = chromadb.EphemeralClient()
        self.collection_name = "case_reports"


    def _report_to_text(self, report: dict) -> str:
        report_id = report.get("report_id", "N/A")
        patient_id = report.get("patient_id","N/A")
        patient_name = report.get("patient_name","N/A")
        service = report.get("requested_service", "N/A")
        code = report.get("procedure_code", "N/A")
        generated_at = report.get("generated_at", "N/A")

        eligibility = report.get("eligibility_result") or {}
        risk = report.get("risk_assessment") or {}
        decision = report.get("decision") or {}
        reflection = report.get("reflection") or {}

        fraud_indicators = risk.get("fraud_indicators") or []
        fraud_str = ", ".join(str(f) for f in fraud_indicators) if fraud_indicators else "None"

        key_findings = decision.get("key_findings") or []
        finding_str = "\n - "+"\n - ".join(str(k) for k in key_findings) if key_findings else "None"

        next_steps = decision.get("next_steps") or []
        steps_str = "\n - "+ "\n - ".join(str(s) for s in next_steps) if next_steps else "None"
        lines = [
            f"Case report: {report_id} (Generated: {generated_at})",
            f"Patient: {patient_name} (ID: {patient_id})",
            f"Requested Service: {service} (Procedure Code: {code})",
            "",
            "Eligibility:",
            f" - Active Plan: {eligibility.get("plan_name","N/A")} (Active : {eligibility.get('is_active',False)})",
            f" - Eligible: {eligibility.get("eligible",False)}",
            f" - Coverage: {eligibility.get("benefit_coverage_percent",0.0)}%",
            f" - Standard Cost : {eligibility.get("standard_cost",0.0)}",
            f" - Patient Responsibility : {eligibility.get("patient_responsibility_estimate",0.0)}"
            f" - Eligibility Reason: {eligibility.get("eligibility_reason","N/A")}",
            "",
            "Risk Assessment:",
            f" - Risk Level: {risk.get('risk_level','unknown')} (Score: {risk.get('risk_score',0.0)})",
            f" - Fraud Indicators: {fraud_str}",
            f" - Required Peer Review: {risk.get('requires_peer_review',False)}",
            "",
            "Decision:",
            f" - Determination: {decision.get("decision",'Pending')} (Confidence: {decision.get('confidence_score',0.0)})"
            f" - Explanation: {decision.get("explanation","N/A")}",
            f" - Key Findings: {finding_str}",
            f" - Next Steps: {steps_str}",
            "",
            "Reflection:",
            f" - Quality Passed: {reflection.get('passed',False)}",
            f" - Groundedness: {reflection.get('groundedness',0.0)}",
            f" - Relevance: {reflection.get('relevance',0.0)}",
            f" - confidence Alignment: {reflection.get('confidence_alignment',0.0)}",
            f" - Feedback: {reflection.get('feedback','None')}",
        ]

        return "\n".join(lines).strip()


    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        paras = text.split("\n\n")
        chunks: list[str] = []
        chunk_size = self.config.get("chunk_size",256)
        chunk_overlap = self.config.get("chunk_overlap",64)
        step = max(1, chunk_size - chunk_overlap)
        for p in paras:
            para = p.strip()
            if not para:
                continue
            if len(para) < chunk_size:
                chunks.append(para)
            else:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk = para[start: end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= len(para):
                        break
                    start += step

        return chunks


    def index_case(self, report: dict, patient_id: str) -> str:
        text = self._report_to_text(report)
        chunks = self._chunk_text(text)
        
        if chunks:
            embeddings = self.embeddings.embed_documents(chunks)
            ids = [f"{patient_id}-{report.get("report_id","x")}-{i}" for i in range(len(chunks))]
            metadatas = [{"patient_id": patient_id, "report_id": report.get("report_id"), "chunk_id": i} for i in range(len(chunks))]
            collection = self.client.get_or_create_collection(self.collection_name)

            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

        return {
            "chunks_indexed": len(chunks),
            "collection_name": self.collection_name,
            "patient_id": patient_id
        }



if __name__ == "__main__":
    import json
    sample_report = {
        "report_id": "RPT-PAT-TEST-A-20260101T000000000000",
        "patient_id": "PAT-TEST-A",
        "patient_name": "Jane Carter",
        "requested_service": "MRI Lumbar Spine without contrast",
        "procedure_code": "72148",
        "eligibility_result": {
            "is_active": True, "plan_name": "MedCore PPO Gold", "benefit_coverage_percent": 80.0,
            "standard_cost": 1200.0, "patient_responsibility_estimate": 240.0,
            "eligible": True, "eligibility_reason": "Active coverage under MedCore PPO Gold.",
        },
        "risk_assessment": {"risk_level": "low", "risk_score": 0.2, "fraud_indicators": [], "requires_peer_review": False},
        "decision": {
            "decision": "APPROVED", "confidence_score": 0.9, "key_findings": [],
            "explanation": "All rules passed.", "next_steps": [],
            "routing": {"escalate_to_human": True, "reason": "policy"}, "rule_results": [],
        },
        "reflection": {"groundedness": 0.9, "relevance": 0.9, "confidence_alignment": 0.9, "passed": True, "feedback": "Grounded."},
        "generated_at": "2026-01-01T00:00:00+00:00",
    }

    agent = IndexingAgent()
    print(json.dumps(agent.index_case(sample_report, "PAT-TEST-A"), indent=2))