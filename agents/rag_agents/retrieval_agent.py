from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
load_dotenv(override=True)

_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "models.yaml"

def _load_models_config() -> dict:
    with open(_MODELS_FILE, encodings="utf-8") as f:
        return yaml.safe_load(f) or {}


class RetrievalAgent:
    def __init__(self, indexing_agent):
        self.indexing_agent = self.indexing_agent
        config_data = _load_models_config()
        self.config = config_data["rag_retrieval_agent"]

    def retrieval(self, query: str, patient_id: str, top_k: Optional[int] = None) -> List[dict]:
        top_k =top_k or self.config.get("top_k",5)
        collection  = self.indexing_agent.client.get_or_create_collection(self.indexing_agent.collection_name)
        if collection.count() == 0:
            return []

        embed_query = self.indexing_agent.embeddings.embed_query(query)
        results = collection.query(
            query_embeddings = [embed_query],
            n_results = top_k,
            where={"patient_id": patient_id}
        )
        distances = results["distances"]
        score = [round((1/(1+distance)),4) for distance in distances[0]]
        final_res = [{"text":results["documents"][0][i], "metadata":results["metadatas"][0][i], "score": score[i] } for i in range(len(score))]
        return final_res



if __name__ == "__main__":
    import json
    import chromadb
    import hashlib
    import math
    import re
    import uuid

    class _StubEmbeddings:
        def _vec(self, text, dim=64):
            v = [0.0] * dim
            for tok in re.findall(r"[a-z0-9_]+", (text or "").lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / norm for x in v]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)


    class IndexingAgent:  # stub stand-in
        def __init__(self):
            self.client = chromadb.EphemeralClient()
            self.collection_name = f"stub_case_reports_{uuid.uuid4().hex}"
            self.embeddings = _StubEmbeddings()

        def index_case(self, report, patient_id):
            def lines(obj, prefix=""):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        yield from lines(value, f"{prefix}{key}.")
                    else:
                        yield f"{prefix}{key}: {value}"
            chunks = list(lines(report))
            collection = self.client.get_or_create_collection(self.collection_name)
            collection.add(
                ids=[f"{patient_id}-{i}" for i in range(len(chunks))],
                embeddings=self.embeddings.embed_documents(chunks),
                documents=chunks,
                metadatas=[{"patient_id": patient_id, "chunk_id": i} for i in range(len(chunks))],
            )
            return {"chunks_indexed": len(chunks)}

    sample_report = {
        "report_id": "RPT-PAT-DEMO-001", "patient_id": "PAT-DEMO-001", "patient_name": "Jane Carter",
        "requested_service": "MRI Lumbar Spine without contrast", "procedure_code": "72148",
        "eligibility_result": {"is_active": True, "plan_name": "MedCore PPO Gold", "benefit_coverage_percent": 80.0,
                               "standard_cost": 1200.0, "patient_responsibility_estimate": 240.0,
                               "eligible": True, "eligibility_reason": "Active coverage."},
        "risk_assessment": {"risk_level": "low", "risk_score": 0.15, "fraud_indicators": [], "requires_peer_review": False},
        "decision": {"decision": "APPROVED", "confidence_score": 0.93, "key_findings": [], "explanation": "All rules passed.",
                     "next_steps": [], "routing": {"escalate_to_human": True, "reason": "policy"}, "rule_results": []},
        "reflection": {"groundedness": 1.0, "relevance": 1.0, "confidence_alignment": 1.0, "passed": True, "feedback": "Grounded"},
    }


    indexing_agent = IndexingAgent()
    indexing_agent.index_case(sample_report, "PAT-DEMO-001")
    retrieval_agent = RetrievalAgent(indexing_agent)

    print("=== Indexed patient (real results expected) ===")
    print(json.dumps(retrieval_agent.retrieve("what was the determination", patient_id="PAT-DEMO-001"), indent=2))

    print("\n=== Never-indexed patient (empty, not an error) ===")
    print(json.dumps(retrieval_agent.retrieve("what was the determination", patient_id="PAT-DOES-NOT-EXIST"), indent=2))