from __future__ import annotations
import glob
import json
import os
from pathlib import Path
from typing import List, Optional

import litellm
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)

from agents.rag_agents.indexing_agent import IndexingAgent
from agents.rag_agents.retrieval_agent import RetrievalAgent


_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
_PERSONA_FILE = Path(__file__).parent.parent.parent / "configs" / "persona.yaml"
_REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "json"


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_case_documents",
            "description": (
                "Semantically search this case's indexed report text for "
                "chunks relevant to a specific query. Use this for narrow, "
                "specific questions about the case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the case's indexed documents.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case_report",
            "description": (
                "Get the full finished case report for this patient, "
                "including eligibility, risk assessment, decision, and "
                "quality reflection."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_assessment",
            "description": (
                "Get only the computed risk_assessment block (risk_level, "
                "risk_score, fraud_indicators, requires_peer_review) for "
                "this patient's case."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_decision",
            "description": (
                "Get only the computed decision block (decision, "
                "confidence_score, key_findings, explanation, next_steps, "
                "routing, rule_results) for this patient's case."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_case",
            "description": (
                "Force a broad retrieval across the whole case (summary, "
                "overview, determination, risk, eligibility). Use this when "
                "the reviewer wants a general summary rather than one "
                "narrow fact."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class GenerationAgent:
    def __init__(self):
        model_file = _load_yaml(_MODELS_FILE)
        rag_agent = model_file.get("rag_generation_agent")
        self.model = rag_agent.get("model")
        self.temperature = rag_agent.get("temperature")
        self.max_tool_calls = rag_agent.get("max_tool_calls",4)
        self.history_turns = rag_agent.get("history_turns",6)

        persona_file = _load_yaml(_PERSONA_FILE)
        rag_agent_persona = persona_file.get("agents").get("rag_generation_agent")
        self.persona = rag_agent_persona.get("persona")

        self.indexing_agent = IndexingAgent()
        self.retrieval_agent = RetrievalAgent(self.indexing_agent)

        self._indexed_patients: set = set()


    def _latest_report_path(self, patient_id: str) -> Optional[Path]:
        res = sorted(glob.glob(str(_REPORTS_DIR / f"RPT-{patient_id}-*.json")))
        return Path(res[-1]) if res else None

    def _load_report(self, patient_id: str) -> dict:
        path = self._latest_report_path(patient_id)
        if not path:
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _ensure_indexed(self, patient_id: str) -> None:
        if patient_id in self._indexed_patients:
            return
        try:
            report = self._load_report(patient_id)
            if report:
                self.indexing_agent.index_case(report,patient_id)
            self._indexed_patients.add(patient_id)
        except Exception:
            return

    def answer(self, patient_id: str, augmented: dict, history: Optional[List[dict]] = None) -> dict:
        try:

            def search_case_documents(query: str):
                self._ensure_indexed(patient_id)
                return self.retrieval_agent.retrieve(query,patient_id=patient_id)
            
            def get_case_report():
                return self._load_report(patient_id)
            
            def get_risk_assessment():
                return self._load_report(patient_id).get("risk_assessment",{})
            
            def get_decision():
                return self._load_report(patient_id).get("decision",{})
            
            def summarize_case():
                self._ensure_indexed(patient_id)
                return self.retrieval_agent.retrieve(
                    "case summary overview determination risk eligibility documents",
                    patient_id=patient_id,
                    top_k=15,
                )
            
            tools = [search_case_documents, get_case_report, get_risk_assessment, get_decision, summarize_case]

            tool_impls = {tool.__qualname__.split(".")[-1] : tool for tool in tools}
            history_msg = history[-1:-self.history_turns] if len(history) >= self.history_turns else history[:]
            messages = [
                {"role":"system", "content":augmented["system_prompt"]},
                *history_msg,
                {"role":"user","content": augmented["expanded_query"]}
            ]

            tool_used: List[str] = []
            context_chunks: List[dict] = []
            final_answer=""
            for i in range(0,self.max_tool_calls):
                response = litellm.completion(
                    model = "openai/"+self.model,
                    messages = messages,
                    tools=_TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    api_base=os.getenv("OPENAI_BASE_URL")
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    messages.append(
                       msg
                    )
                    # print(msg.tool_calls)
                    for tool_call in msg.tool_calls:
                        args = json.loads(tool_call.function.arguments or "{}")
                        name = tool_call.function.name
                        result = tool_impls[name](**args)
                        messages.append(
                            {"role":"tool", "tool_call_id":tool_call.id,"content": json.dumps(result,default=str)}
                        )
                        tool_used.append(name)
                        if isinstance(result,list):
                            context_chunks.extend(result)
                else:
                    final_answer = msg.content
                    break


            if not final_answer:
                fallback = litellm.completion(
                                    model = "openai/"+self.model,
                                    messages = messages,
                                    temperature=self.temperature,
                                    api_key=os.getenv("OPENAI_API_KEY"),
                                    api_base=os.getenv("OPENAI_BASE_URL")
                                )
                final_answer = fallback.choices[0].message.content or ""
            context_used = "\n\n---\n\n".join(c["text"] for c in context_chunks)

            sources = [c["metadata"] for c in context_chunks]

            return {
                "answer": final_answer,
                "tools_used": tool_used,
                "context_used": context_used,
                "sources": sources,
                "query": augmented["expanded_query"]
            }

        except Exception as exc:
            return {
                "answer": f"I ran into an error answering that: {exc}",
                "tools_used": [],
                "context_used": "",
                "sources": [],
                "query": augmented.get("expanded_query","")
            }


if __name__ == "__main__":
    class IndexingAgent:
        def index_case(self, report, patient_id):
            return {"chunks_indexed": 0}

    class RetrievalAgent:
        def __init__(self, indexing_agent):
            self.indexing_agent = indexing_agent

        def retrieve(self, query, patient_id, top_k = None):
            return [{"text": "Determination: APPROVED (confidence 0.93). Risk level: low (score 0.15).",
                     "metadata": {"patient_id": patient_id}, "score": 1.0}]

    augmented = {
        "original_query": "what's the risk level for this case, and what was the final decision?",
        "expanded_query": "what's the risk level for this case, and what was the final decision?",
        "system_prompt": "You are a clinical-review assistant. Answer ONLY using information returned by your tools; if the tools do not contain the answer, say so plainly.",
    }

    agent = GenerationAgent()
    result = agent.answer("PAT-2024-001", augmented, history=[])
    print(json.dumps(result, indent=2))