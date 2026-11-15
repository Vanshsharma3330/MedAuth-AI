from __future__ import annotations
import json
import os
from pathlib import Path

import litellm
import yaml
from dotenv import load_dotenv
load_dotenv(override=True)

_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
_PERSONA_FILE = Path(__file__).parent.parent.parent / "configs" / "persona.yaml"

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _parse_json_content(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


class ReflectionAgent:
    def __init__(self):
        models_config = _load_yaml(_MODELS_FILE).get("rag_reflection_agent", {})
        persona_config = _load_yaml(_PERSONA_FILE).get("agents", {})

        self.model = models_config.get("model", "google.gemma-3-4b-it")
        self.temperature = models_config.get("temperature", 0.0)
        self.min_answer_relevance = models_config.get("min_answer_relevance", 0.60)
        self.min_groundedness = models_config.get("min_groundedness", 0.60)
        self.min_context_relevance = models_config.get("min_context_relevance", 0.60)
        self.persona = persona_config.get("rag_reflection_agent", {}).get("persona", "")


    def evaluate(self, query: str, answer: str, context_used: str) -> dict:
        try:
          prompt = f"""on the basis of answer, query, context_used(maybe empty) score the following in [0.0,1.0]
          -  answer_relevance: does the answer directly address the question?
            - groundedness: is every claim in the answer supported by context_used? If no context is provided, groundedness reflects whether the answer correctly says it couldn't find the information rather than inventing a clinicalfact, code, or number.
          - context_relevance: is context_used actually relevant to the question? If there is no context, this should come back low.
                
            return in JSON:
            {{"answer_relevance": <float>, "groundedness": <float>,"context_relevance": <float>, "feedback": "<1-2 sentence explanation>"}}

            query:
            {query}

            answer:
            {answer}

            context_used:
            {context_used}
            """
          response = litellm.completion(
              model="openai/"+self.model,
              messages = [
                  {"role":"system", "content":self.persona},
                  {"role":"user", "content":prompt},
              ],
              temperature=self.temperature,
              api_key=os.getenv("OPENAI_API_KEY"),
              api_base=os.getenv("OPENAI_BASE_URL"),
              response_format={"type": "json_object"}
          )

          print(response.choices[0].message)

          msg = _parse_json_content(response.choices[0].message.content)
          print("msg: -> ",msg)

          answer_relevance = msg.get("answer_relevance",0.0)
          groundedness = msg.get("groundedness",0.0)
          context_relevance = msg.get("context_relevance",0.0) 

          passed = (msg.get("answer_relevance",0.0) >= self.min_answer_relevance) and (msg.get("groundedness",0.0) >= self.min_groundedness) and (msg.get("context_relevance",0.0) >= self.min_context_relevance)

          feedback = f"passed: {passed}, answer_relevance = {answer_relevance}, context_relevance = {context_relevance}, groundedness = {groundedness}"

          return {
              "answer_relevance": answer_relevance,
              "groundedness":groundedness,
              "context_relevance":context_relevance,
              "passed":passed,
              "feedback":feedback
          }
        except Exception as exc:
            return{  
              "answer_relevance": 0.0,
              "groundedness":0.0,
              "context_relevance":0.0,
              "passed":False,
              "feedback": f"Evaluation could not be completed: {exc}"
            }


if __name__ == "__main__":
    agent = ReflectionAgent()
    result = agent.evaluate(
        "what's the total?",
        "The total is $100.",
        "Total: $100.",
    )
    print(json.dumps(result, indent=2))