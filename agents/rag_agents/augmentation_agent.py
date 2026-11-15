from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional

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


class AugmentationAgent:

    def __init__(self):
        model_data = _load_yaml(_MODELS_FILE).get("rag_augmentation_agent") or {}
        self.model = model_data.get("model", "google.gemma-3-4b-it")
        self.temperature = model_data.get("temperature", 0.0)
        model_persona = _load_yaml(_PERSONA_FILE)["agents"] or {}
        self.persona = model_persona.get("rag_augmentation_agent").get("persona", "you are a prompt engineer")
        self.generation_persona = model_persona.get("rag_generation_agent", {}).get("persona", "you are a clinical assistant")

    def _format_history(self, history: Optional[List[dict]]) -> str:
        if history == None or len(history) == 0:
            return "(no prior conversation)"
        rendered_str = "".join(f"{msg.get("role")}: {msg.get("content", "N/A")}\n" for msg in history)

        return rendered_str

    def _expand_query(self, query: str, history: Optional[List[dict]]) -> str:
        try:
            formated_history = self._format_history(history)
            prompt = f"""expand the given query into a precise and self-contained question using the history provided,
              eg: history has "what's the risk level?" -> assistant answered ->
           user now asks "why is that?" -> the expanded question should
           become something like "Why is the case's risk level what it
           is?".
           
           Query : {query},

           history: {formated_history},

           rule: return the output in the given json format
           {{"expanded_query": "<rewritten, self-contained question>"}}
           """
            response = litellm.complettion(
                model = "openai/" + self.model,
                messages = [{"role": "system", "content": self.persona},
                         {"role": "user", "content": prompt}],
               temperature=self.temperature,
               api_key=os.getenv("OPENAI_API_KEY"),
               api_base=os.getenv("OPENAI_BASE_URL"),
               response_format={"type": "json_object"},
            )

            result = _parse_json_content(response.choices[0].message.content)
            return result['expanded_query']

        except Exception as e:
            print(e)
            return query

    def _build_system_prompt(self) -> str:
        system_prompt = self.generation_persona + "Answer ONLY using information returned by your tools. If the tools don't contain the answer, say so plainly -- never guess or invent clinical facts, codes, or numbers."
        return system_prompt

    def augment(self, query: str, history: Optional[List[dict]] = None) -> dict:
        expanded_query = self._expand_query(query, history)
        system_prompt = self._build_system_prompt()

        return{
            "original_query": query,
            "expanded_query": expanded_query,
            "system_prompt": system_prompt
        }



if __name__ == "__main__":
    agent = AugmentationAgent()
    result = agent.augment(
        "why is that?",
        history=[
            {"role": "user", "content": "what's the risk level for this case?"},
            {"role": "assistant", "content": "The risk level is low (score 0.15)."},
        ],
    )
    print(json.dumps(result, indent=2))