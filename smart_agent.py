import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


@dataclass
class PlanningResult:
    ok: bool
    model: str
    latency_ms: float
    output: Dict[str, Any]
    error: Optional[str] = None
    raw_text: Optional[str] = None


class GeminiMultiAgentPlanner:
    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.models = models or DEFAULT_MODELS
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.timeout_s = 35

    def plan(
        self,
        user_query: str,
        scene_state: Dict[str, Any],
        profile_name: str,
        fov_deg: float,
    ) -> PlanningResult:
        if not self.api_key:
            return PlanningResult(
                ok=False,
                model="none",
                latency_ms=0.0,
                output={},
                error="Missing GEMINI_API_KEY (or GOOGLE_API_KEY).",
            )

        prompt = self._build_prompt(user_query, scene_state, profile_name, fov_deg)
        t0 = time.perf_counter()
        last_error = None
        for model in self.models:
            try:
                text = self._call_generate_content(model, prompt)
                data = self._parse_json_output(text)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return PlanningResult(ok=True, model=model, latency_ms=latency_ms, output=data, raw_text=text)
            except Exception as exc:
                last_error = str(exc)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PlanningResult(ok=False, model=",".join(self.models), latency_ms=latency_ms, output={}, error=last_error)

    def _call_generate_content(self, model: str, prompt: str) -> str:
        url = f"{self.base_url}/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.15,
                "topP": 0.9,
                "maxOutputTokens": 900,
            },
        }
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No Gemini candidate output: {data}")

        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            raise RuntimeError(f"Gemini returned empty text: {data}")
        return text

    @staticmethod
    def _extract_json_block(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        # Try fenced block
        if "```json" in stripped:
            start = stripped.find("```json") + len("```json")
            end = stripped.find("```", start)
            if end > start:
                return stripped[start:end].strip()
        # Fallback: outermost braces
        l = stripped.find("{")
        r = stripped.rfind("}")
        if l != -1 and r != -1 and r > l:
            return stripped[l : r + 1]
        return stripped

    def _parse_json_output(self, text: str) -> Dict[str, Any]:
        block = self._extract_json_block(text)
        data = json.loads(block)
        if not isinstance(data, dict):
            raise RuntimeError("Planner output is not a JSON object.")
        return data

    @staticmethod
    def _build_prompt(user_query: str, scene_state: Dict[str, Any], profile_name: str, fov_deg: float) -> str:
        scene_json = json.dumps(scene_state, ensure_ascii=True)
        return f"""
You are a safety-aware multi-agent assistive planner for vision-impaired object reaching.
Run 4 virtual agents internally and output ONLY valid JSON.

System context:
- Camera-only perception with YOLO + MiDaS fused distance and 3D coordinates.
- Coordinates are in camera frame: x(left-right), y(up-down), z(depth forward), all meters.
- Profile: {profile_name}
- Camera FOV degrees: {fov_deg}
- User query: {user_query}
- Scene state JSON: {scene_json}

Agents:
1) query_reasoner: understand target object + task intent.
2) spatial_reasoner: locate target and reachable approach vector in 3D.
3) safety_assessor: identify collision risks and no-go actions.
4) path_planner: produce executable step-by-step guidance.

Hard constraints:
- Prioritize safety over speed.
- Keep instructions short, imperative, and physically actionable.
- If target is not visible, guide user to search safely.
- Output in English.

Return JSON schema:
{{
  "intent_agent": {{
    "task": "...",
    "target_object": "...",
    "confidence": 0.0
  }},
  "spatial_agent": {{
    "target_visible": true,
    "target_xyz_m": [x, y, z],
    "target_distance_m": 0.0,
    "recommended_approach": "..."
  }},
  "safety_agent": {{
    "risk_level": "low|medium|high",
    "hazards": ["..."],
    "collision_objects": ["..."],
    "safety_rules": ["..."]
  }},
  "path_agent": {{
    "micro_steps": ["..."],
    "stop_conditions": ["..."],
    "fallback_actions": ["..."]
  }},
  "final_guidance": {{
    "summary": "...",
    "speakable_guidance": ["...", "...", "..."]
  }}
}}
""".strip()

