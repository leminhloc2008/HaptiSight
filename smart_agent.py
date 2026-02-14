import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


@dataclass
class PlanningResult:
    ok: bool
    model: str
    latency_ms: float
    output: Dict[str, Any]
    error: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class RealtimeGuidanceResult:
    ok: bool
    model: str
    latency_ms: float
    say: str
    output: Dict[str, Any]
    error: Optional[str] = None
    raw_text: Optional[str] = None


class GeminiMultiAgentPlanner:
    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.models = models or DEFAULT_MODELS
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "12"))

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

    def guide_realtime(
        self,
        user_query: str,
        scene_state: Dict[str, Any],
        profile_name: str,
        fov_deg: float,
        previous_hint: str = "",
    ) -> RealtimeGuidanceResult:
        if not self.api_key:
            return RealtimeGuidanceResult(
                ok=False,
                model="none",
                latency_ms=0.0,
                say="",
                output={},
                error="Missing GEMINI_API_KEY (or GOOGLE_API_KEY).",
            )

        preferences = self._infer_user_preferences(user_query)
        prompt = self._build_realtime_prompt(
            user_query,
            scene_state,
            profile_name,
            fov_deg,
            previous_hint,
            preferences,
        )
        t0 = time.perf_counter()
        last_error = None
        variation_temp = 0.26 + 0.10 * ((int(time.time() * 10) % 3) / 2.0)
        for model in self.models:
            try:
                text = self._call_generate_content(
                    model,
                    prompt,
                    temperature=float(variation_temp),
                    top_p=0.95,
                    max_output_tokens=220,
                )
                data = self._parse_json_output(text)
                say = self._sanitize_say(
                    str(data.get("say") or data.get("guidance") or data.get("next_step") or "")
                )
                if not say:
                    raise RuntimeError("Realtime guidance JSON missing 'say'.")
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return RealtimeGuidanceResult(
                    ok=True,
                    model=model,
                    latency_ms=latency_ms,
                    say=say,
                    output=data,
                    raw_text=text,
                )
            except Exception as exc:
                last_error = str(exc)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RealtimeGuidanceResult(
            ok=False,
            model=",".join(self.models),
            latency_ms=latency_ms,
            say="",
            output={},
            error=last_error,
        )

    @staticmethod
    def _infer_user_preferences(user_query: str) -> Dict[str, str]:
        q = " ".join(str(user_query or "").lower().split())
        prefs = {
            "tone": "calm",
            "pace": "normal",
            "detail": "balanced",
            "style": "direct",
            "user_profile": "general",
        }
        if any(k in q for k in ["elderly", "senior", "old", "grandma", "grandpa"]):
            prefs["user_profile"] = "elderly"
            prefs["pace"] = "slow"
            prefs["detail"] = "detailed"
            prefs["style"] = "gentle"
        elif any(k in q for k in ["blind", "vision-impaired", "visually impaired", "low vision"]):
            prefs["user_profile"] = "vision-impaired"
            prefs["detail"] = "detailed"
            prefs["style"] = "step-by-step"

        if any(k in q for k in ["quick", "fast", "hurry", "asap"]):
            prefs["pace"] = "fast"
            prefs["detail"] = "concise"
        if any(k in q for k in ["slowly", "careful", "carefully", "gentle"]):
            prefs["pace"] = "slow"
            prefs["style"] = "gentle"
        if any(k in q for k in ["detailed", "detail", "explain", "full guidance"]):
            prefs["detail"] = "detailed"
        if any(k in q for k in ["short", "brief", "concise"]):
            prefs["detail"] = "concise"
        if any(k in q for k in ["encourage", "supportive", "reassure"]):
            prefs["tone"] = "encouraging"
        if re.search(r"\bchild|kid|teen\b", q):
            prefs["user_profile"] = "young"
            prefs["style"] = "simple"
        return prefs

    def _call_generate_content(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.15,
        top_p: float = 0.9,
        max_output_tokens: int = 900,
    ) -> str:
        url = f"{self.base_url}/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": int(max_output_tokens),
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
    def _sanitize_say(text: str) -> str:
        clean = " ".join(str(text or "").replace("\n", " ").split())
        if len(clean) > 260:
            head = clean[:260].rstrip()
            punct_idx = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
            if punct_idx >= 80:
                clean = head[: punct_idx + 1].strip()
            else:
                space_idx = head.rfind(" ")
                clean = (head[:space_idx] if space_idx >= 80 else head).strip()
                if clean and clean[-1] not in ".!?":
                    clean = clean + "."
        if clean and clean[-1] not in ".!?":
            clean = clean + "."
        return clean

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

    @staticmethod
    def _build_realtime_prompt(
        user_query: str,
        scene_state: Dict[str, Any],
        profile_name: str,
        fov_deg: float,
        previous_hint: str,
        preferences: Dict[str, str],
    ) -> str:
        scene_json = json.dumps(scene_state, ensure_ascii=True)
        pref_json = json.dumps(preferences or {}, ensure_ascii=True)
        variation_seed = int(time.time() * 1000) % 1000000
        style_variant = (variation_seed % 5) + 1
        return f"""
You are a realtime assistive coach for a vision-impaired user.
Return ONLY compact JSON.

Inputs:
- task: {user_query}
- profile: {profile_name}
- camera_fov_deg: {fov_deg}
- previous_hint: {previous_hint}
- variation_seed: {variation_seed}
- style_variant: {style_variant}
- personalization_preferences: {pref_json}
- scene_state: {scene_json}

Rules:
- Extract the intended target object from task first, then lock onto that target.
- Do not switch to unrelated objects unless requested target is missing.
- Speak calmly, clear, and actionable.
- Use 2 to 3 complete sentences with natural cadence.
- Personalize tone/pace/detail using personalization_preferences.
- Vary sentence structure naturally across turns while preserving clear intent.
- Use depth-informed cues when available: prioritize distance3d_m, xyz_m, depth_rel, fusion_w.
- If scene_state.hand.visible is true, guide based on hand-to-target delta first (left/right/up/down/forward in small centimeters).
- Use scene_state.dangerous_objects to prioritize at most one dangerous obstacle warning on current reach path.
- Translate metric values to human-friendly cues for elderly users: clock direction (e.g., 2 o'clock), finger width, palm width, arm length.
- Avoid technical units like degrees; use relative body-based language.
- Include direction words from x/y and distance from z/d with concrete micro-motion magnitude.
- If requested target is not visible: explicitly say it is not visible and give a short safe scan strategy.
- Focus on one target only; mention at most one hazard.
- If hazard overlaps target corridor in depth, explicitly warn and suggest narrower path.
- If touch/contact is likely (very near hand-to-target), output a completion-style guidance: confirm contact, then gentle grasp and stop.
- Avoid repeating previous_hint verbatim; rephrase naturally if scene is similar.
- Use different sentence openings across consecutive hints.
- Safety first: stop/caution when distance too near or collision risk.
- Keep guidance coherent, no sentence fragments, no abrupt cutoff phrasing.
- Output in clear English only.

JSON schema:
{{
  "say": "short spoken guidance",
  "target": "object name or unknown",
  "distance_m": 0.0,
  "direction": "left|right|center",
  "confidence": 0.0,
  "safety_note": "...",
  "personalization_note": "..."
}}
""".strip()
