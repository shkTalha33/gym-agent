import os
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the SDK
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Older IDs (gemini-pro, gemini-1.5-flash, etc.) return 404 on current v1beta — use current IDs.
# See https://ai.google.dev/gemini-api/docs/models
_LEGACY_ALIASES = {
    "gemini-pro": "gemini-2.5-flash",
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-latest": "gemini-2.5-flash",
    "gemini-1.5-flash-8b": "gemini-2.5-flash-lite",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-pro-latest": "gemini-2.5-pro",
    "gemini-flash-latest": "gemini-2.5-flash",
}

# Try in order after the normalized request model.
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

_models_cache: Optional[List[str]] = None


def _normalize_model_name(requested: Optional[str]) -> str:
    if not requested or not isinstance(requested, str):
        return "gemini-2.5-flash"
    key = requested.strip()
    return _LEGACY_ALIASES.get(key, key)


def _models_from_api() -> List[str]:
    """IDs that actually support generateContent for this API key (cached)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    found: List[str] = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            name = getattr(m, "name", "") or ""
            short = name.split("/", 1)[-1] if "/" in name else name
            if short and short not in found:
                found.append(short)
    except Exception as e:
        print(f"list_models skipped: {e}")
    _models_cache = found
    return found


def _build_model_order(requested: Optional[str]) -> List[str]:
    primary = _normalize_model_name(requested)
    order: List[str] = []
    for m in [primary, *_FALLBACK_MODELS]:
        if m not in order:
            order.append(m)

    # Append any other generateContent-capable IDs for this key (helps when defaults 404).
    for m in _models_from_api():
        if m not in order:
            order.append(m)
    return order


def run_gemini_agent(prompt: str, model: str = "gemini-2.5-flash", json_mode: bool = False) -> str:
    model_order = _build_model_order(model)
    errors: List[str] = []

    def attempt(use_json: bool) -> Optional[str]:
        kwargs = {}
        if use_json:
            kwargs["generation_config"] = genai.GenerationConfig(
                response_mime_type="application/json",
            )
        for name in model_order:
            try:
                inst = genai.GenerativeModel(name)
                response = inst.generate_content(prompt, **kwargs)
                if hasattr(response, "text") and response.text:
                    return response.text
                errors.append(f"{name}: empty or blocked response")
            except Exception as e:
                msg = str(e)
                errors.append(f"{name}: {msg}")
                print(f"GEMINI {name} (json_mode={use_json}): {msg}")
        return None

    if json_mode:
        text = attempt(True)
        if text:
            return text
        print("Gemini JSON mode failed for all models; retrying without application/json…")
        text = attempt(False)
        if text:
            return text
    else:
        text = attempt(False)
        if text:
            return text

    if errors:
        tail = "; ".join(errors[-6:])
        return f"Error from Gemini: {tail}"
    return "AI returned an empty response or was blocked by safety filters."
