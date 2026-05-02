from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.groq_agent import run_groq_agent
from agents.gemini_agent import run_gemini_agent

router = APIRouter()

# --- Request Models ---
class GroqRequest(BaseModel):
    messages: list          # [{"role": "user", "content": "Hello"}]
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 1024

class GeminiRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.5-flash"
    json_mode: bool = False

# --- Groq Endpoint ---
@router.post("/groq")
async def groq_endpoint(req: GroqRequest):
    try:
        result = run_groq_agent(req.messages, req.model, req.max_tokens)
        return {"response": result, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Gemini Endpoint ---
@router.post("/gemini")
async def gemini_endpoint(req: GeminiRequest):
    try:
        result = run_gemini_agent(req.prompt, req.model, req.json_mode)
        return {"response": result, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))