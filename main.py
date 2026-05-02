from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.agent_router import router
import os

app = FastAPI(title="AI Agent Service")

# Allow requests from Express backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # your Express port
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/agent")

@app.get("/health")
def health():
    return {"status": "ok"}

# Run: uvicorn main:app --reload --port 8000