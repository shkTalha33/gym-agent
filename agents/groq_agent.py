import os
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Securely load the client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def run_groq_agent(
    messages: list,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1024,
) -> str:
    """Groq Cloud has a generous free tier; see https://console.groq.com/docs/models"""
    order = []
    seen = set()
    for m in (model, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if m and m not in seen:
            order.append(m)
            seen.add(m)

    last_err: Optional[Exception] = None
    for m in order:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            print(f"GROQ ERROR ({m}): {str(e)}")
    return f"Error from AI Coach: {str(last_err) if last_err else 'unknown error'}"