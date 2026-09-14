"""
LLM client with model fallback chain, using Groq's free-tier models.
Same resilience pattern used in the jd_resume_match project.
"""
import os
from langchain_groq import ChatGroq

# Ordered from most-capable to fastest/most-reliable fallback.
# Same chain that worked well in the jd_resume_match project.
FALLBACK_MODELS = [
    os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b"),
    os.getenv("FALLBACK_MODEL_1", "openai/gpt-oss-20b"),
    os.getenv("FALLBACK_MODEL_2", "llama-3.1-8b-instant"),
]


def get_llm(model_name: str, temperature: float = 0.1):
    return ChatGroq(model=model_name, temperature=temperature)


def invoke_with_fallback(messages, temperature: float = 0.1):
    """
    Try each model in FALLBACK_MODELS in order. Returns the first
    successful response. Raises the last exception if all fail.
    """
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            llm = get_llm(model_name, temperature=temperature)
            response = llm.invoke(messages)
            return response, model_name
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")