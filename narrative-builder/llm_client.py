"""
LLM Client - Factory for a LangChain ChatOpenAI bound to a local vLLM endpoint.

vLLM exposes an OpenAI-compatible HTTP API, so we talk to it with the standard
``langchain_openai.ChatOpenAI`` client by overriding ``base_url``. The same factory
serves both the RAG generator (role="generator") and the RAGAS judge (role="judge").
"""

import json
import os
import re
from typing import Any, Optional

from config import get_config


def _resolve_api_key(config_value: Any) -> str:
    """
    Resolve the API token used to authenticate with the model server.

    Precedence: explicit config value (from .env) > VLLM_API_KEY env var >
    OPENAI_API_KEY env var. Falls back to "EMPTY" for a local, unauthenticated
    vLLM (the OpenAI client requires a non-empty value).
    """
    if config_value and str(config_value).upper() != "EMPTY":
        return str(config_value)
    return os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"


def get_chat_llm(role: str = "generator", **overrides: Any):
    """
    Build a ChatOpenAI client pointed at the configured vLLM endpoint.

    Args:
        role: "generator" uses the VLLM_* settings; "judge" uses RAGAS_JUDGE_*
              and falls back to the VLLM_* settings when those are unset.
        **overrides: Passed through to ChatOpenAI (e.g. temperature=0.0).

    Returns:
        A configured ``langchain_openai.ChatOpenAI`` instance.
    """
    # Imported lazily so the heuristic-only path never requires langchain.
    from langchain_openai import ChatOpenAI

    config = get_config()

    base_url = config.get("VLLM_BASE_URL")
    model = config.get("VLLM_MODEL")
    api_key = _resolve_api_key(config.get("VLLM_API_KEY"))

    if role == "judge":
        base_url = config.get("RAGAS_JUDGE_BASE_URL") or base_url
        model = config.get("RAGAS_JUDGE_MODEL") or model
        # Judge token falls back to the generator token when unset.
        judge_key = config.get("RAGAS_JUDGE_API_KEY")
        api_key = _resolve_api_key(judge_key) if judge_key else api_key

    params: dict = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": config.get("LLM_TEMPERATURE", 0.1),
        "max_tokens": config.get("LLM_MAX_TOKENS", 2048),
        "timeout": config.get("LLM_TIMEOUT", 120),
    }
    params.update(overrides)
    # Note: we intentionally do NOT set streaming on the client. Streaming is scoped to
    # the single generation call (see stream_to_text), not the client or the graph.
    return ChatOpenAI(**params)


def stream_to_text(llm, messages, on_token=None) -> str:
    """
    Stream a single chat completion and return the full accumulated text.

    Streaming is restricted to this one LLM call (it uses ``llm.stream()`` directly);
    it does not affect the client globally or the LangGraph graph execution.

    Args:
        llm: A ChatOpenAI (or compatible) instance.
        messages: Messages to send.
        on_token: Optional callback invoked with each non-empty text chunk
            (e.g. to render live progress).

    Returns:
        The concatenated response text.
    """
    parts = []
    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", "") or ""
        if text:
            parts.append(text)
            if on_token:
                on_token(text)
    return "".join(parts)


def extract_json(text: str) -> Optional[dict]:
    """
    Best-effort extraction of the first JSON object from a model reply.

    Handles models that wrap JSON in prose or ```json fences, or that ignore
    ``response_format``. Returns None if no valid object can be parsed.
    """
    if not text:
        return None

    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    # Fast path: the whole thing is JSON.
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: scan for the first balanced {...} block.
    start = candidate.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == "{":
                depth += 1
            elif candidate[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)

    return None
