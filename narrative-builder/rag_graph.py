"""
RAG Graph - Corrective RAG narrative generation orchestrated with LangGraph.

The pipeline is a small StateGraph:

    retrieve --> generate --> validate --(not grounded & attempts left)--> generate
                                  |
                                  +--(grounded or out of attempts)--> END

``generate`` asks the local vLLM model to write a narrative summary plus per-article
"why it matters" reasoning, grounded strictly in the retrieved articles and citing them
by number. ``validate`` is a programmatic grounding guard that checks citations and
article ids; when it finds fabricated references it feeds corrective feedback back into
``generate`` (bounded by MAX_GENERATION_ATTEMPTS), then drops anything still invalid on
the final pass so the output is always clean.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

import sys

from config import get_config
from llm_client import extract_json, get_chat_llm, stream_to_text

logger = logging.getLogger(__name__)

# A retriever takes (topic, top_k) and returns a list of article dicts.
Retriever = Callable[[str, int], List[Dict[str, Any]]]


SYSTEM_PROMPT = (
    "You are a meticulous news analyst. You build factual narratives STRICTLY from the "
    "numbered source articles provided. Rules:\n"
    "1. Use ONLY information contained in the numbered sources. Never add outside knowledge.\n"
    "2. Support every claim in the summary with citations like [1] or [2][5], referring to the "
    "source numbers.\n"
    "3. If the sources do not support a claim, omit it. If the sources are insufficient to "
    "describe the topic, say so plainly.\n"
    "4. Respond with a SINGLE JSON object and nothing else."
)


class RAGState(TypedDict, total=False):
    topic: str
    top_k: int
    articles: List[Dict[str, Any]]
    summary: str
    timeline: List[Dict[str, Any]]
    attempts: int
    grounded: bool
    feedback: str


def format_contexts(articles: List[Dict[str, Any]], max_chars: Optional[int] = None) -> List[str]:
    """Return the per-article context strings exactly as shown to the LLM."""
    if max_chars is None:
        max_chars = get_config().get("RAG_CONTEXT_CHARS", 1200)
    contexts = []
    for article in articles:
        text = article.get("full_text", "") or article.get("headline", "")
        contexts.append(text[:max_chars])
    return contexts


def check_grounding(articles: List[Dict[str, Any]], summary: str,
                    timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Programmatic grounding guard (pure function, no LLM).

    Verifies that the summary only cites in-range source numbers and that timeline
    entries reference real retrieved article ids. Returns:
        {"problems": [str], "clean_timeline": [entries with a valid id]}
    """
    valid_ids = {a.get("id") for a in articles}
    n_sources = len(articles)
    problems: List[str] = []

    cited = {int(m) for m in re.findall(r"\[(\d+)\]", summary or "")}
    bad_citations = sorted(c for c in cited if c < 1 or c > n_sources)
    if bad_citations:
        problems.append(
            f"The summary cites non-existent source numbers {bad_citations}; "
            f"only [1]..[{n_sources}] exist."
        )
    if n_sources and not cited:
        problems.append("The summary contains no [n] citations; cite the sources you used.")

    bad_timeline = [e for e in (timeline or []) if e.get("id") not in valid_ids]
    if bad_timeline:
        problems.append(
            f"{len(bad_timeline)} timeline entries reference ids that are not among the "
            "retrieved sources; only use the exact id values shown."
        )

    clean_timeline = [e for e in (timeline or []) if e.get("id") in valid_ids]
    return {"problems": problems, "clean_timeline": clean_timeline}


def _build_context_block(articles: List[Dict[str, Any]], max_chars: int) -> str:
    lines = []
    for i, article in enumerate(articles, start=1):
        text = (article.get("full_text", "") or article.get("headline", ""))[:max_chars]
        lines.append(
            f"[{i}] (id={article.get('id')}, date={article.get('date', 'Unknown')}, "
            f"source={article.get('source', 'Unknown')})\n{text}"
        )
    return "\n\n".join(lines)


class RAGNarrator:
    """Compiles and runs the LangGraph RAG pipeline for a given retriever + LLM."""

    def __init__(self, retriever: Retriever, llm: Any = None):
        self.config = get_config()
        self.retriever = retriever
        self.llm = llm if llm is not None else get_chat_llm(role="generator")
        self.max_attempts = self.config.get("MAX_GENERATION_ATTEMPTS", 2)
        self.context_chars = self.config.get("RAG_CONTEXT_CHARS", 1200)
        self.streaming = bool(self.config.get("LLM_STREAMING", True))
        self.app = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate", self._validate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges(
            "validate",
            self._should_retry,
            {"retry": "generate", "done": END},
        )
        return graph.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _retrieve_node(self, state: RAGState) -> RAGState:
        top_k = state.get("top_k") or self.config.get("DEFAULT_TOP_K", 50)
        articles = self.retriever(state["topic"], top_k)
        logger.info("RAG retrieve: %d articles for '%s'", len(articles), state["topic"])
        return {"articles": articles, "attempts": 0}

    def _generate_node(self, state: RAGState) -> RAGState:
        articles = state["articles"]
        attempts = state.get("attempts", 0) + 1

        if not articles:
            return {
                "summary": f"No source articles were retrieved for '{state['topic']}'.",
                "timeline": [],
                "attempts": attempts,
            }

        context_block = _build_context_block(articles, self.context_chars)
        human = (
            f"Topic: {state['topic']}\n\n"
            f"Numbered source articles:\n{context_block}\n\n"
            "Write a JSON object with this exact shape:\n"
            "{\n"
            '  "summary": "<5-10 sentence narrative grounded in the sources, with [n] citations>",\n'
            '  "timeline": [ {"id": <source id>, "why_it_matters": "<1 sentence grounded in that source>"} ]\n'
            "}\n"
            "Include a timeline entry for each source that meaningfully advances the story. "
            "Use the exact id values shown in the sources."
        )
        feedback = state.get("feedback")
        if feedback:
            human += f"\n\nIMPORTANT - fix these problems from your previous attempt:\n{feedback}"

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]
        try:
            if self.streaming:
                # Accumulate streamed tokens, rendering live progress to stderr.
                def _emit(token: str):
                    sys.stderr.write(token)
                    sys.stderr.flush()

                content = stream_to_text(self.llm, messages, on_token=_emit)
                sys.stderr.write("\n")
                sys.stderr.flush()
            else:
                content = self.llm.invoke(messages).content
            parsed = extract_json(content) or {}
        except Exception as exc:  # noqa: BLE001 - surfaced to caller via empty result
            logger.error("LLM generation failed: %s", exc)
            parsed = {}

        return {
            "summary": parsed.get("summary", "") or "",
            "timeline": parsed.get("timeline", []) or [],
            "attempts": attempts,
        }

    def _validate_node(self, state: RAGState) -> RAGState:
        articles = state["articles"]
        summary = state.get("summary", "")
        timeline = state.get("timeline", [])

        check = check_grounding(articles, summary, timeline)
        problems = check["problems"]
        clean_timeline = check["clean_timeline"]

        grounded = not problems
        is_final = state.get("attempts", 0) >= self.max_attempts

        # On the final pass (or when grounded) drop anything still invalid so output is clean.
        if grounded or is_final:
            if not grounded:
                logger.warning(
                    "Grounding guard accepted with cleanup after %d attempts: %s",
                    state.get("attempts"),
                    "; ".join(problems),
                )
            return {"timeline": clean_timeline, "grounded": True, "feedback": ""}

        logger.info("Grounding guard requesting retry: %s", "; ".join(problems))
        return {"grounded": False, "feedback": "\n".join(f"- {p}" for p in problems)}

    def _should_retry(self, state: RAGState) -> str:
        return "done" if state.get("grounded") else "retry"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, topic: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the RAG graph and return the grounded narrative.

        Returns a dict: {"topic", "summary", "timeline", "articles"} where ``timeline``
        is a list of {"id", "why_it_matters"} and ``articles`` are the retrieved sources
        (so the caller can still build heuristic clusters/graph from them).
        """
        final = self.app.invoke({"topic": topic, "top_k": top_k})
        return {
            "topic": topic,
            "summary": final.get("summary", ""),
            "timeline": final.get("timeline", []),
            "articles": final.get("articles", []),
        }
