"""
RAGAS Evaluation - Measure hallucination / grounding quality of the RAG narratives.

For every topic in a golden eval set (JSONL with ``topic`` and ``reference``) this:
  1. runs the LangGraph RAG narrator to produce a grounded summary + retrieved contexts,
  2. assembles a RAGAS evaluation sample, and
  3. scores it with faithfulness, answer relevancy, context precision and context recall.

The judge LLM and embeddings are local: the judge points at the same vLLM endpoint by
default (configurable via RAGAS_JUDGE_*), and embeddings reuse the project's EMBEDDING_MODEL.

Usage:
  python evaluation.py                  # uses config EVAL_SET_PATH and builds the index
  python narrative_builder.py --evaluate
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import get_config
from llm_client import get_chat_llm
from rag_graph import format_contexts
from utils import setup_logging

logger = logging.getLogger(__name__)


def load_eval_set(path: str) -> List[Dict[str, str]]:
    """Load a JSONL eval set; each line is {"topic": ..., "reference": ...}."""
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")

    rows: List[Dict[str, str]] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSONL line %d: %s", line_num, e)
                continue
            if row.get("topic") and row.get("reference"):
                rows.append(row)
            else:
                logger.warning("Skipping line %d: needs both 'topic' and 'reference'", line_num)
    return rows


def _build_samples(eval_rows: List[Dict[str, str]], builder, top_k: int) -> List[Dict[str, Any]]:
    """Run retrieval + RAG generation for each eval row into RAGAS sample dicts."""
    narrator = builder._get_narrator()
    config = get_config()
    context_chars = config.get("RAG_CONTEXT_CHARS", 1200)

    samples = []
    for row in eval_rows:
        topic = row["topic"]
        logger.info("Evaluating topic: %s", topic)
        result = narrator.generate(topic, top_k=top_k)
        articles = result.get("articles", [])
        samples.append({
            "user_input": topic,
            "response": result.get("summary", ""),
            "retrieved_contexts": format_contexts(articles, context_chars) or [""],
            "reference": row["reference"],
        })
    return samples


def evaluate_narratives(eval_set_path: str, builder) -> Dict[str, Any]:
    """
    Run the full RAGAS suite over the eval set and write per-query + aggregate scores.

    Args:
        eval_set_path: Path to the JSONL golden set.
        builder: An initialized NarrativeBuilder (data loaded + index built).

    Returns:
        {"num_samples", "aggregate", "per_query", "results_path"} dict.
    """
    # Imported here so the heuristic path never requires the eval stack.
    from datasets import Dataset  # noqa: F401  (ensures datasets is installed)
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    config = get_config()
    top_k = config.get("EVAL_TOP_K", 20)

    eval_rows = load_eval_set(eval_set_path)
    if not eval_rows:
        raise ValueError(f"No usable rows in eval set: {eval_set_path}")
    logger.info("Loaded %d eval topics from %s", len(eval_rows), eval_set_path)

    samples = _build_samples(eval_rows, builder, top_k)
    dataset = EvaluationDataset.from_list(samples)

    # Local judge LLM + embeddings (no external API calls).
    judge_llm = LangchainLLMWrapper(get_chat_llm(role="judge"))
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.get("EMBEDDING_MODEL"))
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    logger.info("Running RAGAS evaluation (%d samples)...", len(samples))
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb)

    df = result.to_pandas()
    metric_names = [m.name for m in metrics]
    metric_cols = [c for c in df.columns if c in metric_names]

    aggregate = {col: _safe_mean(df[col]) for col in metric_cols}
    per_query = df[["user_input"] + metric_cols].to_dict(orient="records")

    output = {
        "num_samples": len(samples),
        "aggregate": aggregate,
        "per_query": per_query,
        "results_path": config.get("EVAL_RESULTS_PATH", "eval_results.json"),
    }

    results_path = output["results_path"]
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote RAGAS results to %s", results_path)

    _print_summary(aggregate)
    return output


def _safe_mean(series) -> float:
    """Mean ignoring NaNs; returns None if all values are NaN."""
    clean = series.dropna()
    return float(clean.mean()) if len(clean) else None


def _print_summary(aggregate: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("RAGAS Evaluation - Aggregate Scores")
    print("=" * 50)
    for metric, score in aggregate.items():
        shown = f"{score:.4f}" if isinstance(score, float) else "n/a"
        print(f"  {metric:24s} {shown}")
    print("=" * 50 + "\n")


def main():
    """Standalone entry point: build the index then run RAGAS."""
    from narrative_builder import NarrativeBuilder

    config = get_config()
    setup_logging(level=getattr(__import__("logging"), config.get("LOG_LEVEL", "INFO")))

    builder = NarrativeBuilder(use_llm=True)
    if builder.load_and_filter() == 0:
        raise SystemExit("No articles after filtering; check dataset and rating threshold.")
    builder.initialize_embedder()

    evaluate_narratives(config.get("EVAL_SET_PATH"), builder)


if __name__ == "__main__":
    main()
