"""
Evaluation metrics for RAG pipeline.

Includes:
- Retrieval metrics: MRR, Recall@K, Precision@K, NDCG@K
- Generation metrics: Keyword match, Answer relevance, Faithfulness
- Aggregation utilities
"""

from dataclasses import dataclass
import math


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""

    mrr: float  # Mean Reciprocal Rank
    recall_at_k: float  # Recall@K - fraction of relevant docs retrieved
    precision_at_k: float  # Precision@K - fraction of retrieved docs that are relevant
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    hit_rate: float  # Did we find at least one relevant doc?
    k: int


@dataclass
class GenerationMetrics:
    """Metrics for generation quality."""

    answer_relevance: float  # Does answer address the query?
    faithfulness: float  # Is answer grounded in context?
    keyword_match_score: float  # Do expected keywords appear?
    citation_accuracy: float  # Are sources correctly cited?


def mean_reciprocal_rank(
    expected_sources: list[str], retrieved_sources: list[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR measures how highly relevant documents are ranked.
    Returns 1/rank of first relevant document, or 0 if none found.

    Args:
        expected_sources: List of expected source identifiers
        retrieved_sources: List of retrieved source identifiers (in rank order)

    Returns:
        MRR score between 0 and 1
    """
    if not expected_sources:
        return 1.0  # No expected sources means any result is acceptable

    for i, source in enumerate(retrieved_sources, 1):
        # Check if any expected source is a substring of the retrieved source
        for expected in expected_sources:
            if expected.lower() in source.lower():
                return 1.0 / i

    return 0.0


def recall_at_k(
    expected_sources: list[str], retrieved_sources: list[str], k: int = 5
) -> float:
    """
    Calculate Recall@K.

    Recall@K measures what fraction of relevant documents are retrieved in top K.

    Args:
        expected_sources: List of expected source identifiers
        retrieved_sources: List of retrieved source identifiers
        k: Number of top results to consider

    Returns:
        Recall score between 0 and 1
    """
    if not expected_sources:
        return 1.0

    retrieved_k = retrieved_sources[:k]
    found = 0

    for expected in expected_sources:
        for retrieved in retrieved_k:
            if expected.lower() in retrieved.lower():
                found += 1
                break

    return found / len(expected_sources)


def precision_at_k(
    expected_sources: list[str], retrieved_sources: list[str], k: int = 5
) -> float:
    """
    Calculate Precision@K.

    Precision@K measures what fraction of retrieved documents are relevant.

    Args:
        expected_sources: List of expected source identifiers
        retrieved_sources: List of retrieved source identifiers
        k: Number of top results to consider

    Returns:
        Precision score between 0 and 1
    """
    if not retrieved_sources:
        return 0.0

    if not expected_sources:
        # If no expected sources, can't determine relevance
        return 0.0

    retrieved_k = retrieved_sources[:k]
    relevant = 0

    for retrieved in retrieved_k:
        for expected in expected_sources:
            if expected.lower() in retrieved.lower():
                relevant += 1
                break

    return relevant / len(retrieved_k)


def ndcg_at_k(
    expected_sources: list[str], retrieved_sources: list[str], k: int = 5
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG measures ranking quality with position-weighted relevance.
    Higher-ranked relevant documents contribute more to the score.

    Args:
        expected_sources: List of expected source identifiers
        retrieved_sources: List of retrieved source identifiers
        k: Number of top results to consider

    Returns:
        NDCG score between 0 and 1
    """
    if not expected_sources:
        return 1.0

    retrieved_k = retrieved_sources[:k]

    # Calculate DCG
    dcg = 0.0
    for i, retrieved in enumerate(retrieved_k, 1):
        rel = 0
        for expected in expected_sources:
            if expected.lower() in retrieved.lower():
                rel = 1
                break
        dcg += rel / math.log2(i + 1)

    # Calculate ideal DCG (all relevant docs at top)
    ideal_dcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(expected_sources), k) + 1))

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def hit_rate(
    expected_sources: list[str], retrieved_sources: list[str], k: int = 5
) -> float:
    """
    Calculate Hit Rate (binary: did we find at least one relevant doc?).

    Args:
        expected_sources: List of expected source identifiers
        retrieved_sources: List of retrieved source identifiers
        k: Number of top results to consider

    Returns:
        1.0 if at least one relevant doc found, 0.0 otherwise
    """
    if not expected_sources:
        return 1.0

    retrieved_k = retrieved_sources[:k]

    for retrieved in retrieved_k:
        for expected in expected_sources:
            if expected.lower() in retrieved.lower():
                return 1.0

    return 0.0


def keyword_match_score(expected_keywords: list[str], answer: str) -> float:
    """
    Calculate keyword match score.

    Measures what fraction of expected keywords appear in the answer.

    Args:
        expected_keywords: List of keywords expected in the answer
        answer: The generated answer text

    Returns:
        Score between 0 and 1
    """
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)

    return found / len(expected_keywords)


def citation_accuracy(
    answer: str,
    sources: list[str],
    num_sources_cited: int = 0,
) -> float:
    """
    Calculate citation accuracy.

    Measures whether the answer correctly cites sources.

    Args:
        answer: The generated answer text
        sources: List of available source titles
        num_sources_cited: Number of source citations in the answer

    Returns:
        Score between 0 and 1
    """
    if not sources:
        return 1.0

    # Count [Source N] citations in answer
    import re
    citations = re.findall(r'\[Source \d+\]', answer)
    num_citations = len(citations)

    if num_citations == 0:
        return 0.0

    # Check if citation numbers are valid (within source range)
    valid_citations = 0
    for citation in citations:
        num = int(re.search(r'\d+', citation).group())
        if 1 <= num <= len(sources):
            valid_citations += 1

    return valid_citations / num_citations if num_citations > 0 else 0.0


def calculate_retrieval_metrics(
    expected_sources: list[str], retrieved_sources: list[str], k: int = 5
) -> RetrievalMetrics:
    """Calculate all retrieval metrics."""
    return RetrievalMetrics(
        mrr=mean_reciprocal_rank(expected_sources, retrieved_sources),
        recall_at_k=recall_at_k(expected_sources, retrieved_sources, k),
        precision_at_k=precision_at_k(expected_sources, retrieved_sources, k),
        ndcg_at_k=ndcg_at_k(expected_sources, retrieved_sources, k),
        hit_rate=hit_rate(expected_sources, retrieved_sources, k),
        k=k,
    )


def calculate_generation_metrics(
    expected_keywords: list[str],
    answer: str,
    sources: list[str] = None,
    answer_relevance: float = 0.0,
    faithfulness: float = 0.0,
) -> GenerationMetrics:
    """Calculate all generation metrics."""
    return GenerationMetrics(
        answer_relevance=answer_relevance,
        faithfulness=faithfulness,
        keyword_match_score=keyword_match_score(expected_keywords, answer),
        citation_accuracy=citation_accuracy(answer, sources or []) if sources else 0.0,
    )


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Aggregate metrics across all test cases.

    Args:
        results: List of result dicts with 'retrieval' and 'generation' sub-dicts

    Returns:
        Aggregated metrics by overall, category, and difficulty
    """
    if not results:
        return {}

    # Calculate averages
    total = len(results)

    avg_mrr = sum(r["retrieval"]["mrr"] for r in results) / total
    avg_recall = sum(r["retrieval"]["recall_at_k"] for r in results) / total
    avg_precision = sum(r["retrieval"]["precision_at_k"] for r in results) / total
    avg_keyword_match = sum(r["generation"]["keyword_match_score"] for r in results) / total

    # Group by category
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    category_metrics = {}
    for cat, cat_results in by_category.items():
        cat_total = len(cat_results)
        category_metrics[cat] = {
            "count": cat_total,
            "avg_mrr": sum(r["retrieval"]["mrr"] for r in cat_results) / cat_total,
            "avg_recall": sum(r["retrieval"]["recall_at_k"] for r in cat_results) / cat_total,
            "avg_precision": sum(r["retrieval"]["precision_at_k"] for r in cat_results) / cat_total,
            "avg_keyword_match": sum(r["generation"]["keyword_match_score"] for r in cat_results) / cat_total,
        }

    # Group by difficulty
    by_difficulty = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(r)

    difficulty_metrics = {}
    for diff, diff_results in by_difficulty.items():
        diff_total = len(diff_results)
        difficulty_metrics[diff] = {
            "count": diff_total,
            "avg_mrr": sum(r["retrieval"]["mrr"] for r in diff_results) / diff_total,
            "avg_recall": sum(r["retrieval"]["recall_at_k"] for r in diff_results) / diff_total,
            "avg_precision": sum(r["retrieval"]["precision_at_k"] for r in diff_results) / diff_total,
            "avg_keyword_match": sum(r["generation"]["keyword_match_score"] for r in diff_results) / diff_total,
        }

    return {
        "total_cases": total,
        "overall": {
            "avg_mrr": avg_mrr,
            "avg_recall_at_5": avg_recall,
            "avg_precision_at_5": avg_precision,
            "avg_keyword_match": avg_keyword_match,
        },
        "by_category": category_metrics,
        "by_difficulty": difficulty_metrics,
    }


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a readable table string."""
    lines = []

    if metrics.get("overall"):
        overall = metrics["overall"]
        lines.append("Overall Metrics:")
        lines.append(f"  MRR:         {overall.get('avg_mrr', 0):.3f}")
        lines.append(f"  Recall@5:    {overall.get('avg_recall_at_5', 0):.3f}")
        lines.append(f"  Precision@5: {overall.get('avg_precision_at_5', 0):.3f}")
        lines.append(f"  KeywordMatch:{overall.get('avg_keyword_match', 0):.3f}")

    return "\n".join(lines)
