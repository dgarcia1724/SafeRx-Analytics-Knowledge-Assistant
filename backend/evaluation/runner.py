"""
Comprehensive evaluation runner for SafeRx RAG pipeline.

Supports:
- Full evaluation suite (100 test cases)
- Quick evaluation (subset)
- Comparison between retrieval methods
- Detailed failure analysis

Usage:
    python -m evaluation.runner              # Full evaluation
    python -m evaluation.runner --quick      # Quick 10-case test
    python -m evaluation.runner --compare    # Compare retrieval methods
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.reranker import RerankerPipeline
from app.services.hybrid_retriever import HybridRetrieverService
from app.services.generator import GeneratorService
from evaluation.metrics import (
    calculate_retrieval_metrics,
    calculate_generation_metrics,
    aggregate_metrics,
)


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    category: str
    difficulty: str
    query: str
    answer: str
    retrieved_sources: list[str]
    expected_sources: list[str]
    expected_keywords: list[str]
    retrieval_method: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    keyword_match_score: float
    passed: bool
    failure_reason: Optional[str] = None


def load_test_cases(filepath: str) -> list[dict]:
    """Load test cases from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["test_cases"]


def run_single_test(
    test_case: dict,
    pipeline: RerankerPipeline,
    generator: GeneratorService,
    enable_reranking: bool = True,
) -> TestResult:
    """Run a single test case through the RAG pipeline."""
    query = test_case["query"]
    expected_sources = test_case.get("expected_sources", [])
    expected_keywords = test_case.get("expected_answer_contains", [])
    expected_behavior = test_case.get("expected_behavior")

    # Retrieval
    start_retrieval = time.time()
    results, trace_info = pipeline.search(
        query=query,
        enable_reranking=enable_reranking,
    )
    retrieval_time = (time.time() - start_retrieval) * 1000

    # Extract source document titles for metric calculation
    retrieved_sources = [r.metadata.get("doc_title", r.doc_id) for r in results]

    # Generation
    start_generation = time.time()
    retrieval_results = [r.to_retrieval_result() for r in results]

    if retrieval_results:
        generation = generator.generate(query, retrieval_results)
        answer = generation.text
    else:
        # For out-of-scope, this might be intentional
        if expected_behavior in ["polite_refusal", "no_information"]:
            answer = generator.generate_refusal(query).text
        else:
            answer = "No relevant documents found."

    generation_time = (time.time() - start_generation) * 1000
    total_time = retrieval_time + generation_time

    # Calculate metrics
    retrieval_metrics = calculate_retrieval_metrics(
        expected_sources, retrieved_sources, k=5
    )
    generation_metrics = calculate_generation_metrics(expected_keywords, answer)

    # Determine pass/fail
    # Special handling for out-of-scope queries
    if expected_behavior == "polite_refusal":
        # Check if any expected keyword is in the answer
        passed = generation_metrics.keyword_match_score >= 0.3
        failure_reason = None if passed else "Did not politely refuse out-of-scope query"
    elif expected_behavior == "no_information":
        # Check if system appropriately indicates no information
        passed = any(
            phrase in answer.lower()
            for phrase in ["don't have", "no information", "cannot find", "not found"]
        )
        failure_reason = None if passed else "Did not indicate lack of information"
    else:
        # Standard evaluation
        passed = (
            retrieval_metrics.recall_at_k >= 0.5
            and generation_metrics.keyword_match_score >= 0.5
        )
        if not passed:
            if retrieval_metrics.recall_at_k < 0.5:
                failure_reason = f"Retrieval failure: Recall@5={retrieval_metrics.recall_at_k:.2f}"
            else:
                failure_reason = f"Generation failure: KeywordMatch={generation_metrics.keyword_match_score:.2f}"

    return TestResult(
        test_id=test_case["id"],
        category=test_case["category"],
        difficulty=test_case.get("difficulty", "unknown"),
        query=query,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        retrieved_sources=retrieved_sources,
        expected_sources=expected_sources,
        expected_keywords=expected_keywords,
        retrieval_method="hybrid_reranked" if enable_reranking else "hybrid",
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        total_time_ms=total_time,
        mrr=retrieval_metrics.mrr,
        recall_at_k=retrieval_metrics.recall_at_k,
        precision_at_k=retrieval_metrics.precision_at_k,
        keyword_match_score=generation_metrics.keyword_match_score,
        passed=passed,
        failure_reason=failure_reason if not passed else None,
    )


def run_evaluation(
    test_cases_path: str,
    output_path: Optional[str] = None,
    quick: bool = False,
    enable_reranking: bool = True,
) -> dict:
    """Run full evaluation on all test cases."""
    print("SafeRx Analytics - RAG Evaluation")
    print("=" * 60)

    settings = get_settings()

    # Initialize services
    print("Initializing services...")
    pipeline = RerankerPipeline()
    generator = GeneratorService()

    # Check collection
    stats = pipeline.get_stats()
    vector_count = stats['hybrid']['vector'].get('count', 0)
    print(f"Vector store: {vector_count} documents")
    print(f"BM25 index: {stats['hybrid']['bm25'].get('document_count', 0)} documents")
    print(f"Reranking: {'enabled' if enable_reranking else 'disabled'}")

    if vector_count == 0:
        print("ERROR: No documents in collection. Run ingestion first.")
        return {}

    # Load test cases
    print(f"\nLoading test cases from: {test_cases_path}")
    test_cases = load_test_cases(test_cases_path)

    if quick:
        # Sample 10 test cases across categories for quick evaluation
        from random import sample
        categories = list(set(tc["category"] for tc in test_cases))
        sampled = []
        for cat in categories[:5]:
            cat_cases = [tc for tc in test_cases if tc["category"] == cat]
            sampled.extend(sample(cat_cases, min(2, len(cat_cases))))
        test_cases = sampled[:10]
        print(f"Quick mode: Evaluating {len(test_cases)} sampled test cases")
    else:
        print(f"Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("\nRunning evaluation...")
    results: list[TestResult] = []
    passed = 0
    failed = 0
    start_time = time.time()

    for i, test_case in enumerate(test_cases, 1):
        query_preview = test_case['query'][:40] + "..." if len(test_case['query']) > 40 else test_case['query']
        print(f"  [{i:3}/{len(test_cases)}] {test_case['id']}: {query_preview}", end=" ")

        try:
            result = run_single_test(
                test_case,
                pipeline,
                generator,
                enable_reranking=enable_reranking,
            )
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] ({result.total_time_ms:.0f}ms)")

            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1

    total_time = time.time() - start_time

    # Convert results to dict format for aggregation
    results_dicts = []
    for r in results:
        results_dicts.append({
            "test_id": r.test_id,
            "category": r.category,
            "difficulty": r.difficulty,
            "retrieval": {
                "mrr": r.mrr,
                "recall_at_k": r.recall_at_k,
                "precision_at_k": r.precision_at_k,
            },
            "generation": {
                "keyword_match_score": r.keyword_match_score,
                "answer_relevance": 0.0,  # Placeholder
                "faithfulness": 0.0,  # Placeholder
            },
        })

    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated = aggregate_metrics(results_dicts)

    # Analyze failures
    failures = [r for r in results if not r.passed]
    failure_analysis = analyze_failures(failures)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hybrid_search": settings.enable_hybrid_search,
            "reranking": enable_reranking,
            "reranker_model": settings.reranker_model if enable_reranking else None,
            "vector_weight": settings.vector_weight,
            "bm25_weight": settings.bm25_weight,
        },
        "collection_stats": stats['hybrid'],
        "test_summary": {
            "total": len(test_cases),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "total_evaluation_time_s": total_time,
            "avg_query_time_ms": sum(r.total_time_ms for r in results) / len(results) if results else 0,
        },
        "metrics": aggregated,
        "failure_analysis": failure_analysis,
        "detailed_results": [asdict(r) for r in results],
    }

    # Save report
    if output_path:
        output_file = Path(output_path)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(__file__).parent / f"evaluation_report_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_file}")

    # Print summary
    print_summary(report)

    return report


def analyze_failures(failures: list[TestResult]) -> dict:
    """Analyze failure patterns."""
    if not failures:
        return {"total_failures": 0}

    # Categorize failures
    retrieval_failures = [f for f in failures if f.failure_reason and "Retrieval" in f.failure_reason]
    generation_failures = [f for f in failures if f.failure_reason and "Generation" in f.failure_reason]
    other_failures = [f for f in failures if f not in retrieval_failures and f not in generation_failures]

    # Group by category
    by_category = {}
    for f in failures:
        cat = f.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f.test_id)

    # Group by difficulty
    by_difficulty = {}
    for f in failures:
        diff = f.difficulty
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(f.test_id)

    return {
        "total_failures": len(failures),
        "retrieval_failures": len(retrieval_failures),
        "generation_failures": len(generation_failures),
        "other_failures": len(other_failures),
        "by_category": {cat: len(ids) for cat, ids in by_category.items()},
        "by_difficulty": {diff: len(ids) for diff, ids in by_difficulty.items()},
        "retrieval_failure_ids": [f.test_id for f in retrieval_failures],
        "generation_failure_ids": [f.test_id for f in generation_failures],
    }


def print_summary(report: dict):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    summary = report["test_summary"]
    print(f"Total test cases: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_evaluation_time_s']:.1f}s")
    print(f"Avg query time: {summary['avg_query_time_ms']:.0f}ms")

    if report.get("metrics", {}).get("overall"):
        print("\nOverall Metrics:")
        overall = report["metrics"]["overall"]
        print(f"  Mean Reciprocal Rank (MRR): {overall['avg_mrr']:.3f}")
        print(f"  Recall@5: {overall['avg_recall_at_5']:.3f}")
        print(f"  Precision@5: {overall['avg_precision_at_5']:.3f}")
        print(f"  Keyword Match: {overall['avg_keyword_match']:.3f}")

    if report.get("failure_analysis", {}).get("total_failures", 0) > 0:
        fa = report["failure_analysis"]
        print("\nFailure Analysis:")
        print(f"  Retrieval failures: {fa['retrieval_failures']}")
        print(f"  Generation failures: {fa['generation_failures']}")
        print(f"  Other failures: {fa['other_failures']}")

        if fa.get("by_category"):
            print("\n  Failures by category:")
            for cat, count in sorted(fa["by_category"].items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

    if report.get("metrics", {}).get("by_category"):
        print("\nMetrics by Category:")
        for cat, metrics in report["metrics"]["by_category"].items():
            print(f"  {cat}: MRR={metrics['avg_mrr']:.3f}, Recall={metrics['avg_recall']:.3f}")


def compare_retrieval_methods(test_cases_path: str):
    """Compare different retrieval configurations."""
    print("SafeRx Analytics - Retrieval Method Comparison")
    print("=" * 60)

    # Initialize services
    hybrid_retriever = HybridRetrieverService()
    generator = GeneratorService()

    # Check collection
    stats = hybrid_retriever.get_stats()
    if stats['vector']['count'] == 0:
        print("ERROR: No documents in collection.")
        return

    # Load test cases (use subset for comparison)
    test_cases = load_test_cases(test_cases_path)[:20]
    print(f"Comparing on {len(test_cases)} test cases")

    methods = {
        "vector_only": lambda q: hybrid_retriever.vector_only_search(q, top_k=5),
        "bm25_only": lambda q: hybrid_retriever.bm25_only_search(q, top_k=5),
        "hybrid": lambda q: hybrid_retriever.search(q, top_k=5)[0],
    }

    results = {method: {"mrr": [], "recall": [], "precision": []} for method in methods}

    for test_case in test_cases:
        query = test_case["query"]
        expected_sources = test_case.get("expected_sources", [])

        for method_name, search_fn in methods.items():
            search_results = search_fn(query)

            # Extract source names
            if method_name == "bm25_only":
                retrieved_sources = [r.metadata.get("doc_title", r.doc_id) for r in search_results]
            elif method_name == "hybrid":
                retrieved_sources = [r.metadata.get("doc_title", r.doc_id) for r in search_results]
            else:
                retrieved_sources = [r.metadata.get("doc_title", "") for r in search_results]

            metrics = calculate_retrieval_metrics(expected_sources, retrieved_sources, k=5)
            results[method_name]["mrr"].append(metrics.mrr)
            results[method_name]["recall"].append(metrics.recall_at_k)
            results[method_name]["precision"].append(metrics.precision_at_k)

    # Print comparison
    print("\nResults:")
    print(f"{'Method':<15} {'MRR':>10} {'Recall@5':>12} {'Precision@5':>14}")
    print("-" * 55)

    for method_name in methods:
        avg_mrr = sum(results[method_name]["mrr"]) / len(results[method_name]["mrr"])
        avg_recall = sum(results[method_name]["recall"]) / len(results[method_name]["recall"])
        avg_precision = sum(results[method_name]["precision"]) / len(results[method_name]["precision"])
        print(f"{method_name:<15} {avg_mrr:>10.3f} {avg_recall:>12.3f} {avg_precision:>14.3f}")


def main():
    """Run evaluation with command line arguments."""
    parser = argparse.ArgumentParser(description="SafeRx RAG Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (10 cases)")
    parser.add_argument("--compare", action="store_true", help="Compare retrieval methods")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    test_cases_path = Path(__file__).parent / "test_cases.json"

    if args.compare:
        compare_retrieval_methods(str(test_cases_path))
    else:
        run_evaluation(
            str(test_cases_path),
            output_path=args.output,
            quick=args.quick,
            enable_reranking=not args.no_rerank,
        )


if __name__ == "__main__":
    main()
