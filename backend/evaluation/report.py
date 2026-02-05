"""
Report generation utilities for evaluation results.

Generates:
- JSON reports (full data)
- Markdown summaries (human-readable)
- Comparison reports (A/B testing)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_markdown_report(evaluation_report: dict, output_path: Optional[str] = None) -> str:
    """
    Generate a human-readable Markdown report from evaluation results.

    Args:
        evaluation_report: The evaluation report dict from runner.py
        output_path: Optional path to save the report

    Returns:
        Markdown string
    """
    lines = []

    # Header
    lines.append("# SafeRx RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {evaluation_report.get('timestamp', datetime.now().isoformat())}")
    lines.append("")

    # Configuration
    config = evaluation_report.get("config", {})
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Hybrid Search | {'Enabled' if config.get('hybrid_search') else 'Disabled'} |")
    lines.append(f"| Reranking | {'Enabled' if config.get('reranking') else 'Disabled'} |")
    if config.get("reranking"):
        lines.append(f"| Reranker Model | {config.get('reranker_model', 'N/A')} |")
    lines.append(f"| Vector Weight | {config.get('vector_weight', 'N/A')} |")
    lines.append(f"| BM25 Weight | {config.get('bm25_weight', 'N/A')} |")
    lines.append("")

    # Summary
    summary = evaluation_report.get("test_summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Test Cases:** {summary.get('total', 0)}")
    lines.append(f"- **Passed:** {summary.get('passed', 0)} ({summary.get('pass_rate', 0)*100:.1f}%)")
    lines.append(f"- **Failed:** {summary.get('failed', 0)}")
    lines.append(f"- **Evaluation Time:** {summary.get('total_evaluation_time_s', 0):.1f} seconds")
    lines.append(f"- **Avg Query Time:** {summary.get('avg_query_time_ms', 0):.0f}ms")
    lines.append("")

    # Overall Metrics
    metrics = evaluation_report.get("metrics", {})
    overall = metrics.get("overall", {})
    if overall:
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("| Metric | Value | Target |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| MRR | {overall.get('avg_mrr', 0):.3f} | > 0.70 |")
        lines.append(f"| Recall@5 | {overall.get('avg_recall_at_5', 0):.3f} | > 0.85 |")
        lines.append(f"| Precision@5 | {overall.get('avg_precision_at_5', 0):.3f} | > 0.75 |")
        lines.append(f"| Keyword Match | {overall.get('avg_keyword_match', 0):.3f} | > 0.70 |")
        lines.append("")

    # Metrics by Category
    by_category = metrics.get("by_category", {})
    if by_category:
        lines.append("## Metrics by Category")
        lines.append("")
        lines.append("| Category | Count | MRR | Recall@5 | Keyword Match |")
        lines.append("|----------|-------|-----|----------|---------------|")
        for cat, cat_metrics in sorted(by_category.items()):
            lines.append(
                f"| {cat} | {cat_metrics.get('count', 0)} | "
                f"{cat_metrics.get('avg_mrr', 0):.3f} | "
                f"{cat_metrics.get('avg_recall', 0):.3f} | "
                f"{cat_metrics.get('avg_keyword_match', 0):.3f} |"
            )
        lines.append("")

    # Metrics by Difficulty
    by_difficulty = metrics.get("by_difficulty", {})
    if by_difficulty:
        lines.append("## Metrics by Difficulty")
        lines.append("")
        lines.append("| Difficulty | Count | MRR | Recall@5 |")
        lines.append("|------------|-------|-----|----------|")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                diff_metrics = by_difficulty[diff]
                lines.append(
                    f"| {diff} | {diff_metrics.get('count', 0)} | "
                    f"{diff_metrics.get('avg_mrr', 0):.3f} | "
                    f"{diff_metrics.get('avg_recall', 0):.3f} |"
                )
        lines.append("")

    # Failure Analysis
    fa = evaluation_report.get("failure_analysis", {})
    if fa.get("total_failures", 0) > 0:
        lines.append("## Failure Analysis")
        lines.append("")
        lines.append(f"**Total Failures:** {fa.get('total_failures', 0)}")
        lines.append("")
        lines.append("### Failure Types")
        lines.append("")
        lines.append(f"- **Retrieval Failures:** {fa.get('retrieval_failures', 0)} (document not retrieved)")
        lines.append(f"- **Generation Failures:** {fa.get('generation_failures', 0)} (answer missing keywords)")
        lines.append(f"- **Other Failures:** {fa.get('other_failures', 0)}")
        lines.append("")

        # Failures by category
        failures_by_cat = fa.get("by_category", {})
        if failures_by_cat:
            lines.append("### Failures by Category")
            lines.append("")
            for cat, count in sorted(failures_by_cat.items(), key=lambda x: -x[1]):
                lines.append(f"- {cat}: {count}")
            lines.append("")

    # Known Limitations (placeholder for manual annotation)
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("_Add failure pattern analysis here after review._")
    lines.append("")
    lines.append("### Common Failure Patterns")
    lines.append("")
    lines.append("1. **[Pattern 1]:** Description")
    lines.append("2. **[Pattern 2]:** Description")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Generate recommendations based on metrics
    if overall:
        if overall.get('avg_recall_at_5', 0) < 0.85:
            lines.append("- [ ] Improve retrieval recall - consider adjusting BM25 weight or chunk size")
        if overall.get('avg_mrr', 0) < 0.70:
            lines.append("- [ ] Improve ranking - tune reranker or RRF parameters")
        if overall.get('avg_keyword_match', 0) < 0.70:
            lines.append("- [ ] Improve answer quality - review system prompt or add more context")

    lines.append("")

    # Join and return
    markdown = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    return markdown


def generate_comparison_report(
    report_a: dict,
    report_b: dict,
    label_a: str = "Baseline",
    label_b: str = "Improved",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a comparison report between two evaluation runs.

    Useful for A/B testing different configurations.
    """
    lines = []

    lines.append("# SafeRx RAG Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")
    lines.append(f"Comparing **{label_a}** vs **{label_b}**")
    lines.append("")

    # Configuration comparison
    lines.append("## Configuration Comparison")
    lines.append("")
    lines.append(f"| Setting | {label_a} | {label_b} |")
    lines.append("|---------|----------|----------|")

    config_a = report_a.get("config", {})
    config_b = report_b.get("config", {})

    for key in set(list(config_a.keys()) + list(config_b.keys())):
        val_a = config_a.get(key, "N/A")
        val_b = config_b.get(key, "N/A")
        lines.append(f"| {key} | {val_a} | {val_b} |")
    lines.append("")

    # Metrics comparison
    lines.append("## Metrics Comparison")
    lines.append("")

    overall_a = report_a.get("metrics", {}).get("overall", {})
    overall_b = report_b.get("metrics", {}).get("overall", {})

    lines.append(f"| Metric | {label_a} | {label_b} | Change |")
    lines.append("|--------|----------|----------|--------|")

    metric_keys = [
        ("avg_mrr", "MRR"),
        ("avg_recall_at_5", "Recall@5"),
        ("avg_precision_at_5", "Precision@5"),
        ("avg_keyword_match", "Keyword Match"),
    ]

    for key, display_name in metric_keys:
        val_a = overall_a.get(key, 0)
        val_b = overall_b.get(key, 0)
        change = val_b - val_a
        change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
        emoji = "+" if change > 0.01 else ("-" if change < -0.01 else "=")
        lines.append(f"| {display_name} | {val_a:.3f} | {val_b:.3f} | {change_str} {emoji} |")

    lines.append("")

    # Pass rate comparison
    summary_a = report_a.get("test_summary", {})
    summary_b = report_b.get("test_summary", {})

    lines.append("## Pass Rate Comparison")
    lines.append("")
    rate_a = summary_a.get("pass_rate", 0) * 100
    rate_b = summary_b.get("pass_rate", 0) * 100
    change = rate_b - rate_a
    lines.append(f"- **{label_a}:** {rate_a:.1f}%")
    lines.append(f"- **{label_b}:** {rate_b:.1f}%")
    lines.append(f"- **Change:** {'+' if change > 0 else ''}{change:.1f}%")
    lines.append("")

    # Performance comparison
    lines.append("## Performance Comparison")
    lines.append("")
    time_a = summary_a.get("avg_query_time_ms", 0)
    time_b = summary_b.get("avg_query_time_ms", 0)
    lines.append(f"| Metric | {label_a} | {label_b} |")
    lines.append("|--------|----------|----------|")
    lines.append(f"| Avg Query Time | {time_a:.0f}ms | {time_b:.0f}ms |")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")

    improvements = []
    regressions = []

    for key, display_name in metric_keys:
        val_a = overall_a.get(key, 0)
        val_b = overall_b.get(key, 0)
        change = val_b - val_a
        if change > 0.01:
            improvements.append(f"{display_name} improved by {change:.3f}")
        elif change < -0.01:
            regressions.append(f"{display_name} regressed by {abs(change):.3f}")

    if improvements:
        lines.append("**Improvements:**")
        for imp in improvements:
            lines.append(f"- {imp}")
        lines.append("")

    if regressions:
        lines.append("**Regressions:**")
        for reg in regressions:
            lines.append(f"- {reg}")
        lines.append("")

    if not improvements and not regressions:
        lines.append("No significant changes detected.")
        lines.append("")

    markdown = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    return markdown


def load_report(filepath: str) -> dict:
    """Load an evaluation report from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def main():
    """Generate reports from evaluation results."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("report", type=str, help="Path to evaluation report JSON")
    parser.add_argument("--output", "-o", type=str, help="Output path for Markdown report")
    parser.add_argument("--compare", type=str, help="Path to second report for comparison")
    args = parser.parse_args()

    report = load_report(args.report)

    if args.compare:
        report_b = load_report(args.compare)
        output = args.output or "comparison_report.md"
        markdown = generate_comparison_report(report, report_b, output_path=output)
        print(f"Comparison report saved to: {output}")
    else:
        output = args.output or args.report.replace(".json", ".md")
        markdown = generate_markdown_report(report, output_path=output)
        print(f"Markdown report saved to: {output}")


if __name__ == "__main__":
    main()
