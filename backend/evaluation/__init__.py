"""
SafeRx RAG Evaluation Framework.

Provides comprehensive evaluation of the RAG pipeline including:
- Retrieval metrics (MRR, Recall@K, Precision@K, NDCG@K)
- Generation metrics (Keyword match, Faithfulness, Citation accuracy)
- Failure analysis and categorization
- Report generation (JSON and Markdown)

Usage:
    # Run full evaluation
    python -m evaluation.runner

    # Run quick evaluation (10 cases)
    python -m evaluation.runner --quick

    # Compare retrieval methods
    python -m evaluation.runner --compare

    # Generate markdown report from JSON
    python -m evaluation.report evaluation_report.json
"""
