"""
Backward-compatible evaluation runner for SafeRx Analytics RAG pipeline.

This module provides the original evaluation interface while delegating
to the enhanced runner.py for actual execution.

Usage:
    python evaluation/evaluate.py

For the full-featured evaluation, use:
    python -m evaluation.runner
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.runner import run_evaluation, compare_retrieval_methods


def main():
    """Run evaluation with default paths (backward compatible)."""
    test_cases_path = Path(__file__).parent / "test_cases.json"
    run_evaluation(str(test_cases_path))


if __name__ == "__main__":
    main()
