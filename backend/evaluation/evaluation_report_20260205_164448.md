# SafeRx RAG Evaluation Report

**Generated:** 2026-02-05T16:44:48.040095

## Configuration

| Setting | Value |
|---------|-------|
| Hybrid Search | Enabled |
| Reranking | Enabled |
| Reranker Model | cross-encoder |
| Vector Weight | 0.7 |
| BM25 Weight | 0.3 |

## Summary

- **Total Test Cases:** 100
- **Passed:** 62 (62.0%)
- **Failed:** 38
- **Evaluation Time:** 570.7 seconds
- **Avg Query Time:** 5707ms

## Overall Metrics

| Metric | Value | Target |
|--------|-------|--------|
| MRR | 0.942 | > 0.70 |
| Recall@5 | 0.963 | > 0.85 |
| Precision@5 | 0.736 | > 0.75 |
| Keyword Match | 0.540 | > 0.70 |

## Metrics by Category

| Category | Count | MRR | Recall@5 | Keyword Match |
|----------|-------|-----|----------|---------------|
| adverse_reactions | 20 | 0.975 | 1.000 | 0.629 |
| contraindications | 15 | 0.967 | 1.000 | 0.450 |
| dosage | 15 | 0.833 | 0.933 | 0.677 |
| drug_interactions | 20 | 0.950 | 0.900 | 0.571 |
| edge_cases | 5 | 1.000 | 0.850 | 0.677 |
| out_of_scope | 10 | 1.000 | 1.000 | 0.000 |
| special_populations | 15 | 0.913 | 1.000 | 0.648 |

## Metrics by Difficulty

| Difficulty | Count | MRR | Recall@5 |
|------------|-------|-----|----------|
| easy | 47 | 0.926 | 0.979 |
| medium | 50 | 0.954 | 0.945 |
| hard | 3 | 1.000 | 1.000 |

## Failure Analysis

**Total Failures:** 38

### Failure Types

- **Retrieval Failures:** 1 (document not retrieved)
- **Generation Failures:** 27 (answer missing keywords)
- **Other Failures:** 10

### Failures by Category

- out_of_scope: 10
- drug_interactions: 7
- adverse_reactions: 7
- contraindications: 6
- dosage: 4
- special_populations: 4

## Known Limitations

_Add failure pattern analysis here after review._

### Common Failure Patterns

1. **[Pattern 1]:** Description
2. **[Pattern 2]:** Description

## Recommendations

- [ ] Improve answer quality - review system prompt or add more context
