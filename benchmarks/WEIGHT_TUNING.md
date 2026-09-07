# AgentRecall retrieval weight tuning report

Date: 2026-08-21

Historical development measurements: the private corpus and labeled queries are not shipped, and these results were not independently reproduced in the public-commit review. Use the synthetic gate for a reproducible public check; these small-corpus timings are not a production-scale latency guarantee.

## Safety and corpus

- Source: a private multi-agent production corpus; its path and host identity are intentionally omitted.
- Benchmark input: a SQLite online-backup copy in a disposable temporary directory; the source DB was opened read-only and never migrated or benchmarked in place.
- Corpus size: approximately 100 memories, with ACL-dependent visible subsets.
- Query embeddings: a configured local OpenAI-compatible embedding endpoint; endpoint and model identifiers are intentionally omitted.
- Equivalent duplicate memories were treated as one relevance target by normalized title/content, preventing private/shared duplicate rows from being scored as false failures.
- No memory contents, embeddings, credentials, or chat/session logs are stored in this report.

## Configurations evaluated

Initial BM25-heavy implementation:

```text
bm25=0.55 lexical=0.20 vector=0.20 importance=0.03 recency=0.02 exact_boost=0.00
```

Final selected blend:

```text
bm25=0.25 lexical=0.15 vector=0.55 importance=0.03 recency=0.02 exact_boost=0.16
```

The exact boost is applied only when every query term appears lexically. It lets a true all-terms match survive a misleading vector neighbor while preserving vector dominance over merely partial lexical overlap.

## Search process

Two grid passes evaluated 4,744 candidate blends:

1. Coarse sweep across BM25, lexical, vector, and exact-match boost.
2. Fine sweep around the strongest region, additionally varying importance and recency weights.

Selection objective combined:

- 20 hand-labeled semantic/paraphrase queries over real memories with real query embeddings.
- Duplicate-aware real title queries with real query embeddings.
- Lexical fallback queries with no query embedding.
- Stored-embedding oracle checks.
- Two synthetic adversarial guards:
  - all-terms lexical match versus misleading perfect vector neighbor;
  - strong semantic match versus merely partial lexical neighbor.

A broad plateau tied for the best aggregate objective. The selected point retains small importance/recency signals while staying in that plateau.

## Iteration results

| Configuration | Semantic top-1 | Semantic top-5 | Semantic MRR | Exact guard | Semantic guard | Aggregate objective |
|---|---:|---:|---:|---:|---:|---:|
| Initial BM25-heavy | 0.80 | 0.95 | 0.8670 | pass | fail | 0.8902 |
| Balanced, no exact boost (`0.20/0.30/0.45`) | 0.85 | 0.95 | 0.9083 | fixture-dependent | pass | 0.9588 |
| Final (`0.25/0.15/0.55`, boost `0.16`) | 0.85 | 1.00 | 0.9250 | pass | pass | 0.9663 |

## Untuned holdout

Ten additional paraphrase queries were authored only after selecting the final weights, then run five times against the real DB copy.

| Metric | Initial BM25-heavy | Final | Delta |
|---|---:|---:|---:|
| Top-1 accuracy | 0.80 | 0.90 | +0.10 |
| Top-5 recall | 1.00 | 1.00 | 0.00 |
| MRR | 0.8833 | 0.9500 | +0.0667 |
| Median-run p95 retrieval | 45.241 ms | 45.042 ms | -0.199 ms |
| ACL leaks | 0 | 0 | 0 |
| Expired hits | 0 | 0 | 0 |

This holdout improved rather than regressed, reducing concern that the selected weights only fit the tuning queries.

## Reproducible real-copy benchmark

The updated benchmark script generated 56 duplicate-aware title queries for the primary ACL view and embedded each title through a configured OpenAI-compatible endpoint.

| Scenario | Baseline top-1 | Final top-1 | Baseline MRR | Final MRR | Baseline p95 | Final p95 |
|---|---:|---:|---:|---:|---:|---:|
| Lexical fallback | 0.9643 | 1.0000 | 0.9762 | 1.0000 | 31.192 ms | 30.466 ms |
| Real query embeddings | 0.9643 | 1.0000 | 0.9747 | 1.0000 | 42.214 ms | 46.247 ms |
| Stored-embedding oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 42.553 ms | 46.328 ms |

Here “baseline” is the pre-BM25 legacy vector-weighted overlap algorithm. The final blend added about 4 ms p95 in the real-query-embedding title test while remaining below the 50 ms gate.

## Cross-identity real-copy checks

All used live query embeddings and the same copied DB:

| Agent view | Cases | Baseline top-1 | Final top-1 | Baseline MRR | Final MRR | Final p95 |
|---|---:|---:|---:|---:|---:|---:|
| View A | 56 | 0.9643 | 1.0000 | 0.9747 | 1.0000 | 46.247 ms |
| View B | 60 | 0.9667 | 1.0000 | 0.9750 | 1.0000 | 37.026 ms |
| View C | 49 | 0.9592 | 1.0000 | 0.9745 | 1.0000 | 30.090 ms |
| View D | 48 | 0.9583 | 1.0000 | 0.9740 | 1.0000 | 29.164 ms |

All runs had zero ACL leaks and zero expired-memory hits.

## Synthetic gate

Final synthetic adversarial result:

```text
top1_accuracy=1.0 top5_recall=1.0 MRR=1.0 p95=0.688ms ACL_leaks=0 expired_hits=0 gate_passed=true
```

## Decision

Keep the final blend:

```text
bm25=0.25 lexical=0.15 vector=0.55 importance=0.03 recency=0.02 exact_lexical_boost=0.16
```

Rationale:

- It passes both opposing adversarial guards.
- It improves the tuned semantic set and the independently authored holdout.
- It preserves perfect duplicate-aware title retrieval, lexical fallback, and stored-embedding oracle behavior.
- It produced no ACL/expiration leakage.
- Its measured retrieval latency remained within the 50 ms gate across four agent views.

## Limitations

- The real corpus is still small (approximately 100 rows), so future growth may change latency and ranking behavior.
- Title-derived queries are easier than arbitrary user questions; the hand-labeled semantic and holdout sets partially compensate for this.
- Human relevance labels can be imperfect, especially where multiple memories overlap semantically.
- Query-embedding HTTP latency is not included in store retrieval p95 because it is shared by both ranking algorithms and occurs before `store.search()`.
