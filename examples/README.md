# Examples

## Baseline metrics (`baseline_metrics.json`)

Committed **regression floors** for CI. On every PR, GitHub Actions runs:

```bash
eval-harness run configs/matrices/rag_qa_baseline.yaml --mock \
  --baseline examples/baseline_metrics.json
```

If aggregate retrieval or generation metrics drop more than `tolerance` below these values, the job fails.

## Sample leaderboard (mock run)

Typical outcome on the 5-sample MLflow Q&A golden set:

| Rank | Variant | recall@k | mrr | faithfulness |
|------|---------|----------|-----|--------------|
| 1 | bm25-only | 1.000 | 1.000 | ~0.47 |
| 2 | hybrid-rerank | 1.000 | 1.000 | ~0.47 |

Both variants retrieve the reference context; mock answers score low on `exact_match` by design. Use `--live` for real generation metrics.

## Reproduce locally

```bash
eval-harness run configs/matrices/rag_qa_baseline.yaml --mock \
  --baseline examples/baseline_metrics.json
open results/runs/*/leaderboard.html
```
