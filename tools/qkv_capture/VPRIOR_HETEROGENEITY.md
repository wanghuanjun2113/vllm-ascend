# V prior heterogeneity diagnostic

Run analyze_vprior_heterogeneity.py inside vllm-023-qwen35-qkv.
It reads the frozen qwen35_35B development/validation banks and the pinned
21690.13 SOTA's statistic extractors; it does not run or edit any candidate.

The output is data/qwen35_35B/prior_heterogeneity:
- PLAN.json: fixed K=2 and evaluation boundaries.
- development_records.json / validation_records.json: every head's statistics.
- FROZEN_TWO_TYPE_MODEL.json: development-only normalization, centers and templates,
  written before reading validation.
- SUMMARY.json: variance decomposition, document-holdout diagnostic, predictions.
- RESULTS.md: interpretation and limits.

Feature vectors contain calibration lag1..8 and normalized tail at the five
calibration lengths; there are no layer/head/document identity features.
Layer/head IDs are used only to measure stability and oracle diagnostic bounds.
MSE is standardized statistic reconstruction error, not Attention output MSE
or contest points. Per-T templates in this diagnostic are not identical to the
previous solution's State injection behavior for missing calibration lengths.
The fixed K=2 partition is an operational hypothesis, not proof of two natural
clusters. Only three documents per bank are available.
No NPU, online submission, live service, or candidate mutation is involved.
