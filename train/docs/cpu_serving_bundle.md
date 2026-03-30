# ViT-L CPU Serving Bundle

Mirip_v2 CPU serving bundle is built under `train/serving` and always includes:

- `manifest.json`
- `encoder_fp32.onnx`
- `preprocessor.json`
- `benchmarks.json`
- `quality_report.json`
- `model_sha256.txt`

Optional core files:

- `encoder_int8.onnx`

Production diagnosis bundles must also provide manifest extras for:

- `diagnosis_head`
- `anchors`

Promotion rule:

- FP32 ONNX is the baseline.
- INT8 becomes the default encoder only when `int8_tier_agreement_vs_fp32 >= 0.99`
  and INT8 p50 latency improves by at least 20% over FP32.
- Otherwise `manifest.json` keeps `encoder_fp32.onnx` as `default_encoder`.
- `train/serving/export_bundle.py` keeps the agreement at `0.0` unless a measured value is
  passed explicitly via `--int8-tier-agreement`.

Benchmark report expectations:

- Each encoder benchmark stores `latency_ms_p50`, `latency_ms_p95`, `startup_time_ms`,
  `rss_mib`, and `thread_count`.
- Export uses an intra-op thread sweep of `8`, `12`, and `16`, keeps per-thread sweep
  details in `benchmarks.json`, and records `best_intra_op_num_threads` for runtime reuse.

`train/training/train_dinov3.py` now accepts a local Hugging Face export directory as
`--model-name`, but it validates that the directory exists and contains `config.json`
before training starts.
