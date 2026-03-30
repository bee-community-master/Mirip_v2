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

`train/training/train_dinov3.py` now accepts a local Hugging Face export directory as
`--model-name`, but it validates that the directory exists and contains `config.json`
before training starts.
