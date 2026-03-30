# Mirip_v2 DINOv3-ViTL16 + Vast.ai 학습 계획

## Summary

- 이 문서는 `Mirip_v2/train` 실험용 DINOv3 학습 파이프라인과 Vast.ai 실행 절차의 기준 문서다.
- 최종 데이터소스는 `Mirip_v2/train/data/metadata`와 `Mirip_v2/train/data/raw_images`이며, 크롤링 완료 후 JPG 변환과 metadata path 정규화가 끝난 snapshot만 학습에 사용한다.
- 구현 범위는 실험 전용이다. `Mirip` 운영 추론과 직접 호환시키지 않고, checkpoint, reports, anchors를 독립 산출물로 관리한다.
- 실제 학습 단계는 `인터뷰 파싱 -> tier_score 보강 -> anchor_group 보강 -> pair 생성 -> pairwise 학습 -> 평가 -> anchor 재생성` 순서로 고정한다.

## Data Pipeline

- `prepare_snapshot.py`
  - `data/metadata`를 읽어 학습 적격 manifest를 생성한다.
  - `interview_raw`에서 `competition_ratio`, `exam_topic`을 추출한다.
  - `tier_refined.tier_score` 규칙을 적용하되 경쟁률이 없으면 tier 기본점수 fallback을 사용한다.
  - `anchor_group = "{university}_{normalized_dept}"` 규칙을 적용하고 최소 그룹 크기 15 미만은 제외한다.
  - metadata의 `images` 값은 staged JPG 상대경로만 허용하고 manifest에는 `raw_images/<name>.jpg`로 기록한다.
  - 산출물: `train/training/data/snapshot_manifest.csv`, `train/reports/snapshot_report.json`
- `build_pairs.py`
  - snapshot manifest를 읽어 `metadata_{train,val,test}.csv`, `pairs_{train,val}.csv`, `pair_statistics.json`을 생성한다.
  - split: image 기준 `80/10/10`, seed `42`
  - pair: same-dept 50%, cross-dept 50%
  - 최소 `tier_score` gap `5.0`
  - 이미지당 최대 등장 횟수 `30`
  - same-dept에서 `재현작` 우선
  - 목표 pair 수를 채우지 못하면 partial output을 남기고 즉시 실패한다.

## Training Pipeline

- 모델: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- 정책: frozen backbone + trainable projector/head
- 입력/출력: `(img1, img2) -> (score1, score2)`
- loss: `MarginRankingLoss(margin=0.3)`
- feature head:
  - backbone output -> projector `1024 -> 512 -> 256`
  - score head `256 -> 64 -> 1`
- 전처리: Hugging Face `AutoImageProcessor` 기본 설정 재사용
- 기본 학습 설정:
  - optimizer: `AdamW`
  - lr: `1e-4`
  - weight decay: `0.05`
  - scheduler: cosine decay
  - epochs: `50`
  - early stopping patience: `10`
  - seed: `42`
  - precision: `bf16` 우선, 미지원 시 `fp16`
  - effective batch size 목표: `64`
- 32GB GPU 배치 fallback:
  - `8 x 8`
  - OOM 시 `6 x 10`
  - 다시 OOM 시 `4 x 16`

## Evaluation

- 핵심 지표:
  - `val_accuracy`
  - `same_dept_accuracy`
  - `anchor_tier_accuracy`
  - `latency_ms_per_pair`
  - `peak_vram_gb`
- `build_anchors_dinov3.py`는 `metadata_train.csv`를 기준으로 tier별 기본 10개 앵커를 생성한다.
- 산출물:
  - checkpoints: `train/checkpoints/dinov3_vitl16/`
  - reports: `train/reports/dinov3_vitl16_{smoke,full}.json`
  - anchors: `train/anchors/anchors.pt`
- full 후처리 비교 결과는 `train/reports/dinov3_vit7b16_postprocess_registry.json`에 누적한다.
- hourly checkpoint cleanup은 위 registry를 source of truth로 사용하고, checkpoint를 다시 평가하지 않는다.

## Vast.ai Runbook

1. crawl 종료 후 `train/data`를 freeze한다.
2. local에서 JPG 변환 + metadata path 정규화를 끝낸다.
3. `python3 train/training/validate_training_readiness.py`
4. `python3 train/training/prepare_snapshot.py`
5. `python3 train/training/build_pairs.py`
6. `python3 train/training/validate_training_readiness.py --mode prepared`
7. Vast offer search
8. 인스턴스 create + SSH attach + wait
9. `Mirip_v2/train/data`, `Mirip_v2/train/training/data`, `Mirip_v2/train/reports`를 rsync 업로드
10. remote bootstrap: Python env, `transformers`/`wandb` 등 설치
11. remote `validate-upload` 실행
12. smoke run
13. smoke 통과 시 full run
14. `train/checkpoints`, `train/reports`, `train/anchors`를 local로 회수
15. 인스턴스 destroy

Smoke check:

- 모델 다운로드 성공
- processor forward 성공
- train loader 1 epoch
- checkpoint save
- checkpoint resume
- evaluate 1회 통과
- anchor build 1회 통과
- peak VRAM 기록
