# Mirip_v2 DINOv3-ViTL16 + Vast.ai 학습 계획

## Summary

- 이 문서는 `Mirip_v2` 실험용 DINOv3 학습 파이프라인과 Vast.ai 실행 절차의 기준 문서다.
- 최종 데이터소스는 `Mirip_v2/data/crawled/metadata`와 대응 이미지이며, 크롤링 완료 후 snapshot manifest를 고정한 뒤에만 `training/data`를 재생성한다.
- 구현 범위는 실험 전용이다. `Mirip` 운영 추론과 직접 호환시키지 않고, checkpoint, reports, anchors를 독립 산출물로 관리한다.
- 실제 학습 단계는 `인터뷰 파싱 -> tier_score 보강 -> anchor_group 보강 -> pair 생성 -> pairwise 학습 -> 평가 -> anchor 재생성` 순서로 고정한다.

## Data Pipeline

- `prepare_snapshot.py`
  - `data/crawled/metadata`를 읽어 학습 적격 manifest를 생성한다.
  - `interview_raw`에서 `competition_ratio`, `exam_topic`을 추출한다.
  - `tier_refined.tier_score` 규칙을 적용하되 경쟁률이 없으면 tier 기본점수 fallback을 사용한다.
  - `anchor_group = "{university}_{normalized_dept}"` 규칙을 적용하고 최소 그룹 크기 15 미만은 제외한다.
  - 산출물: `training/data/snapshot_manifest.csv`, `reports/snapshot_report.json`
- `build_pairs.py`
  - snapshot manifest를 읽어 `metadata_{train,val,test}.csv`, `pairs_{train,val}.csv`, `pair_statistics.json`을 생성한다.
  - split: image 기준 `80/10/10`, seed `42`
  - pair: same-dept 50%, cross-dept 50%
  - 최소 `tier_score` gap `5.0`
  - 이미지당 최대 등장 횟수 `20`
  - same-dept에서 `재현작` 우선

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
  - checkpoints: `checkpoints/dinov3_vitl16/`
  - reports: `reports/dinov3_vitl16_{smoke,full}.json`
  - anchors: `anchors/anchors.pt`

## Vast.ai Runbook

1. crawl 완료 후 local snapshot manifest 생성
2. local에서 `training/data` 재생성
3. Vast offer search
4. 인스턴스 create + SSH attach + wait
5. `Mirip_v2` 학습 관련 파일과 snapshot/pairs를 rsync 업로드
6. remote bootstrap: Python env, `transformers`/`wandb` 등 설치
7. smoke run
8. smoke 통과 시 full run
9. checkpoint/reports/anchors를 local로 회수
10. 인스턴스 destroy

Smoke check:

- 모델 다운로드 성공
- processor forward 성공
- train loader 1 epoch
- checkpoint save
- checkpoint resume
- evaluate 1회 통과
- anchor build 1회 통과
- peak VRAM 기록
