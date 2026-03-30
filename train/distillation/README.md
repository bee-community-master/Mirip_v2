# Mirip_v2 DINOv3 ViT-L Distillation

## 프로젝트 개요

이 디렉터리는 `PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m` teacher를 완전히 freeze한 상태로 사용하고, `PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m` student를 patch/local feature 중심으로 추가 distillation fine-tuning 하기 위한 학습 코드다.

목표는 분류 정확도보다 이미지의 섬세한 dense feature 보존이다. 그래서 loss 설계는 patch token alignment와 relational patch alignment를 중심에 두고, cls/pool/mid loss는 보조적으로만 사용한다.

## 디렉터리 구조

```text
train/distillation/
  train.py
  eval.py
  config.py
  utils.py
  datasets.py
  losses.py
  models.py
  engine.py
  requirements.txt
  README.md
  configs/
    vitl_distill.yaml
```

## 설치 방법

```bash
cd train/distillation
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch`는 Vast.ai의 PyTorch base image에서 제공된다고 가정한다.

## teacher / student 체크포인트 준비

기본 config는 아래 체크포인트를 사용한다.

- teacher: `PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m`
- student: `PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m`

기본 backend 순서는 `huggingface -> timm` 이다. Hugging Face 로딩이 실패하면 timm fallback을 시도한다. pairwise handoff에 바로 쓰려면 HF backend로 export 가능한 경로를 유지하는 것이 공식 지원 경로다.

공식 Facebook DINOv3 gated checkpoint를 쓰고 싶다면 Hugging Face 인증이 필요하다. 현재 예시 config는 공개 mirror 기준이다.

## 데이터셋 준비 형식

기본 데이터 계약은 Mirip staged 데이터다.

- metadata: `train/data/metadata/*.json`
- raw images: `train/data/raw_images/*.jpg`

각 metadata JSON은 최소한 `images` 배열을 가져야 한다. 가능한 경우 아래 필드를 함께 유지한다.

- `post_no`
- `tier`
- `department`
- `normalized_dept`
- `anchor_group`
- `university`
- `work_type`

prepared split CSV가 존재하면 우선 사용한다.

- `train/training/data/metadata_train.csv`
- `train/training/data/metadata_val.csv`

CSV가 없으면 `post_no` 또는 image path 기반 deterministic hash split으로 train/val을 나눈다.

추가 확장 경로도 지원한다.

- `source_type=imagefolder`
- `source_type=webdataset`

`webdataset`를 사용할 때는 `webdataset_url_pattern`에 반드시 `{split}` placeholder를 넣어 train/val shard를 분리해야 한다. 예:

```text
s3://bucket/mirip-{split}-%06d.tar
```

## 학습 방법

학습 진입점:

```bash
cd train/distillation
python train.py --config configs/vitl_distill.yaml
```

smoke 학습:

```bash
cd train/distillation
python train.py --config configs/vitl_distill.yaml --smoke
```

resume:

```bash
cd train/distillation
python train.py --config configs/vitl_distill.yaml --resume checkpoints/distill_vitl/<run>/last.pt
```

학습 전 데이터 검증:

```bash
cd train/distillation
python train.py --config configs/vitl_distill.yaml --validate-data --report reports/distillation/validate_data.json
```

## high-resolution phase

학습은 3-stage 구조를 기본으로 한다.

1. `stage1_main`
   - 해상도 `256`
   - 주된 distillation 구간
2. `stage2_highres`
   - 해상도 `384` 또는 `518`
   - 고해상도 적응 구간
3. `stage3_refine`
   - 마지막 미세 조정
   - global loss 비중을 낮추고 patch/local feature 유지에 더 집중

모든 해상도는 patch size 16의 배수로 자동 보정된다.

## 주요 loss 설명

- `L_patch`
  - teacher/student patch token alignment
  - 기본값은 cosine loss
- `L_rel`
  - 이미지 내부 patch-to-patch similarity matrix distillation
  - local structure를 더 안정적으로 맞추기 위한 loss
- `L_cls`
  - cls token alignment
  - 보조 목적
- `L_pool`
  - pooled/global representation alignment
  - retrieval/전역 표현 안정화용 보조 loss
- `L_mid`
  - teacher/student intermediate feature alignment
  - 깊이 차이가 큰 teacher/student를 약하게 연결

기본 총 loss:

```text
L_total =
  1.0 * L_cls
+ 2.0 * L_patch
+ 1.0 * L_rel
+ 0.3 * L_mid
+ 0.2 * L_pool
```

## 평가 방법

```bash
cd train/distillation
python eval.py --config configs/vitl_distill.yaml --ckpt checkpoints/distill_vitl/<run>/best.pt
```

평가 결과는 아래를 저장한다.

- teacher/student feature cosine similarity
- patch similarity matrix alignment score
- intermediate feature alignment score
- offline nearest-neighbor retrieval (`Recall@1`, `Recall@5`, `MRR`)
- patch embedding visualization PNG

## pairwise handoff

best checkpoint가 갱신되면 HF 호환 student backbone export를 아래 위치에 저장한다.

```text
train/checkpoints/distill_vitl/<run>/best_student_backbone/
```

이 export는 기존 pairwise 학습에 그대로 넘길 수 있다.

```bash
python3 train/training/train_dinov3.py --model-name train/checkpoints/distill_vitl/<run>/best_student_backbone
```

## 메모리 부족 시 대응법

- `configs/vitl_distill.yaml`에서 `batch_size`를 먼저 줄인다.
- 그 다음 `gradient_accumulation_steps`를 올려 effective batch를 복원한다.
- `stage2_highres`를 `384`로 유지하고 `518`은 충분한 VRAM이 있을 때만 사용한다.
- `evaluation.batch_size`도 별도로 줄일 수 있다.
- `models.gradient_checkpointing=true`를 유지한다.
- 필요하면 `distillation.rel_patch_sample_size`를 128보다 더 낮춰 relational loss 메모리를 줄인다.

## 자주 수정할 하이퍼파라미터

- `stages[].resolution`
- `stages[].batch_size`
- `stages[].gradient_accumulation_steps`
- `stages[].learning_rate`
- `distillation.rel_patch_sample_size`
- `models.teacher_dtype`
- `models.student_dtype`
- `data.num_workers`
- `evaluation.batch_size`

## Vast.ai 사용

distillation 전용 runner:

```bash
python3 train/scripts/vast_ai_distillation_runner.py print-command --stage smoke-distill
python3 train/scripts/vast_ai_distillation_runner.py execute-stage --stage bootstrap --instance-id <ID>
python3 train/scripts/vast_ai_distillation_runner.py execute-stage --stage validate-data --instance-id <ID>
python3 train/scripts/vast_ai_distillation_runner.py execute-stage --stage smoke-distill --instance-id <ID>
python3 train/scripts/vast_ai_distillation_runner.py execute-stage --stage full-distill --instance-id <ID>
```
