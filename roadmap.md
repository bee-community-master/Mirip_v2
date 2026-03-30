# Mirip_v2 DINOv3 Teacher 로드맵

> 최종 업데이트: 2026-03-30  
> 문서 위치: `./roadmap.md`
> 참고: baseline 분석에 필요한 레거시 소스 경로 표기는 원문 기준을 유지한다.

## 1. 한눈 요약

### 목표

`Mirip_v2`의 1차 목표는 현재 MIRIP 진단 파이프라인을 사실 기반으로 다시 정리하고, 아래 순서대로 teacher 모델을 교체하는 것이다.

1. 현재 DINOv2 baseline을 재현한다.
2. 같은 데이터/같은 평가 기준으로 `facebook/dinov3-vitl16-pretrain-lvd1689m` teacher를 학습한다.
3. DINOv3 teacher가 baseline보다 낫다는 것이 확인되면 그때 `facebook/dinov3-convnext-large-pretrain-lvd1689m` distillation 단계로 넘어간다.

### 이번 문서에서 고정하는 것

- 기준 데이터 소스: sibling `metadata`, `Mirip/backend/data/crawled/metadata`
- 제외 데이터 소스: `metadata2`
- 기준 UI 계약: `mirip_diagnosis_html.html`
- 기준 teacher 승격 조건: `val_accuracy`, `same_dept_accuracy`, anchor-tier accuracy, 출력 스키마 정합성
- 기준 GPU 운영 계획: Vast.ai on-demand RTX PRO 4500 Blackwell 32GB, AMP fp16, effective batch 64 목표

### 이번 문서에서 하지 않는 것

- `metadata2` 재가공 설계
- 실제 distillation 구현
- learned university probability model
- FE/BE 코드 수정 자체

## 2. 기준 소스

### 로컬 기준 파일

- HTML 결과 계약: [mirip_diagnosis_html.html](/Users/shawn/Documents/Project/antlter_bootcamp3-1/mirip_diagnosis_html.html)
- 현재 페어 생성: [prepare_training_pairs.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/scripts/prepare_training_pairs.py)
- 현재 추론: [inference.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/services/inference.py)
- 현재 프론트 응답 매핑: [diagnosisService.js](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/my-app/src/services/diagnosisService.js)
- 현재 DINOv2 extractor: [feature_extractor.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/ml/feature_extractor.py)
- 현재 ranking model: [ranking_model.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/ml/ranking_model.py)
- 현재 ranker: [tier_ranker.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/ml/tier_ranker.py)
- 현재 anchor 생성: [build_anchors.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/scripts/build_anchors.py)
- 현재 pair 통계: [pair_statistics.json](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/training/data/pair_statistics.json)

### 데이터 소스 고정

- Source A: `../antlter_bootcamp3-1/metadata`
- Source B: `../antlter_bootcamp3-1/Mirip/backend/data/crawled/metadata`

정책:

- `metadata2`는 baseline 소스에서 제외한다.
- `Mirip_v2`는 우선 문서/실험 workspace로 두고, `Mirip`를 현재 기준 구현 저장소로 취급한다.

### 외부 참고

- DINOv3 공식 저장소: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- DINOv3 distillation 관련 이슈: [Issue #25](https://github.com/facebookresearch/dinov3/issues/25)
- Vast.ai 인스턴스 개요: [docs.vast.ai/documentation/instances](https://docs.vast.ai/documentation/instances/)
- Vast.ai 인스턴스 타입: [docs.vast.ai/documentation/instances/choosing/instance-types](https://docs.vast.ai/documentation/instances/choosing/instance-types)
- Vast.ai 가격 문서: [docs.vast.ai/documentation/instances/pricing](https://docs.vast.ai/documentation/instances/pricing)

## 3. 현재 상태 진단

### 데이터 현황

| 소스 | 전체 파일 | 디코드 성공 | 파손 | 비고 |
| --- | ---: | ---: | ---: | --- |
| sibling `metadata` | 3,747 | 3,746 | 1 | `221.json` 파손 |
| backend `data/crawled/metadata` | 3,747 | 3,746 | 1 | 실제 학습 파생물 생성에 사용 |

### backend metadata 핵심 수치

`Mirip/backend/data/crawled/metadata` 기준:

- 총 디코드 가능: 3,746
- 파손: 1
- 이미지 경로 있음: 3,630
- `tier_refined.tier_score` 있음: 3,746
- `anchor_group` 있음: 2,734
- 학습 적격 샘플(`image + anchor_group + tier_score`): 2,734
- `interview_parsed` 있음: 3,629
- `competition_ratio` 있음: 3,467
- `exam_topic` 있음: 3,404

### 티어 분포

| Tier | Count |
| --- | ---: |
| S | 144 |
| A | 1,229 |
| B | 1,579 |
| C | 794 |

### 상위 학과 분포

| normalized_dept | Count |
| --- | ---: |
| visual_design | 715 |
| design_general | 700 |
| fine_art | 605 |
| industrial_design | 456 |
| craft | 418 |
| fashion | 238 |
| animation | 232 |
| interior | 150 |
| other | 126 |
| sculpture | 106 |

### 작품 타입 분포

| work_type | Count |
| --- | ---: |
| 재현작 | 2,106 |
| 평소작 | 1,421 |
| unknown | 219 |

### 이미 존재하는 pair artifact

`pair_statistics.json` 기준:

| Split | Total pairs | Same dept | Cross dept | Unique images | Eligible items |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 40,000 | 20,000 | 20,000 | 2,187 | 2,187 |
| val | 4,940 | 2,440 | 2,500 | 273 | 273 |

Train pair 품질 계층:

- 둘 다 재현작: 12,139
- 재현작 1개: 989
- 평소작만: 6,872
- cross-dept: 20,000

Val pair 품질 계층:

- 둘 다 재현작: 1,372
- 재현작 1개: 282
- 평소작만: 786
- cross-dept: 2,500

## 4. 현재 파이프라인 추정

### 현재 backbone / model 구조

[feature_extractor.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/ml/feature_extractor.py), [ranking_model.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/ml/ranking_model.py) 기준:

- backbone: frozen `facebook/dinov2-large`
- feature dim: 1024
- projector: `1024 -> 512 -> 256`
- score head: `256 -> 64 -> 1`
- loss: `MarginRankingLoss`
- 입력 크기: `448 x 448`

### 실제 학습 흐름

현재 저장소 기준 유효한 학습 흐름은 아래로 본다.

1. 메타데이터 크롤링/정규화
2. 인터뷰 파싱
3. 경쟁률 기반 `tier_score` 생성
4. precomputed pair 생성
5. pairwise ranking 학습
6. 학습된 모델로 anchor 재생성
7. anchor 비교 기반 tier 추론

### `tier_score`가 만들어지는 방식

[augment_tier_labels.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/scripts/augment_tier_labels.py) 기준:

- 기본 티어별 점수대가 있다.
- 경쟁률 percentile로 티어 내부 위치를 보정한다.
- 경쟁률이 없으면 기본 점수로 떨어진다.

현재 티어 점수 범위:

| Tier | Range |
| --- | --- |
| S | 82 - 98 |
| A | 65 - 84 |
| B | 48 - 68 |
| C | 30 - 50 |

### pair 생성 규칙

[prepare_training_pairs.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/scripts/prepare_training_pairs.py) 기준:

- discrete tier만 보지 않고 `tier_score`를 사용
- same-dept 50%
- cross-dept 50%
- 최소 `tier_score` gap 5.0
- 이미지별 최대 등장 횟수 20
- same-dept 안에서는 재현작 우선
- 기본 총 pair 수 50,000
- 현재 커밋된 artifact는 train 40,000 / val 4,940

same-dept 품질 우선순위:

- 1단계: 둘 다 `재현작`
- 2단계: 하나만 `재현작`
- 3단계: 둘 다 평소작
- 4단계: cross-dept

### 현재 FE/BE 계약 불일치

현재 backend 축 이름:

- `composition`
- `technique`
- `creativity`
- `completeness`

현재 frontend remap:

- `composition -> composition`
- `technique -> color`
- `creativity -> technique`
- `completeness -> creativity`

문제:

- 이 매핑은 의미가 뒤틀려 있다.
- HTML이 요구하는 축 이름과 직접 연결되지 않는다.
- `Mirip_v2`에서는 새 공개 스키마를 고정해야 한다.

### 현재 추론 한계

[inference.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/app/services/inference.py) 기준:

- 대학 확률은 learned model이 아니라 heuristic
- fallback 축 점수는 실제 rubric head 결과가 아니라 feature 통계 기반
- `anchors.pt`는 있지만 baseline checkpoint는 커밋돼 있지 않음

결론:

- v2는 checkpoint 이관이 아니라 DINOv2 baseline 재현부터 시작해야 한다.

### 현재 평가 harness 이슈

[evaluate.py](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/training/scripts/evaluate.py)와 [training_guide.md](/Users/shawn/Documents/Project/antlter_bootcamp3-1/Mirip/backend/docs/training_guide.md)의 일부는 현재 `PairwiseRankingModel` 시그니처와 완전히 맞지 않는다.

정책:

- baseline 재현 전에 평가 harness 정합성부터 맞춘다.
- DINOv3 비교는 이 정합성 수정이 끝난 뒤에만 유효하다.

## 5. V2 산출물 계약

v2 실행 시 아래 산출물을 생성하거나 유지한다.

| 경로 | 용도 |
| --- | --- |
| `metadata_manifest.csv` | 최종 학습 적격 manifest |
| `pairs_train.csv` | train pair |
| `pairs_val.csv` | val pair |
| `train/output_models/logs/data_audit.md` | 데이터 품질 점검 보고서 |
| `train/output_models/logs/dinov2_baseline.json` | DINOv2 baseline 결과 |
| `train/output_models/logs/dinov3_teacher.json` | DINOv3 teacher 결과 |
| `train/output_models/checkpoints/dinov2_baseline/` | baseline checkpoint |
| `train/output_models/checkpoints/dinov3_vitl16/` | DINOv3 teacher checkpoint |
| `train/output_models/anchors/anchors.pt` | 승인된 모델에서 다시 만든 anchor |

추가로 남겨도 되는 비필수 산출물:

- raw log
- W&B run id
- GPU profiling 결과
- 임시 캐시 파일

## 6. V2 출력 스키마 계약

### 공개 출력 스키마

```json
{
  "composition": 0,
  "light_texture": 0,
  "form_completion": 0,
  "topic_interpretation": 0,
  "tier": "B",
  "university_probabilities": [
    {
      "university": "홍익대학교",
      "department": "시각디자인과",
      "probability": 0.65
    }
  ]
}
```

### HTML 라벨 대응

| v2 field | 화면 라벨 |
| --- | --- |
| `composition` | 구성력 |
| `light_texture` | 명암/질감 |
| `form_completion` | 조형완성도 |
| `topic_interpretation` | 주제해석 |

### 레거시 필드 처리 원칙

- `composition/technique/creativity/completeness`는 레거시 backend 필드로만 본다.
- `composition/color/technique/creativity`는 레거시 frontend 표시용으로만 본다.
- v2 공개 계약은 이 두 레거시 계약을 그대로 쓰지 않는다.

### 대학 확률 정책

teacher 단계에서는 `university_probabilities`를 계속 heuristic으로 둔다.

정책:

- 제품 출력에는 포함한다.
- 모델 승격 metric에서는 제외한다.

## 7. 실행 로드맵

### Phase 0. Workspace 준비

할 일:

1. 현재 저장소 루트를 기준 workspace로 사용
2. 로드맵 문서 유지
3. baseline 재현에 필요한 문서/스크립트만 복사 또는 참조
4. `Mirip`는 소스 오브 트루스로 두고 v2 검증 전까지는 read-only 기준 저장소처럼 다룸

### Phase 1. 데이터 감사와 manifest 확정

산출물:

- `train/output_models/logs/data_audit.md`
- `metadata_manifest.csv`

할 일:

1. sibling `metadata` 감사
2. backend metadata 감사
3. 파손/제외 사유 목록화
4. 학습 적격 이미지 경로 실존 여부 확인
5. 2,734개 학습 적격 샘플 manifest 확정
6. `metadata2`는 baseline 범위 밖이라고 명시

종료 조건:

- 학습 적격 샘플의 이미지 경로가 모두 유효
- 파손 파일이 고정 목록으로 관리됨
- baseline manifest row 수가 확정됨

### Phase 2. DINOv2 baseline 재현

산출물:

- `train/output_models/logs/dinov2_baseline.json`
- `train/output_models/checkpoints/dinov2_baseline/`
- 재생성된 `train/output_models/anchors/anchors.pt`

고정 원칙:

- 입력 크기 `448` 유지
- frozen backbone 유지
- projector/head 구조는 호환성 수정이 필요할 때만 손댐
- pair 생성 규칙은 현재 `prepare_training_pairs.py`를 따른다
- 평가 metric은 새로 만들지 말고 기존 harness를 정합화해서 사용한다

실행 순서:

1. `pairs_train.csv`, `pairs_val.csv` 재생성 또는 검증
2. `evaluate.py` 정합성 수정
3. DINOv2 baseline 학습
4. `val_accuracy`, `same_dept_accuracy`, anchor-tier accuracy, latency 측정
5. baseline checkpoint로 anchor 재생성

기록할 metric:

- pairwise `val_accuracy`
- `same_dept_accuracy`
- anchor-tier classification accuracy
- per-image inference latency
- RTX PRO 4500 Blackwell 32GB 기준 peak VRAM

### Phase 3. DINOv3 teacher 전환

산출물:

- `train/output_models/logs/dinov3_teacher.json`
- `train/output_models/checkpoints/dinov3_vitl16/`

backbone 고정:

- `facebook/dinov3-vitl16-pretrain-lvd1689m`

환경 원칙:

- DINOv3 공식 repo / 공식 지원 경로 사용
- 현재 환경이 `transformers < 4.56.0`이면 legacy stack을 덮어쓰지 말고 v2 전용 환경을 만든다

모델 전환 원칙:

- 첫 단계는 frozen-backbone, head-only training
- feature extractor abstraction은 기존 ranking model과 shape-compatible하게 유지
- backbone hidden size는 config에서 동적으로 읽고 DINOv2 가정값을 하드코딩하지 않는다
- projector/head는 backbone과 분리해 baseline 비교를 쉽게 유지한다

비교 원칙:

- 같은 manifest
- 같은 split
- 같은 pair 생성 규칙
- 같은 metric
- 학습 뒤 같은 방식으로 anchor 재생성

### Phase 4. Vast.ai RTX PRO 4500 Blackwell 32GB 실행 계획

인프라 고정:

- 플랫폼: Vast.ai
- GPU: RTX PRO 4500 Blackwell 32GB
- 인스턴스 타입: on-demand 우선
- precision: AMP fp16 기본
- interruptible: 첫 smoke test 성공 전에는 사용 금지

런 프로파일:

- 목표 effective batch size: 64
- 메모리가 허용하면 per-device batch를 높인다
- 메모리가 부족하면 per-device batch를 낮추고 gradient accumulation으로 복구한다
- full run 전에 checkpoint resume 검증이 반드시 끝나야 한다

Smoke test 체크리스트:

1. 환경 부팅
2. 모델 가중치 다운로드
3. processor/model forward pass
4. 1 epoch dry-run
5. checkpoint save
6. checkpoint resume
7. peak VRAM 기록

정책:

- 7개를 모두 통과해야 full run으로 넘어간다.

### Phase 5. 승격 게이트

DINOv3 teacher가 baseline을 대체하려면 아래 조건을 모두 만족해야 한다.

1. `val_accuracy`가 개선되거나 최소 비열화
2. `same_dept_accuracy`가 개선되거나 최소 비열화
3. 둘 중 최소 하나는 실제로 개선
4. anchor-tier accuracy 비열화
5. 출력 스키마가 v2 rubric 계약과 일치
6. 화면 축 라벨이 레거시 remap 없이 직접 연결 가능

하나라도 실패하면:

- serving teacher는 그대로 DINOv2 baseline 유지

## 8. Future Distillation

이번 문서에서는 계획만 고정한다.

대상 student:

- `facebook/dinov3-convnext-large-pretrain-lvd1689m`

지금 확정하는 것:

- student 목표 아키텍처는 ConvNeXt-Large
- distillation은 teacher 승인 후 시작
- 구현 시 DINOv3 공식 distillation 지원 경로를 참고

이번 문서에서 미루는 것:

- loss 조합
- feature matching 단위
- temperature
- fine-tuning 스케줄
- 배포 패키징

teacher 승인 전제:

- `dinov3_teacher.json` 승인
- anchor-tier accuracy 승인
- 출력 스키마 안정화
- RTX PRO 4500 Blackwell 32GB smoke test와 full run 1회 성공

## 9. 테스트 계획

### 데이터 감사 테스트

- sibling `metadata` 파일 수 검증
- sibling `metadata` 파손 파일 목록 검증
- backend metadata 파일 수 검증
- backend metadata 파손 파일 목록 검증
- tier 분포 검증
- dept 분포 검증
- work_type 분포 검증
- 학습 적격 이미지 경로 실존 검증

### DINOv2 baseline 테스트

- pair CSV 재생성 또는 검증
- baseline 학습
- 평가 수행
- anchor 재생성
- anchor-tier 샘플 테스트
- latency / VRAM 기록

### DINOv3 smoke test

- `facebook/dinov3-vitl16-pretrain-lvd1689m` 로드
- processor 출력 shape 확인
- 1 epoch 실행
- checkpoint 저장
- checkpoint resume
- resume 이후 metric 이상 없음 확인

### DINOv3 full 비교 테스트

- baseline과 teacher를 같은 validation artifact로 비교
- 새 anchor로 anchor-tier accuracy 비교
- v2 스키마로 직접 출력 가능한지 확인

### 제품 계약 테스트

- HTML 요구 4축이 모두 존재
- 축 이름이 의미적으로 직접 대응
- 대학 확률은 heuristic임을 문서에 명시
- 대학 확률은 승격 metric에서 제외

## 10. 가정과 기본값

- `metadata1`은 sibling `../antlter_bootcamp3-1/metadata`
- `Mirip_v2`는 `Mirip`의 브랜치가 아니라 sibling workspace
- `metadata2`는 baseline 범위에서 제외
- `anchors.pt`는 있으나 baseline checkpoint는 커밋돼 있지 않음
- DINOv3 모델명과 지원 상태는 공식 DINOv3 저장소 기준
- Vast.ai 운영 지침은 현재 문서 기준

## 11. 바로 다음 액션

1. `train/output_models/logs/data_audit.md` 작성
2. 2,734개 학습 적격 샘플 기준 `metadata_manifest.csv` 생성
3. `evaluate.py`와 실제 ranking model 시그니처 정합성 수정
4. DINOv2 baseline 재학습 및 anchor 재생성
5. Vast.ai RTX PRO 4500 Blackwell 32GB on-demand 인스턴스 확보
6. DINOv3 smoke test 수행
7. smoke test 통과 후 full run 수행
