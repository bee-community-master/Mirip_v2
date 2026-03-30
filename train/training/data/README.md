# Generated Training Data

이 디렉터리의 `snapshot_manifest.csv`, `metadata_{train,val,test}.csv`, `pairs_{train,val}.csv`, `pair_statistics.json`은 모두 생성 산출물이다.

- 입력 소스 오브 트루스: `Mirip_v2/train/data/metadata`
- 대응 이미지 루트: `Mirip_v2/train/data/raw_images`
- metadata의 `images` 값은 절대경로가 아니라 staged JPG 상대경로여야 한다. 허용 형식은 `foo.jpg` 또는 `raw_images/foo.jpg`이며, manifest에는 항상 `raw_images/foo.jpg`로 기록된다.
- 재생성 순서:
  1. `python3 train/training/validate_training_readiness.py`
  1. `python3 train/training/prepare_snapshot.py`
  2. `python3 train/training/build_pairs.py`

`build_pairs.py`는 기본 설정 기준 목표 pair 수를 끝까지 채우지 못하면 실패한다. 현재 남아 있는 CSV는 참조용일 수 있으며, 최종 crawl snapshot이 고정되면 다시 덮어써야 한다.
