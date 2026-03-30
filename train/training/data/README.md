# Generated Training Data

이 디렉터리의 `snapshot_manifest.csv`, `metadata_{train,val,test}.csv`, `pairs_{train,val}.csv`, `pair_statistics.json`은 모두 생성 산출물이다.

- 소스 오브 트루스: `Mirip_v2/data/crawled/metadata`
- 재생성 순서:
  1. `python3 train/training/prepare_snapshot.py`
  2. `python3 train/training/build_pairs.py`

현재 남아 있는 CSV는 참조용일 수 있으며, 최종 crawl snapshot이 고정되면 다시 덮어써야 한다.
