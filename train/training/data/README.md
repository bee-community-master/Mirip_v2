# Generated Training Data

이 디렉터리의 `snapshot_manifest.csv`, `metadata_{train,val,test}.csv`, `pairs_{train,val}.csv`, `pair_statistics.json`은 모두 freeze 시점에 재생성하는 산출물이다.

- 입력 소스 오브 트루스: `Mirip_v2/train/data/metadata`
- 대응 이미지 루트: `Mirip_v2/train/data/raw_images`
- metadata의 `images` 값은 절대경로가 아니라 staged JPG 상대경로여야 한다. 허용 형식은 `foo.jpg` 또는 `raw_images/foo.jpg`이며, manifest에는 항상 `raw_images/foo.jpg`로 기록된다.
- 이 디렉터리의 CSV/JSON은 git 커밋 대상이 아니라 local/Vast sync 대상이다.
- 재생성 순서:
  1. `python3 train/training/validate_training_readiness.py`
  2. `python3 train/training/prepare_snapshot.py`
  3. `python3 train/training/build_pairs.py`
  4. `python3 train/training/validate_training_readiness.py --mode prepared`

`build_pairs.py`는 기본 설정 기준 목표 pair 수를 끝까지 채우지 못하면 실패한다. freeze 이후 `train/data`가 바뀌면 raw validator부터 다시 수행해야 한다.
현재 reboot 기준 기본 정책은 아래와 같다.

- split: `train=0.8`, `val=0.1`, `test=0.1`
- pairs: `train=40000`, `val=5000`
- same/cross dept ratio: `0.5 / 0.5`
- tier distance quota: `distance1=0.6`, `distance2=0.3`, `distance3=0.1`
- max appearances: `48`
