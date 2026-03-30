# Vast.ai Preparation Notes

## 조사 결론

`vast.ai`는 학습 작업을 자동화할 때 브라우저보다 API 기준으로 다루는 편이 안정적입니다. 공식 문서 기준 핵심 흐름은 아래와 같습니다.

1. offer 검색
2. offer id로 instance 생성
3. instance 상태 확인
4. 필요 시 SSH 키 부착
5. SSH 또는 `execute` API로 원격 명령 실행
6. 끝나면 stop 또는 destroy

## 학습 운영 관점 핵심 포인트

### 1. 인스턴스 생성은 2단계

- search offers: `POST /api/v0/bundles/`
- create instance: `PUT /api/v0/asks/{offer_id}/`

공식 문서는 offer churn이 흔하다고 명시합니다. 그래서 검색 결과를 하나만 믿지 말고 상위 몇 개를 후보로 들고 있는 쪽이 안전합니다.

### 2. 학습 작업은 `ssh_direct` 또는 `jupyter_direct`가 실용적

- `ssh_direct`: SSH 전용, 원격 커맨드 실행과 파일 전송이 단순함
- `jupyter_direct`: SSH + Jupyter 같이 열림

SSH/Jupyter 런타입은 Docker entrypoint를 Vast 쪽 초기화 스크립트로 교체하므로, 초기화 작업은 `onstart`에 넣어야 합니다.

### 3. 환경 변수 포맷이 CLI와 API에서 다름

- API instance create에서는 `env`가 JSON object여야 함
- 포트는 `"-p 8080:8080": "1"` 형태로 넣어야 함

### 4. SSH 키는 인스턴스 생성 뒤 API로 붙일 수 있음

- attach ssh-key: `POST /api/v0/instances/{id}/ssh/`

그래서 로컬 공개키만 준비되면 Codex에서 바로 인스턴스에 접속 가능한 상태를 만들 수 있습니다.

### 5. 원격 명령은 API만으로도 실행 가능

- execute: `PUT /api/v0/instances/command/{id}/`

긴 학습은 SSH 세션이나 `tmux`가 더 편하지만, smoke test나 bootstrap은 `execute` API만으로도 충분합니다.

### 6. 운영 정리

- smoke test 전에는 interruptible 대신 on-demand 우선
- 끝난 뒤 보존이 필요 없으면 `destroy`
- 잠깐 중지면 `stop`, 다만 재시작 시 GPU 보장이 약해질 수 있음

## Mirip_v2에서 준비해둔 항목

- RTX PRO 4500 Blackwell 32GB on-demand 기본 offer filter
- 표준 image / disk / env / onstart 템플릿
- 로컬 파일 rsync 업로드 루틴
- SSH / execute / destroy 제어 스크립트
- DINOv3 학습용 stage runner (`train/scripts/vast_ai_training_runner.py`)
- bootstrap / validate-upload / smoke / full 원격 실행 커맨드

## 현재 Mirip_v2 학습 직전 절차

1. `train/data` freeze
2. local raw validator 실행
3. local snapshot/pairs/reports 생성
4. local prepared validator 실행
5. Vast instance 준비
6. `train/data`, `train/training/data`, `train/reports` 업로드
7. remote `bootstrap`
8. remote `validate-upload`
9. remote `smoke`
10. smoke 통과 후 `full`

## 공식 문서

- Quickstart: <https://docs.vast.ai/quickstart>
- Creating Instances with the API: <https://docs.vast.ai/api-reference/creating-instances-with-api>
- Attach SSH Key: <https://docs.vast.ai/api-reference/instances/attach-ssh-key>
- Execute: <https://docs.vast.ai/api-reference/instances/execute>
- Manage Instance: <https://docs.vast.ai/api-reference/instances/manage-instance>
- Destroy Instance: <https://docs.vast.ai/api-reference/instances/destroy-instance>
- Vast.ai SDK repo: <https://github.com/vast-ai/vast-sdk>
