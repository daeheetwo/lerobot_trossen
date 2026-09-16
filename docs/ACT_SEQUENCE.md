# ACT Task08 → Task09 연속 평가

기존 08·09 ACT 체크포인트를 따로 불러와, **08이 끝난 실제 자세에서 09를 시작하는** 평가 도구다. 로봇은 한 번 연결하고 두 단계 사이에는 초기 위치 이동·재연결·리더암 제어를 실행하지 않는다. 기본 시험 횟수는 `08→09` 한 쌍을 기준으로 10회다.

현재 버전은 사람이 완료 여부와 전환 시점을 지정한다. 이 시험은 단계 사이 초기화 문제를 제거하고 연결 성공률을 측정하기 위한 것이며, 자동 완료 판정이나 11단계 자율 실행은 포함하지 않는다. 09가 08의 종료 자세에서 성공하는지는 실제 평가로 확인해야 한다.

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/eval_act_sequence.py` | 명령행 실행, 터미널 키 입력, 모의 실행 |
| `scripts/sequence_core.py` | 연결·대기·08·전환·09·시험 종료 상태 제어 |
| `scripts/sequence_lerobot.py` | 실제 ACT 추론, 로봇 입출력, 영상·결과 저장 |
| `configs/task08_09_sequence.json` | 모델, 데이터셋 메타데이터, 장치, 시험 시간 설정 |
| `tests/test_act_sequence.py` | 로봇 없이 실행하는 동작 검증 |

기존 `lerobot-record` 실행 경로는 유지했다. 팔 설정에 추가한 `park_on_disconnect`의 기본값은 `true`이므로 기존 종료 시 staged/sleep 자세 이동도 유지된다. 연속 평가만 이 값을 `false`로 설정하고, 사용자가 `P`로 종료할 때만 주차 자세로 이동한다.

## 1. Windows에서 모의 실행

`D:\trossenai`에서 Python 3.11 이상으로 실행한다. 이 모드는 Python 표준 라이브러리만 사용하며 모델 다운로드, GPU, 카메라, 로봇이 필요 없다.

```powershell
cd D:\trossenai
python scripts/eval_act_sequence.py --dry-run
python -m unittest discover -s tests -v
```

현재 PC에서 `python`이 잡히지 않으면, 이번 검증에 사용한 설치 경로로도 실행할 수 있다.

```powershell
cd D:\trossenai
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 -B scripts/eval_act_sequence.py --dry-run
```

완료 시 `MOCK PASS: 10 sequences; connect=1, disconnect=1, teleop=0`과 보고서 경로가 표시된다. 이는 전환 코드 검증 결과이며 ACT의 실제 성공률을 의미하지 않는다. Windows 개발 폴더에서 로봇용 전체 의존성을 설치할 필요는 없다.

## 2. 실제 모델 경로 설정

`configs/task08_09_sequence.json`의 두 `policy`를 실제 학습 완료된 체크포인트로 지정한다. 현재 입력된 이름은 앞서 정한 학습 이름이며, 해당 모델이 Hub에 업로드되었는지는 별도로 확인해야 한다.

```json
"policy": "kiroaiseoul/act_task08_takeout_and_put_beaker_100k_251data"
```

또는 로봇 PC에 복사한 체크포인트 디렉터리를 지정한다.

```json
"policy": "/home/USER/lerobot_trossen/output/RUN/checkpoints/100000/pretrained_model"
```

각 체크포인트에 `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json` 및 이들이 참조하는 통계 파일이 있어야 한다. **08과 09가 각자 저장된 정규화·역정규화 설정을 사용한다.** 평가 데이터를 기준으로 통계를 새로 계산하지 않는다. ACT의 `n_action_steps`와 temporal ensemble 설정은 체크포인트 값을 유지하며 확인 로그에 출력한다.

`dataset`은 해당 모델 학습에 사용한 데이터셋의 Hub ID 또는 로컬 데이터셋 폴더다. 연결 전에 `meta/info.json`으로 state/action 채널 이름·순서·차원, 카메라 크기, FPS를 확인한다. Hub에서는 메타데이터 파일만 가져온다.

기본 설정은 **observation.state 14차원 / action 16차원**, 카메라 3대, 30 FPS다. `include_base_in_state=false`는 입력에서 베이스 속도 두 채널만 제외한다. 09의 후진 등 베이스 출력은 유지된다. IP·카메라 시리얼은 기존 평가 명령에서 가져왔으므로 실제 로봇 PC와 일치하는지 확인한다.

## 3. 로봇 PC에서 사전 확인과 실행

실제 실행은 이 저장소의 **Linux 로봇 PC 환경(LeRobot 0.4.4, CUDA)**에서 한다. Windows 폴더 수정만으로 다른 PC의 코드는 바뀌지 않는다. 배포할 때는 새 스크립트 3개, 설정 파일과 수정한 로봇 패키지를 함께 옮겨야 한다. 이번 작업에서는 다른 서버나 로봇 PC를 변경하지 않았다.

프로젝트 폴더에서 먼저 다음 명령을 실행한다.

```bash
uv run python scripts/eval_act_sequence.py --check
```

`--check`는 실제 두 모델·처리기를 불러오고 모의 관측으로 CUDA 추론까지 확인한다. 로봇의 `connect()`나 동작 명령은 호출하지 않는다. 비공개 모델·데이터셋이면 해당 로봇 PC에서 Hugging Face 로그인이 필요하다.

준비가 끝난 뒤 실제 동작은 다음 명령으로 시작한다.

```bash
uv run python scripts/eval_act_sequence.py --run
```

**실행 최초 연결 시에는 기존 코드처럼 팔이 staged 자세로 이동한다.** 이후 08→09 사이에는 이 이동을 반복하지 않는다. 키 입력은 실행 중인 터미널에 포커스가 있어야 하며 영문 입력 상태로 사용한다. Rerun 화면이 필요 없으면 설정의 `display_data`를 `false`로 바꾼다.

## 4. 한 쌍을 평가하는 순서

```text
최초 연결 및 staged 이동
  → READY: 08 시작 환경 준비
  → S: 08 실행
  → N: 사람이 08 성공 판정 → 현재 자세 유지, 베이스 정지
  → HANDOFF: 팔·그리퍼·물체를 재배치하지 않음
  → S: 같은 자세의 관측으로 09 실행
  → N: 사람이 09 성공 판정 → 08→09 한 쌍 저장
  → READY: 다음 시험 환경 재배치
  → 10쌍 종료 후 DONE에서 유지 → 물체를 확보한 뒤 P 또는 X로 종료
```

| 키 | 동작 |
|---|---|
| `S` | READY에서 08 시작, HANDOFF에서 09 시작, PAUSED에서 현재 단계 재개 |
| `N` | 실행 중인 단계의 성공을 사람이 기록. 08이면 HANDOFF, 09이면 한 쌍 종료 |
| Space | 현재 단계를 일시 정지하고 자세 유지 |
| `F` | 현재 시험 실패 처리 및 저장. 08 실패 시 09는 실행하지 않음 |
| `T` | READY에서만 리더암 제어 켜기/끄기. 시험 간 재배치용 |
| `Q` 또는 Ctrl+C | 평가 중단 및 자세 유지. 아직 연결을 해제하지 않음 |
| `P` | DONE에서 staged/sleep 자세로 이동 후 종료 |
| `X` | DONE에서 주차 자세로 이동하지 않고 연결 정리 후 종료 |

READY의 리더암 제어는 자동으로 켜지지 않는다. `T`를 켜면 리더암의 현재 관절 위치가 바로 목표가 되므로 팔 자세를 맞춘 뒤 사용한다. 이 도구에서 리더암 제어 중 베이스 속도 명령은 0으로 고정된다. 수동 베이스 재배치는 기존 장치 절차에 따라 시험 사이에 수행한다. HANDOFF에서는 `T`가 동작하지 않는다.

`timeout_s`는 단계별 누적 실행 시간 제한이다. 일시 정지 시간은 제외하고 재개해도 제한을 처음부터 다시 주지 않는다. 시간 초과는 성공이 아니라 `timeout`으로 저장하고 다음 단계로 자동 진행하지 않는다.

전환·대기 시 팔은 관측된 관절 위치를 유지하며, 그리퍼는 마지막 목표를 유지해 잡고 있던 물체를 임의로 놓지 않게 한다. 모델의 `reset()`은 남은 action chunk와 추론 상태를 비우는 동작으로, 로봇을 초기 위치로 움직이지 않는다.

**종료 전에는 잡고 있는 비커를 확보해야 한다.** `P`는 실제로 팔을 움직이며, `X`도 드라이버 정리로 토크가 해제될 수 있다. 오류·프로세스 종료 후의 자세 유지는 보장하지 않는다. 터미널 중단 키는 다음 제어 처리 시점에 반영되므로 하드웨어 비상 정지를 대체하지 않는다.

## 5. 저장 결과와 판정

기본 경로는 `output/act_sequence_날짜_시간/`이다. `--output`으로 새 폴더를 지정할 수 있고, 기존 폴더는 덮어쓰지 않는다. **Hugging Face 업로드는 하지 않는다.**

| 결과 | 내용 |
|---|---|
| `run_config.json` | 실행 설정과 실제 모델 캐시/로컬 경로 |
| `results.jsonl` | 시험별 08/09 판정, 최종 상태, 에피소드 번호·프레임 수 |
| `events.jsonl` | 전환·정지 이벤트와 프레임별 단계·phase·실제 경과 시간 |
| `dataset/` | 08→09 전체를 한 에피소드로 저장한 LeRobot 영상·state·action |
| `dry_run.json` | 모의 실행에서만 생성되는 보고서 |

성공한 경우뿐 아니라 실패·시간 초과·중단도 남긴다. 저장 과정 오류가 나면 `recording_failed` 이벤트와 남은 파일을 확인해야 한다. 실제 루프가 목표 FPS보다 느릴 수 있으므로 성능 분석에는 `events.jsonl`의 실제 시간도 사용한다. 영상과 LeRobot timestamp는 설정한 FPS 기준이다.

연속 성공률은 **08과 09 모두 성공한 시험 수 / 시작한 시험 수**로 계산한다. 08 성공 후 09 성공률도 따로 보면 연결부 문제를 파악하기 쉽다. 중단을 제외하는 실험 규칙이 있다면 제외 횟수와 사유를 함께 보고한다.

이 기록에는 사람의 전환 판단, 대기 구간과 실패도 포함된다. HANDOFF 프레임은 다음 단계 task 문구로 기록되지만 아직 다음 모델이 실행된 것은 아니며, 정확한 구분은 `events.jsonl`의 `phase`를 사용한다. 재학습용 성공 시연으로 바로 사용하지 말고 구간과 결과를 검토해야 한다.

## 검증 범위

Windows에서는 같은 상태 제어기를 사용하는 모의 실행과 단위 테스트를 실행할 수 있다. 연결 1회, 단계 사이 자세·그리퍼 유지, 08 실패 시 09 미실행, 시간 초과·정지·재개, 잘못된 차원·채널·action 거부, 기존 종료 동작의 기본값을 확인한다. 실제 카메라·모터·CUDA 추론은 로봇 PC의 `--check`와 현장 시험으로 검증해야 한다.
