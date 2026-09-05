#!/usr/bin/env bash
# Preflight + full experiment run, meant to be executed ON the keti host.
#
#   ssh keti_1
#   cd ~/CV && git pull origin claude/lst-keti-server-setup-l2rpd1
#   bash experiments/quantile-anomaly/run_on_keti.sh
#
# Every step prints what it found, so a failure says which one and why rather
# than dying silently. Nothing here is destructive: it creates a venv and
# writes only under experiments/quantile-anomaly/.
set -uo pipefail
cd "$(dirname "$0")"

N=${N:-300}
NEMP=${NEMP:-500}
SEEDS=${SEEDS:-"0 1 2"}
CHRONOS_PATH=${CHRONOS_PATH:-}
VENV=${VENV:-.venv-keti}

hr() { printf '%s\n' "------------------------------------------------------------"; }
step() { hr; printf '%s\n' "$1"; hr; }

step "0. where am I"
echo "  host   : $(hostname)"
echo "  user   : $(whoami)"
echo "  pwd    : $(pwd)"
echo "  python : $(command -v python3 || echo MISSING) ($(python3 -V 2>&1))"

step "1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
else
  echo "  nvidia-smi 없음 -> CPU로 실행된다 (느리지만 동작한다)"
fi

step "2. huggingface.co 도달 여부"
HF_CODE=$(curl -s -m 20 -o /dev/null -w '%{http_code}' https://huggingface.co 2>/dev/null || echo 000)
echo "  https://huggingface.co -> HTTP $HF_CODE"
if [ "$HF_CODE" = "200" ]; then
  echo "  => Chronos-2를 여기서 바로 받을 수 있다. 모델 전송 불필요."
elif [ -n "$CHRONOS_PATH" ]; then
  echo "  => 허브는 막혔지만 CHRONOS_PATH가 주어졌다: $CHRONOS_PATH"
  [ -f "$CHRONOS_PATH/config.json" ] \
    && echo "     config.json 확인됨" \
    || { echo "     !! config.json 없음. 복사가 덜 되었거나 경로가 한 단계 어긋났다."; ls -la "$CHRONOS_PATH" 2>&1 | head; }
else
  echo "  => 허브가 막혔고 로컬 체크포인트도 없다."
  echo "     CHRONOS_PATH=~/models/chronos-2 를 주거나, HF가 되는 곳에서 받아 옮겨야 한다."
  echo "     이대로 진행하면 ORACLE + MLPQR 만으로 돌아간다 (Chronos-2 열은 비게 된다)."
fi

step "3. python 환경"
if [ ! -d "$VENV" ]; then
  echo "  $VENV 생성"
  python3 -m venv "$VENV" || { echo "  !! venv 생성 실패"; exit 1; }
fi
PY="$VENV/bin/python"
"$VENV/bin/pip" install -q --upgrade pip
echo "  의존성 설치 (수 분 걸릴 수 있다)"
"$VENV/bin/pip" install -q -r requirements.txt 2>&1 | tail -3
"$PY" - <<'EOF'
import torch, numpy, scipy, pandas, sklearn
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      f"  device_count={torch.cuda.device_count()}")
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  pandas {pandas.__version__}")
EOF

step "4. 단위 테스트"
"$PY" test_core.py 2>&1 | tail -6
"$PY" test_core.py >/dev/null 2>&1 || { echo "  !! 테스트 실패. 여기서 멈춘다."; exit 1; }

step "5. 실험 실행  (N=$N, seeds=$SEEDS)"
ARGS=(--n "$N" --n-emp "$NEMP" --seeds $SEEDS)
if [ -n "$CHRONOS_PATH" ]; then
  ARGS+=(--chronos-path "$CHRONOS_PATH")
  export HF_HUB_OFFLINE=1
  echo "  로컬 체크포인트 사용, HF_HUB_OFFLINE=1"
fi
"$PY" run.py "${ARGS[@]}" 2>&1 | tail -25 || { echo "  !! run.py 실패"; exit 1; }

step "6. 평가와 요약"
"$PY" evaluate.py 2>&1 | tail -3
"$PY" summarize.py > results/summary.txt 2>&1
echo "  results/summary.txt 기록됨"

step "7. 어떤 모델이 실제로 들어갔나"
"$PY" - <<'EOF'
import json, pandas as pd
print("  provenance:", json.load(open("results/provenance.json"))["models"])
d = pd.read_csv("results/scores.csv")
print("  models in scores.csv:", sorted(d.model.unique()))
print("  rows:", len(d))
EOF

hr
echo "완료. 다음 파일을 대화에 붙여 주면 보고서를 갱신한다."
echo "  results/summary.txt"
echo "  results/metrics.csv"
echo "  results/diagnostics.csv"
