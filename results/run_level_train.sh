#!/bin/bash
# 학습 모델 수준 향상 실험 — 목표 기준과 설계는 paper/수준향상_사전기준.md (학습 전 기록)
#   bash results/run_level_train.sh b1     # 기본안 B1: 상대 풀 + 격추만 승리 + 1,000세대 + α 1.0, 시드 0–4
#   bash results/run_level_train.sh e1     # 학습 연장 E1(낮은 우선순위, 추가실험_사전기준.md)
# 컨테이너가 다시 시작되어도 같은 명령을 다시 실행하면 작업마다 마지막 체크포인트에서 이어 학습한다
# (train_es.py --resume 은 끊김 없이 돌린 학습과 비트 단위로 같다). 끝난 작업은 건너뛴다.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
STAGE=${1:-b1}
C="--pop 32 --episodes 8 --sigma 0.05 --lr 0.01 --workers 1"

latest() {   # 출력: history 가 그 세대까지 이어져 있는 가장 늦은 체크포인트 (없으면 빈 줄)
python3 - "$1" "$2" <<'PY'
import glob, json, os, re, sys
d, tag = sys.argv[1], sys.argv[2]
h = os.path.join(d, f"history_{tag}.json")
try:
    n = len(json.load(open(h)))
except Exception:
    n = 0
best = None
for p in glob.glob(os.path.join(d, f"ckpt_{tag}_gen*.npz")):
    g = int(re.search(r"gen(\d+)", p).group(1))
    if g <= n and (best is None or g > best[0]):
        best = (g, p)
print(best[1] if best else "")
PY
}

if [ "$STAGE" = b1 ]; then
  D=results/es_pool; mkdir -p $D
  BTO="results/es_bto/ckpt_seed0_gen00300.npz results/es_bto/ckpt_seed1_gen00300.npz results/es_bto/ckpt_seed2_gen00300.npz"
  RL=$(ls results/es/ckpt_seed*_gen00300.npz | tr '\n' ' ')
  P="--timeout-rule draw --alpha-fixed 1.0 --pool-bto $BTO --pool-rl $RL --pool-slots bt1,bt2,bt2,bto,bto,rl,rl,rl --league-every 100"
  for s in 0 1 2 3 4; do
    [ -f $D/ckpt_seed${s}_gen01000.npz ] && continue
    ck=$(latest $D seed$s)
    if [ -n "$ck" ]; then START="--resume $ck --resume-history $D/history_seed$s.json"
    else START="--init results/es/bc_init.npz"; fi
    echo "python -m dfxai.rl.train_es --generations 1000 $C --checkpoint-every 50 --seed $s --tag seed$s $P $START --outdir $D >> $D/train_seed$s.log 2>&1"
  done | xargs -P 5 -I{} bash -c "{}"
  echo "=== B1 끝 $(date)"
fi

if [ "$STAGE" = e1 ]; then
  D=results/es_long; mkdir -p $D
  for s in 1 11 14 2 18; do
    [ -f $D/ckpt_seed${s}_gen01000.npz ] && continue
    ck=$(latest $D seed$s)
    if [ -n "$ck" ]; then START="--resume $ck --resume-history $D/history_seed$s.json"
    else START="--resume results/es/ckpt_seed${s}_gen00300.npz --resume-history results/es/history_seed$s.json"; fi
    echo "nice -n 19 python -m dfxai.rl.train_es --generations 1000 $C --checkpoint-every 50 --seed $s --tag seed$s $START --outdir $D >> $D/train_seed$s.log 2>&1"
  done | xargs -P 5 -I{} bash -c "{}"
  echo "=== E1 끝 $(date)"
fi
