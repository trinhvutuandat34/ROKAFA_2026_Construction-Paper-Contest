#!/bin/bash
# 추가 실험 학습 — 판정 기준은 paper/추가실험_사전기준.md (학습 전에 기록)
#   E1 학습 연장: 학습 시드 1·11·14·2·18 을 300 -> 1,000세대 (results/es_long)
#   E2 하이브리드 보강: 잔차형·게이팅형 시드 0–2 를 200 -> 300세대, 시드 3–4 새로 300세대
#   E3 격추만 승리 규칙 재학습: 시드 0–4, 300세대 (results/es_draw)
# 4코어 기준 약 6–7시간. 작업마다 워커 1개로 돌리고 xargs 로 코어 수만큼 동시에 돌린다.
#   bash results/run_extra_train.sh        # 전부 (E1 다음 E2·E3)
#   bash results/run_extra_train.sh e1     # E1 만
#   bash results/run_extra_train.sh e23    # E2·E3 만
STAGE=${1:-all}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
mkdir -p results/es_long results/es_draw
C="--pop 32 --episodes 8 --sigma 0.05 --lr 0.01 --workers 1"
G="results/es/ckpt_seed11_gen00300.npz"

if [ "$STAGE" = all ] || [ "$STAGE" = e1 ]; then
echo "=== E1 시작 $(date)"
for s in 1 11 14 2 18; do
  echo "python -m dfxai.rl.train_es --generations 1000 $C --checkpoint-every 50 --seed $s --tag seed$s \
--resume results/es/ckpt_seed${s}_gen00300.npz --resume-history results/es/history_seed$s.json \
--outdir results/es_long > results/es_long/train_seed$s.log 2>&1"
done | xargs -P 5 -I{} bash -c "{}"
echo "=== E1 끝 $(date)"
fi
[ "$STAGE" = e1 ] && exit 0

echo "=== E2·E3 시작 $(date)"
{
  for s in 0 1 2 3 4; do      # E3 (300세대, 오래 걸리는 작업부터)
    echo "python -m dfxai.rl.train_es --generations 300 $C --checkpoint-every 25 --seed $s --tag seed$s \
--init results/es/bc_init.npz --timeout-rule draw --outdir results/es_draw > results/es_draw/train_seed$s.log 2>&1"
  done
  for s in 3 4; do            # E2 새 시드 (300세대)
    echo "python -m dfxai.rl.train_es_hybrid --kind residual --beta 0.35 --generations 300 $C --checkpoint-every 25 \
--seed $s --tag seed$s --outdir results/es_res > results/es_res/train_seed$s.log 2>&1"
    echo "python -m dfxai.rl.train_es_hybrid --kind gating --rl-ckpt $G --generations 300 $C --checkpoint-every 25 \
--seed $s --tag seed$s --outdir results/es_gate > results/es_gate/train_seed$s.log 2>&1"
  done
  for s in 0 1 2; do          # E2 기존 시드 이어 학습 (200 -> 300세대)
    echo "python -m dfxai.rl.train_es_hybrid --kind residual --beta 0.35 --generations 300 $C --checkpoint-every 25 \
--seed $s --tag seed$s --resume results/es_res/ckpt_seed${s}_gen00200.npz --resume-history results/es_res/history_seed$s.json \
--outdir results/es_res > results/es_res/train_resume_seed$s.log 2>&1"
    echo "python -m dfxai.rl.train_es_hybrid --kind gating --rl-ckpt $G --generations 300 $C --checkpoint-every 25 \
--seed $s --tag seed$s --resume results/es_gate/ckpt_seed${s}_gen00200.npz --resume-history results/es_gate/history_seed$s.json \
--outdir results/es_gate > results/es_gate/train_resume_seed$s.log 2>&1"
  done
} | xargs -P 4 -I{} bash -c "{}"
echo "=== E2·E3 끝 $(date)"
touch results/EXTRA_TRAIN_DONE
