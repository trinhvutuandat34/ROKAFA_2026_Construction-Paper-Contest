#!/bin/bash
# 수준 향상 실험 평가 — 기준은 paper/수준향상_사전기준.md
#   bash results/run_level_eval.sh b1 [세대…]   (기본: 100–1,000세대 중 체크포인트가 있는 것)
#   bash results/run_level_eval.sh b1h_ref       보류 상대 보조 점검의 비교 기준(행동트리·BTO·본실험 학습 모델 20개)
#   bash results/run_level_eval.sh b1h [세대…]  보류 상대 보조 점검(격추 우선 학습 모델 2개 + PPO 2개 상대)
# 1) 본실험 규칙(체력 비교)에서 BT-v2·BT-v3 상대, 2) 격추만 승리 규칙에서 같은 평가(+ BT·BTO 비교),
# 3) 1,000세대 모델이 있으면 본실험 26개 + 새 모델 5개의 전원 대전.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
W=${WORKERS:-4}
STAGE=${1:-b1}; shift
GENS=${*:-"00100 00200 00300 00400 00500 00600 00700 00800 00900 01000"}

if [ "$STAGE" = b1 ]; then
  D=results/es_pool
  CK=""
  for s in 0 1 2 3 4; do for g in $GENS; do
    [ -f $D/ckpt_seed${s}_gen${g}.npz ] && CK="$CK $D/ckpt_seed${s}_gen${g}.npz"
  done; done
  python -m dfxai.experiments.run_eval --no-bt --ckpts $CK --alphas 1.0 --rl-prefix POOL --opponents 2 3 \
         --n-seeds 100 --workers $W --outdir results/level_b1_hp
  python -m dfxai.analysis.report --outdir results/level_b1_hp > /dev/null
  python -m dfxai.experiments.run_eval --ckpts $CK --alphas 1.0 --rl-prefix POOL \
         --bto-ckpts results/es_bto/ckpt_seed{0,1,2}_gen00300.npz --opponents 2 3 \
         --n-seeds 100 --workers $W --timeout-rule draw --outdir results/level_b1_draw
  python -m dfxai.analysis.report --outdir results/level_b1_draw > /dev/null
  if [ -f $D/ckpt_seed0_gen01000.npz ]; then
    python -m dfxai.experiments.round_robin --rl-ckpts "results/es/ckpt_seed*_gen00300.npz" \
           --bto-ckpts "results/es_bto/ckpt_seed*_gen00300.npz" --pool-ckpts "$D/ckpt_seed*_gen01000.npz" \
           --n-seeds 30 --workers $W --outdir results/level_b1_pool > /dev/null
  fi
fi

# 보류 상대 보조 점검 (수준향상_사전기준.md 1-1절): 학습 풀에 없고 BT-v3 구조도 아닌 상대
H="KF-s0=results/es_killfirst/ckpt_seed0_gen00300.npz KF-s1=results/es_killfirst/ckpt_seed1_gen00300.npz \
PPO-s0=results/ppo/ckpt_seed0_step001500000.npz PPO-s1=results/ppo/ckpt_seed1_step001500000.npz"
if [ "$STAGE" = b1h_ref ]; then
  python -m dfxai.experiments.run_eval --ckpts results/es/ckpt_seed*_gen00300.npz --alphas 1.0 \
         --bto-ckpts results/es_bto/ckpt_seed{0,1,2}_gen00300.npz --opponents --opp-rl $H \
         --n-seeds 100 --record-episodes 0 --workers $W --outdir results/level_heldout_ref
fi
if [ "$STAGE" = b1h ]; then
  D=results/es_pool; CK=""
  for s in 0 1 2 3 4; do for g in $GENS; do
    [ -f $D/ckpt_seed${s}_gen${g}.npz ] && CK="$CK $D/ckpt_seed${s}_gen${g}.npz"
  done; done
  python -m dfxai.experiments.run_eval --no-bt --ckpts $CK --alphas 1.0 --rl-prefix POOL --opponents --opp-rl $H \
         --n-seeds 100 --record-episodes 0 --workers $W --outdir results/level_b1_heldout
fi
