#!/bin/bash
# 추가 실험 평가 — 판정 기준은 paper/추가실험_사전기준.md
#   bash results/run_extra_eval.sh e1 | e2 | e3 | e4 | all
# 평가 절차는 본실험과 같다(상대 BT-v2·BT-v3, 초기조건 100개 × 진영 2, 대리모델 궤적 30교전).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
W=${WORKERS:-4}
STAGE=${1:-all}

if [ "$STAGE" = all ] || [ "$STAGE" = e1 ]; then
  # E1 학습 연장: 300세대(본실험 체크포인트) + 400–1,000세대(이어 학습)
  CK=""
  for s in 1 11 14 2 18; do
    CK="$CK results/es/ckpt_seed${s}_gen00300.npz"
    for g in 00400 00500 00600 00700 00800 00900 01000; do CK="$CK results/es_long/ckpt_seed${s}_gen${g}.npz"; done
  done
  python -m dfxai.experiments.run_eval --no-bt --ckpts $CK --alphas 1.0 --opponents 2 3 \
         --n-seeds 100 --workers $W --outdir results/extra_long
  python -m dfxai.analysis.report --outdir results/extra_long > /dev/null
fi

if [ "$STAGE" = all ] || [ "$STAGE" = e2 ]; then
  # E2 하이브리드 보강: 잔차형·게이팅형 시드 0–4 × 100·200·300세대 + 원 모델(BT-v2, 학습 시드 11)
  R=""; Q=""
  for s in 0 1 2 3 4; do for g in 00100 00200 00300; do
    R="$R results/es_res/ckpt_seed${s}_gen${g}.npz"; Q="$Q results/es_gate/ckpt_seed${s}_gen${g}.npz"
  done; done
  python -m dfxai.experiments.run_eval --ckpts results/es/ckpt_seed11_gen00300.npz --alphas 1.0 \
         --residual-ckpts $R --gate-ckpts $Q --opponents 2 3 --n-seeds 100 --workers $W --outdir results/extra_hyb
  python -m dfxai.analysis.report --outdir results/extra_hyb > /dev/null
fi

if [ "$STAGE" = all ] || [ "$STAGE" = e3 ]; then
  # E3 격추만 승리 규칙 재학습: 무승부 규칙에서 100·200·300세대, 비교용으로 BT 3종·BTO 300세대
  D=""
  for s in 0 1 2 3 4; do for g in 00100 00200 00300; do D="$D results/es_draw/ckpt_seed${s}_gen${g}.npz"; done; done
  python -m dfxai.experiments.run_eval --ckpts $D --alphas 1.0 \
         --bto-ckpts results/es_bto/ckpt_seed{0,1,2}_gen00300.npz --opponents 2 3 --n-seeds 100 \
         --workers $W --timeout-rule draw --outdir results/extra_draw
  python -m dfxai.analysis.report --outdir results/extra_draw > /dev/null
  # 같은 300세대 모델을 본실험 규칙(시간종료 체력 비교)에서도 평가
  D3=""; for s in 0 1 2 3 4; do D3="$D3 results/es_draw/ckpt_seed${s}_gen00300.npz"; done
  python -m dfxai.experiments.run_eval --no-bt --ckpts $D3 --alphas 1.0 --opponents 2 3 --n-seeds 100 \
         --workers $W --outdir results/extra_draw_hp
  python -m dfxai.analysis.report --outdir results/extra_draw_hp > /dev/null
fi

if [ "$STAGE" = all ] || [ "$STAGE" = e4 ]; then
  # E4 학습 규모 확대: 개체군 64 × 후보당 16교전, 시드 0–2 × 100·200·300세대
  B=""
  for s in 0 1 2; do for g in 00100 00200 00300; do B="$B results/es_big/ckpt_seed${s}_gen${g}.npz"; done; done
  python -m dfxai.experiments.run_eval --no-bt --ckpts $B --alphas 1.0 --opponents 2 3 \
         --n-seeds 100 --workers $W --outdir results/extra_big
  python -m dfxai.analysis.report --outdir results/extra_big > /dev/null
fi
