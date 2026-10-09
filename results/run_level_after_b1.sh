#!/bin/bash
# B1 학습이 1,000세대까지 끝나면 전체 평가(본실험 규칙·격추만 승리·전원 대전)와 보류 상대 점검을 이어서 돌린다.
cd "$(dirname "$0")/.."
until [ $(ls results/es_pool/ckpt_seed*_gen01000.npz 2>/dev/null | wc -l) -ge 5 ]; do sleep 60; done
echo "=== B1 1,000세대 도달 $(date)"
WORKERS=4 bash results/run_level_eval.sh b1
WORKERS=4 bash results/run_level_eval.sh b1h
echo "=== 평가 끝 $(date)"
echo DONE
