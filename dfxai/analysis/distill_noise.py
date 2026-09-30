"""
트리 적합의 우연이 성능 차 Δ 에 얼마나 실리는지 (R10 보조).

    python -m dfxai.analysis.distill_noise --outdir results/distill --workers 4

R10 은 깊이마다 트리를 한 번만 적합했다. BT-v2 처럼 깊이 4 트리가 성능을 회복했는데 깊이 6·8 트리는 못하는
비단조 결과가 나왔으므로, 같은 깊이에서 **훈련 교전을 복원추출**(부트스트랩)해 트리를 여러 번 적합하고 폐루프
성능이 얼마나 흔들리는지 잰다. 평가 시드는 R10 과 같다.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ..experiments.run_eval import make_env
from .distill_perf import TreePolicy, _play, _alpha_of, _spec_lookup, SEED0
from .dstar_validity import _episode_ids


def _conditions() -> list[str]:
    return ["BT-v2", "BT-v3", "BTO-s0-b300", "RL-s0-b300-a1.00", "RL-s11-b300-a1.00", "RL-s1-b300-a1.00", "RL-s9-b300-a1.00"]


def _job(args) -> list[dict]:
    name, main_dir, spec, depth, boot, n_eval, seed0 = args
    alpha = _alpha_of(spec)
    env = make_env({})
    z = np.load(os.path.join(main_dir, f"traj_{name}.npz"))
    X, Y = z["obs"].astype(np.float64), z["act"].astype(np.float64)
    ep = _episode_ids(X)
    n_ep = int(ep.max() + 1)
    rng = np.random.default_rng(1000 + boot)
    pick = rng.integers(0, n_ep, n_ep)
    idx = np.concatenate([np.where(ep == e)[0] for e in pick])
    tree = DecisionTreeRegressor(max_depth=depth, random_state=boot).fit(X[idx], Y[idx])
    tp = TreePolicy(tree)
    rows = []
    for ov in (2, 3):
        for i in range(n_eval):
            for side in (0, 1):
                row, _, _ = _play(tp, ov, seed0 + i, side, alpha, env)
                row.update(cond=name, depth=depth, boot=boot)
                rows.append(row)
    return rows


def run(outdir: str, main_dir: str, hyb_dir: str, workers: int, depths=(8, 16), n_boot: int = 5,
        n_eval: int = 100, seed0: int = SEED0) -> None:
    specs = _spec_lookup(main_dir, hyb_dir)
    jobs = [(n, specs[n][0], specs[n][1], d, b, n_eval, seed0) for n in _conditions() for d in depths for b in range(n_boot)]
    print(f"트리 적합 노이즈: 작업 {len(jobs)}개", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_job, jobs, chunksize=1))
    else:
        res = [_job(j) for j in jobs]
    pd.DataFrame([r for x in res for r in x]).to_csv(os.path.join(outdir, "noise_games.csv"), index=False)
    with open(os.path.join(outdir, "noise_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conditions=_conditions(), depths=list(depths), n_boot=n_boot, n_eval=n_eval, eval_seed0=seed0,
                       elapsed_s=round(time.time() - t0, 1)), f, ensure_ascii=False, indent=2)
    print(f"완료 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="트리 적합 노이즈")
    ap.add_argument("--outdir", default="results/distill")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--hyb-dir", default="results/main_hyb")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.outdir, a.main_dir, a.hyb_dir, a.workers)
