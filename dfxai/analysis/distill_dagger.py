"""
대리 트리 정책의 성능 손실이 분포 이동 때문인지 확인하는 보조 시험 (R10 보조, DAgger 방식).

    python -m dfxai.analysis.distill_dagger --outdir results/distill --workers 4

R10 의 트리는 원본 정책이 방문한 상태에서만 학습했다. 트리 정책이 돌면 원본이 가지 않던 상태에 들어가
오차가 쌓일 수 있다(모방 학습의 공변량 이동). 이 시험은 표준 처방인 DAgger 를 그대로 적용한다.

  반복 i = 1..N:
    1. 현재 트리를 정책으로 삼아 학습용 시드(3,000,000 대, 평가 시드와 다름)에서 BT-v2·BT-v3 와 싸운다(청군).
    2. 트리가 방문한 관측마다 **원본 정책에 지령을 물어** 정답을 붙인다.
    3. 기존 데이터에 합쳐 같은 깊이로 트리를 다시 적합한다.
    4. 평가 시드에서 성능을 잰다(R10 과 같은 시드·같은 절차).

결과가 원본 성능에 가까워지면 R10 의 손실은 분포 이동 탓이고, 그대로면 깊이 d 트리가 정책의 성능 관련 부분을
담지 못한다는 뜻이다. 반복마다 "트리가 방문한 상태에서 원본 지령을 재현한 R²" 도 기록해 이동의 크기를 본다.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ..experiments.run_eval import build_condition, make_env
from .distill_perf import (TreePolicy, _play, _weighted_r2, _alpha_of, _spec_lookup, SEED0)

TRAIN_SEED0 = 3_000_000


def _dagger_conditions() -> list[str]:
    return [f"RL-s{s}-b300-a1.00" for s in range(20)] + ["BT-v2", "RL-s0-b0-a1.00"]


def _job(args) -> dict:
    name, main_dir, spec, depths, n_iter, n_train, n_eval, seed0 = args
    pol = build_condition(spec)
    alpha = _alpha_of(spec)
    env = make_env({})
    log = np.load(os.path.join(main_dir, f"traj_{name}.npz"))
    X0, Y0 = log["obs"].astype(np.float64), log["act"].astype(np.float64)
    games, fits = [], []
    eval_seeds = [seed0 + i for i in range(n_eval)]
    for d in depths:
        X, Y = X0.copy(), Y0.copy()
        tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, Y)
        for it in range(1, n_iter + 1):
            tp = TreePolicy(tree)
            new_obs = []
            for ov in (2, 3):
                for i in range(n_train):
                    s = TRAIN_SEED0 + (it - 1) * 1000 + i
                    _, ob, _ = _play(tp, ov, s, 0, alpha, env, record=True)
                    new_obs.append(ob)
            No = np.concatenate(new_obs)
            Ne = np.asarray([pol.act(o) for o in No], dtype=np.float64)           # 원본에게 정답을 묻는다
            r2_on = _weighted_r2(Ne, tree.predict(No))                            # 트리가 방문한 상태에서의 충실도
            X, Y = np.concatenate([X, No]), np.concatenate([Y, Ne])
            tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, Y)
            tp = TreePolicy(tree)
            fits.append(dict(cond=name, depth=d, iteration=it, n_data=len(X), n_leaves=tp.n_leaves,
                             r2_on_policy_before=r2_on, r2_train_after=_weighted_r2(Y, tree.predict(X))))
            for ov in (2, 3):
                for s in eval_seeds:
                    for side in (0, 1):
                        row, _, _ = _play(tp, ov, s, side, alpha, env)
                        row.update(cond=name, depth=d, iteration=it)
                        games.append(row)
    return dict(games=games, fits=fits)


def run(outdir: str, main_dir: str, hyb_dir: str, workers: int, depths=(4, 8, 16), n_iter: int = 3,
        n_train: int = 30, n_eval: int = 100, seed0: int = SEED0, conds: list[str] | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    specs = _spec_lookup(main_dir, hyb_dir)
    names = conds or _dagger_conditions()
    jobs = [(n, specs[n][0], specs[n][1], tuple(depths), n_iter, n_train, n_eval, seed0) for n in names]
    print(f"DAgger: 조건 {len(jobs)}개, 깊이 {depths}, 반복 {n_iter}, 학습 시드 {n_train}×2상대, 평가 시드 {n_eval}", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_job, jobs, chunksize=1))
    else:
        res = [_job(j) for j in jobs]
    pd.DataFrame([g for r in res for g in r["games"]]).to_csv(os.path.join(outdir, "dagger_games.csv"), index=False)
    pd.DataFrame([f for r in res for f in r["fits"]]).to_csv(os.path.join(outdir, "dagger_fits.csv"), index=False)
    with open(os.path.join(outdir, "dagger_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conditions=names, depths=list(depths), n_iter=n_iter, n_train=n_train, n_eval=n_eval,
                       train_seed0=TRAIN_SEED0, eval_seed0=seed0, elapsed_s=round(time.time() - t0, 1)),
                  f, ensure_ascii=False, indent=2)
    print(f"완료 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="대리 트리 DAgger 보조 시험")
    ap.add_argument("--outdir", default="results/distill")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--hyb-dir", default="results/main_hyb")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--depths", nargs="*", type=int, default=[4, 8, 16])
    a = ap.parse_args()
    run(a.outdir, a.main_dir, a.hyb_dir, a.workers, tuple(a.depths), a.n_iter)
