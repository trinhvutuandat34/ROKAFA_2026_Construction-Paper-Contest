"""
대리 트리 정책이 원본과 갈라지는 곳은 어느 전술 영역인가 (R10 보조, R11 과 연결).

    python -m dfxai.analysis.distill_regions --outdir results/distill --workers 4

트리를 정책으로 돌려 방문한 상태마다 원본 정책의 지령과 트리의 지령을 비교하고, 제곱오차를 Saldiran 외(2024)의
전술 영역(정면 조우·공격·방어·중립)으로 나눈다. 비교를 위해 같은 트리가 **원본이 방문한 상태**(새 시드, 원본이 직접
싸운 교전)에서 낸 오차의 영역별 몫도 함께 잰다. 두 값의 차이가 클수록 트리 정책이 원본이 가지 않는 상황으로 들어가
오차가 쌓인다는 뜻이다.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ..experiments.run_eval import build_condition, make_env
from .distill_perf import TreePolicy, _play, _alpha_of, _spec_lookup
from .tactical_regions import region_codes, REGIONS
from ..xai.surrogate import EPS_VAR

SEED0 = 4_000_000


def _conditions() -> list[str]:
    return [f"RL-s{s}-b300-a1.00" for s in range(20)] + ["BT-v2", "RL-s0-b0-a1.00"]


def _shares(obs: np.ndarray, err: np.ndarray, act: np.ndarray) -> dict:
    reg = region_codes(obs)
    tot = float(err.sum())
    var = float(np.var(act, axis=0).sum())
    out = dict(unexplained=tot / (len(obs) * var) if var > EPS_VAR else np.nan)
    for k, rn in enumerate(REGIONS):
        m = reg == k
        out[f"time_{rn}"] = float(m.mean())
        out[f"err_{rn}"] = float(err[m].sum() / tot) if tot > 0 else np.nan
    return out


def _job(args) -> list[dict]:
    name, main_dir, spec, depths, n_seeds, seed0 = args
    pol = build_condition(spec)
    alpha = _alpha_of(spec)
    env = make_env({})
    z = np.load(os.path.join(main_dir, f"traj_{name}.npz"))
    X, Y = z["obs"].astype(np.float64), z["act"].astype(np.float64)
    # 원본이 새 시드에서 직접 싸운 교전의 상태와 지령
    ob_o, ac_o = [], []
    for ov in (2, 3):
        for i in range(n_seeds):
            _, ob, ac = _play(pol, ov, seed0 + i, 0, alpha, env, record=True)
            ob_o.append(ob); ac_o.append(ac)
    Xo, Yo = np.concatenate(ob_o), np.concatenate(ac_o)
    rows = []
    for d in depths:
        tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, Y)
        tp = TreePolicy(tree)
        ob_t = []
        for ov in (2, 3):
            for i in range(n_seeds):
                _, ob, _ = _play(tp, ov, seed0 + i, 0, alpha, env, record=True)
                ob_t.append(ob)
        Xt = np.concatenate(ob_t)
        At = np.asarray([pol.act(o) for o in Xt], dtype=np.float64)
        e_t = ((At - tree.predict(Xt)) ** 2).sum(axis=1)
        e_o = ((Yo - tree.predict(Xo)) ** 2).sum(axis=1)
        rows.append(dict(cond=name, depth=d, states="tree-visited", **_shares(Xt, e_t, At)))
        rows.append(dict(cond=name, depth=d, states="original-visited", **_shares(Xo, e_o, Yo)))
    return rows


def run(outdir: str, main_dir: str, hyb_dir: str, workers: int, depths=(8, 16), n_seeds: int = 30, seed0: int = SEED0) -> None:
    specs = _spec_lookup(main_dir, hyb_dir)
    jobs = [(n, specs[n][0], specs[n][1], tuple(depths), n_seeds, seed0) for n in _conditions()]
    print(f"트리 정책 이탈 영역: 조건 {len(jobs)}개, 깊이 {depths}", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_job, jobs, chunksize=1))
    else:
        res = [_job(j) for j in jobs]
    pd.DataFrame([r for x in res for r in x]).to_csv(os.path.join(outdir, "region_error.csv"), index=False)
    print(f"완료 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="대리 트리 정책이 원본과 갈라지는 영역")
    ap.add_argument("--outdir", default="results/distill")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--hyb-dir", default="results/main_hyb")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.outdir, a.main_dir, a.hyb_dir, a.workers)
