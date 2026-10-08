"""
정책 풀 상호 대전 (round robin).

본실험의 성능은 "고정 BT 2종을 이기는 능력"이라 좁다. 여기서는 학습 정책들과
BT 들을 한 풀에 넣고 서로 전부 붙여, **풀 전체에 대한 평균 점수**를 두 번째 성능
지표로 만든다. 진영 교대 짝지은 설계는 그대로 쓴다.

    python -m dfxai.experiments.round_robin --rl-ckpts results/es/ckpt_seed*_gen00300.npz \
        --n-seeds 50 --workers 10 --outdir results/pool

출력: results/pool/pool_matrix.csv (행=청, 열=홍 점수), pool_scores.csv (정책별 평균)
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ..env import DogfightEnv, run_episode
from .run_eval import build_condition


def _alpha(spec):
    return float(spec.get("alpha", 1.0 if spec["kind"] in ("rl", "shield") else 0.0))


def _job(args):
    sa, sb, seed = args
    pa, pb = build_condition(sa), build_condition(sb)
    env = DogfightEnv()
    r1 = run_episode(pa, pb, seed=seed, alpha=_alpha(sa), alpha_red=_alpha(sb), env=env)
    r2 = run_episode(pb, pa, seed=seed, alpha=_alpha(sb), alpha_red=_alpha(sa), env=env)
    s1 = 1.0 if r1.winner == 1 else (0.5 if r1.winner == 0 else 0.0)
    s2 = 1.0 if r2.winner == -1 else (0.5 if r2.winner == 0 else 0.0)   # a 가 홍군
    return dict(a=sa["name"], b=sb["name"], seed=seed, score_a=(s1 + s2) / 2)


def specs_from_args(rl_ckpts, bto_ckpts=(), shield_ckpts=(), with_bt=True, pool_ckpts=()):
    specs = []
    if with_bt:
        specs += [dict(name=f"BT-v{v}", kind="bt", bt_version=v) for v in (1, 2, 3)]
    for ck in rl_ckpts:
        ms, mg = re.search(r"seed(\d+)", ck), re.search(r"gen(\d+)", ck)
        specs.append(dict(name=f"RL-s{ms.group(1)}-b{int(mg.group(1))}-a1.00", kind="rl",
                          ckpt=ck, alpha=1.0))
    for ck in bto_ckpts:
        ms, mg = re.search(r"seed(\d+)", ck), re.search(r"gen(\d+)", ck)
        specs.append(dict(name=f"BTO-s{ms.group(1)}-b{int(mg.group(1))}", kind="bto", ckpt=ck))
    for ck in shield_ckpts:
        ms, mg = re.search(r"seed(\d+)", ck), re.search(r"gen(\d+)", ck)
        specs.append(dict(name=f"SHD-s{ms.group(1)}-b{int(mg.group(1))}", kind="shield",
                          ckpt=ck, alpha=1.0, bt_version=3))
    for ck in pool_ckpts:            # 수준 향상 실험(상대 풀 학습) 모델
        ms, mg = re.search(r"seed(\d+)", ck), re.search(r"gen(\d+)", ck)
        specs.append(dict(name=f"POOL-s{ms.group(1)}-b{int(mg.group(1))}", kind="rl",
                          ckpt=ck, alpha=1.0))
    return specs


def run(specs, n_seeds=50, workers=1, outdir="results/pool", seed0=20_000):
    os.makedirs(outdir, exist_ok=True)
    jobs = [(specs[i], specs[j], seed0 + s)
            for i in range(len(specs)) for j in range(i + 1, len(specs))
            for s in range(n_seeds)]
    print(f"정책 {len(specs)}개, 쌍 {len(specs)*(len(specs)-1)//2}개, 교전 {2*len(jobs)}회", flush=True)
    if workers > 1:
        with Pool(workers) as p:
            rows = p.map(_job, jobs, chunksize=8)
    else:
        rows = [_job(j) for j in jobs]
    df = pd.DataFrame(rows)
    names = [s["name"] for s in specs]
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for (a, b), g in df.groupby(["a", "b"]):
        m = g["score_a"].mean()
        mat.loc[a, b] = m; mat.loc[b, a] = 1.0 - m
    mat.to_csv(os.path.join(outdir, "pool_matrix.csv"))
    fam = lambda n: "BT" if n.startswith("BT-v") else ("BTO" if n.startswith("BTO") else
                    ("Shield" if n.startswith("SHD") else ("Pool" if n.startswith("POOL") else "RL")))
    out = pd.DataFrame(dict(cond=names, pool_score=mat.mean(axis=1).values,
                            pool_score_vs_rl=[mat.loc[n, [m for m in names if fam(m) == "RL" and m != n]].mean()
                                              for n in names],
                            family=[fam(n) for n in names]))
    out.to_csv(os.path.join(outdir, "pool_scores.csv"), index=False)
    df.to_csv(os.path.join(outdir, "pool_episodes.csv"), index=False)
    print(out.sort_values("pool_score", ascending=False).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="정책 풀 상호 대전")
    ap.add_argument("--rl-ckpts", nargs="*", default=[])
    ap.add_argument("--bto-ckpts", nargs="*", default=[])
    ap.add_argument("--shield-ckpts", nargs="*", default=[])
    ap.add_argument("--pool-ckpts", nargs="*", default=[], help="수준 향상 실험 모델 (조건명 POOL-s<S>-b<G>)")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/pool")
    a = ap.parse_args()
    rl = sorted(sum([glob.glob(p) for p in a.rl_ckpts], []))
    bto = sorted(sum([glob.glob(p) for p in a.bto_ckpts], []))
    shd = sorted(sum([glob.glob(p) for p in a.shield_ckpts], []))
    pl = sorted(sum([glob.glob(p) for p in a.pool_ckpts], []))
    run(specs_from_args(rl, bto, shd, pool_ckpts=pl), a.n_seeds, a.workers, a.outdir)


if __name__ == "__main__":
    main()
