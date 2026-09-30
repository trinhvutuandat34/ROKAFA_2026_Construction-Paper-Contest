"""
대리 결정트리를 실제 정책으로 돌렸을 때의 성능 (R10).

    python -m dfxai.analysis.distill_perf --outdir results/distill --workers 4
    python -m dfxai.analysis.distill_perf --outdir results/distill --report-only

무엇을 확인하는가
-----------------
D\\* 는 대리 트리가 정책의 **지령을 재현한 정도**(R²)이고, 트리를 정책으로 돌렸을 때의
**성능**이 아니다. Saldiran 외(2024, 3쪽)는 정책 단순화(신경망을 결정트리로 줄이기)가 성능을
희생하고, 성능을 맞추려 깊이를 늘리면 트리가 커져 설명의 이점이 사라진다고 서술했다(시험한 것은 아님).
이 모듈은 우리 정책들로 그 진술을 직접 확인한다.

  1. 정책의 로그 (관측, 지령) 에 깊이 d 인 결정트리를 적합한다 (surrogate.py 와 같은 회귀, 같은 로그).
  2. 그 트리를 정책으로 삼아 **새 시드**에서 BT-v2·BT-v3 와 진영 교대로 싸운다(원본과 같은 시드·같은 절차).
  3. 원본 대비 점수 차 Δ(d) 를 시드 단위로 대응 비교하고, Δ 가 −0.05 이상이 되는 최소 깊이 d_perf 를
     충실도 기준 최소 깊이 D\\* 와 비교한다. 허용 한계 0.05 는 같은 예산으로 독립 학습한 정책 20개의
     점수 표준편차(0.051)에서 정했다.

평가 시드는 2,000,000 부터다. ES 학습이 뽑은 시드(0–10^6 미만)와 본실험의 시드(10000–10099) 어느
쪽과도 겹치지 않는다. 원본 정책도 같은 새 시드에서 다시 평가한다.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ..agents.base import Policy
from ..agents.bt import BTPolicy
from ..env import run_episode
from ..experiments.run_eval import build_condition, make_env
from ..xai.surrogate import _r2_per_output, EPS_VAR

SEED0 = 2_000_000
GRID = (1, 2, 3, 4, 6, 8, 12, 16)
MARGIN = 0.05                    # 성능 허용 한계(점수 단위)
CHASE = {"s1", "s10", "s12", "s14"}   # 초안 4.5절의 '추격 격추' 4개 (나머지 16개는 '치고 빠지기')


# ------------------------------------------------------------------ 트리 정책
class TreePolicy(Policy):
    """sklearn 회귀트리를 정책으로 쓰기 위한 경량 순회기.

    sklearn 의 predict 는 호출당 오버헤드가 커서 스텝마다 부르면 느리다. 트리 배열을 꺼내 파이썬
    루프로 순회한다. sklearn 은 입력을 float32 로 바꿔 임계값과 비교하므로 같은 방식으로 맞춘다
    (같은 결과인지는 `_selftest` 가 확인한다).
    """

    def __init__(self, tree: DecisionTreeRegressor, name: str = "TREE"):
        t = tree.tree_
        self.left = t.children_left.tolist()
        self.right = t.children_right.tolist()
        self.feat = t.feature.tolist()
        self.thr = t.threshold.tolist()
        self.val = [v[:, 0].tolist() for v in t.value]        # (n_nodes, n_out)
        self.n_leaves = int(tree.get_n_leaves())
        self.depth = int(tree.get_depth())
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).tolist()          # float32 값을 파이썬 float 로
        left, right, feat, thr = self.left, self.right, self.feat, self.thr
        n = 0
        while left[n] != -1:
            n = left[n] if x[feat[n]] <= thr[n] else right[n]
        return np.clip(np.asarray(self.val[n], dtype=np.float64), -1.0, 1.0)

    def complexity(self) -> dict:
        return {"leaves": self.n_leaves, "depth": self.depth}


def _selftest(seed: int = 0) -> None:
    """TreePolicy 가 sklearn predict 와 같은 값을 내는지(공학적 확인)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, 20))
    y = np.stack([np.sin(X[:, 0] * 2) + X[:, 1], np.tanh(X[:, 2] * X[:, 3]), (X[:, 4] > 0.2) * 1.0], axis=1)
    tr = DecisionTreeRegressor(max_depth=10, random_state=0).fit(X, y)
    pol = TreePolicy(tr)
    Xt = rng.normal(size=(500, 20))
    a = np.stack([pol.act(x) for x in Xt])
    b = np.clip(tr.predict(Xt), -1, 1)
    assert np.allclose(a, b, atol=1e-12), float(np.abs(a - b).max())


# ------------------------------------------------------------------ 정책 목록
def _spec_lookup(main_dir: str, hyb_dir: str) -> dict:
    specs = {}
    for d in (main_dir, hyb_dir):
        p = os.path.join(d, "manifest.json")
        if os.path.exists(p):
            for c in json.load(open(p, encoding="utf-8"))["conditions"]:
                specs.setdefault(c["name"], (d, c))
    return specs


def default_conditions() -> list[str]:
    """규칙 기반 대조군, 복제본, 학습 정책 20개, 혼합·감독형 각 일부, 잔차형·게이팅형."""
    names = ["BT-v1", "BT-v2", "BT-v3", "RL-s0-b0-a1.00"]
    names += [f"RL-s{s}-b300-a1.00" for s in range(20)]
    names += [f"BTO-s{s}-b300" for s in range(3)]
    names += [f"HYB-s{s}-b300-a0.50" for s in (0, 1, 11)]
    names += [f"SHD-s{s}-b300" for s in (0, 1, 11)]
    names += [f"RES-s{s}-b200" for s in range(3)] + [f"GATE-s{s}-b200" for s in range(3)]
    return names


def _alpha_of(spec: dict) -> float:
    return float(spec.get("alpha", 1.0 if spec["kind"] in ("rl", "shield", "residual", "gating") else 0.0))


# ------------------------------------------------------------------ 교전
def _play(pol: Policy, opp_v: int, seed: int, side: int, alpha: float, env, record: bool = False):
    opp = BTPolicy(version=opp_v)
    ob = ac = None
    if side == 0:
        out = run_episode(pol, opp, seed=seed, alpha=alpha, alpha_red=0.0, env=env, record=record)
        res, ob, ac = out if record else (out, None, None)
    else:
        res = run_episode(opp, pol, seed=seed, alpha=0.0, alpha_red=alpha, env=env, record=False)
    win = res.winner if side == 0 else -res.winner
    v = res.violations_blue if side == 0 else res.violations_red
    row = dict(opponent=opp_v, seed=seed, side=side,
               score=1.0 if win == 1 else (0.5 if win == 0 else 0.0),
               outcome=res.outcome, dur=res.duration,
               viol=v.get("deck", 0.0) + v.get("over_g", 0.0) + v.get("separation", 0.0),
               viol_deck=v.get("deck", 0.0), viol_over_g=v.get("over_g", 0.0))
    return row, ob, ac


def _weighted_r2(y: np.ndarray, p: np.ndarray) -> float:
    var = np.var(y, axis=0)
    w = var / var.sum() if var.sum() > EPS_VAR else np.ones(y.shape[1]) / y.shape[1]
    return float(np.dot(w, _r2_per_output(y, p)))


def _job(args) -> dict:
    """조건 하나: 원본을 새 시드에서 평가하고, 깊이별 트리를 적합해 같은 방식으로 평가한다."""
    name, main_dir, spec, dstar, n_seeds, seed0, depths_extra = args
    pol = build_condition(spec)
    alpha = _alpha_of(spec)
    env = make_env({})
    log = np.load(os.path.join(main_dir, f"traj_{name}.npz"))
    X, Y = log["obs"].astype(np.float64), log["act"].astype(np.float64)
    seeds = [seed0 + i for i in range(n_seeds)]

    games, fits = [], []
    fresh_obs, fresh_act = [], []
    for ov in (2, 3):
        for s in seeds:
            for side in (0, 1):
                row, ob, ac = _play(pol, ov, s, side, alpha, env, record=(side == 0))
                row.update(cond=name, depth=0)
                games.append(row)
                if ob is not None:
                    fresh_obs.append(ob); fresh_act.append(ac)
    Xf, Yf = np.concatenate(fresh_obs), np.concatenate(fresh_act)

    depths = sorted(set(GRID) | set(depths_extra))
    for d in depths:
        tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, Y)
        tp = TreePolicy(tree, name=f"{name}@{d}")
        fits.append(dict(cond=name, depth=d, n_leaves=tp.n_leaves,
                         r2_train=_weighted_r2(Y, tree.predict(X)),
                         r2_fresh=_weighted_r2(Yf, tree.predict(Xf))))
        for ov in (2, 3):
            for s in seeds:
                for side in (0, 1):
                    row, _, _ = _play(tp, ov, s, side, alpha, env)
                    row.update(cond=name, depth=d)
                    games.append(row)
    return dict(cond=name, dstar=dstar, games=games, fits=fits, n_train=len(X), n_fresh=len(Xf))


def run(outdir: str, main_dir: str, hyb_dir: str, workers: int, n_seeds: int, seed0: int,
        conds: list[str] | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    _selftest()
    specs = _spec_lookup(main_dir, hyb_dir)
    names = conds or default_conditions()
    dmap = {}
    for d in (main_dir, hyb_dir):
        p = os.path.join(d, "merged.csv")
        if os.path.exists(p):
            m = pd.read_csv(p)
            for c, v in zip(m["cond"], m["d_star"]):
                dmap.setdefault(c, float(v))
    jobs = []
    for n in names:
        d, spec = specs[n]
        ds = dmap.get(n, np.nan)
        extra = [min(16, int(ds))] if np.isfinite(ds) else []
        jobs.append((n, d, spec, ds, n_seeds, seed0, extra))
    print(f"조건 {len(jobs)}개, 시드 {n_seeds}개(시작 {seed0}), 깊이 {GRID} + D*", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_job, jobs, chunksize=1))
    else:
        res = [_job(j) for j in jobs]
    games = pd.DataFrame([g for r in res for g in r["games"]])
    fits = pd.DataFrame([f for r in res for f in r["fits"]])
    dstar = pd.DataFrame([dict(cond=r["cond"], d_star=r["dstar"], n_train=r["n_train"], n_fresh=r["n_fresh"]) for r in res])
    games.to_csv(os.path.join(outdir, "games.csv"), index=False)
    fits.to_csv(os.path.join(outdir, "fits.csv"), index=False)
    dstar.to_csv(os.path.join(outdir, "conds.csv"), index=False)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conditions=names, n_seeds=n_seeds, seed0=seed0, grid=list(GRID), margin=MARGIN,
                       opponents=[2, 3], elapsed_s=round(time.time() - t0, 1)), f, ensure_ascii=False, indent=2)
    print(f"완료 {time.time() - t0:.0f}s, 교전 {len(games)}회", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="대리 트리 정책 성능 확인")
    ap.add_argument("--outdir", default="results/distill")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--hyb-dir", default="results/main_hyb")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.outdir, a.main_dir, a.hyb_dir, a.workers, a.n_seeds, a.seed0, a.conds)
    from .distill_report import build
    print(build(a.outdir))
