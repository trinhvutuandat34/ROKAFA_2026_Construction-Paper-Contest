"""
전술 영역별 설명 비용 (R11).

    python -m dfxai.analysis.tactical_regions --outdir results/tactical --workers 4
    python -m dfxai.analysis.tactical_regions --outdir results/tactical --report-only

질문
----
D\\* 는 정책 전체를 하나의 깊이로 요약한다. 학습 정책이 규칙 모델과 어디에서 갈라지는지, 곧 **어느
전술 상황에서 행동이 복잡해지는지**는 보이지 않는다. Saldiran 외(2024, 21쪽 식 (39)–(42))의
전술 영역으로 상태를 나눠 본다.

  정면 조우 ATA ≤ 45° 이고 AA ≥ 135°
  공격      ATA ≤ 90° 이면서 정면 조우가 아님
  방어      ATA > 90° 이고 AA > 90°
  중립      ATA > 90° 이고 AA ≤ 90°

네 영역은 겹치지 않고 전체를 덮는다. 우리 관측의 ATA(`ata_total`)와 AA(`aa`)는 Saldiran 의 정의와
같다(AA=0 이면 내가 상대의 정확한 6시). 영역은 부호 없는 각으로 나눴고 Saldiran 의 4사분면 확장은 쓰지 않았다.

측정 (조건마다, 영역마다)
------------------------
 1. 자기 로그 대리트리: 정책이 그 영역에서 남긴 (관측, 지령) 에 깊이 1–16 결정트리를 적합하고
    교전 단위로 나눈 시험 교전에서 충실도 F(d) 를 잰다(R9d 와 같은 이유로 스텝 무작위 분할은 쓰지 않는다).
    D\\* 는 문턱 통계량이라 불안정하므로(R9b) 깊이 4·8 의 충실도 F(4), F(8) 을 주 지표로 하고
    D\\*(0.90) 을 보조로 쓴다.
 2. 공통 상태 대리트리: 모든 조건의 로그에서 균일 추출한 40,000개 관측(`common_pool.npy`)에 정책을
    질의해 같은 방식으로 잰다. 방문 여부와 무관하게 "그 영역에서 정책 함수가 얼마나 복잡한가"를 본다.
    (관측이 독립 추출이라 스텝 무작위 분할을 쓴다.)
 3. BT-v2 대비 지령 편차: 같은 관측에서 BT-v2 가 낼 지령과의 차이(뱅크는 원형 차이). 학습 정책은 BT-v2
    복제본에서 출발했으므로 "규칙에서 얼마나 벗어났는가"의 직접 척도다.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

from ..agents.bt import BTPolicy
from ..config import FEATURE_NAMES
from ..experiments.run_eval import build_condition
from ..xai.surrogate import _r2_per_output, EPS_VAR
from .dstar_validity import _episode_ids, _first_hit, DEPTHS

REGIONS = ("head_on", "offensive", "defensive", "neutral")
REGION_KO = {"head_on": "정면 조우", "offensive": "공격", "defensive": "방어", "neutral": "중립"}
I_ATA, I_AA, I_ALPHA = FEATURE_NAMES.index("ata_total"), FEATURE_NAMES.index("aa"), FEATURE_NAMES.index("alpha")
MIN_N_OWN, MIN_TEST_OWN, MIN_N_POOL = 1000, 250, 800
N_SPLITS = 5          # 자기 로그 교전 단위 분할의 난수 시드 수(영역 안 표본이 적어 한 번의 분할은 흔들린다)
CHASE = {"s1", "s10", "s12", "s14"}


def region_codes(obs: np.ndarray) -> np.ndarray:
    """0 정면 조우, 1 공격, 2 방어, 3 중립. 각은 도 단위, 부호 없음(관측이 이미 0–π)."""
    ata = obs[:, I_ATA] * 180.0
    aa = obs[:, I_AA] * 180.0
    head_on = (ata <= 45.0) & (aa >= 135.0)
    off = (ata <= 90.0) & ~head_on
    dfn = (ata > 90.0) & (aa > 90.0)
    neu = (ata > 90.0) & (aa <= 90.0)
    code = np.full(len(obs), -1, dtype=int)
    code[head_on], code[off], code[dfn], code[neu] = 0, 1, 2, 3
    assert (code >= 0).all() and (head_on.astype(int) + off + dfn + neu == 1).all()
    return code


def _weighted_r2(y, p) -> float:
    var = np.var(y, axis=0)
    w = var / var.sum() if var.sum() > EPS_VAR else np.ones(y.shape[1]) / y.shape[1]
    return float(np.dot(w, _r2_per_output(y, p)))


def _curve(Xtr, ytr, Xte, yte) -> np.ndarray:
    out = []
    for d in DEPTHS:
        tr = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xtr, ytr)
        out.append(_weighted_r2(yte, tr.predict(Xte)))
    return np.asarray(out)


def _summ(f: np.ndarray) -> dict:
    return dict(F4=f[3], F8=f[7], F16=f[15], D90=_first_hit(f, 0.90), D95=_first_hit(f, 0.95))


def deviation(act_pol: np.ndarray, act_ref: np.ndarray) -> dict:
    """정책과 BT-v2 지령의 편차. 뱅크는 원형 차이[도], 하중배수·스로틀은 정규화 지령 차이(0–2)."""
    d_bank = np.minimum(np.abs(act_pol[:, 0] - act_ref[:, 0]), 2.0 - np.abs(act_pol[:, 0] - act_ref[:, 0])) * 180.0
    d_load = np.abs(act_pol[:, 1] - act_ref[:, 1])
    d_thr = np.abs(act_pol[:, 2] - act_ref[:, 2])
    return dict(dev_bank_deg=float(d_bank.mean()), dev_load=float(d_load.mean()), dev_thr=float(d_thr.mean()),
                bank_flip=float((d_bank > 90.0).mean()))


# ------------------------------------------------------------------ 조건 하나
def _job(args) -> dict:
    name, log_dir, spec, pool_path, seed = args
    bt2 = BTPolicy(version=2)
    z = np.load(os.path.join(log_dir, f"traj_{name}.npz"))
    X, Y = z["obs"].astype(np.float64), z["act"].astype(np.float64)
    reg = region_codes(X)
    ep = _episode_ids(X)
    alpha = float(spec.get("alpha", 1.0 if spec["kind"] in ("rl", "shield", "residual", "gating") else 0.0))

    rows, dev_rows = [], []
    share = {REGIONS[k]: float((reg == k).mean()) for k in range(4)}

    # ---- 1. 자기 로그: 교전 단위 분할 (주, 분할 시드 N_SPLITS 개) 과 스텝 무작위 분할 (보조, 1개)
    for sd in range(N_SPLITS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed + sd)
        tr_idx, te_idx = next(gss.split(X, Y, groups=ep))
        is_tr = np.zeros(len(X), bool); is_tr[tr_idx] = True
        for k, rn in enumerate(REGIONS):
            m = reg == k
            n_ep = int(len(np.unique(ep[m]))) if m.any() else 0
            row = dict(cond=name, source="own", split="grouped", split_seed=sd, region=rn, n=int(m.sum()), n_ep=n_ep)
            mtr, mte = m & is_tr, m & ~is_tr
            if m.sum() >= MIN_N_OWN and mte.sum() >= MIN_TEST_OWN and mtr.sum() >= MIN_TEST_OWN:
                row.update(_summ(_curve(X[mtr], Y[mtr], X[mte], Y[mte])))
            rows.append(row)
    for k, rn in enumerate(REGIONS):
        m = reg == k
        n_ep = int(len(np.unique(ep[m]))) if m.any() else 0
        row = dict(cond=name, source="own", split="random", split_seed=0, region=rn, n=int(m.sum()), n_ep=n_ep)
        if m.sum() >= MIN_N_OWN:
            Xa, Xb, ya, yb = train_test_split(X[m], Y[m], test_size=0.3, random_state=seed, shuffle=True)
            row.update(_summ(_curve(Xa, ya, Xb, yb)))
        rows.append(row)

    # ---- 3a. BT-v2 대비 지령 편차: 자기 로그 상태
    ref = np.asarray([bt2.act(o) for o in X], dtype=np.float64)
    for k, rn in enumerate(REGIONS):
        m = reg == k
        if m.sum() >= 50:
            dev_rows.append(dict(cond=name, source="own", region=rn, n=int(m.sum()), **deviation(Y[m], ref[m])))

    # ---- 2, 3b. 공통 상태
    pool = np.load(pool_path).astype(np.float64)
    pool[:, I_ALPHA] = alpha
    pol = build_condition(spec)
    Ap = np.asarray([pol.act(o) for o in pool], dtype=np.float64)
    Rp = np.asarray([bt2.act(o) for o in pool], dtype=np.float64)
    rp = region_codes(pool)
    for k, rn in enumerate(REGIONS):
        m = rp == k
        row = dict(cond=name, source="pool", split="random", split_seed=0, region=rn, n=int(m.sum()), n_ep=0)
        if m.sum() >= MIN_N_POOL:
            Xa, Xb, ya, yb = train_test_split(pool[m], Ap[m], test_size=0.3, random_state=seed, shuffle=True)
            row.update(_summ(_curve(Xa, ya, Xb, yb)))
            dev_rows.append(dict(cond=name, source="pool", region=rn, n=int(m.sum()), **deviation(Ap[m], Rp[m])))
        rows.append(row)
    return dict(cond=name, rows=rows, dev=dev_rows, share=share)


def default_conditions() -> list[str]:
    names = ["BT-v1", "BT-v2", "BT-v3", "RL-s0-b0-a1.00"]
    names += [f"BTO-s{s}-b300" for s in range(3)]
    names += [f"RL-s{s}-b300-a1.00" for s in range(20)]
    names += [f"HYB-s{s}-b300-a0.50" for s in range(20)]
    names += [f"SHD-s{s}-b300" for s in range(20)]
    names += [f"RES-s{s}-b200" for s in range(3)] + [f"GATE-s{s}-b200" for s in range(3)]
    return names


def run(outdir: str, main_dir: str, hyb_dir: str, workers: int, conds: list[str] | None = None, seed: int = 0):
    os.makedirs(outdir, exist_ok=True)
    specs = {}
    for d in (main_dir, hyb_dir):
        p = os.path.join(d, "manifest.json")
        if os.path.exists(p):
            for c in json.load(open(p, encoding="utf-8"))["conditions"]:
                specs.setdefault(c["name"], (d, c))
    pool_path = os.path.join(main_dir, "common_pool.npy")
    names = conds or default_conditions()
    jobs = [(n, specs[n][0], specs[n][1], pool_path, seed) for n in names]
    print(f"조건 {len(jobs)}개", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_job, jobs, chunksize=1))
    else:
        res = [_job(j) for j in jobs]
    pd.DataFrame([r for x in res for r in x["rows"]]).to_csv(os.path.join(outdir, "regional.csv"), index=False)
    pd.DataFrame([r for x in res for r in x["dev"]]).to_csv(os.path.join(outdir, "deviation.csv"), index=False)
    pd.DataFrame([dict(cond=x["cond"], **x["share"]) for x in res]).to_csv(os.path.join(outdir, "share.csv"), index=False)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conditions=names, regions=list(REGIONS), min_n_own=MIN_N_OWN, min_test_own=MIN_TEST_OWN,
                       min_n_pool=MIN_N_POOL, seed=seed, elapsed_s=round(time.time() - t0, 1)), f, ensure_ascii=False, indent=2)
    print(f"완료 {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ 4. 오차 집중도
def _error_job(args) -> list[dict]:
    """전역 트리(깊이 4·8)를 교전 단위 분할로 적합했을 때, 시험 표본의 제곱오차가 어느 영역에서 나오는가.

    분산 가중 R² 의 미설명 부분은 (모든 채널의 제곱오차 합) / (N × 채널 분산 합) 이므로 영역 r 이 미설명
    분산에 기여하는 몫은 그 영역의 제곱오차 합 / 전체 제곱오차 합이다. 시간 비율과 비교하면 어느 영역이
    행동 재현을 어렵게 만드는지 드러난다.
    """
    name, log_dir, seed = args
    z = np.load(os.path.join(log_dir, f"traj_{name}.npz"))
    X, Y = z["obs"].astype(np.float64), z["act"].astype(np.float64)
    reg, ep = region_codes(X), _episode_ids(X)
    rows = []
    for sd in range(N_SPLITS):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed + sd).split(X, Y, groups=ep))
        for d in (4, 8):
            tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X[tr], Y[tr])
            se = ((Y[te] - tree.predict(X[te])) ** 2).sum(axis=1)
            tot = float(se.sum())
            var = float(np.var(Y[te], axis=0).sum())
            unexplained = tot / (len(te) * var) if var > EPS_VAR else np.nan
            for k, rn in enumerate(REGIONS):
                m = reg[te] == k
                rows.append(dict(cond=name, depth=d, split_seed=sd, region=rn, time_share=float(m.mean()),
                                 err_share=float(se[m].sum() / tot) if tot > 0 else np.nan, unexplained=unexplained))
    return rows


def run_error_share(outdir: str, main_dir: str, hyb_dir: str, workers: int, conds: list[str] | None = None,
                    seed: int = 0) -> None:
    specs = {}
    for d in (main_dir, hyb_dir):
        p = os.path.join(d, "manifest.json")
        if os.path.exists(p):
            for c in json.load(open(p, encoding="utf-8"))["conditions"]:
                specs.setdefault(c["name"], (d, c))
    names = conds or default_conditions()
    jobs = [(n, specs[n][0], seed) for n in names]
    if workers > 1:
        with Pool(workers) as p:
            res = list(p.imap_unordered(_error_job, jobs, chunksize=1))
    else:
        res = [_error_job(j) for j in jobs]
    pd.DataFrame([r for x in res for r in x]).to_csv(os.path.join(outdir, "error_share.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="전술 영역별 설명 비용")
    ap.add_argument("--outdir", default="results/tactical")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--hyb-dir", default="results/main_hyb")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.outdir, a.main_dir, a.hyb_dir, a.workers, a.conds)
        run_error_share(a.outdir, a.main_dir, a.hyb_dir, a.workers, a.conds)
    from .tactical_report import build
    print(build(a.outdir))
