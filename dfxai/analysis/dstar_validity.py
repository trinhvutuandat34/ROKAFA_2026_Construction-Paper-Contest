"""
D* 측정 타당성 점검 (R9). 기존 R1–R8 을 다시 돌리지 않고 저장된 결과만 읽는다.

    python -m dfxai.analysis.dstar_validity --outdir results/main \
        --out paper/results_dstar_validity.md --workers 4

각 절이 답하는 질문
-------------------
R9a R3 정오표     "공통 상태 D* 의 +0.19 는 방문 범위를 통제하면 사라지는가"
                  → robustness.py R3 표의 편상관 열은 D*(자기 로그) 기준이다. 공통 상태 D*
                    기준 편상관과 동등성 검정을 따로 계산한다.
R9b 임계값 민감도  "D* 는 0.95 라는 숫자에 걸려 있는 문턱 통계량이다"
                  → 충실도 곡선 F(d) 로 임계값을 바꿔 다시 세고, 문턱 없는 요약(F(4), 평균 F)과 비교.
R9c 절단의 의미    "D*=17 은 '16단계로도 설명 안 됨'인가"
                  → 절단 조건의 F(16) 분포와 F(d) 곡선의 단조성.
R9d 에피소드 분할  "대리트리의 시험 집합이 훈련 집합과 같은 교전에서 나오지 않았는가"
                  → surrogate.py 는 시간 스텝을 무작위로 섞어 7:3 으로 나누므로 이웃 스텝이 양쪽에
                    들어간다. 교전 단위로 나눈 D* 를 같은 실행 안에서 무작위 분할 D* 와 나란히 잰다.
"""
from __future__ import annotations
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

from .paper import md_table
from .robustness import _rho, _partial_spearman, bootstrap_equivalence
from ..config import FEATURE_NAMES
from ..xai.surrogate import _r2_per_output, EPS_VAR

T_IDX = FEATURE_NAMES.index("t_frac")
DEPTHS = tuple(range(1, 17))
FAMILIES = ("BT", "BTO", "RL", "Hybrid", "Shield", "PPO")


# ------------------------------------------------------------------ 도우미
def _first_hit(fids: np.ndarray, target: float, max_depth: int = 16) -> float:
    hit = np.where(np.asarray(fids) >= target)[0]
    return float(hit[0] + 1) if hit.size else float(max_depth + 1)


def _episode_ids(obs: np.ndarray) -> np.ndarray:
    """경과시간 비율이 줄어드는 지점이 새 교전의 시작이다(traj 는 교전을 이어 붙인 것)."""
    starts = np.r_[0, np.where(np.diff(obs[:, T_IDX]) < 0)[0] + 1]
    ids = np.zeros(len(obs), dtype=int)
    for k, s in enumerate(starts):
        ids[s:] = k
    return ids


def _fit_curve(Xtr, ytr, Xte, yte, seed: int) -> np.ndarray:
    var = np.var(yte, axis=0)
    w = var / var.sum() if var.sum() > EPS_VAR else np.ones(yte.shape[1]) / yte.shape[1]
    out = []
    for d in DEPTHS:
        tree = DecisionTreeRegressor(max_depth=d, random_state=seed).fit(Xtr, ytr)
        out.append(float(np.dot(w, _r2_per_output(yte, tree.predict(Xte)))))
    return np.asarray(out)


def _one(args) -> dict:
    """조건 하나: 같은 로그로 무작위 분할과 교전 단위 분할의 F(d) 곡선을 잰다."""
    path, cond, seed = args
    z = np.load(path)
    obs, act = z["obs"].astype(np.float64), z["act"].astype(np.float64)
    ids = _episode_ids(obs)
    Xtr, Xte, ytr, yte = train_test_split(obs, act, test_size=0.3, random_state=seed, shuffle=True)
    f_rand = _fit_curve(Xtr, ytr, Xte, yte, seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(obs, act, groups=ids))
    f_grp = _fit_curve(obs[tr], act[tr], obs[te], act[te], seed)
    return dict(cond=cond, n_episodes=int(ids.max() + 1), n_samples=len(obs),
                f_rand=f_rand, f_grp=f_grp)


# ------------------------------------------------------------------ R9a
def r3_erratum(m: pd.DataFrame) -> str:
    rows = []
    for nm, d in [("전체", m), ("학습·혼합만", m[m["family"].isin(["RL", "Hybrid"])])]:
        r_own, p_own, n = _rho(d["score"], d["d_star"])
        r_com, p_com, _ = _rho(d["score"], d["d_star_common"])
        pr_own, pp_own, _ = _partial_spearman(d["score"], d["d_star"], d["coverage_entropy"])
        pr_com, pp_com, _ = _partial_spearman(d["score"], d["d_star_common"], d["coverage_entropy"])
        eq = bootstrap_equivalence(d.reset_index(drop=True), xcol="score", ycol="d_star_common")
        rows.append({"부분집합": nm, "n": n,
                     "ρ(점수, D* 자기)": r_own, "편상관(D* 자기; 방문범위 통제)": pr_own, "p(편, 자기)": pp_own,
                     "ρ(점수, D* 공통)": r_com, "p(공통)": p_com,
                     "편상관(D* 공통; 방문범위 통제)": pr_com, "p(편, 공통)": pp_com,
                     "CI90 조건(공통)": f"[{eq['ci90_cond'][0]:.2f}, {eq['ci90_cond'][1]:.2f}]",
                     "CI90 시드클러스터(공통)": f"[{eq['ci90_cluster'][0]:.2f}, {eq['ci90_cluster'][1]:.2f}]",
                     "TOST ±0.2 (조건/클러스터)": f"{eq['tost_cond_0.2']}/{eq['tost_cluster_0.2']}",
                     "TOST ±0.3 (조건/클러스터)": f"{eq['tost_cond_0.3']}/{eq['tost_cluster_0.3']}"})
    fmt = {c: "{:.4f}" for c in ("p(편, 자기)", "p(공통)", "p(편, 공통)")}
    fam = m.groupby("family").agg(n=("cond", "size"), D_자기=("d_star", "median"),
                                  D_공통=("d_star_common", "median"),
                                  F4_공통=("f4_common", "mean")).reindex(FAMILIES).reset_index()
    out = "### R9a. R3 정오표: 공통 상태 D* 의 편상관\n\n"
    out += ("`results_ext.md` R3 표의 '편상관 ρ(점수, D* ; 방문범위 통제)' 열은 **D*(자기 로그)** 로 계산된 값이다"
            "(`robustness.py` 의 `_partial_spearman(score, d_star, coverage_entropy)`). 공통 상태 D* 의 +0.19 가 "
            "방문 범위를 통제하면 사라지는지는 계산된 적이 없었다. 아래가 공통 상태 D* 기준 값이다.\n\n")
    out += md_table(pd.DataFrame(rows), fmt) + "\n\n"
    out += "**계열별 D\\* (중앙값, 자기 로그 대 공통 상태)**\n\n" + md_table(fam, {"F4_공통": "{:.3f}"}) + "\n"
    return out


# ------------------------------------------------------------------ R9b, R9c
def _load_curves(outdir: str, conds) -> np.ndarray:
    z = np.load(os.path.join(outdir, "fidelity_curves.npz"), allow_pickle=True)
    return np.array([np.asarray(z[c])[1] for c in conds])


def threshold_and_truncation(outdir: str, m: pd.DataFrame) -> str:
    F = _load_curves(outdir, m["cond"])
    sub = m["family"].isin(["RL", "Hybrid"]).values
    rows = []
    for thr in (0.90, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98):
        ds = np.array([_first_hit(f, thr) for f in F])
        r, p, _ = _rho(m["score"], ds)
        r2, p2, _ = _rho(m["score"][sub], ds[sub])
        rows.append({"임계값": f"{thr:.2f}", "절단 비율": float((ds >= 17).mean()),
                     "ρ(점수, D*) 전체": r, "p": p, "ρ 학습·혼합만": r2, "p(학습·혼합)": p2})
    free = []
    for nm, v in (("F(4)", F[:, 3]), ("F(16)", F[:, 15]), ("F(1..16) 평균", F.mean(1))):
        r, p, _ = _rho(m["score"], -v)
        r2, p2, _ = _rho(m["score"][sub], -v[sub])
        free.append({"문턱 없는 요약": nm + " (부호 반전: 클수록 설명 어려움)", "ρ 전체": r, "p": p,
                     "ρ 학습·혼합만": r2, "p(학습·혼합)": p2})
    tr = m["d_star"].values >= 17
    f16 = F[:, 15]
    mono_bad = int((np.diff(F, axis=1) < -1e-9).any(axis=1).sum())
    q = pd.Series(f16[tr]).quantile([0, .1, .25, .5, .75, .9, 1])
    near = (~tr) & (F[np.arange(len(F)), np.clip(m["d_star"].values.astype(int) - 1, 0, 15)] < 0.955)
    out = "### R9b. 임계값 민감도와 문턱 없는 충실도 요약\n\n"
    out += ("D\\* 는 충실도 곡선이 0.95 를 처음 넘는 깊이다. 곡선은 깊이 6–16 에서 0.93–0.95 부근에 "
            "납작하게 놓이는 조건이 많아, 문턱을 조금 옮기면 D\\* 가 크게 바뀐다. 아래는 저장된 곡선으로 "
            "임계값만 바꿔 다시 센 결과다(대리트리는 다시 적합하지 않음).\n\n")
    tf = {"절단 비율": "{:.2f}", "p": "{:.4f}", "p(학습·혼합)": "{:.4f}"}
    out += md_table(pd.DataFrame(rows), tf) + "\n\n"
    out += md_table(pd.DataFrame(free), {"p": "{:.4f}", "p(학습·혼합)": "{:.4f}"}) + "\n\n"
    out += "### R9c. 절단(D\\*=17)의 의미\n\n"
    out += (f"* 절단 조건 {int(tr.sum())}개(전체 {len(m)}개의 {tr.mean():.0%}). 절단 조건의 F(16) 분위수"
            f"(0/10/25/50/75/90/100%): " + " / ".join(f"{v:.3f}" for v in q.values) + ".\n")
    out += (f"* 절단 조건 중 F(16) ≥ 0.94 는 {int((f16[tr] >= 0.94).sum())}개, ≥ 0.93 은 "
            f"{int((f16[tr] >= 0.93).sum())}개, ≥ 0.90 은 {int((f16[tr] >= 0.90).sum())}개. "
            "'16단계로도 설명되지 않는다'는 R² 0.95 미달이라는 뜻이며 대부분 0.90–0.94 다.\n")
    out += (f"* F(d) 가 깊이에 대해 단조가 아닌 조건 {mono_bad}개(전체 {len(m)}개). 시험 집합의 "
            "표본 잡음이 곡선에 실려 있어 D\\* 는 0.95 근방에서 잡음에 민감하다.\n")
    out += (f"* 절단이 아닌 조건 {int((~tr).sum())}개 중 D\\* 에서의 F 가 0.955 미만(문턱을 겨우 넘음)인 "
            f"조건은 {int(near.sum())}개.\n")
    fam = pd.DataFrame({"family": m["family"], "F(4)": F[:, 3], "F(16)": F[:, 15], "절단": tr})
    fam = fam.groupby("family").agg(n=("F(4)", "size"), F4=("F(4)", "mean"), F16=("F(16)", "mean"),
                                    절단비율=("절단", "mean")).reindex(FAMILIES).reset_index()
    out += "\n**계열별 F(4), F(16), 절단 비율**\n\n" + md_table(fam) + "\n"
    return out


# ------------------------------------------------------------------ R9d
def grouped_split(outdir: str, m: pd.DataFrame, workers: int, seed: int = 0) -> tuple[str, pd.DataFrame]:
    jobs = [(os.path.join(outdir, f"traj_{c}.npz"), c, seed) for c in m["cond"]]
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(_one, jobs, chunksize=4)
    else:
        res = [_one(j) for j in jobs]
    g = pd.DataFrame(res)
    g["d_rand"] = [_first_hit(f, 0.95) for f in g["f_rand"]]
    g["d_grp"] = [_first_hit(f, 0.95) for f in g["f_grp"]]
    g["d_grp_090"] = [_first_hit(f, 0.90) for f in g["f_grp"]]
    g["f4_rand"] = [f[3] for f in g["f_rand"]]
    g["f4_grp"] = [f[3] for f in g["f_grp"]]
    g["f16_rand"] = [f[15] for f in g["f_rand"]]
    g["f16_grp"] = [f[15] for f in g["f_grp"]]
    z = m.merge(g.drop(columns=["f_rand", "f_grp"]), on="cond")
    z.to_csv(os.path.join(outdir, "dstar_grouped.csv"), index=False)

    sub = z["family"].isin(["RL", "Hybrid"])
    rows = []
    for nm, col in (("무작위 스텝 분할 (기존 방식, 같은 실행)", "d_rand"),
                    ("교전 단위 분할 D*(0.95)", "d_grp"),
                    ("교전 단위 분할 D*(0.90)", "d_grp_090")):
        r, p, n = _rho(z["score"], z[col])
        r2, p2, _ = _rho(z["score"][sub], z[col][sub])
        eq = bootstrap_equivalence(z, xcol="score", ycol=col)
        rows.append({"측정": nm, "절단 비율": float((z[col] >= 17).mean()),
                     "ρ(점수, D*) 전체": r, "p": p, "ρ 학습·혼합만": r2, "p(학습·혼합)": p2,
                     "CI90 시드클러스터": f"[{eq['ci90_cluster'][0]:.2f}, {eq['ci90_cluster'][1]:.2f}]",
                     "TOST ±0.2 (조건/클러스터)": f"{eq['tost_cond_0.2']}/{eq['tost_cluster_0.2']}"})
    r_same, _, _ = _rho(z["d_rand"], z["d_grp"])
    r_stored, _, _ = _rho(z["d_star"], z["d_rand"])
    fam = z.groupby("family").agg(n=("cond", "size"), D_무작위=("d_rand", "median"), D_교전단위=("d_grp", "median"),
                                  F4_무작위=("f4_rand", "mean"), F4_교전단위=("f4_grp", "mean"),
                                  F16_무작위=("f16_rand", "mean"), F16_교전단위=("f16_grp", "mean")
                                  ).reindex(FAMILIES).reset_index()
    out = "### R9d. 교전 단위 분할 D*\n\n"
    out += ("`surrogate.py` 는 (관측, 지령) 스텝을 무작위로 섞어 7:3 으로 나누므로 0.2 초 간격의 이웃 스텝이 훈련·시험 "
            "양쪽에 들어간다. 여기서는 저장된 궤적(조건당 교전 60개, 경과시간 비율이 되감기는 지점으로 복원)을 교전 단위로 "
            "나눠 30% 교전을 시험에 쓴다. 같은 실행 안에서 두 방식을 함께 계산했으므로 라이브러리 버전 차이는 "
            f"비교에 섞이지 않는다(분할 시드 {seed} 하나, 조건 {len(z)}개).\n\n")
    out += md_table(pd.DataFrame(rows), {"절단 비율": "{:.2f}", "p": "{:.4f}", "p(학습·혼합)": "{:.4f}"}) + "\n\n"
    out += (f"저장된 D\\*(merged.csv)와 이 실행의 무작위 분할 D\\* 의 순위상관 {r_stored:.3f}; "
            f"무작위 분할 D\\* 와 교전 단위 D\\* 의 순위상관 {r_same:.3f}.\n\n")
    out += "**계열별 (D\\* 는 중앙값, 나머지는 평균)**\n\n" + md_table(fam) + "\n"
    return out, z


def build(outdir: str, out_path: str, workers: int) -> str:
    m = pd.read_csv(os.path.join(outdir, "merged_common.csv"))
    parts = ["# D* 측정 타당성 점검 (자동 생성, R9)\n",
             f"원본: `{outdir}` · 조건 {len(m)}개. 생성: `python -m dfxai.analysis.dstar_validity`. "
             "기존 R1–R8(`results_ext.md`)은 다시 계산하지 않았다.\n",
             r3_erratum(m), threshold_and_truncation(outdir, m)]
    s, _ = grouped_split(outdir, m, workers)
    parts.append(s)
    doc = "\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="D* 측정 타당성 점검")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--out", default="paper/results_dstar_validity.md")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    print(build(a.outdir, a.out, a.workers))


if __name__ == "__main__":
    main()
