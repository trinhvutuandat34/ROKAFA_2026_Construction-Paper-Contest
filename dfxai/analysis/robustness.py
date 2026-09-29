"""
박사 수준 심사를 겨냥한 강건성 분석 묶음.

    python -m dfxai.analysis.robustness --outdir results/main --out paper/results_ext.md

각 절이 답하는 반론
-------------------
R1 신뢰도    "D* 가 너무 시끄러워 상관이 0 으로 감쇠된 것 아닌가"
             → 반분(split-half) 신뢰도와 감쇠 보정 상관.
R2 동등성    "유의하지 않다 ≠ 없다"
             → ρ 의 부트스트랩 신뢰구간(조건 재추출·학습 시드 클러스터 재추출)과
               동등성 검정(TOST, ±0.2 / ±0.3).
R3 공통 상태  "D* 는 상황의 다양성을 재는 것 아닌가"
             → 공통 관측 집합에서 잰 D*_common, 상태 방문 범위 공변량, 편상관.
R4 표본 독립  "67개 조건은 시드 5개에서 파생된 것이라 독립이 아니다"
             → 학습 시드 임의효과 혼합모형, 시드 단위 집계 상관.
R5 전술 군집  "'치고 빠지기'/'추격'은 사후 명명이다"
             → 종료 사유·WEZ 특징의 가우시안 혼합 군집(BIC), 군집별 D*.
R6 예산 세밀  "산 모양은 표준편차 안에 있다"  (25세대 간격 체크포인트 평가가 있을 때)
R7 민감도    "결과는 120초·HP 비교 규칙의 산물이다" (results/sens_* 가 있을 때)
R8 정책 풀   "성능 = BT-v2 를 이기는 능력일 뿐" (results/pool 이 있을 때)
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from .figures import load_merged, parse_cond, read_json
from .paper import md_table, _spearman
from ..xai.surrogate import surrogate_fidelity


# ------------------------------------------------------------------ 도우미
def _rho(x, y):
    r, p, n = _spearman(x, y)
    return r, p, n


def _partial_spearman(x, y, z):
    """z 를 통제한 x–y 편상관 (순위 변환 후 잔차 상관)."""
    d = pd.DataFrame(dict(x=x, y=y, z=z)).dropna().rank()
    if len(d) < 6:
        return np.nan, np.nan, len(d)
    rx = d["x"] - np.polyval(np.polyfit(d["z"], d["x"], 1), d["z"])
    ry = d["y"] - np.polyval(np.polyfit(d["z"], d["y"], 1), d["z"])
    r = stats.pearsonr(rx, ry)
    return float(r.statistic), float(r.pvalue), len(d)


# ------------------------------------------------------------------ R1 신뢰도
def split_half(outdir: str, merged: pd.DataFrame, target: float = 0.95) -> dict:
    """반분 신뢰도. D*: 조건별 로그를 전반/후반(≈교전 1~15 / 16~30)으로 나눠 각각 계산.
    점수: 홀수/짝수 시드로 나눠 각각 평균. 조건 간 Spearman 이 신뢰도 r_xx."""
    rows = []
    for _, m in merged.iterrows():
        p = os.path.join(outdir, f"traj_{m['cond']}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p); ob, ac = d["obs"], d["act"]
        h = len(ob) // 2
        r1 = surrogate_fidelity(ob[:h], ac[:h], target=target)
        r2 = surrogate_fidelity(ob[h:], ac[h:], target=target)
        rows.append(dict(cond=m["cond"], d1=r1.min_depth_at_target, d2=r2.min_depth_at_target,
                         f1=r1.fidelity_at_fixed_depth, f2=r2.fidelity_at_fixed_depth))
    dd = pd.DataFrame(rows)
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    ep["half"] = ep["seed"] % 2
    sc = ep.groupby(["cond", "half"])["score"].mean().unstack()
    sc.columns = ["s_even", "s_odd"]
    dd = dd.merge(sc.reset_index(), on="cond")
    r_d, p_d, n = _rho(dd["d1"], dd["d2"])
    r_f, _, _ = _rho(dd["f1"], dd["f2"])
    r_s, _, _ = _rho(dd["s_even"], dd["s_odd"])
    # Spearman-Brown: 반분 → 전체 길이 신뢰도
    sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan
    full = dd.merge(merged[["cond", "score", "d_star", "fidelity_at_4"]], on="cond")
    rho_obs, p_obs, _ = _rho(full["score"], full["d_star"])
    rho_f_obs, _, _ = _rho(full["score"], full["fidelity_at_4"])
    rel_d, rel_s, rel_f = sb(r_d), sb(r_s), sb(r_f)
    dis = rho_obs / np.sqrt(rel_d * rel_s) if rel_d > 0 and rel_s > 0 else np.nan
    dis_f = rho_f_obs / np.sqrt(rel_f * rel_s) if rel_f > 0 and rel_s > 0 else np.nan
    # 절단(17) 비율
    cens = float((merged["d_star"] >= 17).mean())
    dd.to_csv(os.path.join(outdir, "reliability_split.csv"), index=False)
    return dict(n=n, r_half_dstar=r_d, r_half_f4=r_f, r_half_score=r_s,
                rel_dstar=rel_d, rel_f4=rel_f, rel_score=rel_s,
                rho_obs=rho_obs, rho_disattenuated=dis,
                rho_f4_obs=rho_f_obs, rho_f4_disattenuated=dis_f,
                censored_frac=cens,
                mean_abs_diff_dstar=float((dd["d1"] - dd["d2"]).abs().mean()))


# ------------------------------------------------------------------ R2 동등성
def bootstrap_equivalence(merged: pd.DataFrame, n_boot: int = 5000, seed: int = 0,
                          xcol="score", ycol="d_star") -> dict:
    rng = np.random.default_rng(seed)
    d = merged.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    x, y = d[xcol].values, d[ycol].values
    obs = stats.spearmanr(x, y).statistic
    # (a) 조건 재추출
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        if np.std(y[i]) < 1e-9 or np.std(x[i]) < 1e-9:
            continue
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    bs = np.asarray(bs)
    # (b) 학습 시드 클러스터 재추출 (BT 는 항상 포함)
    seeds = sorted(set(d.loc[d["train_seed"] >= 0, "train_seed"]))
    cb = []
    if len(seeds) >= 3:
        base = d[d["train_seed"] < 0]
        groups = {s: d[d["train_seed"] == s] for s in seeds}
        for _ in range(n_boot):
            pick = rng.choice(seeds, len(seeds), replace=True)
            dd = pd.concat([base] + [groups[s] for s in pick])
            if dd[ycol].std() < 1e-9:
                continue
            cb.append(stats.spearmanr(dd[xcol], dd[ycol]).statistic)
    cb = np.asarray(cb)

    def ci(a, q):
        return (float(np.percentile(a, 100 * (1 - q) / 2)), float(np.percentile(a, 100 * (1 + q) / 2))) if len(a) else (np.nan, np.nan)
    out = dict(n=len(d), rho=float(obs), n_seeds=len(seeds),
               ci95_cond=ci(bs, 0.95), ci90_cond=ci(bs, 0.90),
               ci95_cluster=ci(cb, 0.95), ci90_cluster=ci(cb, 0.90))
    for bound in (0.2, 0.3):
        lo, hi = out["ci90_cond"]
        out[f"tost_cond_{bound}"] = bool(lo > -bound and hi < bound)
        lo, hi = out["ci90_cluster"]
        out[f"tost_cluster_{bound}"] = bool(lo > -bound and hi < bound) if len(cb) else None
    return out


# ------------------------------------------------------------------ R3 공통 상태
def common_state_section(outdir: str, merged: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    p = os.path.join(outdir, "xai_common.csv")
    if not os.path.exists(p):
        return "### R3. 공통 상태 D*\n\n(xai_common.csv 없음 — `python -m dfxai.xai.common_state` 를 먼저 실행)\n", None
    xc = pd.read_csv(p)
    m = merged.merge(xc, on="cond", how="left")
    rows = []
    for nm, dsub in [("전체", m), ("학습·혼합만", m[m["family"].isin(["RL", "Hybrid"])])]:
        r1, p1, n = _rho(dsub["score"], dsub["d_star"])
        r2, p2, _ = _rho(dsub["score"], dsub["d_star_common"])
        r3, p3, _ = _rho(dsub["d_star"], dsub["d_star_common"])
        r4, p4, _ = _rho(dsub["d_star"], dsub["coverage_entropy"])
        r5, p5, _ = _rho(dsub["score"], dsub["coverage_entropy"])
        r6, p6, n6 = _partial_spearman(dsub["score"], dsub["d_star"], dsub["coverage_entropy"])
        r7, p7, _ = _rho(dsub["score"], dsub["f4_common"])
        rows.append({"부분집합": nm, "n": n,
                     "ρ(점수, D* 자기로그)": r1, "p1": p1,
                     "ρ(점수, D* 공통)": r2, "p2": p2,
                     "ρ(점수, F4 공통)": r7, "p7": p7,
                     "ρ(D* 자기, D* 공통)": r3, "p3": p3,
                     "ρ(D*, 방문범위)": r4, "p4": p4,
                     "ρ(점수, 방문범위)": r5, "p5": p5,
                     "편상관 ρ(점수, D* 자기로그 ; 방문범위 통제)": r6, "p6": p6})
    out = "### R3. 공통 상태 집합에서 잰 D* 와 상태 방문 범위\n\n"
    out += ("모든 조건의 로그를 합쳐 균일 추출한 관측 40,000개를 각 정책에 똑같이 질의해 "
            "대리트리를 적합한 D*_common 과, 각 정책 자기 로그의 상태 방문 범위(거리·총각·"
            "애스펙트각 3차원 히스토그램 정규화 엔트로피)를 함께 봅니다. D* 가 상황의 다양성이 "
            "아니라 규칙의 복잡성을 재는지 확인하는 표입니다.\n\n")
    out += md_table(pd.DataFrame(rows), {c: "{:.4f}" for c in ("p1", "p2", "p3", "p4", "p5", "p6", "p7")}) + "\n\n"
    fam = m.groupby("family").agg(n=("cond", "size"), D_own=("d_star", "mean"),
                                  D_common=("d_star_common", "mean"),
                                  F4_common=("f4_common", "mean"),
                                  방문범위=("coverage_entropy", "mean")).reset_index()
    out += "**계열별 평균**\n\n" + md_table(fam) + "\n"
    m.to_csv(os.path.join(outdir, "merged_common.csv"), index=False)
    return out, m


# ------------------------------------------------------------------ R4 혼합모형
def mixed_model(merged: pd.DataFrame) -> dict:
    d = merged[merged["family"].isin(["RL", "Hybrid"])].dropna(subset=["d_star", "score"]).copy()
    out = dict(n=len(d), n_seeds=int(d["train_seed"].nunique()))
    try:
        import statsmodels.formula.api as smf
        d["d_z"] = (d["d_star"] - d["d_star"].mean()) / d["d_star"].std()
        d["b_z"] = (d["budget"] - d["budget"].mean()) / d["budget"].std()
        md = smf.mixedlm("score ~ d_z + b_z + alpha", d, groups=d["train_seed"]).fit(reml=False)
        out.update(coef_dstar=float(md.params["d_z"]), p_dstar=float(md.pvalues["d_z"]),
                   coef_budget=float(md.params["b_z"]), p_budget=float(md.pvalues["b_z"]),
                   coef_alpha=float(md.params["alpha"]), p_alpha=float(md.pvalues["alpha"]),
                   seed_var=float(md.cov_re.iloc[0, 0]), resid_var=float(md.scale))
        # D* 단독 모형
        md2 = smf.mixedlm("score ~ d_z", d, groups=d["train_seed"]).fit(reml=False)
        out.update(coef_dstar_only=float(md2.params["d_z"]), p_dstar_only=float(md2.pvalues["d_z"]))
    except Exception as e:                       # pragma: no cover
        out["error"] = str(e)
    # 시드 단위 집계 (독립 표본 = 학습 시드)
    g = d[d["family"] == "RL"].groupby("train_seed").agg(score=("score", "mean"), d_star=("d_star", "mean"))
    r, p, n = _rho(g["score"], g["d_star"])
    out.update(seed_level_rho=r, seed_level_p=p, seed_level_n=n)
    g3 = d[(d["family"] == "RL") & (d["budget"] == d["budget"].max())]
    r, p, n = _rho(g3["score"], g3["d_star"])
    out.update(final_budget_rho=r, final_budget_p=p, final_budget_n=n)
    return out


# ------------------------------------------------------------------ R5 전술 군집
def tactic_clusters(outdir: str, merged: pd.DataFrame, seed: int = 0) -> tuple[str, pd.DataFrame]:
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    ep["kill_win"] = ((ep["outcome"] == "gun_kill") & (ep["win"] == 1)).astype(float)
    ep["timeout"] = (ep["outcome"] == "timeout").astype(float)
    f = ep.groupby("cond").agg(kill_win=("kill_win", "mean"), timeout=("timeout", "mean"),
                               wez=("wez_time", "mean"), duration=("duration", "mean"),
                               dmg=("damage_dealt", "mean")).reset_index()
    f = f.merge(merged[["cond", "family", "score", "d_star", "fidelity_at_4", "budget"]], on="cond")
    learn = f[f["family"].isin(["RL", "Hybrid", "Shield"])].reset_index(drop=True)
    X = learn[["kill_win", "timeout", "wez", "duration"]].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    from sklearn.mixture import GaussianMixture
    best, bics = None, []
    for k in range(1, 5):
        gm = GaussianMixture(k, n_init=5, random_state=seed).fit(X)
        bics.append((k, gm.bic(X)))
        if best is None or gm.bic(X) < best[1]:
            best = (k, gm.bic(X), gm)
    k, _, gm = best
    learn["cluster"] = gm.predict(X)
    # 군집 이름: 격추 비율 순서
    order = learn.groupby("cluster")["kill_win"].mean().sort_values().index
    name = {c: f"C{i+1}" for i, c in enumerate(order)}
    learn["cluster"] = learn["cluster"].map(name)
    summ = learn.groupby("cluster").agg(n=("cond", "size"), 격추승비율=("kill_win", "mean"),
                                        시간종료비율=("timeout", "mean"), WEZ초=("wez", "mean"),
                                        점수=("score", "mean"), D평균=("d_star", "mean"),
                                        D중앙=("d_star", "median"), F4=("fidelity_at_4", "mean")).reset_index()
    groups = [g["d_star"].values for _, g in learn.groupby("cluster")]
    kw = stats.kruskal(*groups) if len(groups) > 1 else None
    within = []
    for c, g in learn.groupby("cluster"):
        r, p, n = _rho(g["score"], g["d_star"])
        within.append(dict(cluster=c, n=n, rho=r, p=p))
    learn.to_csv(os.path.join(outdir, "tactic_clusters.csv"), index=False)
    # 순수 학습 정책, 최종 예산만 (README 6.4 의 표본과 같은 집합)
    rl = learn[(learn["family"] == "RL") & (learn["budget"] == learn["budget"].max())].reset_index(drop=True)
    rl_txt = ""
    if len(rl) >= 6:
        X2 = rl[["kill_win", "timeout", "wez", "duration"]].values
        X2 = (X2 - X2.mean(0)) / (X2.std(0) + 1e-9)
        best2, bics2 = None, []
        for k2 in range(1, min(4, len(rl) // 3) + 1):
            gm2 = GaussianMixture(k2, n_init=5, random_state=seed).fit(X2)
            bics2.append((k2, gm2.bic(X2)))
            if best2 is None or gm2.bic(X2) < best2[1]:
                best2 = (k2, gm2.bic(X2), gm2)
        rl["cluster"] = best2[2].predict(X2)
        order2 = rl.groupby("cluster")["kill_win"].mean().sort_values().index
        rl["cluster"] = rl["cluster"].map({c: f"C{i+1}" for i, c in enumerate(order2)})
        s2 = rl.groupby("cluster").agg(n=("cond", "size"), 격추승비율=("kill_win", "mean"),
                                       시간종료비율=("timeout", "mean"), WEZ초=("wez", "mean"),
                                       점수=("score", "mean"), D평균=("d_star", "mean"),
                                       D중앙=("d_star", "median")).reset_index()
        rl_txt = ("**순수 학습 정책, 최종 예산만** (BIC: " +
                  ", ".join(f"k={k_}: {b:.1f}" for k_, b in bics2) + f"; 선택 k={best2[0]})\n\n" +
                  md_table(s2) + "\n\n")
        rl.to_csv(os.path.join(outdir, "tactic_clusters_rl_final.csv"), index=False)
    # 군집과 무관한 연속 전술 축: 격추승 비율·WEZ 체류와 D* 의 상관
    rk, pk, nk = _rho(learn["kill_win"], learn["d_star"])
    rw, pw_, _ = _rho(learn["wez"], learn["d_star"])
    rk2, pk2, nk2 = _rho(rl["kill_win"], rl["d_star"]) if len(rl) >= 4 else (np.nan, np.nan, len(rl))
    rs2, ps2, _ = _rho(rl["kill_win"], rl["score"]) if len(rl) >= 4 else (np.nan, np.nan, 0)
    out = "### R5. 전술 군집 (가우시안 혼합, BIC 선택)\n\n"
    out += ("학습·혼합 조건의 종료 사유(격추승 비율, 시간종료 비율), WEZ 체류초, 교전 시간을 "
            "표준화해 가우시안 혼합모형으로 군집화했습니다. 군집 수는 BIC 최소로 골랐습니다"
            f"(BIC: {', '.join(f'k={k_}: {b:.1f}' for k_, b in bics)}; 선택 k={k}). "
            "사람이 보고 이름 붙인 전술이 아니라 객관 기준으로 나뉘는지, 군집 간 D* 가 다른지 봅니다.\n\n")
    out += md_table(summ) + "\n\n"
    if kw is not None:
        out += f"군집 간 D* Kruskal–Wallis: H={kw.statistic:.2f}, p={kw.pvalue:.4f}\n\n"
    out += "**군집 내 점수–D* Spearman** (전술을 고정했을 때 성능과 설명 비용의 관계)\n\n"
    out += md_table(pd.DataFrame(within), {"p": "{:.4f}"}) + "\n\n"
    out += rl_txt
    out += ("**연속 전술 축과 D***: 학습·혼합 전체에서 ρ(격추승 비율, D*)="
            f"{rk:.2f} (p={pk:.4f}, n={nk}), ρ(WEZ 체류초, D*)={rw:.2f} (p={pw_:.4f}); "
            f"순수 학습 최종 예산에서 ρ(격추승 비율, D*)={rk2:.2f} (p={pk2:.4f}, n={nk2}), "
            f"ρ(격추승 비율, 점수)={rs2:.2f} (p={ps2:.4f}).\n")
    return out, learn


# ------------------------------------------------------------------ R6 예산 세밀
def dense_budget_section(dense_dir: str) -> str:
    p = os.path.join(dense_dir, "merged.csv")
    if not os.path.exists(p):
        return ""
    m = load_merged(dense_dir)
    xc_p = os.path.join(dense_dir, "xai_common.csv")
    if os.path.exists(xc_p):
        m = m.merge(pd.read_csv(xc_p), on="cond", how="left")
    d = m[m["family"] == "RL"]
    g = d.groupby("budget").agg(n=("cond", "size"), 점수=("score", "mean"), 점수_sd=("score", "std"),
                                D=("d_star", "mean"), D_sd=("d_star", "std"), D_중앙=("d_star", "median"),
                                F4=("fidelity_at_4", "mean"),
                                **({"D_공통": ("d_star_common", "mean"),
                                    "방문범위": ("coverage_entropy", "mean")} if "d_star_common" in d.columns else {})
                                ).reset_index()
    out = "### R6. 25세대 간격 학습 예산에 따른 D* (순수 학습 정책, 시드 평균)\n\n"
    out += md_table(g, {"budget": "{:.0f}"}) + "\n\n"
    # 시드별 정점 위치
    peaks = d.groupby("train_seed").apply(lambda x: x.loc[x["d_star"].idxmax(), "budget"])
    out += (f"시드별 D* 최대 세대: 중앙값 {peaks.median():.0f} "
            f"(시드 {len(peaks)}개, 범위 {peaks.min():.0f}–{peaks.max():.0f}).\n")
    r1, p1, n = _rho(d["budget"], d["score"]); r2, p2, _ = _rho(d["budget"], d["d_star"])
    out += f"예산–점수 ρ={r1:.2f} (p={p1:.4f}), 예산–D* ρ={r2:.2f} (p={p2:.4f}), n={n}.\n"
    if "coverage_entropy" in d.columns:
        r3, p3, _ = _rho(d["budget"], d["coverage_entropy"]); r4, p4, _ = _rho(d["d_star"], d["coverage_entropy"])
        out += f"예산–방문범위 ρ={r3:.2f} (p={p3:.4f}), D*–방문범위 ρ={r4:.2f} (p={p4:.4f}).\n"
    # 초기(≤100) vs 후기(≥200) D* 대응 비교 (같은 시드)
    piv = d.pivot_table(index="train_seed", columns="budget", values="d_star")
    early = [b for b in piv.columns if 50 <= b <= 100]; late = [b for b in piv.columns if b >= 250]
    if early and late:
        e, l = piv[early].mean(axis=1), piv[late].mean(axis=1)
        ok = e.notna() & l.notna()
        if ok.sum() >= 6:
            w = stats.wilcoxon(e[ok], l[ok], zero_method="zsplit")
            out += (f"같은 시드 내 초기(50–100세대) 평균 D* {e[ok].mean():.1f} 대 후기(≥250세대) "
                    f"{l[ok].mean():.1f}: Wilcoxon p={w.pvalue:.4f} (n={int(ok.sum())}).\n")
    return out


# ------------------------------------------------------------------ R7 민감도
def sensitivity_section(main_dir: str, sens_dirs: list[str]) -> str:
    main = load_merged(main_dir)
    rows = []
    for sd in sens_dirs:
        p = os.path.join(sd, "merged.csv")
        if not os.path.exists(p):
            continue
        s = load_merged(sd)
        man = read_json(os.path.join(sd, "manifest.json"))
        both = main.merge(s, on="cond", suffixes=("_main", "_s"))
        r_s, p_s, n = _rho(both["score_main"], both["score_s"])
        r_d, p_d, _ = _rho(both["d_star_main"], both["d_star_s"])
        r_sd, p_sd, _ = _rho(s["score"], s["d_star"])
        rl = s[s["family"] == "RL"]; bt = s[s["family"] == "BT"]
        rows.append({"설정": os.path.basename(sd), "env_opts": json.dumps(man.get("env_opts", {})),
                     "n": n, "ρ(점수 본실험, 점수 변형)": r_s, "ρ(D* 본실험, D* 변형)": r_d,
                     "변형 내 ρ(점수, D*)": r_sd, "p": p_sd,
                     "RL 평균점수": rl["score"].mean(), "BT-v2 점수": float(bt.loc[bt["cond"] == "BT-v2", "score"].iloc[0]) if len(bt) else np.nan,
                     "RL 평균 D*": rl["d_star"].mean()})
    if not rows:
        return ""
    out = "### R7. 환경 규칙 민감도\n\n"
    out += ("교전 제한시간과 시간종료 규칙을 바꿔 같은 정책들을 다시 평가했습니다. "
            "정책 순위와 성능–D* 관계가 규칙에 따라 뒤집히는지 봅니다.\n\n")
    out += md_table(pd.DataFrame(rows), {"p": "{:.4f}"}) + "\n"
    return out


# ------------------------------------------------------------------ R8 정책 풀
def pool_section(pool_dir: str, merged: pd.DataFrame) -> str:
    p = os.path.join(pool_dir, "pool_scores.csv")
    if not os.path.exists(p):
        return ""
    ps = pd.read_csv(p)
    m = ps.merge(merged[["cond", "score", "d_star", "fidelity_at_4"]], on="cond", how="left")
    r1, p1, n = _rho(m["pool_score"], m["score"])
    r2, p2, _ = _rho(m["pool_score"], m["d_star"])
    r3, p3, _ = _rho(m["pool_score_vs_rl"], m["d_star"])
    out = "### R8. 정책 풀 상호 대전\n\n"
    out += ("학습 정책들과 BT 를 한 풀에 넣고 서로 전부 붙인 평균 점수(풀 점수)입니다. "
            "본실험 점수(고정 BT 2종 상대)와 풀 점수의 순위가 일치하는지, 풀 점수로 봐도 D* 와 "
            "무관한지 봅니다.\n\n")
    out += md_table(m.sort_values("pool_score", ascending=False)) + "\n\n"
    out += (f"ρ(풀 점수, 본실험 점수)={r1:.2f} (p={p1:.4f}, n={n}); "
            f"ρ(풀 점수, D*)={r2:.2f} (p={p2:.4f}); ρ(학습 정책 상대 풀 점수, D*)={r3:.2f} (p={p3:.4f}).\n")
    return out


# ------------------------------------------------------------------ 문서
def build(outdir: str, out_path: str, dense_dir: str = "results/dense",
          sens_dirs: list[str] | None = None, pool_dir: str = "results/pool") -> str:
    merged = load_merged(outdir)
    parts = ["# 강건성 분석 (자동 생성)\n",
             f"원본: `{outdir}` · 조건 {len(merged)}개 · 학습 시드 "
             f"{merged.loc[merged['train_seed'] >= 0, 'train_seed'].nunique()}개\n"]

    r1 = split_half(outdir, merged)
    parts.append("### R1. D* 와 점수의 반분 신뢰도, 감쇠 보정 상관\n\n"
                 "각 조건의 로그를 전반/후반으로 나눠 D* 를 두 번 계산하고(점수는 홀수/짝수 시드), "
                 "두 값의 조건 간 순위상관을 Spearman–Brown 으로 전체 길이 신뢰도로 올렸습니다. "
                 "관측 상관 ρ_obs 를 √(신뢰도_D × 신뢰도_점수) 로 나눈 것이 감쇠 보정 상관입니다.\n\n"
                 + md_table(pd.DataFrame([r1]), {"p": "{:.4f}"}) + "\n")

    r2 = bootstrap_equivalence(merged)
    r2f = bootstrap_equivalence(merged, ycol="fidelity_at_4")
    tab = pd.DataFrame([
        dict(지표="점수 vs D*(0.95)", rho=r2["rho"], n=r2["n"],
             CI95_조건=f"[{r2['ci95_cond'][0]:.2f}, {r2['ci95_cond'][1]:.2f}]",
             CI90_조건=f"[{r2['ci90_cond'][0]:.2f}, {r2['ci90_cond'][1]:.2f}]",
             CI95_시드클러스터=f"[{r2['ci95_cluster'][0]:.2f}, {r2['ci95_cluster'][1]:.2f}]",
             CI90_시드클러스터=f"[{r2['ci90_cluster'][0]:.2f}, {r2['ci90_cluster'][1]:.2f}]",
             동등_0p2_조건=r2["tost_cond_0.2"], 동등_0p3_조건=r2["tost_cond_0.3"],
             동등_0p2_클러스터=r2["tost_cluster_0.2"], 동등_0p3_클러스터=r2["tost_cluster_0.3"]),
        dict(지표="점수 vs F(4)", rho=r2f["rho"], n=r2f["n"],
             CI95_조건=f"[{r2f['ci95_cond'][0]:.2f}, {r2f['ci95_cond'][1]:.2f}]",
             CI90_조건=f"[{r2f['ci90_cond'][0]:.2f}, {r2f['ci90_cond'][1]:.2f}]",
             CI95_시드클러스터=f"[{r2f['ci95_cluster'][0]:.2f}, {r2f['ci95_cluster'][1]:.2f}]",
             CI90_시드클러스터=f"[{r2f['ci90_cluster'][0]:.2f}, {r2f['ci90_cluster'][1]:.2f}]",
             동등_0p2_조건=r2f["tost_cond_0.2"], 동등_0p3_조건=r2f["tost_cond_0.3"],
             동등_0p2_클러스터=r2f["tost_cluster_0.2"], 동등_0p3_클러스터=r2f["tost_cluster_0.3"])])
    parts.append("### R2. 상관의 부트스트랩 신뢰구간과 동등성 검정(TOST)\n\n"
                 "조건을 재추출한 부트스트랩(5,000회)과, 학습 시드를 통째로 재추출하는 클러스터 "
                 "부트스트랩(같은 시드에서 나온 조건들이 독립이 아님을 반영)의 신뢰구간입니다. "
                 "90% 신뢰구간이 ±0.2(또는 ±0.3) 안에 들어오면 '상관이 그 한계보다 작다'가 "
                 "α=0.05 에서 성립합니다(두 단측검정 절차).\n\n" + md_table(tab) + "\n")

    s3, m3 = common_state_section(outdir, merged)
    parts.append(s3)

    r4 = mixed_model(merged)
    parts.append("### R4. 학습 시드 임의효과 혼합모형과 시드 단위 상관\n\n"
                 "score \~ D*(표준화) + 예산(표준화) + α, 학습 시드 임의절편. 같은 시드에서 나온 "
                 "조건들의 상관을 모형이 흡수하므로 D* 계수의 p 값이 표본 독립 가정에 기대지 않습니다. "
                 "시드 단위 상관은 순수 학습 정책을 시드별로 평균해(독립 표본 = 시드) 구한 값입니다.\n\n"
                 + md_table(pd.DataFrame([r4]), {"p_dstar": "{:.4f}", "p_budget": "{:.4f}", "p_alpha": "{:.4f}",
                                                 "p_dstar_only": "{:.4f}", "seed_level_p": "{:.4f}",
                                                 "final_budget_p": "{:.4f}"}) + "\n")

    s5, _ = tactic_clusters(outdir, merged)
    parts.append(s5)

    s6 = dense_budget_section(dense_dir)
    if s6:
        parts.append(s6)
    s7 = sensitivity_section(outdir, sens_dirs or sorted(glob.glob("results/sens_*")))
    if s7:
        parts.append(s7)
    s8 = pool_section(pool_dir, merged)
    if s8:
        parts.append(s8)

    doc = "\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(outdir, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(dict(R1=r1, R2=r2, R2_f4=r2f, R4=r4), f, indent=2, ensure_ascii=False, default=str)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="강건성 분석")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--out", default="paper/results_ext.md")
    ap.add_argument("--dense-dir", default="results/dense")
    ap.add_argument("--pool-dir", default="results/pool")
    ap.add_argument("--sens-dirs", nargs="*", default=None)
    a = ap.parse_args()
    print(build(a.outdir, a.out, a.dense_dir, a.sens_dirs, a.pool_dir))


if __name__ == "__main__":
    main()
