"""
논문·README 에 적힌 핵심 수치가 저장된 결과와 맞는지 다시 계산해 대조한다.

    python -m dfxai.analysis.check_claims            # 하나라도 어긋나면 종료 코드 1

원고를 고칠 때마다 돌리면 "문서 숫자 ≠ 결과 파일" 실수를 잡을 수 있다. 기대값은 이 파일 안에
적혀 있으므로, 논문의 숫자를 일부러 바꿨다면 여기도 같이 바꾼다. 실험을 다시 돌려 결과가 바뀌면
어긋나는 항목이 그대로 목록으로 나온다.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

from .robustness import _partial_spearman

ROWS: list[tuple[str, float, float, float, str]] = []


def check(name: str, got: float, want: float, tol: float, where: str):
    ROWS.append((name, float(got), float(want), tol, where))


def rho(x, y):
    d = pd.concat([pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))], axis=1).dropna()
    r = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return float(r.statistic), float(r.pvalue)


def run(main: str, hyb: str) -> int:
    m = pd.read_csv(os.path.join(main, "merged_common.csv"))
    ep = pd.read_csv(os.path.join(main, "episodes.csv"))
    rb = json.load(open(os.path.join(main, "robustness.json"), encoding="utf-8"))
    mh = pd.read_csv(os.path.join(hyb, "merged.csv"))
    C = m["cond"]

    check("조건 수", len(m), 282, 0, "README 5.3")
    check("교전 수", len(ep), 112800, 0, "README 5.3")

    # 표 5: 점수와 D*·F(4)·위반율의 순위상관
    sub = m[m.family.isin(["RL", "Hybrid"])]
    r, p = rho(m.score, m.d_star)
    check("ρ(점수, D*) 전체", r, -0.02, 0.005, "표 5"); check("p", p, 0.75, 0.005, "표 5")
    r, p = rho(sub.score, sub.d_star)
    check("ρ(점수, D*) 학습·혼합", r, -0.18, 0.005, "표 5"); check("p", p, 0.006, 0.0005, "표 5")
    check("ρ(점수, F4) 전체", rho(m.score, m.fidelity_at_4)[0], 0.05, 0.005, "표 5")
    check("ρ(점수, 위반율) 전체", rho(m.score, m.viol_total)[0], 0.18, 0.005, "표 5")
    check("ρ(D*, 위반율) 전체", rho(m.d_star, m.viol_total)[0], 0.29, 0.005, "표 5")
    check("절단(17) 비율", (m.d_star >= 17).mean(), 0.58, 0.005, "초안 6장")

    # 표 3: 순수 학습 정책의 예산별 결과 (표본표준편차, n−1)
    rl = m[C.str.match(r"RL-s\d+-b\d+-a1.00")].copy()
    rl["b"] = rl.cond.str.extract(r"-b(\d+)-")[0].astype(int)
    for b, sc, sd, dd, f4, vi in [(100, .646, .084, 8.1, .926, .062), (200, .674, .062, 7.2, .932, .046),
                                  (300, .716, .051, 7.6, .929, .050)]:
        g = rl[rl.b == b]
        check(f"RL b{b} 점수", g.score.mean(), sc, .0006, "표 3"); check(f"RL b{b} 점수 sd", g.score.std(), sd, .0006, "표 3")
        check(f"RL b{b} D*", g.d_star.mean(), dd, .06, "표 3"); check(f"RL b{b} F4", g.fidelity_at_4.mean(), f4, .0006, "표 3")
        check(f"RL b{b} 위반율", g.viol_total.mean(), vi, .0006, "표 3")
    g = rl[rl.b == 300]
    bt2 = m.loc[C == "BT-v2", "score"].iloc[0]
    check("300세대 중 BT-v2 초과 개수", (g.score > bt2).sum(), 17, 0, "4.1")
    check("300세대 점수 최소", g.score.min(), .597, .0006, "4.1"); check("300세대 점수 최대", g.score.max(), .785, .0006, "4.1")
    check("300세대 보류 상대(BT-v3) 평균", ep[ep.cond.isin(g.cond) & (ep.opponent == "BT-v3")].score.mean(), .746, .0006, "4.1")

    # 표 4: BT 3종과 계열 집계
    for c, sc, dd in [("BT-v1", .241, 1), ("BT-v2", .671, 4), ("BT-v3", .329, 5), ("RL-s0-b0-a1.00", .639, 2)]:
        row = m[C == c].iloc[0]
        check(f"{c} 점수", row.score, sc, .0006, "표 4"); check(f"{c} D*", row.d_star, dd, 0, "표 4")
    for a, sc, sd in [("0.25", .619, .040), ("0.50", .639, .047), ("0.75", .721, .045)]:
        g = m[C.str.match(rf"HYB-s\d+-b300-a{a}")]
        check(f"선형 혼합 α={a} 점수", g.score.mean(), sc, .0006, "표 4"); check(f"선형 혼합 α={a} sd", g.score.std(), sd, .0006, "표 4")
    for fam, sc, sd in [("Shield", .543, .072), ("PPO", .385, .009)]:
        g = m[m.family == fam]
        check(f"{fam} 점수", g.score.mean(), sc, .0006, "표 4"); check(f"{fam} sd", g.score.std(), sd, .0006, "표 4")
    g = m[C.str.match(r"BTO-s\d-b300")]
    check("BTO 300세대 점수", g.score.mean(), .602, .0006, "표 10·4.8"); check("BTO 300세대 sd", g.score.std(), .008, .0006, "4.8")

    # 강건성 R1–R2, R3(정오표 반영)
    check("R1 D* 신뢰도", rb["R1"]["rel_dstar"], .82, .005, "표 6"); check("R1 점수 신뢰도", rb["R1"]["rel_score"], .96, .005, "표 6")
    check("R2 CI95 조건 하한", rb["R2"]["ci95_cond"][0], -.14, .005, "표 6"); check("R2 CI95 조건 상한", rb["R2"]["ci95_cond"][1], .10, .005, "표 6")
    check("R2 CI95 클러스터 하한", rb["R2"]["ci95_cluster"][0], -.21, .005, "표 6"); check("R2 CI95 클러스터 상한", rb["R2"]["ci95_cluster"][1], .18, .005, "표 6")
    check("R3 ρ(점수, D* 공통)", rho(m.score, m.d_star_common)[0], .19, .005, "표 6")
    check("R3 편상관(D* 공통; 방문범위 통제)", _partial_spearman(m.score, m.d_star_common, m.coverage_entropy)[0], .16, .005, "표 6 (정오표)")
    check("R3 편상관(D* 자기; 방문범위 통제)", _partial_spearman(m.score, m.d_star, m.coverage_entropy)[0], .00, .005, "이전 판본의 '0.00'")

    # 표 8: 전술
    d = rl[rl.b == 300].set_index("cond")
    e = ep[ep.cond.isin(d.index)]
    kill = e.assign(k=e.outcome == "gun_kill").groupby("cond").k.mean()
    chase = [c for c in d.index if c.split("-")[1] in ("s1", "s10", "s12", "s14")]
    hit = d.drop(index=chase)
    check("치고 빠지기 정책 수", len(hit), 16, 0, "4.5"); check("치고 빠지기 D* 중앙값", hit.d_star.median(), 5, 0, "4.5")
    check("치고 빠지기 중 D* 3–9 개수", ((hit.d_star >= 3) & (hit.d_star <= 9)).sum(), 15, 0, "4.5")
    check("추격 격추 정책 격추 종료 비율 최소", kill[chase].min(), .26, .006, "4.5"); check("추격 격추 정책 격추 종료 비율 최대", kill[chase].max(), .35, .006, "4.5")

    # 안전규칙: "보장"이 아니라 관측 빈도 (표 10, 5.3)
    def eps(pat, col):
        x = ep[ep.cond.str.contains(pat, regex=True)]
        return int((x[col] > 0).sum()), len(x)
    check("BT-v3 하드덱 위반 교전 수", eps(r"^BT-v3$", "viol_deck")[0], 0, 0, "4.6.1·5.3")
    check("BT-v3 과G 위반 교전 수", eps(r"^BT-v3$", "viol_over_g")[0], 0, 0, "4.6.1·5.3")
    check("BT-v3 최소 이격 위반 교전 수(400 중)", eps(r"^BT-v3$", "viol_sep")[0], 180, 0, "4.6.1")
    check("Shield 하드덱 위반 교전 수(8,000 중)", eps(r"^SHD-", "viol_deck")[0], 30, 0, "4.6.1·5.3 (정오표)")
    check("Shield 과G 위반 교전 수(8,000 중)", eps(r"^SHD-", "viol_over_g")[0], 3, 0, "4.6.1·5.3 (정오표)")
    shd = ep[ep.cond.str.startswith("SHD-")]
    check("Shield 평균 하드덱 위반율", shd.viol_deck.mean(), .0005, .00005, "표 10")
    check("Shield 위반 교전이 나온 시드 수(하드덱)", shd[shd.viol_deck > 0].cond.nunique(), 4, 0, "4.6.1")

    # 확장 하이브리드 (표 9b)
    for pat, sc, sd in [(r"RES-s\d-b200", .628, .024), (r"GATE-s\d-b200", .722, .047)]:
        g = mh[mh.cond.str.match(pat)]
        check(f"{pat} 점수", g.score.mean(), sc, .0006, "표 9b"); check(f"{pat} sd", g.score.std(), sd, .0006, "표 9b")
    check("게이팅형 200세대 D* 모두 17", (mh[mh.cond.str.match(r"GATE-s\d-b200")].d_star == 17).sum(), 3, 0, "표 9b")

    _check_tactical(os.path.join("results", "tactical"))
    _check_distill(os.path.join("results", "distill"))

    bad = 0
    print(f"{'항목':40s} {'계산값':>9s} {'문서값':>9s}  판정  위치")
    for name, got, want, tol, where in ROWS:
        ok = abs(got - want) <= tol + 1e-12
        bad += (not ok)
        print(f"{name:40s} {got:9.4f} {want:9.4f}  {'OK  ' if ok else 'FAIL'}  {where}")
    print(f"\n{len(ROWS) - bad}/{len(ROWS)} 일치" + ("" if not bad else f", {bad}개 어긋남"))
    return 1 if bad else 0


def _check_tactical(tdir: str) -> None:
    """R11 (전술 영역별 설명 비용): README 6.11 에 적은 값. 결과 폴더가 없으면 건너뛴다."""
    if not os.path.exists(os.path.join(tdir, "regional.csv")):
        return
    from .tactical_report import _agg_regional, group_of
    reg = _agg_regional(pd.read_csv(os.path.join(tdir, "regional.csv")))
    own = reg[(reg.source == "own") & (reg.split == "grouped")]

    def med(group: str, region: str, col: str = "F4") -> float:
        return float(own[(own.group == group) & (own.region == region)][col].median())

    for grp, vals in {"학습(치고 빠지기)": (.857, .942, .997, .995), "학습(추격)": (.788, .858, .985, .992),
                      "BT-v2": (.817, .991, .998, 1.000)}.items():
        for rn, v in zip(("head_on", "offensive", "defensive", "neutral"), vals):
            check(f"R11 F4 {grp} {rn}", med(grp, rn), v, .0006, "6.11 표 R11-2")
    check("R11 F4 혼합 α=0.5 정면 조우", med("혼합 α=0.5", "head_on"), .608, .0006, "6.11")
    check("R11 F4 Shield 정면 조우", med("Shield", "head_on"), .654, .0006, "6.11")

    es = pd.read_csv(os.path.join(tdir, "error_share.csv"))
    es["group"] = es.cond.map(group_of)
    e8 = es[(es.depth == 8) & (es.region == "head_on")]
    per = e8.groupby("cond").agg(err=("err_share", "mean"), t=("time_share", "mean"), grp=("group", "first"))
    check("R11 정면 조우 오차 몫 최소(깊이 8, 조건별)", per.err.min(), .19, .006, "6.11 표 R11-8")
    check("R11 정면 조우 오차 몫 최대(깊이 8, 조건별)", per.err.max(), .73, .006, "6.11 표 R11-8")
    check("R11 정면 조우 시간 비율 최대(깊이 8, 조건별)", per.t.max(), .10, .006, "6.11 표 R11-8")
    check("R11 정면 조우 오차 몫 > 시간 비율인 조건 수(73개 중)", int((per.err > per.t).sum()), 73, 0, "6.11 표 R11-8")


def _check_distill(ddir: str) -> None:
    """R10 (트리를 정책으로 돌린 성능)과 보조 시험: README 6.11 에 적은 값. 결과 폴더가 없으면 건너뛴다."""
    if not os.path.exists(os.path.join(ddir, "games.csv")):
        return
    from .distill_report import load, paired, group_of, _seed_scores, MARGIN
    g, f, c, _ = load(ddir)
    a, _s0 = paired(g)
    ds = c.set_index("cond")["d_star"]
    learned = [x for x in c.cond if group_of(x) in ("학습(치고 빠지기)", "학습(추격)")]
    la = a[a.cond.isin(learned)]
    check("R10 학습 정책 수", len(learned), 20, 0, "6.11")
    check("R10 학습 정책 중 깊이 ≤16 에서 한계 안에 든 정책 수", la[la.delta >= -MARGIN].cond.nunique(), 0, 0, "6.11")
    check("R10 깊이 16 Δ 중앙값(학습)", la[la.depth == 16].delta.median(), -.246, .0006, "6.11")
    at_d = pd.Series({x: float(a[(a.cond == x) & (a.depth == int(min(16, ds[x])))].delta.iloc[0]) for x in learned})
    check("R10 D* 깊이 Δ 최소(학습)", at_d.min(), -.676, .0006, "6.11"); check("R10 D* 깊이 Δ 최대(학습)", at_d.max(), -.238, .0006, "6.11")
    check("R10 D* 깊이 Δ 중앙값(학습)", at_d.median(), -.556, .0006, "6.11")
    m = f[f.cond.isin(learned)].merge(a[["cond", "depth", "delta"]], on=["cond", "depth"])
    hi = m[m.r2_fresh >= .97]
    check("R10 새 교전 R² ≥ 0.97 쌍의 수", len(hi), 20, 0, "6.11"); check("R10 새 교전 R² ≥ 0.97 쌍의 Δ 중앙값", hi.delta.median(), -.384, .0006, "6.11")
    f4 = f[(f.depth == 4) & f.cond.isin(learned)]
    grp = f4.cond.map(group_of)
    check("R10 깊이 4 새 교전 R²(치고 빠지기)", f4[grp == "학습(치고 빠지기)"].r2_fresh.median(), .945, .0006, "6.11")
    check("R10 깊이 4 새 교전 R²(추격)", f4[grp == "학습(추격)"].r2_fresh.median(), .904, .0006, "6.11")
    for gp, sc0, sc16 in [("학습(치고 빠지기)", .704, .450)]:
        cs = [x for x in c.cond if group_of(x) == gp]
        sg = g[g.cond.isin(cs)]
        check(f"R10 {gp} 원본 점수", sg[sg.depth == 0].score.mean(), sc0, .0006, "6.11 표 R10-5")
        check(f"R10 {gp} 깊이 16 트리 점수", sg[sg.depth == 16].score.mean(), sc16, .0006, "6.11 표 R10-5")

    # DAgger
    dgp = os.path.join(ddir, "dagger_games.csv")
    if os.path.exists(dgp):
        dg = pd.read_csv(dgp); df = pd.read_csv(os.path.join(ddir, "dagger_fits.csv"))
        orig = _seed_scores(g[g.depth == 0], ["cond"]).rename(columns={"score": "s0"})
        da = _seed_scores(dg[dg.cond.isin(learned)], ["cond", "depth", "iteration"]).merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0)
        for d, med, ok in [(4, -.374, 0), (8, -.088, 2), (16, -.028, 16)]:
            x = da[(da.depth == d) & (da.iteration == 3)].groupby("cond").delta.agg(["mean", "std", "size"])
            check(f"DAgger 3회 깊이 {d} Δ 중앙값", x["mean"].median(), med, .0006, "6.11 표 R10-6")
            check(f"DAgger 3회 깊이 {d} 한계 안 정책 수", int((x["mean"] >= -MARGIN).sum()), ok, 0, "6.11 표 R10-6")
            if d == 16:
                up = x["mean"] + 1.96 * x["std"] / np.sqrt(x["size"])
                check("DAgger 3회 깊이 16 상한<0 정책 수", int((up < 0).sum()), 10, 0, "6.11 표 R10-6")
        l16 = df[(df.depth == 16) & (df.iteration == 3) & df.cond.isin(learned)].n_leaves
        check("DAgger 깊이 16 잎 수 중앙값(학습)", l16.median(), 11997, 1, "6.11 표 R10-6")
        r1 = df[(df.depth == 4) & (df.iteration == 1)]
        check("DAgger 1회차 시작 R² 깊이 4(치고 빠지기)", r1[r1.cond.map(group_of) == "학습(치고 빠지기)"].r2_on_policy_before.median(), .291, .0006, "6.11")
        check("DAgger 1회차 시작 R² 깊이 4(추격)", r1[r1.cond.map(group_of) == "학습(추격)"].r2_on_policy_before.median(), -.414, .0006, "6.11")

    # 트리 적합 노이즈
    ngp = os.path.join(ddir, "noise_games.csv")
    if os.path.exists(ngp):
        ng = pd.read_csv(ngp)
        base = _seed_scores(g[g.depth == 0], ["cond"]).rename(columns={"score": "s0"})
        nb = _seed_scores(ng, ["cond", "depth", "boot"]).merge(base, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0)
        nb = nb.groupby(["cond", "depth", "boot"]).delta.mean().reset_index()
        v2 = nb[nb.cond == "BT-v2"]
        check("노이즈 BT-v2 깊이 8 부트스트랩 Δ 최대", v2[v2.depth == 8].delta.max(), -.115, .0006, "6.11 표 R10-7")
        check("노이즈 BT-v2 깊이 16 부트스트랩 Δ 최소", v2[v2.depth == 16].delta.min(), .006, .0006, "6.11 표 R10-7")

    # 트리 정책 이탈 영역
    rgp = os.path.join(ddir, "region_error.csv")
    if os.path.exists(rgp):
        r = pd.read_csv(rgp)
        r["group"] = r.cond.map(group_of)
        x = r[(r.group == "학습(치고 빠지기)") & (r.depth == 8)]
        t, o = x[x.states == "tree-visited"], x[x.states == "original-visited"]
        check("이탈 영역 트리 방문 방어 시간 비율(깊이 8, 치고 빠지기)", t.time_defensive.median(), .58, .006, "6.11 표 R10-8")
        check("이탈 영역 트리 방문 방어 오차 몫(깊이 8, 치고 빠지기)", t.err_defensive.median(), .59, .006, "6.11 표 R10-8")
        check("이탈 영역 원본 방문 방어 시간 비율(깊이 8, 치고 빠지기)", o.time_defensive.median(), .20, .006, "6.11 표 R10-8")
        check("이탈 영역 원본 방문 방어 오차 몫(깊이 8, 치고 빠지기)", o.err_defensive.median(), .01, .006, "6.11 표 R10-8")


def main():
    ap = argparse.ArgumentParser(description="문서 수치 대 결과 파일 대조")
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--hyb", default="results/main_hyb")
    a = ap.parse_args()
    sys.exit(run(a.main, a.hyb))


if __name__ == "__main__":
    main()
