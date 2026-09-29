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

    # 확장 하이브리드 (표 9b)
    for pat, sc, sd in [(r"RES-s\d-b200", .628, .024), (r"GATE-s\d-b200", .722, .047)]:
        g = mh[mh.cond.str.match(pat)]
        check(f"{pat} 점수", g.score.mean(), sc, .0006, "표 9b"); check(f"{pat} sd", g.score.std(), sd, .0006, "표 9b")
    check("게이팅형 200세대 D* 모두 17", (mh[mh.cond.str.match(r"GATE-s\d-b200")].d_star == 17).sum(), 3, 0, "표 9b")

    bad = 0
    print(f"{'항목':40s} {'계산값':>9s} {'문서값':>9s}  판정  위치")
    for name, got, want, tol, where in ROWS:
        ok = abs(got - want) <= tol + 1e-12
        bad += (not ok)
        print(f"{name:40s} {got:9.4f} {want:9.4f}  {'OK  ' if ok else 'FAIL'}  {where}")
    print(f"\n{len(ROWS) - bad}/{len(ROWS)} 일치" + ("" if not bad else f", {bad}개 어긋남"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="문서 수치 대 결과 파일 대조")
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--hyb", default="results/main_hyb")
    a = ap.parse_args()
    sys.exit(run(a.main, a.hyb))


if __name__ == "__main__":
    main()
