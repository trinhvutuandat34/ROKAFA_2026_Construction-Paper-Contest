"""
학습 모델 수준 향상 실험(B1 …) 분석과 사전 판정 — 기준은 paper/수준향상_사전기준.md (학습 전 기록).

    bash results/run_level_eval.sh b1
    python -m dfxai.analysis.level --out paper/results_level.md

  results/level_b1_hp     본실험 규칙(시간종료 체력 비교)에서의 평가  ← 판정에 쓰는 값
  results/level_b1_draw   격추만 승리 규칙에서의 평가(+ BT 3종·BTO 비교)
  results/level_b1_pool   본실험 26개 모델 + B1 1,000세대 5개의 전원 대전
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
from scipy import stats

from .paper import md_table
from .extra import cond_table, _md, _parse
from .tradeoff_tests import price_cv

T1_BTO, T1_BT2 = 0.882, 0.843          # 본실험: BT-v3 상대 BTO 평균, BT-v2
FINAL = 1000


def _with_meta(t: pd.DataFrame) -> pd.DataFrame:
    t = t[[c.startswith("POOL-") for c in t.index]].copy()
    t["seed"] = [_parse(c)[1] for c in t.index]
    t["gen"] = [_parse(c)[2] for c in t.index]
    return t


def win_share(outdir: str) -> pd.Series:
    """승리 가운데 격추승 비율(상대 2종 합산)."""
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    g = ep.groupby("cond")
    return g.apply(lambda d: ((d.outcome == "gun_kill") & (d.win == 1)).sum() / max(1, d.win.sum()),
                   include_groups=False)


def judge(hp: pd.DataFrame, pool: pd.DataFrame | None, gen: int = FINAL) -> dict:
    last = hp[hp.gen == gen].sort_values("seed")
    n_bto = int((last.v3 > T1_BTO).sum())
    n_bt2 = int((last.v3 > T1_BT2).sum())
    t1 = "충족" if n_bto >= 2 else ("부분 충족" if (n_bto == 1 or n_bt2 >= 2) else "미충족")
    t2, n_pool, ref = "평가 전", None, None
    if pool is not None:
        ps = pool.set_index("cond").pool_score
        ref = float(ps[[c for c in ps.index if c.startswith("BT")]].max())   # BT-v1–3, BTO
        new = ps[[c for c in ps.index if c.startswith("POOL-") and c.endswith(f"-b{gen}")]]
        n_pool = int((new > ref).sum())
        t2 = "충족" if n_pool == len(new) == 5 else ("부분 충족" if n_pool >= 3 else "미충족")
    n_kill = int((last.kill_share >= 0.5).sum())
    t3 = "충족" if n_kill >= 3 else ("부분 충족" if n_kill >= 1 else "미충족")
    marks = [t1, t2, t3]
    if all(m == "충족" for m in marks):
        overall = "높은 수준에 도달"
    elif "미충족" in marks:
        overall = "도달하지 못함"
    else:
        overall = "부분 도달"
    return dict(t1=t1, t2=t2, t3=t3, overall=overall, n_bto=n_bto, n_bt2=n_bt2, n_pool=n_pool,
                pool_ref=ref, n_kill=n_kill, last=last)


def tradeoff(hp_dir: str, main_dir: str = "results/main", gens=(300, 600, 1000)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """검토 문서 6.3: 연관성(Spearman)과 제약 비용을 (가) 본실험 + 새 모델, (나) 새 모델만으로."""
    m = pd.read_csv(os.path.join(main_dir, "merged.csv"))
    n = pd.read_csv(os.path.join(hp_dir, "merged.csv"))
    n = n[[bool(re.match(r"POOL-s\d+-b(\d+)$", c)) and _parse(c)[2] in gens for c in n.cond]]
    n["family"] = "Pool"
    allm = pd.concat([m, n], ignore_index=True)
    for d in (allm, n):
        d["vs_v2"] = 2 * d["score"] - d["score_heldout"]
        d["vs_v3"] = d["score_heldout"]
        d["cost_f4"] = 1 - d["fidelity_at_4"]
    sets = {"본실험 + 새 모델(전체)": allm,
            "본실험 혼합 제외 + 새 모델": allm[allm.family.isin(["BT", "BTO", "RL", "PPO", "Pool"])],
            "새 모델만": n}
    perfs = [("학습 상대", "vs_v2"), ("보류 상대", "vs_v3"), ("상대 2종 평균", "score")]
    costs = [("D*95", "d_star_095"), ("D*90", "d_star_090"), ("1-F(4)", "cost_f4")]
    rows = []
    for sn, d in sets.items():
        for pn, pc in perfs:
            for cn, cc in costs:
                r, p = stats.spearmanr(d[pc], d[cc])
                rows.append(dict(집합=sn, n=len(d), 성능=pn, 설명비용=cn, rho=r, p=p))
    assoc = pd.DataFrame(rows)
    e1 = pd.read_csv(os.path.join(main_dir, "episodes.csv"), usecols=["cond", "opponent", "seed", "score"])
    e2 = pd.read_csv(os.path.join(hp_dir, "episodes.csv"), usecols=["cond", "opponent", "seed", "score"])
    e2 = e2[e2.cond.isin(n.cond)]
    price = price_cv(allm, pd.concat([e1, e2], ignore_index=True))
    return assoc, price


def analyse(prefix="results/level_b1", main_dir="results/main") -> tuple[str, dict]:
    hp_dir, draw_dir, pool_dir = f"{prefix}_hp", f"{prefix}_draw", f"{prefix}_pool"
    hp = _with_meta(cond_table(hp_dir))
    hp["kill_share"] = win_share(hp_dir).reindex(hp.index)
    pool = pd.read_csv(os.path.join(pool_dir, "pool_scores.csv")) if os.path.exists(
        os.path.join(pool_dir, "pool_scores.csv")) else None
    gen = FINAL if (hp.gen == FINAL).any() else int(hp.gen.max())
    j = judge(hp, pool, gen)
    out = [f"## B1 기본안 (상대 풀 + 격추만 승리 + α 1.0, 판정 세대 {gen})\n"]
    if gen != FINAL:
        out.append(f"> 중간 결과: {FINAL}세대 체크포인트가 아직 없어 {gen}세대로 계산했다. 판정은 {FINAL}세대에서 한다.\n")
    out.append(f"**사전 판정: {j['overall']}.** T1 일반 교전 능력 {j['t1']}(BT-v3 상대 0.882 초과 {j['n_bto']}개, "
               f"0.843 초과 {j['n_bt2']}개/5), T2 상대적 우열 {j['t2']}"
               + (f"(행동트리·BTO 최고 {j['pool_ref']:.3f}보다 높은 모델 {j['n_pool']}개/5)" if j['n_pool'] is not None else "")
               + f", T3 규칙 허점 의존 {j['t3']}(격추승 비율 0.5 이상 {j['n_kill']}개/5).\n")
    last = j["last"]
    out.append("### 표 B1-1. 판정 세대의 시드별 결과 (본실험 규칙)\n")
    cols = ["v2", "v3", "avg", "d95", "d90", "f4", "kill_share", "timeout"]
    t = last.copy()
    t["kill_share"] = t["kill_share"].astype(float)
    from .extra import KO, FMT
    KO.setdefault("kill_share", "승리 중 격추승"); FMT.setdefault("kill_share", "{:.2f}")
    out.append(_md(t, cols, "조건"))
    if pool is not None:
        ps = pool.sort_values("pool_score", ascending=False)
        out.append("\n### 표 B1-2. 전원 대전 점수 (본실험 26개 + B1 5개, 초기조건 30개 × 진영 교대)\n")
        out.append(md_table(ps[["cond", "family", "pool_score"]], {"pool_score": "{:.3f}"}))
    out.append("\n### 표 B1-3. 세대별 5개 시드 평균 (본실험 규칙)\n")
    mg = hp.groupby("gen")[["v2", "v3", "avg", "d95", "d90", "f4", "kill_share"]].mean()
    mg[["d95", "d90"]] = mg[["d95", "d90"]].round(1)
    mg.index = mg.index.astype(str)
    out.append(_md(mg, ["v2", "v3", "avg", "d95", "d90", "f4", "kill_share"], "세대"))
    if os.path.exists(os.path.join(draw_dir, "episodes.csv")):
        dr = cond_table(draw_dir)
        ref = dr.loc[[c for c in dr.index if not c.startswith("POOL-")]]
        new = _with_meta(dr)
        out.append("\n### 표 B1-4. 격추만 승리 규칙에서의 평가 (참고)\n")
        nl = new[new.gen == gen]
        out.append(_md(pd.concat([ref, nl[ref.columns]]), ["v2", "v3", "avg", "d95", "kill_win", "timeout"], "조건"))
    res = dict(judge=j, hp=hp)
    try:
        assoc, price = tradeoff(hp_dir, main_dir)
        res["assoc"], res["price"] = assoc, price
        out.append("\n### 표 B1-5. 트레이드오프 검정 다시 계산 (검토 문서 6.3)\n")
        out.append("연관성(Spearman ρ, 양수가 트레이드오프 방향). 새 모델은 300·600·1,000세대.\n")
        a = assoc.pivot_table(index=["집합", "성능"], columns="설명비용", values="rho").round(2)
        out.append(md_table(a.reset_index(), {}))
        out.append("\n제약 비용(본실험 + 새 모델, 교차 검증, 양수 = 상한 때문에 잃은 성능).\n")
        pr = price[price["설명비용"] == "d_star_095"].pivot_table(index="성능", columns="k", values="가격").round(3)
        out.append(md_table(pr.reset_index(), {}))
    except Exception as ex:          # 평가가 덜 끝나 300·600·1,000세대 모델이 없을 때
        out.append(f"\n(트레이드오프 재계산 생략: {ex})\n")
    return "\n".join(out) + "\n", res


def main():
    ap = argparse.ArgumentParser(description="수준 향상 실험 분석")
    ap.add_argument("--prefix", default="results/level_b1")
    ap.add_argument("--out", default="paper/results_level.md")
    a = ap.parse_args()
    txt, res = analyse(a.prefix)
    head = ("# 학습 모델 수준 향상 실험 결과 (자동 생성)\n\n"
            "기준: `paper/수준향상_사전기준.md`(학습 전에 기록). 생성: `python -m dfxai.analysis.level`.\n\n")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(head + txt)
    j = res["judge"]
    print(f"저장: {a.out}\n종합 {j['overall']} | T1 {j['t1']} ({j['n_bto']}/{j['n_bt2']}) | T2 {j['t2']} | T3 {j['t3']} ({j['n_kill']})")


if __name__ == "__main__":
    main()
