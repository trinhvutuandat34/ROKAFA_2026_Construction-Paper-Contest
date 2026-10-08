"""
추가 실험(E1–E3) 분석과 사전 판정 — 판정 기준은 paper/추가실험_사전기준.md (학습 전에 기록).

    bash results/run_extra_eval.sh all
    python -m dfxai.analysis.extra --out paper/results_extra.md

  E1 학습 연장     results/extra_long      학습 시드 1·11·14·2·18, 300–1,000세대
  E2 하이브리드     results/extra_hyb       잔차형·게이팅형 시드 0–4, 100·200·300세대 + 원 모델
  E3 격추만 승리    results/extra_draw      무승부 규칙에서 학습·평가한 시드 0–4 (+ BT·BTO 비교)
                  results/extra_draw_hp   같은 300세대 모델을 본실험 규칙에서 평가
  E4 학습 규모 확대  results/extra_big       개체군 64 × 후보당 16교전, 시드 0–2, 100·200·300세대

평가가 끝나지 않은 실험은 건너뛰고 문서에 "평가 전"으로 적는다.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
from scipy import stats

from .paper import md_table
from .stats import holm

BASE_V3 = {"BT-v2": 0.843, "BTO": 0.882}      # 본실험: BT-v3 상대 최고 규칙 모델


# ------------------------------------------------------------ 공통
def cond_table(outdir: str) -> pd.DataFrame:
    """조건별 상대 점수, 설명 비용, 종료 사유."""
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    xai = pd.read_csv(os.path.join(outdir, "xai.csv")).set_index("cond")
    sc = ep.pivot_table(index="cond", columns="opponent", values="score", aggfunc="mean")
    out = pd.DataFrame({"v2": sc.get("BT-v2"), "v3": sc.get("BT-v3")})
    out["avg"] = out[["v2", "v3"]].mean(axis=1)
    g = ep.groupby("cond")
    out["kill_win"] = g.apply(lambda d: ((d.outcome == "gun_kill") & (d.win == 1)).mean(),
                              include_groups=False)
    out["timeout"] = g.apply(lambda d: (d.outcome == "timeout").mean(), include_groups=False)
    out["wez"] = g["wez_time"].mean()
    out["viol"] = g.apply(lambda d: (d.viol_deck + d.viol_over_g + d.viol_sep).mean(),
                          include_groups=False)
    out["d95"] = xai["d_star_095"]
    out["d90"] = xai["d_star_090"]
    out["f4"] = xai["fidelity_at_4"]
    return out


def per_init(outdir: str) -> pd.DataFrame:
    """(조건) x (상대, 초기조건) 점수. 진영 교대 두 판을 평균한 대응 단위."""
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    return ep.groupby(["cond", "opponent", "seed"]).score.mean().unstack(["opponent", "seed"])


def paired_p(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).values
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="zsplit").pvalue)


def _parse(cond: str) -> tuple[str, int, int]:
    m = re.match(r"([A-Z]+)-s(\d+)-b(\d+)", cond)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (cond, -1, -1)


def _done(outdir: str) -> bool:
    return all(os.path.exists(os.path.join(outdir, f)) for f in ("episodes.csv", "xai.csv"))


FMT = {"v2": "{:.3f}", "v3": "{:.3f}", "avg": "{:.3f}", "d95": "{:g}", "d90": "{:g}",
       "f4": "{:.3f}", "kill_win": "{:.2f}", "timeout": "{:.2f}", "wez": "{:.2f}", "viol": "{:.3f}"}
KO = {"v2": "BT-v2 상대", "v3": "BT-v3 상대", "avg": "상대 2종 평균", "d95": "D*95", "d90": "D*90",
      "f4": "F(4)", "kill_win": "격추승 비율", "timeout": "시간종료 비율", "wez": "WEZ(초)", "viol": "위반율"}


def _md(df: pd.DataFrame, cols: list[str], first: str | None = None) -> str:
    d = df[cols].rename(columns=KO)
    d.index.name = first or "조건"
    return md_table(d.reset_index(), {KO[c]: FMT[c] for c in cols if c in FMT})


# ------------------------------------------------------------ E1
def analyse_e1(outdir="results/extra_long", base_gen: int = 300) -> tuple[str, dict]:
    t = cond_table(outdir)
    t = t[[c.startswith("RL-") for c in t.index]].copy()
    t["seed"] = [_parse(c)[1] for c in t.index]
    t["gen"] = [_parse(c)[2] for c in t.index]
    pi = per_init(outdir)
    seeds = [1, 11, 14, 2, 18]
    t = t[t.seed.isin(seeds)]
    gens = sorted(t.gen.unique())
    g_last = max(gens)
    rows = []
    for s in seeds:
        a = t[(t.seed == s) & (t.gen == base_gen)].iloc[0]
        b = t[(t.seed == s) & (t.gen == g_last)].iloc[0]
        ca, cb = f"RL-s{s}-b{base_gen}-a1.00", f"RL-s{s}-b{g_last}-a1.00"
        rows.append(dict(seed=s, s300=a.avg, s_last=b.avg, dS=b.avg - a.avg,
                         v2_300=a.v2, v2_last=b.v2, v3_300=a.v3, v3_last=b.v3,
                         d300=a.d95, d_last=b.d95, dD=b.d95 - a.d95,
                         d90_300=a.d90, d90_last=b.d90,
                         kw300=a.kill_win, kw_last=b.kill_win,
                         p=paired_p(pi.loc[cb], pi.loc[ca])))
    r = pd.DataFrame(rows)
    dS, dD = float(r.dS.mean()), float(r.dD.mean())
    n_up = int((r.dS > 0).sum())
    sub = t[t.gen >= base_gen]
    rho, p_rho = stats.spearmanr(sub.gen, sub.d95)
    rho90, p90 = stats.spearmanr(sub.gen, sub.d90)
    rho_s, p_s = stats.spearmanr(sub.gen, sub.avg)
    if dS < 0.03:
        perf = "정체"
    elif dS >= 0.05 and n_up >= 4:
        perf = "뚜렷한 향상"
    else:
        perf = "소폭 향상"
    dj = "증가" if (dD >= 2 and rho > 0 and p_rho < 0.05) else "증가 없음"
    last = t[t.gen == g_last]
    allck = t[t.gen > base_gen]
    res = dict(dS=dS, dD=dD, n_up=n_up, rho=float(rho), p_rho=float(p_rho), perf=perf, dj=dj,
               g_last=int(g_last), rho90=float(rho90), p90=float(p90), rho_s=float(rho_s), p_s=float(p_s),
               n_v3_bt2=int((last.v3 > BASE_V3["BT-v2"]).sum()),
               n_v3_bto=int((last.v3 > BASE_V3["BTO"]).sum()),
               max_v3=float(allck.v3.max()), max_v3_cond=str(allck.v3.idxmax()),
               mean_by_gen=t.groupby("gen")[["v2", "v3", "avg", "d95", "d90", "f4", "kill_win"]].mean())

    out = ["## E1. 학습 연장 (%d → %d세대)\n" % (base_gen, g_last)]
    out.append(f"**사전 판정: 성능 {perf}, 설명 비용 {dj}.** "
               f"ΔS = {dS:+.3f} (5개 중 {n_up}개 시드에서 향상), ΔD = {dD:+.1f}, "
               f"세대–D\\*95 Spearman ρ = {rho:.2f} (p = {p_rho:.3f}, {len(sub)}개 점).\n")
    out.append("보조: 세대–점수 ρ = %.2f (p = %.3f), 세대–D\\*90 ρ = %.2f (p = %.3f). "
               "%d세대 모델 가운데 BT-v3 상대 점수가 BT-v2(0.843)를 넘은 모델 %d개, BTO(0.882)를 넘은 모델 %d개. "
               "%d세대보다 뒤의 모든 체크포인트 가운데 BT-v3 상대 최고 점수는 %.3f(%s).\n"
               % (rho_s, p_s, rho90, p90, g_last, res["n_v3_bt2"], res["n_v3_bto"],
                  base_gen, res["max_v3"], res["max_v3_cond"]))
    out.append("### 표 E1-1. 시드별 %d세대 대 %d세대\n" % (base_gen, g_last))
    r2 = r.copy()
    r2["seed"] = r2.seed.astype(str)
    b0 = base_gen
    r2.columns = ["학습 시드", f"S({b0})", f"S({g_last})", "ΔS", f"BT-v2 {b0}", f"BT-v2 {g_last}",
                  f"BT-v3 {b0}", f"BT-v3 {g_last}", f"D*95 {b0}", f"D*95 {g_last}", "ΔD",
                  f"D*90 {b0}", f"D*90 {g_last}", f"격추승 {b0}", f"격추승 {g_last}", "p(대응)"]
    out.append(md_table(r2, {"p(대응)": "{:.3f}", f"D*95 {b0}": "{:.0f}", f"D*95 {g_last}": "{:.0f}",
                             "ΔD": "{:+.0f}", f"D*90 {b0}": "{:.0f}", f"D*90 {g_last}": "{:.0f}",
                             "ΔS": "{:+.3f}", f"격추승 {b0}": "{:.2f}", f"격추승 {g_last}": "{:.2f}"}))
    out.append("\nS = 상대 2종 평균 점수. p(대응) = 같은 시드의 두 체크포인트를 (상대, 초기조건) 200단위로 짝지은 Wilcoxon 부호순위 검정.\n")
    out.append("### 표 E1-2. 세대별 5개 시드 평균\n")
    mg = res["mean_by_gen"].copy()
    mg.index = mg.index.astype(str)
    mg[["d95", "d90"]] = mg[["d95", "d90"]].round(1)
    out.append(_md(mg, ["v2", "v3", "avg", "d95", "d90", "f4", "kill_win"], "세대"))
    out.append("\n### 표 E1-3. 시드 × 세대 (상대 2종 평균 점수 / D\\*95)\n")
    piv_s = t.pivot_table(index="gen", columns="seed", values="avg")[seeds]
    piv_d = t.pivot_table(index="gen", columns="seed", values="d95")[seeds]
    cell = piv_s.copy().astype(object)
    for g in piv_s.index:
        for s in seeds:
            cell.loc[g, s] = f"{piv_s.loc[g, s]:.3f} / {piv_d.loc[g, s]:.0f}"
    cell.columns = [f"시드 {s}" for s in seeds]
    out.append(md_table(cell.reset_index().rename(columns={"gen": "세대"})))
    return "\n".join(out) + "\n", res


# ------------------------------------------------------------ E2
def mixing_diag(ckpt: str, kind: str, n_init: int = 30, seed0: int = 10_000) -> tuple[float, float]:
    """평가 초기조건(청군)에서 게이트 g 또는 잔차 |δ| 의 평균과 교전 안 표준편차의 평균."""
    from ..env import DogfightEnv
    from ..agents.bt import BTPolicy
    if kind == "gating":
        from ..agents.gating import GatingPolicy as P
    else:
        from ..agents.residual import ResidualPolicy as P
    pol = P.load(ckpt)
    env = DogfightEnv()
    means, sds = [], []
    for ov in (2, 3):
        for i in range(n_init):
            red = BTPolicy(version=ov)
            ob, orr = env.reset(seed=seed0 + i, alpha=1.0)
            v = []
            for _ in range(env.max_steps + 1):
                a = pol.act(ob)
                v.append(float(pol.last_gate) if kind == "gating" else float(np.abs(pol.last_delta).mean()))
                ob, orr, done = env.step(a, red.act(orr))
                if done:
                    break
            means.append(np.mean(v)); sds.append(np.std(v))
    return float(np.mean(means)), float(np.mean(sds))


def analyse_e2(outdir="results/extra_hyb", diag=True) -> tuple[str, dict]:
    t = cond_table(outdir)
    pi = per_init(outdir)
    parents = {"BT-v2": "BT-v2", "학습 시드 11": "RL-s11-b300-a1.00"}
    out = ["## E2. 하이브리드 보강 (잔차형·게이팅형, 시드 5개 × 300세대)\n"]
    res = {}
    for prefix, kind, name in (("RES", "residual", "잔차형"), ("GATE", "gating", "게이팅형")):
        rows = []
        for c in sorted([c for c in t.index if c.startswith(prefix + "-")],
                        key=lambda c: (_parse(c)[2], _parse(c)[1])):
            _, s, g = _parse(c)
            row = dict(cond=c, seed=s, gen=g, **{k: t.loc[c, k] for k in ("v2", "v3", "avg", "d95", "d90", "f4")})
            comp = []
            for pname, pc in parents.items():
                diff = t.loc[c, "avg"] - t.loc[pc, "avg"]
                p = paired_p(pi.loc[c], pi.loc[pc])
                dom = diff > 0 and p < 0.05 and t.loc[c, "d95"] <= t.loc[pc, "d95"]
                row[f"Δ vs {pname}"] = diff
                row[f"p vs {pname}"] = p
                if dom:
                    comp.append(pname)
            row["절충점"] = ", ".join(comp) if comp else "아님"
            rows.append(row)
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        last = r[r.gen == r.gen.max()]
        n_comp = int((last["절충점"] != "아님").sum())
        verdict = "절충점이 될 수 있다" if n_comp >= 3 else "절충점이 관측되지 않음(기존 판정 유지)"
        res[kind] = dict(n_comp=n_comp, verdict=verdict, last=last, all=r,
                         mean_last=last[["v2", "v3", "avg", "d95", "d90"]].mean())
        out.append(f"### {name}\n")
        out.append(f"**사전 판정: {verdict}.** {int(last.gen.max())}세대 시드 5개 중 원 모델 하나를 두 축에서 동시에 "
                   f"앞선 시드 {n_comp}개. {int(last.gen.max())}세대 평균 점수 {last.avg.mean():.3f} "
                   f"(범위 {last.avg.min():.3f}–{last.avg.max():.3f}), BT-v2 상대 {last.v2.mean():.3f}, "
                   f"BT-v3 상대 {last.v3.mean():.3f}, D\\*95 {', '.join(str(int(x)) for x in last.d95)}.\n")
        if diag:
            dg = []
            for _, x in last.iterrows():
                d = "results/es_res" if kind == "residual" else "results/es_gate"
                ck = os.path.join(d, f"ckpt_seed{x.seed}_gen{x.gen:05d}.npz")
                if os.path.exists(ck):
                    m, sd = mixing_diag(ck, kind)
                    dg.append(f"시드 {x.seed} {m:.2f}(교전 안 표준편차 {sd:.3f})")
            lab = "게이트 값 g" if kind == "gating" else "잔차 |δ|(정규화 단위)"
            out.append(f"평가 교전(청군, 상대별 초기조건 30개)에서 {lab}: " + "; ".join(dg) + ".\n")
        rr = r.drop(columns=["seed", "gen"]).set_index("cond")
        fmt = {"v2": "{:.3f}", "v3": "{:.3f}", "avg": "{:.3f}", "d95": "{:.0f}", "d90": "{:.0f}", "f4": "{:.3f}"}
        for pname in parents:
            fmt[f"Δ vs {pname}"] = "{:+.3f}"; fmt[f"p vs {pname}"] = "{:.3f}"
        out.append(md_table(rr.rename(columns=KO).reset_index(),
                            {KO.get(k, k): v for k, v in fmt.items()}))
        out.append("")
    pr = t.loc[list(parents.values()), ["v2", "v3", "avg", "d95", "d90", "f4"]]
    out.append("원 모델(같은 실행):\n")
    out.append(_md(pr, ["v2", "v3", "avg", "d95", "d90", "f4"], "조건"))
    out.append("\n절충점 = 상대 2종 평균 점수가 원 모델보다 높고(초기조건 단위 Wilcoxon p < 0.05) D\\*95가 그 원 모델 이하. "
               "BT-v2의 BT-v2 상대 점수는 자기 대전(0.500)이다.\n")
    return "\n".join(out) + "\n", res


# ------------------------------------------------------------ E3
def analyse_e3(outdir="results/extra_draw", hp_dir="results/extra_draw_hp",
               base_dir="results/sens_draw") -> tuple[str, dict]:
    t = cond_table(outdir)
    rl = t[[c.startswith("RL-") for c in t.index]].copy()
    rl["seed"] = [_parse(c)[1] for c in rl.index]
    rl["gen"] = [_parse(c)[2] for c in rl.index]
    g_last = rl.gen.max()
    last = rl[rl.gen == g_last].sort_values("seed")
    ep = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    bt = []
    for c in last.index:
        e = ep[(ep.cond == c) & (ep.opponent == "BT-v2")]
        w, l = int(e.win.sum()), int(e.loss.sum())
        p = float(stats.binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0
        bt.append(dict(cond=c, wins=w, losses=l, draws=int(e.draw.sum()), p=p))
    bt = pd.DataFrame(bt)
    bt["p_holm"], _ = holm(bt.p.values)
    bt["sig_win"] = (bt.p_holm < 0.05) & (bt.wins > bt.losses)
    n_sig = int(bt.sig_win.sum())
    verdict = ("규칙 허점 없이도 학습 상대를 이긴다" if n_sig >= 3 else "규칙 허점 없이는 학습 상대를 이기지 못한다")
    res = dict(n_sig=n_sig, verdict=verdict, last=last, bt=bt)

    base = cond_table(base_dir) if _done(base_dir) else None
    out = ["## E3. \"격추만 승리\" 규칙 재학습 (시드 5개 × %d세대)\n" % g_last]
    out.append(f"**사전 판정: {verdict}.** 무승부 규칙에서 BT-v2 상대 점수가 0.5보다 유의하게 높은 시드 "
               f"{n_sig}개/5개(승·패 이항 검정, Holm 보정). {g_last}세대 평균: BT-v2 상대 {last.v2.mean():.3f}, "
               f"BT-v3 상대 {last.v3.mean():.3f}, D\\*95 중앙값 {last.d95.median():.0f} "
               f"(범위 {last.d95.min():.0f}–{last.d95.max():.0f}), 격추승 비율 {last.kill_win.mean():.2f}.\n")
    if base is not None:
        b_rl = base[[c.startswith("RL-") for c in base.index]]
        b_bto = base[[c.startswith("BTO-") for c in base.index]]
        out.append(f"비교(무승부 규칙 재평가, 다시 학습하지 않음): 본실험 300세대 학습 모델 20개 평균 BT-v2 상대 "
                   f"{b_rl.v2.mean():.3f}, BT-v3 상대 {b_rl.v3.mean():.3f}, 격추승 비율 {b_rl.kill_win.mean():.2f}; "
                   f"BT-v2의 BT-v3 상대 {base.loc['BT-v2', 'v3']:.3f}; BTO 3개 평균 BT-v2 상대 {b_bto.v2.mean():.3f}, "
                   f"BT-v3 상대 {b_bto.v3.mean():.3f}.\n")
    out.append("### 표 E3-1. 무승부 규칙에서의 결과\n")
    show = t.copy()
    out.append(_md(show.sort_index(), ["v2", "v3", "avg", "d95", "d90", "f4", "kill_win", "timeout"], "조건"))
    out.append("\n### 표 E3-2. %d세대 모델의 BT-v2 상대 승·패 (무승부 규칙)\n" % g_last)
    out.append(md_table(bt, {"p": "{:.4f}", "p_holm": "{:.4f}"}))
    if _done(hp_dir):
        h = cond_table(hp_dir)
        res["hp"] = h
        out.append("\n### 표 E3-3. 같은 %d세대 모델을 본실험 규칙(시간종료 체력 비교)에서 평가\n" % g_last)
        out.append(_md(h, ["v2", "v3", "avg", "d95", "d90", "kill_win", "timeout"], "조건"))
    return "\n".join(out) + "\n", res


# ------------------------------------------------------------ E4
def analyse_e4(outdir="results/extra_big", main_dir="results/main") -> tuple[str, dict]:
    t = cond_table(outdir)
    t = t[[c.startswith("RL-") for c in t.index]].copy()
    t["seed"] = [_parse(c)[1] for c in t.index]
    t["gen"] = [_parse(c)[2] for c in t.index]
    g_last = t.gen.max()
    last = t[t.gen == g_last].sort_values("seed")
    base = cond_table(main_dir)
    base = base[[bool(re.match(r"RL-s\d+-b300-a1\.00$", c)) for c in base.index]]
    b_mean = float(base.avg.mean())
    m = float(last.avg.mean())
    if m >= b_mean + 0.05 and bool((last.avg > b_mean).all()):
        perf = "규모 확대로 성능이 오른다"
    elif m < b_mean + 0.02:
        perf = "오르지 않는다"
    else:
        perf = "소폭"
    q75 = float(np.percentile(base.d95, 75))
    high_d = perf == "규모 확대로 성능이 오른다" and bool((last.d95 >= q75).all())
    res = dict(perf=perf, mean=m, base=b_mean, last=last, high_d=high_d, q75=q75)
    out = ["## E4. 학습 규모 확대 (개체군 64 × 후보당 16교전, 시드 3개 × %d세대)\n" % g_last]
    out.append(f"**사전 판정: {perf}.** {g_last}세대 3개 시드 상대 2종 평균 {m:.3f} "
               f"(시드별 {', '.join(f'{x:.3f}' for x in last.avg)}), 본실험 300세대 20개 평균 {b_mean:.3f}. "
               f"BT-v2 상대 {last.v2.mean():.3f}(본실험 {base.v2.mean():.3f}), BT-v3 상대 {last.v3.mean():.3f}"
               f"(본실험 {base.v3.mean():.3f}). D\\*95 {', '.join(str(int(x)) for x in last.d95)} "
               f"(본실험 중앙값 {base.d95.median():.0f}, 상위 사분위 {q75:.0f})"
               + ("; 고성능 영역에서 설명 비용이 커질 가능성으로 보고한다." if high_d else ".") + "\n")
    out.append(_md(t.sort_values(["seed", "gen"]), ["v2", "v3", "avg", "d95", "d90", "f4", "kill_win", "timeout"], "조건"))
    return "\n".join(out) + "\n", res


def main():
    ap = argparse.ArgumentParser(description="추가 실험 분석")
    ap.add_argument("--out", default="paper/results_extra.md")
    ap.add_argument("--no-diag", action="store_true", help="게이트·잔차 진단 교전을 건너뜀")
    a = ap.parse_args()
    parts = ["# 추가 실험 결과 (자동 생성)\n",
             "판정 기준: `paper/추가실험_사전기준.md`(학습 전에 기록). 생성: `python -m dfxai.analysis.extra`.\n"]
    summary = {}
    for name, fn, d in (("E1", analyse_e1, "results/extra_long"),
                        ("E2", lambda: analyse_e2(diag=not a.no_diag), "results/extra_hyb"),
                        ("E3", analyse_e3, "results/extra_draw"),
                        ("E4", analyse_e4, "results/extra_big")):
        if not _done(d):
            parts.append(f"## {name}\n\n평가 전.\n")
            continue
        txt, res = fn()
        parts.append(txt)
        summary[name] = res
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"저장: {a.out}")
    for k, v in summary.items():
        if k == "E1":
            print(k, v["perf"], v["dj"], f"dS={v['dS']:+.3f} dD={v['dD']:+.1f} rho={v['rho']:.2f} p={v['p_rho']:.3f}")
        elif k == "E2":
            for kind, r in v.items():
                print(k, kind, r["verdict"], r["n_comp"])
        elif k == "E3":
            print(k, v["verdict"], v["n_sig"])
        else:
            print(k, v["perf"], f"{v['mean']:.3f} vs {v['base']:.3f}")


if __name__ == "__main__":
    main()
