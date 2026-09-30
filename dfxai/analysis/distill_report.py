"""대리 트리 정책 성능 확인(R10) 결과를 Markdown 으로 정리한다. 측정은 distill_perf.py."""
from __future__ import annotations
import json, os, re
import numpy as np
import pandas as pd
from scipy import stats

from .paper import md_table
from .distill_perf import GRID, MARGIN, CHASE
from .tactical_report import group_of, ROWS


def _fmt_delta(d: pd.Series) -> str:
    d = d.dropna()
    if d.empty:
        return "–"
    k = int((d >= -MARGIN).sum())
    return f"{d.median():+.2f} ({k}/{len(d)})"


def load(outdir: str):
    g = pd.read_csv(os.path.join(outdir, "games.csv"))
    f = pd.read_csv(os.path.join(outdir, "fits.csv"))
    c = pd.read_csv(os.path.join(outdir, "conds.csv"))
    man = json.load(open(os.path.join(outdir, "manifest.json"), encoding="utf-8"))
    return g, f, c, man


def paired(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """시드 단위(4교전 평균)로 원본과 대응시킨 트리 정책의 점수 차."""
    ps = games.groupby(["cond", "depth", "seed"]).score.mean().reset_index()
    orig = ps[ps.depth == 0].drop(columns="depth").rename(columns={"score": "s0"})
    tr = ps[ps.depth > 0].merge(orig, on=["cond", "seed"])
    tr["delta"] = tr["score"] - tr["s0"]
    a = tr.groupby(["cond", "depth"]).agg(score=("score", "mean"), delta=("delta", "mean"),
                                          sd=("delta", "std"), n=("delta", "size")).reset_index()
    a["se"] = a["sd"] / np.sqrt(a["n"])
    a["lo"], a["hi"] = a["delta"] - 1.96 * a["se"], a["delta"] + 1.96 * a["se"]
    return a, orig.groupby("cond").s0.mean()


def d_perf(a: pd.DataFrame) -> pd.Series:
    out = {}
    for c, s in a.groupby("cond"):
        s = s.sort_values("depth")
        ok = s[s["delta"] >= -MARGIN]
        out[c] = float(ok["depth"].iloc[0]) if len(ok) else np.nan
    return pd.Series(out)


def build(outdir: str, out_path: str = "paper/results_distill.md") -> str:
    games, fits, conds, man = load(outdir)
    a, s0 = paired(games)
    a = a.merge(fits, on=["cond", "depth"]).merge(conds[["cond", "d_star"]], on="cond")
    a["group"] = a["cond"].map(group_of)
    dp = d_perf(a)
    dstar = conds.set_index("cond")["d_star"]
    n_seeds = man["n_seeds"]

    out = ["# 대리 트리를 정책으로 돌렸을 때의 성능 (자동 생성, R10)\n",
           f"조건 {len(man['conditions'])}개, 새 시드 {n_seeds}개(시작 {man['seed0']}) × 상대 BT-v2·BT-v3 × 진영 교대 = 조건당 {4 * n_seeds}교전. "
           "트리는 각 정책의 본실험 로그(`results/main/traj_*.npz`)에 적합했고, 원본 정책도 같은 새 시드에서 다시 평가했다. "
           "Δ 는 트리 정책 점수 − 원본 점수를 시드 단위(시드마다 4교전 평균)로 대응시킨 평균이다. "
           f"허용 한계는 −{MARGIN:.2f}(같은 예산으로 독립 학습한 정책 20개의 점수 표준편차 0.051 에서 정함)이며, "
           "Δ ≥ −0.05 인 최소 깊이를 d_perf 라 한다. 셀의 괄호는 (Δ ≥ −0.05 인 정책 수 / 정책 수)다. "
           "D\\* 는 본실험의 충실도 기준 최소 깊이(깊이 16 까지, 17 은 절단)다.\n"]

    # ---------------- 표 1: 묶음 × 깊이
    rows = []
    for g in ROWS:
        sub = a[a.group == g]
        if sub.empty:
            continue
        r = {"정책 묶음": f"{g} (n={sub.cond.nunique()})", "원본 점수": float(s0[sub.cond.unique()].median())}
        for d in GRID:
            r[f"깊이 {d}"] = _fmt_delta(sub[sub.depth == d].set_index("cond")["delta"])
        rows.append(r)
    out.append("### 표 R10-1. 깊이별 성능 차 Δ — 묶음별 중앙값(Δ ≥ −0.05 인 정책 수/전체)\n")
    out.append(md_table(pd.DataFrame(rows), {"원본 점수": "{:.3f}"}) + "\n")

    # ---------------- 표 2: 학습 정책 20개
    rl = a[a.group.isin(["학습(치고 빠지기)", "학습(추격)"])]
    rows = []
    for c in sorted(rl.cond.unique(), key=lambda x: (dstar[x], x)):
        s = rl[rl.cond == c].set_index("depth")
        dd = int(min(16, dstar[c]))
        seed = re.search(r"-s(\d+)-", c).group(1)
        dpv = dp[c]
        r = {"학습 시드": f"s{seed}", "전술": "추격" if f"s{seed}" in CHASE else "치고 빠지기", "D*": int(dstar[c]),
             "원본 점수": float(s0[c]), "충실도 R²(새 교전, 깊이 D*)": float(s.loc[dd, "r2_fresh"]),
             "Δ(깊이 D*)": float(s.loc[dd, "delta"]), "Δ(4)": float(s.loc[4, "delta"]), "Δ(8)": float(s.loc[8, "delta"]),
             "Δ(16)": float(s.loc[16, "delta"]),
             "d_perf": "없음(>16)" if np.isnan(dpv) else int(dpv),
             "d_perf 의 잎 수": "–" if np.isnan(dpv) else int(s.loc[int(dpv), "n_leaves"]),
             "깊이 16 의 잎 수": int(s.loc[16, "n_leaves"])}
        rows.append(r)
    out.append("### 표 R10-2. 학습 정책 20개(300세대)의 개별 결과 (D\\* 순)\n")
    out.append(md_table(pd.DataFrame(rows), {"원본 점수": "{:.3f}", "충실도 R²(새 교전, 깊이 D*)": "{:.3f}",
                                              "Δ(깊이 D*)": "{:+.3f}", "Δ(4)": "{:+.3f}", "Δ(8)": "{:+.3f}", "Δ(16)": "{:+.3f}"}) + "\n")

    # ---------------- 표 3: 충실도와 성능
    base = a[a.group.isin(["학습(치고 빠지기)", "학습(추격)"])].copy()
    bins = [(-np.inf, 0.90, "< 0.90"), (0.90, 0.95, "0.90–0.95"), (0.95, 0.97, "0.95–0.97"), (0.97, np.inf, "≥ 0.97")]
    rows = []
    for lo, hi, lab in bins:
        s = base[(base.r2_fresh >= lo) & (base.r2_fresh < hi)]
        if len(s):
            rows.append({"새 교전 충실도 R²": lab, "정책×깊이 쌍": len(s), "Δ 중앙값": float(s.delta.median()),
                         "Δ ≥ −0.05 비율": float((s.delta >= -MARGIN).mean())})
    out.append("### 표 R10-3. 충실도가 높으면 성능도 회복되는가 — 학습 정책 20개 × 깊이 8–9개\n")
    out.append("충실도는 트리 학습에 쓰지 않은 새 시드의 원본 교전(청군)에서 잰 분산 가중 R² 다.\n\n")
    out.append(md_table(pd.DataFrame(rows), {"Δ 중앙값": "{:+.3f}", "Δ ≥ −0.05 비율": "{:.2f}"}) + "\n")
    r, p = stats.spearmanr(base.r2_fresh, base.delta)
    out.append(f"\n학습 정책 전체의 (충실도, Δ) 순위상관 ρ={r:.2f} (n={len(base)}쌍; 같은 정책의 여러 깊이를 독립 표본으로 "
               f"본 값이라 p={p:.1e} 는 실제보다 작게 나온다 — 독립 단위는 정책 20개). ")
    d16 = base[base.depth == 16]
    out.append(f"깊이 16 트리의 새 교전 충실도 중앙값 {d16.r2_fresh.median():.3f}, Δ 중앙값 {d16.delta.median():+.3f}.\n")
    ok = dp[[c for c in dp.index if group_of(c) in ("학습(치고 빠지기)", "학습(추격)")]]
    dd = dstar[ok.index]
    dpv = ok.dropna()
    if len(dpv) >= 5:
        r2, p2 = stats.spearmanr(dstar[dpv.index], dpv)
        out.append(f"D\\* 와 d_perf 의 순위상관(d_perf 가 있는 {len(dpv)}개) ρ={r2:.2f} (p={p2:.3f}). ")
    out.append(f"d_perf 가 깊이 16 이하에 없는 정책 {int(ok.isna().sum())}/{len(ok)}개; d_perf ≤ D\\* 인 정책 {int((ok <= np.minimum(dd, 16)).sum())}개.\n")

    # ---------------- 표 4: 잎 수
    lv = fits.merge(conds[["cond"]], on="cond")
    lv["group"] = lv["cond"].map(group_of)
    rows = []
    for g in ("BT-v2", "BTO", "복제본", "학습(치고 빠지기)", "학습(추격)", "혼합 α=0.5"):
        sub = lv[lv.group == g]
        if sub.empty:
            continue
        r = {"정책 묶음": g}
        for d in GRID:
            r[f"깊이 {d}"] = int(sub[sub.depth == d].n_leaves.median())
        rows.append(r)
    out.append("\n### 표 R10-4. 트리 잎 수(중앙값) — 깊이가 같아도 정책에 따라 트리 크기가 다르다\n")
    out.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---------------- 표 5: 구성 요소 (점수 낮아지는 방식)
    tr16 = games[(games.depth == 16)]
    orig = games[games.depth == 0]
    rows = []
    for g in ("BT-v2", "학습(치고 빠지기)", "학습(추격)"):
        cs = [c for c in games.cond.unique() if group_of(c) == g]
        for lab, dfx in (("원본", orig), ("깊이 16 트리", tr16)):
            x = dfx[dfx.cond.isin(cs)]
            rows.append({"정책 묶음": g, "정책": lab, "점수": x.score.mean(), "시간종료": (x.outcome == "timeout").mean(),
                         "격추로 끝남": (x.outcome == "gun_kill").mean(), "지면충돌": (x.outcome == "crash").mean(),
                         "공중충돌": (x.outcome == "collision").mean(), "위반율": x.viol.mean()})
    out.append("\n### 표 R10-5. 종료 사유와 위반율: 원본 대 깊이 16 트리\n")
    out.append(md_table(pd.DataFrame(rows), {k: "{:.3f}" for k in ("점수", "시간종료", "격추로 끝남", "지면충돌", "공중충돌", "위반율")}) + "\n")

    out.extend(_dagger_sections(outdir, games))
    out.extend(_noise_sections(outdir, games))
    out.extend(_region_sections(outdir))
    doc = "\n".join(out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(doc)
    a.to_csv(os.path.join(outdir, "retention.csv"), index=False)
    return out_path


def _seed_scores(g: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return g.groupby(keys + ["seed"]).score.mean().reset_index()


def _dagger_sections(outdir: str, games: pd.DataFrame) -> list[str]:
    gp, fp = os.path.join(outdir, "dagger_games.csv"), os.path.join(outdir, "dagger_fits.csv")
    if not (os.path.exists(gp) and os.path.exists(fp)):
        return ["\n> DAgger 보조 시험(`distill_dagger.py`)은 아직 실행되지 않았다.\n"]
    dg, df = pd.read_csv(gp), pd.read_csv(fp)
    man = json.load(open(os.path.join(outdir, "dagger_manifest.json"), encoding="utf-8"))
    orig = _seed_scores(games[games.depth == 0], ["cond"]).rename(columns={"score": "s0"})
    plain = _seed_scores(games[games.depth > 0], ["cond", "depth"])
    da = _seed_scores(dg, ["cond", "depth", "iteration"])
    d_plain = plain.merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0).groupby(["cond", "depth"]).delta.mean()
    d_it = da.merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0).groupby(["cond", "depth", "iteration"]).delta.mean()
    depths = man["depths"]
    out = ["\n### 표 R10-6. DAgger 보조 시험: 트리가 방문한 상태를 원본에게 물어 데이터에 더한 뒤 다시 적합 (분포 이동 확인)\n",
           f"반복 {man['n_iter']}회, 반복마다 학습용 시드 {man['n_train']}개 × 상대 2종(청군)에서 트리를 돌려 방문 상태를 모으고 원본 정책으로 정답을 붙였다. "
           "평가는 R10 과 같은 새 시드다. 셀은 Δ 중앙값(Δ ≥ −0.05 인 정책 수/전체)이며 '기본'은 R10-1 의 트리(로그만 사용)다.\n"]
    rows = []
    for g in ("BT-v2", "복제본", "학습(치고 빠지기)", "학습(추격)"):
        conds = [c for c in d_plain.index.get_level_values(0).unique() if group_of(c) == g and c in set(dg.cond)]
        if not conds:
            continue
        for d in depths:
            r = {"정책 묶음": f"{g} (n={len(conds)})", "깊이": d, "기본": _fmt_delta(pd.Series({c: d_plain[(c, d)] for c in conds}))}
            for it in range(1, man["n_iter"] + 1):
                r[f"DAgger {it}회"] = _fmt_delta(pd.Series({c: d_it[(c, d, it)] for c in conds}))
            rows.append(r)
    out.append(md_table(pd.DataFrame(rows)) + "\n")
    rows = []
    df["group"] = df["cond"].map(group_of)
    for g in ("BT-v2", "복제본", "학습(치고 빠지기)", "학습(추격)"):
        for d in depths:
            sub = df[(df.group == g) & (df.depth == d)]
            if sub.empty:
                continue
            r = {"정책 묶음": g, "깊이": d}
            for it in range(1, man["n_iter"] + 1):
                r[f"{it}회차 시작 시 R²"] = float(sub[sub.iteration == it].r2_on_policy_before.median())
            r["최종 잎 수"] = int(sub[sub.iteration == man["n_iter"]].n_leaves.median())
            rows.append(r)
    out.append("**트리가 방문한 상태에서 원본 지령을 재현한 R²** (각 회차에서 트리를 갱신하기 전, 정책 묶음 중앙값). 원본이 방문한 상태에서의 충실도(R10-2 의 새 교전 R²)와 비교하면 분포 이동의 크기가 보인다.\n")
    out.append(md_table(pd.DataFrame(rows), {f"{i}회차 시작 시 R²": "{:.3f}" for i in range(1, man["n_iter"] + 1)}) + "\n")

    # 학습 정책 20개 요약: 마지막 회차의 Δ 가 허용 한계 안인가, 원본보다 유의하게 낮은가
    last = man["n_iter"]
    lp = [c for c in d_plain.index.get_level_values(0).unique() if group_of(c) in ("학습(치고 빠지기)", "학습(추격)") and c in set(dg.cond)]
    conds_tbl = pd.read_csv(os.path.join(outdir, "conds.csv")).set_index("cond")
    ds = da[da.iteration == last].merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0)
    rows, ok_depth = [], {}
    for d in depths:
        x = ds[(ds.depth == d) & ds.cond.isin(lp)].groupby("cond").delta.agg(["mean", "std", "size"])
        hi = x["mean"] + 1.96 * x["std"] / np.sqrt(x["size"])
        base = pd.Series({c: d_plain[(c, d)] for c in x.index})
        rows.append({"깊이": d, "잎 수(중앙값)": int(df[(df.depth == d) & (df.iteration == last) & df.cond.isin(lp)].n_leaves.median()),
                     "기본 Δ 중앙값": float(base.median()), f"DAgger {last}회 Δ 중앙값": float(x["mean"].median()),
                     "Δ ≥ −0.05 정책 수": int((x["mean"] >= -MARGIN).sum()), "Δ 95% 상한 < 0 정책 수": int((hi < 0).sum()),
                     "기본보다 Δ 가 커진 정책 수": int(((x["mean"] - base.loc[x.index]) > 0).sum())})
        for c in x.index[x["mean"] >= -MARGIN]:
            ok_depth[c] = min(ok_depth.get(c, 99), d)
    out.append(f"**학습 정책 {len(lp)}개 요약 (DAgger {last}회 후)**. Δ 95% 상한은 시드 100개 대응 차이의 정규근사 구간이다.\n")
    out.append(md_table(pd.DataFrame(rows), {"깊이": "{:.0f}", "잎 수(중앙값)": "{:.0f}", "기본 Δ 중앙값": "{:+.3f}",
                                             f"DAgger {last}회 Δ 중앙값": "{:+.3f}", "Δ ≥ −0.05 정책 수": "{:.0f}",
                                             "Δ 95% 상한 < 0 정책 수": "{:.0f}", "기본보다 Δ 가 커진 정책 수": "{:.0f}"}) + "\n")
    dd_ok = pd.Series({c: ok_depth.get(c, np.nan) for c in lp})
    cnt = {int(k): int(v) for k, v in dd_ok.value_counts().items()}
    line = f"허용 한계 안에 들어온 가장 얕은 깊이(격자 {tuple(depths)} 중): " + ", ".join(f"깊이 {k}: {v}개" for k, v in sorted(cnt.items())) + \
           f", 격자 안에 없음 {int(dd_ok.isna().sum())}개."
    v = dd_ok.dropna()
    if len(v) >= 5 and v.nunique() > 1:
        r_, p_ = stats.spearmanr(conds_tbl.loc[v.index, "d_star"], v)
        line += f" 이 깊이와 D\\* 의 순위상관 ρ={r_:+.2f} (p={p_:.2f}, n={len(v)}; 깊이가 격자 3점이라 검정력이 낮다)."
    out.append(line + "\n")
    return out


def _region_sections(outdir: str) -> list[str]:
    p = os.path.join(outdir, "region_error.csv")
    if not os.path.exists(p):
        return []
    r = pd.read_csv(p)
    r["group"] = r["cond"].map(group_of)
    regions = [("head_on", "정면 조우"), ("offensive", "공격"), ("defensive", "방어"), ("neutral", "중립")]
    out = ["\n### 표 R10-8. 트리 정책의 오차는 어느 전술 영역에서 나오는가\n",
           "트리를 정책으로 돌려 방문한 상태(트리 방문)와, 같은 시드에서 원본이 직접 싸워 방문한 상태(원본 방문) 각각에서 "
           "트리 지령과 원본 지령의 제곱오차를 4개 전술 영역으로 나눴다(상대 BT-v2·BT-v3, 시드 30개, 진영 0). "
           "셀은 `머문 시간 비율 / 오차 몫`의 묶음 중앙값이고 미설명 분산은 1 − R² 다. R² 의 분모는 각 상태 집합에서 원본 지령의 분산이라 "
           "두 행의 미설명 분산은 같은 기준의 절대 오차가 아니다. 영역별 몫과 머문 시간을 비교하는 용도로 읽는다.\n"]
    rows = []
    for g in ("학습(치고 빠지기)", "학습(추격)", "BT-v2", "복제본"):
        for d in sorted(r.depth.unique()):
            for st, lab in (("original-visited", "원본 방문"), ("tree-visited", "트리 방문")):
                x = r[(r.group == g) & (r.depth == d) & (r.states == st)]
                if x.empty:
                    continue
                row = {"정책 묶음": f"{g} (n={x.cond.nunique()})", "깊이": int(d), "상태": lab, "미설명 분산": float(x.unexplained.median())}
                for k, ko in regions:
                    row[ko] = f"{x['time_' + k].median():.2f} / {x['err_' + k].median():.2f}"
                rows.append(row)
    out.append(md_table(pd.DataFrame(rows), {"깊이": "{:.0f}", "미설명 분산": "{:.3f}"}) + "\n")
    return out


def _noise_sections(outdir: str, games: pd.DataFrame) -> list[str]:
    p = os.path.join(outdir, "noise_games.csv")
    if not os.path.exists(p):
        return []
    ng = pd.read_csv(p)
    man = json.load(open(os.path.join(outdir, "noise_manifest.json"), encoding="utf-8"))
    orig = _seed_scores(games[games.depth == 0], ["cond"]).rename(columns={"score": "s0"})
    plain = _seed_scores(games[games.depth > 0], ["cond", "depth"])
    sn = _seed_scores(ng, ["cond", "depth", "boot"])
    dn = sn.merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0).groupby(["cond", "depth", "boot"]).delta.mean().reset_index()
    dp = plain.merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0).groupby(["cond", "depth"]).delta.mean()
    rows = []
    for (c, d), s in dn.groupby(["cond", "depth"]):
        rows.append({"정책": c, "깊이": d, "R10-1 의 트리 Δ": float(dp[(c, d)]), "부트스트랩 트리 Δ 평균": float(s.delta.mean()),
                     "표준편차": float(s.delta.std()), "최소": float(s.delta.min()), "최대": float(s.delta.max())})
    out = ["\n### 표 R10-7. 트리 적합의 우연: 훈련 교전을 복원추출해 같은 깊이에서 트리를 다시 적합했을 때의 Δ\n",
           f"부트스트랩 {man['n_boot']}회, 평가 시드는 R10 과 같다. BT-v2 처럼 깊이에 따라 Δ 가 비단조로 움직이는 정도가 트리 적합의 우연으로 설명되는지 본다.\n"]
    out.append(md_table(pd.DataFrame(rows).sort_values(["정책", "깊이"]), {k: "{:+.3f}" for k in ("R10-1 의 트리 Δ", "부트스트랩 트리 Δ 평균", "최소", "최대")} | {"표준편차": "{:.3f}"}) + "\n")
    return out


# ------------------------------------------------------------------ 그림
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # 범주 팔레트 슬롯 1–4 (validate_palette.js 통과)
_FIG_GROUPS = (
    ("Rule-based (BT-v1/v2/v3, BTO)", "규칙 기반 (BT-v1/v2/v3, BTO)", ("BT-v1", "BT-v2", "BT-v3", "BTO")),
    ("Learned: hit-and-run", "학습: 치고 빠지기", ("학습(치고 빠지기)",)),
    ("Learned: chase", "학습: 추격", ("학습(추격)",)),
    ("Mixed (hybrid, Shield, residual, gating)", "혼합 (선형·Shield·잔차·게이팅)", ("혼합 α=0.5", "Shield", "잔차형", "게이팅형")),
)


def plot_retention(outdir: str, path: str, lang: str = "en") -> str:
    """깊이별 성능 차 Δ. 정책마다 옅은 선, 묶음 중앙값은 굵은 선. 0 이 원본 수준, 점선이 허용 한계."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .labels import T, set_language
    set_language(lang)
    a = pd.read_csv(os.path.join(outdir, "retention.csv"))
    a["group"] = a["cond"].map(group_of)
    depths = list(GRID)
    xs = {d: i for i, d in enumerate(depths)}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color=muted, lw=1.0, zorder=1)
    ax.axhline(-MARGIN, color=muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(depths) - 0.45, 0.012, T("original\nlevel", "원본\n수준"), color=muted, fontsize=8, ha="left", va="bottom")
    ax.text(len(depths) - 0.45, -MARGIN, T("tolerance\n−0.05", "허용 한계\n−0.05"), color=muted, fontsize=8, ha="left", va="top")
    ends = []
    for (en, ko, groups), col in zip(_FIG_GROUPS, _SERIES_COLORS):
        sub = a[a.group.isin(groups) & a.depth.isin(depths)]
        for c, s in sub.groupby("cond"):
            s = s.sort_values("depth")
            ax.plot([xs[d] for d in s.depth], s.delta, color=col, lw=0.8, alpha=0.28, zorder=2)
        med = sub.groupby("depth").delta.median().reindex(depths)
        ax.plot([xs[d] for d in med.index], med.values, color=col, lw=2.0, marker="o", ms=6, mfc=col, mec="#fcfcfb", mew=1.2, zorder=4,
                label=f"{T(en, ko)} (n={sub.cond.nunique()})")
        ends.append((med.values[-1], col, T(en, ko)))
    ax.set_xticks(range(len(depths))); ax.set_xticklabels([str(d) for d in depths], color=muted, fontsize=9)
    ax.set_xlabel(T("Depth of the distilled decision tree", "대리 결정트리 깊이"), color=ink, fontsize=10)
    ax.set_ylabel(T("Score change vs original policy\n(tree − original, paired by seed)", "원본 대비 점수 변화\n(트리 − 원본, 시드 대응)"), color=ink, fontsize=10)
    ax.set_ylim(-0.85, 0.18)
    ax.set_xlim(-0.4, len(depths) + 0.45)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5, labelcolor=ink, handlelength=1.6)
    ax.set_title(T("Running the distilled tree as the policy: performance vs tree depth",
                   "대리 트리를 정책으로 돌렸을 때: 깊이별 성능"), color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_dagger(outdir: str, path: str, lang: str = "en") -> str:
    """DAgger 반복에 따른 Δ. 학습 정책 20개(300세대)마다 옅은 선, 깊이별 중앙값은 굵은 선. 0회차는 로그만으로 적합한 트리."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .labels import T, set_language
    set_language(lang)
    games = pd.read_csv(os.path.join(outdir, "games.csv"))
    dg = pd.read_csv(os.path.join(outdir, "dagger_games.csv"))
    fits = pd.read_csv(os.path.join(outdir, "dagger_fits.csv"))
    man = json.load(open(os.path.join(outdir, "dagger_manifest.json"), encoding="utf-8"))
    lp = [c for c in dg.cond.unique() if group_of(c) in ("학습(치고 빠지기)", "학습(추격)")]
    orig = _seed_scores(games[(games.depth == 0) & games.cond.isin(lp)], ["cond"]).rename(columns={"score": "s0"})
    plain = _seed_scores(games[(games.depth > 0) & games.cond.isin(lp)], ["cond", "depth"]).assign(iteration=0)
    it = _seed_scores(dg[dg.cond.isin(lp)], ["cond", "depth", "iteration"])
    allp = pd.concat([plain, it]).merge(orig, on=["cond", "seed"]).assign(delta=lambda x: x.score - x.s0)
    d = allp.groupby(["cond", "depth", "iteration"]).delta.mean().reset_index()
    depths = man["depths"]
    leaves = {dp: int(fits[(fits.depth == dp) & (fits.iteration == man["n_iter"]) & fits.cond.isin(lp)].n_leaves.median()) for dp in depths}
    xs = list(range(man["n_iter"] + 1))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color=muted, lw=1.0, zorder=1)
    ax.axhline(-MARGIN, color=muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(xs[-1] + 0.08, 0.012, T("original level", "원본 수준"), color=muted, fontsize=8, ha="left", va="bottom")
    ax.text(xs[-1] + 0.08, -MARGIN - 0.008, T("tolerance −0.05", "허용 한계 −0.05"), color=muted, fontsize=8, ha="left", va="top")
    for dp, col in zip(depths, _SERIES_COLORS):
        sub = d[d.depth == dp]
        for c, s in sub.groupby("cond"):
            s = s.sort_values("iteration")
            ax.plot(s.iteration, s.delta, color=col, lw=0.8, alpha=0.28, zorder=2)
        med = sub.groupby("iteration").delta.median()
        ax.plot(med.index, med.values, color=col, lw=2.0, marker="o", ms=6, mfc=col, mec="#fcfcfb", mew=1.2, zorder=4,
                label=T(f"Depth {dp} (≈{leaves[dp]:,} leaves after DAgger)", f"깊이 {dp} (DAgger 후 잎 약 {leaves[dp]:,}개)"))
    ax.set_xticks(xs); ax.set_xticklabels([T("log only", "로그만")] + [str(i) for i in xs[1:]], color=muted, fontsize=9)
    ax.set_xlabel(T("DAgger iterations (tree-visited states relabelled by the original policy)", "DAgger 반복 횟수 (트리가 방문한 상태를 원본이 다시 라벨)"), color=ink, fontsize=10)
    ax.set_ylabel(T("Score change vs original policy\n(tree − original, paired by seed)", "원본 대비 점수 변화\n(트리 − 원본, 시드 대응)"), color=ink, fontsize=10)
    ax.set_ylim(-0.85, 0.12); ax.set_xlim(-0.2, xs[-1] + 0.95)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1, frameon=False, fontsize=8.5, labelcolor=ink, handlelength=1.6)
    ax.set_title(T(f"Performance gap of the tree policy over DAgger iterations (n={len(lp)} learned policies)",
                   f"DAgger 반복에 따른 트리 정책의 성능 격차 (학습 정책 {len(lp)}개)"), color=ink, fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)
    return path
