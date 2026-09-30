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
    out.append(f"\n학습 정책 전체의 (충실도, Δ) 순위상관 ρ={r:.2f} (p={p:.1e}, n={len(base)}쌍). ")
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

    doc = "\n".join(out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(doc)
    a.to_csv(os.path.join(outdir, "retention.csv"), index=False)
    return out_path


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
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color=muted, lw=1.0, zorder=1)
    ax.axhline(-MARGIN, color=muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(depths) - 0.55, -MARGIN + 0.012, T("tolerance −0.05", "허용 한계 −0.05"), color=muted, fontsize=8, ha="right", va="bottom")
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
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=ink, handlelength=1.6)
    ax.set_title(T("Running the distilled tree as the policy: performance vs tree depth",
                   "대리 트리를 정책으로 돌렸을 때: 깊이별 성능"), color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
