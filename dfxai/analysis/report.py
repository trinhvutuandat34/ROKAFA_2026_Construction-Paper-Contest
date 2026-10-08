"""
분석 리포트 생성기.

실행하면 results/<run>/ 안에 논문에 바로 넣을 표와 그림이 생깁니다.

  summary.csv        조건별 성능·규칙준수 요약
  xai.csv            조건별 설명가능성 지표 (D*, F(4))
  merged.csv         세 축을 합친 최종 표 (논문 본문 표)
  pairwise.csv       쌍별 검정 + Holm 보정
  fig_pareto.png     성능-설명가능성 평면 + 파레토 프론티어
  fig_compliance.png 성능-규칙준수 평면
  fig_fidelity.png   깊이별 충실도 곡선
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .stats import summarize, pairwise_tests, binomial_vs_chance
from ..xai.surrogate import surrogate_fidelity
from .labels import T, set_language, language


def compute_xai(outdir: str, target: float = 0.95) -> pd.DataFrame:
    """저장된 궤적마다 대리모델 충실도를 계산."""
    rows, curves = [], {}
    for path in sorted(glob.glob(os.path.join(outdir, "traj_*.npz"))):
        cond = os.path.basename(path)[len("traj_"):-len(".npz")]
        d = np.load(path)
        r = surrogate_fidelity(d["obs"], d["act"], target=target)
        rows.append(dict(cond=cond, d_star=r.min_depth_at_target,
                         d_star_090=r.d_star_multi["0.9"],
                         d_star_095=r.d_star_multi["0.95"],
                         d_star_099=r.d_star_multi["0.99"],
                         fidelity_at_4=r.fidelity_at_fixed_depth,
                         n_leaves=r.n_leaves_at_target, n_samples=r.n_samples))
        curves[cond] = (r.depths, r.fidelity)
    df = pd.DataFrame(rows)
    np.savez(os.path.join(outdir, "fidelity_curves.npz"),
             **{k: np.array(v) for k, v in curves.items()})
    return df


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x는 클수록, y는 작을수록 좋다고 가정한 비지배 집합."""
    idx = np.argsort(-x)
    best, keep = np.inf, []
    for i in idx:
        if y[i] < best:
            keep.append(i); best = y[i]
    return np.array(sorted(keep))


def _sfx() -> str:
    """한글 라벨 그림은 _ko 접미사로 따로 저장합니다."""
    return "" if language() == "en" else "_ko"


def _family(cond: str) -> str:
    if cond.startswith("BTO"):
        return "BTO"
    if cond.startswith("SHD"):
        return "Shield"
    if cond.startswith("PPO"):
        return "PPO"
    if cond.startswith("RES"):
        return "Residual"
    if cond.startswith("GATE"):
        return "Gated"
    if cond.startswith("POOL"):
        return "Pool"
    if cond.startswith("BT"):
        return "BT"
    if cond.startswith("RL"):
        return "RL"
    return "Hybrid"


def make_report(outdir: str, target: float = 0.95) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(outdir, "episodes.csv"))
    summ = summarize(df)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    pw = pairwise_tests(df)
    pw.to_csv(os.path.join(outdir, "pairwise.csv"), index=False)
    binomial_vs_chance(df).to_csv(os.path.join(outdir, "binomial.csv"), index=False)

    xai = compute_xai(outdir, target)
    xai.to_csv(os.path.join(outdir, "xai.csv"), index=False)

    # 성능은 상대 2종 평균으로 집계 (보류 상대 단독 결과도 별도 열로 유지)
    perf = summ.pivot_table(index="cond", values=["score", "viol_total",
                                                  "wez_time", "surv_time"],
                            aggfunc="mean").reset_index()
    held = summ[summ["opponent"] == summ["opponent"].max()][["cond", "score"]] \
        .rename(columns={"score": "score_heldout"})
    merged = perf.merge(held, on="cond", how="left").merge(xai, on="cond", how="left")
    merged["family"] = merged["cond"].map(_family)
    merged = merged.sort_values("score", ascending=False)
    merged.to_csv(os.path.join(outdir, "merged.csv"), index=False)

    _plot_pareto(merged, outdir)
    _plot_compliance(merged, outdir)
    _plot_fidelity(outdir)
    return merged


def _scatter_by_family(ax, d, xcol, ycol):
    styles = {"BT": ("o", "tab:blue"), "RL": ("s", "tab:red"),
              "Hybrid": ("^", "tab:green"), "BTO": ("D", "tab:cyan"),
              "Shield": ("P", "tab:purple"), "PPO": ("v", "tab:orange"),
              "Residual": ("*", "tab:pink"), "Gated": ("h", "tab:brown")}
    for fam, g in d.groupby("family"):
        m, c = styles.get(fam, ("x", "gray"))
        ax.scatter(g[xcol], g[ycol], marker=m, c=c, s=70, label=fam,
                   edgecolors="k", linewidths=0.5, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["cond"], (r[xcol], r[ycol]), fontsize=6,
                    xytext=(4, 4), textcoords="offset points")


def _plot_pareto(merged: pd.DataFrame, outdir: str):
    d = merged.dropna(subset=["d_star"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _scatter_by_family(ax, d, "score", "d_star")
    pf = pareto_front(d["score"].values, d["d_star"].values)
    ax.plot(d["score"].values[pf], d["d_star"].values[pf], "k--", lw=1.2,
            alpha=0.7, label=T("Pareto front", "파레토 프론티어"), zorder=2)
    ax.set_xlabel(T("Combat performance  (score: win=1, draw=0.5)", "전투 성능  (점수: 승 1, 무 0.5)"))
    ax.set_ylabel(T(r"Explainability cost  $D^*$ (min tree depth)", r"설명 비용  $D^*$ (최소 트리 깊이)"))
    ax.set_title(T("Performance vs. Explainability", "성능 대 설명가능성"))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_pareto{_sfx()}.png"), dpi=180)
    plt.close(fig)


def _plot_compliance(merged: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _scatter_by_family(ax, merged, "score", "viol_total")
    ax.set_xlabel(T("Combat performance  (score)", "전투 성능  (점수)"))
    ax.set_ylabel(T("Rule violation rate  (per sim step)", "규칙 위반율  (적분 스텝당)"))
    ax.set_title(T("Performance vs. Rule Compliance", "성능 대 규칙 준수"))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_compliance{_sfx()}.png"), dpi=180)
    plt.close(fig)


def _plot_fidelity(outdir: str):
    path = os.path.join(outdir, "fidelity_curves.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for k in d.files:
        depths, fid = d[k]
        ax.plot(depths, fid, marker="o", ms=3, lw=1.2, label=k)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_xlabel(T("Surrogate decision-tree depth", "대리 결정트리 깊이"))
    ax.set_ylabel(T(r"Fidelity  $R^2$ (variance-weighted)", r"충실도  $R^2$ (분산가중)"))
    ax.set_title(T("Surrogate fidelity vs. tree depth", "트리 깊이별 대리모델 충실도"))
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_fidelity{_sfx()}.png"), dpi=180)
    plt.close(fig)


def make_figures_only(outdir: str) -> None:
    """CSV 를 다시 계산하지 않고 그림 3종만 다시 그립니다 (라벨 언어 바꿀 때)."""
    merged = pd.read_csv(os.path.join(outdir, "merged.csv"))
    _plot_pareto(merged, outdir)
    _plot_compliance(merged, outdir)
    _plot_fidelity(outdir)


def main():
    ap = argparse.ArgumentParser(description="분석 리포트 생성")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--lang", default="en", choices=("en", "ko"), help="그림 라벨 언어")
    ap.add_argument("--figs-only", action="store_true", help="CSV 재계산 없이 그림만")
    a = ap.parse_args()
    set_language(a.lang)
    if a.figs_only:
        make_figures_only(a.outdir); print("figures written"); return
    m = make_report(a.outdir, a.target)
    cols = ["cond", "family", "score", "score_heldout", "d_star",
            "fidelity_at_4", "viol_total"]
    print(m[[c for c in cols if c in m.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
