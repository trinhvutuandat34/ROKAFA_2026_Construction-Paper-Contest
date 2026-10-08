"""
추가 실험(E1–E4)을 반영한 논문 그림.

    python -m dfxai.viz.paper_figs_extra --out results/paper_figs_rewrite

  fig4_1_plane.png        <그림 4-1> 최종 예산 모델의 성능–설명 비용 평면.
                          잔차형·게이팅형은 results/extra_hyb 의 300세대(시드 5개)가 있으면 그것을,
                          없으면 results/main_hyb 의 200세대(시드 3개)를 쓴다.
  fig4_5_extension.png    학습 연장(E1) 300 → 1,000세대의 시드별 점수와 D*95.

한글 글꼴은 koreanize-matplotlib(NanumGothic)이 있으면 쓴다.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:            # 글꼴이 없으면 기본 글꼴로 그린다(한글이 네모로 나올 수 있음)
    pass
plt.rcParams["axes.unicode_minus"] = False

C_RULE, C_LEARN, C_HYB = "#2f6fd6", "#ee6a3a", "#22a86b"


def _jitter(n: int, rng, w: float = 0.12) -> np.ndarray:
    return rng.uniform(-w, w, n)


def _final(m: pd.DataFrame, hyb: pd.DataFrame, hyb_gen: int) -> list[tuple]:
    """(라벨, 묶음, 표식, 색, 데이터) 목록."""
    pick = lambda pat: m[m.cond.str.match(pat)]
    rows = [
        ("행동트리 BT-v1·v2·v3", "규칙 기반", "o", C_RULE, pick(r"BT-v\d$")),
        ("상수 최적화 BTO", "규칙 기반", "D", C_RULE, pick(r"BTO-s\d+-b300$")),
        ("신경망 (300세대)", "학습 기반", "o", C_LEARN, pick(r"RL-s\d+-b300-a1\.00$")),
        ("행동복제 초기 모델", "학습 기반", "s", C_LEARN, pick(r"RL-s0-b0-a1\.00$")),
        ("무작위 초기화 PPO", "학습 기반", "X", C_LEARN, pick(r"PPO-s\d+-b1500000$")),
        ("선형 혼합형 (α 0.25~0.75)", "하이브리드", "^", C_HYB, pick(r"HYB-s\d+-b300-a0\.\d+$")),
        ("감독형", "하이브리드", "v", C_HYB, pick(r"SHD-s\d+-b300$")),
        (f"잔차형 ({hyb_gen}세대)", "하이브리드", "*", C_HYB, hyb[hyb.cond.str.match(rf"RES-s\d+-b{hyb_gen}$")]),
        (f"게이팅형 ({hyb_gen}세대)", "하이브리드", "h", C_HYB, hyb[hyb.cond.str.match(rf"GATE-s\d+-b{hyb_gen}$")]),
    ]
    return rows


def fig_plane(out: str, seed: int = 0) -> str:
    m = pd.read_csv("results/main/merged.csv")
    if os.path.exists("results/extra_hyb/merged.csv"):
        hyb, gen = pd.read_csv("results/extra_hyb/merged.csv"), 300
    else:
        hyb, gen = pd.read_csv("results/main_hyb/merged.csv"), 200
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    groups = _final(m, hyb, gen)
    for lab, grp, mk, col, d in groups:
        if d.empty:
            continue
        y = d["d_star_095"].values.astype(float)
        y = y + np.where(y >= 17, _jitter(len(y), rng), _jitter(len(y), rng, 0.05))
        big = grp == "규칙 기반" and mk == "o"
        alpha = 0.55 if lab.startswith("선형") else 0.9
        ax.scatter(d["score"], y, marker=mk, s=110 if big else (95 if mk == "*" else 55),
                   c=col, alpha=alpha, edgecolors="white", linewidths=0.6, zorder=3)
        if big:
            for _, r in d.iterrows():
                dx, dy = (0.012, 0.35) if r.cond != "BT-v2" else (0.012, -0.75)
                ax.annotate(r.cond, (r.score, r.d_star_095), xytext=(r.score + dx, r.d_star_095 + dy),
                            fontsize=10, color="#555555")
    ax.axhline(17, ls="--", lw=1, color="#888888", zorder=1)
    ax.text(0.205, 17.6, "절단: 깊이 16으로도 95% 재현 불가(D* = 17로 기록)", fontsize=10, color="#444444")
    ax.set_xlim(0.2, 0.83); ax.set_ylim(0, 18.3)
    ax.set_yticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
    ax.set_xlabel("전투 성능: 점수 (상대 2종 평균, 0.5 = 대등)", fontsize=12)
    ax.set_ylabel("설명 비용: 최소 트리 깊이 D*", fontsize=12)
    ax.grid(alpha=0.35)
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    handles = {}
    for lab, grp, mk, col, d in groups:
        handles.setdefault(grp, []).append(Line2D([], [], marker=mk, ls="", color=col, markersize=9 if mk != "*" else 12,
                                                  label=lab))
    x0 = 0.0
    for grp in ("규칙 기반", "학습 기반", "하이브리드"):
        leg = fig.legend(handles=handles[grp], title=grp, loc="upper left", bbox_to_anchor=(0.02 + x0, 0.215),
                         frameon=False, fontsize=10, title_fontproperties={"weight": "bold", "size": 11})
        leg._legend_box.align = "left"
        x0 += 0.32
    fig.subplots_adjust(bottom=0.30, top=0.98, left=0.10, right=0.98)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "fig4_1_plane.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_extension(out: str) -> str | None:
    path = "results/extra_long/merged.csv"
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    ep = pd.read_csv("results/extra_long/episodes.csv")
    m = m[m.cond.str.startswith("RL-")].copy()
    m["seed"] = m.cond.str.extract(r"-s(\d+)-").astype(int)
    m["gen"] = m.cond.str.extract(r"-b(\d+)-").astype(int)
    sc = ep.pivot_table(index="cond", columns="opponent", values="score")
    m["v3"] = m.cond.map(sc["BT-v3"])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    colors = dict(zip([1, 11, 14, 2, 18], ["#d1495b", "#ee6a3a", "#edae49", "#66a182", "#2e4057"]))
    for s, d in m.sort_values("gen").groupby("seed"):
        axs[0].plot(d.gen, d.score, "-o", ms=4, color=colors.get(s, "gray"), label=f"시드 {s}")
        axs[1].plot(d.gen, d.d_star_095 + (list(colors).index(s) - 2) * 0.12, "-o", ms=4,
                    color=colors.get(s, "gray"))
    g = m.groupby("gen")
    axs[0].plot(g.score.mean().index, g.score.mean().values, "k--", lw=2, label="평균")
    axs[1].plot(g.d_star_095.mean().index, g.d_star_095.mean().values, "k--", lw=2)
    axs[0].axhline(0.671, color=C_RULE, ls=":", lw=1.5)
    axs[0].text(305, 0.676, "BT-v2 점수", color=C_RULE, fontsize=9)
    axs[0].set_title("(a) 전투 성능 (상대 2종 평균 점수)", fontsize=12)
    axs[1].set_title("(b) 설명 비용 (D*95, 17 = 절단)", fontsize=12)
    for a in axs:
        a.set_xlabel("학습 예산 (세대)", fontsize=11); a.grid(alpha=0.35)
        a.set_xticks(range(300, 1001, 100))
    axs[1].set_ylim(0, 18)
    axs[0].legend(fontsize=8.5, ncol=3, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out, "fig4_5_extension.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="추가 실험 반영 그림")
    ap.add_argument("--out", default="results/paper_figs_rewrite")
    a = ap.parse_args()
    for p in (fig_plane(a.out), fig_extension(a.out)):
        if p:
            print("저장:", p)


if __name__ == "__main__":
    main()
