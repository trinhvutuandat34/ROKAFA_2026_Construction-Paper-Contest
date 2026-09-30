"""전술 영역별 설명 비용(R11) 결과를 Markdown 으로 정리한다. 측정은 tactical_regions.py."""
from __future__ import annotations
import itertools, json, os, re
import numpy as np
import pandas as pd
from scipy import stats

from .paper import md_table
from .tactical_regions import REGIONS, REGION_KO, CHASE

ROWS = ["BT-v1", "BT-v2", "BT-v3", "BTO", "복제본", "학습(치고 빠지기)", "학습(추격)", "혼합 α=0.5", "Shield", "잔차형", "게이팅형"]


def group_of(c: str) -> str:
    if c.startswith("BT-v"):
        return c
    if c.startswith("BTO"):
        return "BTO"
    if c == "RL-s0-b0-a1.00":
        return "복제본"
    if c.startswith("RL-"):
        s = re.search(r"-s(\d+)-", c).group(1)
        return "학습(추격)" if f"s{s}" in CHASE else "학습(치고 빠지기)"
    if c.startswith("HYB"):
        return "혼합 α=0.5"
    if c.startswith("SHD"):
        return "Shield"
    if c.startswith("RES"):
        return "잔차형"
    if c.startswith("GATE"):
        return "게이팅형"
    return "기타"


def _cell(v: pd.Series, fmt="{:.2f}", n_all: int | None = None) -> str:
    v = v.dropna()
    tail = f" [{len(v)}/{n_all}]" if n_all is not None and len(v) < n_all else ""
    if len(v) == 0:
        return "–" + tail
    if len(v) == 1:
        return fmt.format(v.iloc[0]) + tail
    return f"{fmt.format(v.median())} ({fmt.format(v.min())}–{fmt.format(v.max())})" + tail


def _table(df: pd.DataFrame, col: str, fmt="{:.2f}") -> str:
    rows = []
    for g in ROWS:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        r = {"정책 묶음": f"{g} (n={sub['cond'].nunique()})"}
        for rn in REGIONS:
            r[REGION_KO[rn]] = _cell(sub[sub["region"] == rn][col], fmt, sub["cond"].nunique())
        rows.append(r)
    return md_table(pd.DataFrame(rows))


def _agg_regional(df: pd.DataFrame, min_valid: int = 3) -> pd.DataFrame:
    """분할 시드 여러 개의 결과를 (조건, 원천, 분할, 영역) 마다 합친다. F 는 평균, D\\* 는 중앙값.
    유효한 분할이 min_valid 개 미만이면 결측으로 둔다."""
    keys = ["cond", "source", "split", "region"]
    g = df.groupby(keys)
    a = g.agg(n=("n", "first"), n_ep=("n_ep", "first"), k=("F4", "count"), F4=("F4", "mean"), F8=("F8", "mean"),
              F16=("F16", "mean"), D90=("D90", "median"), D95=("D95", "median"), F4_sd=("F4", "std")).reset_index()
    need = np.where(a["split"] == "grouped", min_valid, 1)
    for c in ("F4", "F8", "F16", "D90", "D95", "F4_sd"):
        a.loc[a["k"] < need, c] = np.nan
    a["group"] = a["cond"].map(group_of)
    return a


def _holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def _pairwise(wide: pd.DataFrame, label: str) -> str:
    """영역 쌍마다 시드 대응 Wilcoxon(Holm 보정). wide: index=조건, columns=영역."""
    rows, ps = [], []
    for a, b in itertools.combinations(REGIONS, 2):
        d = wide[[a, b]].dropna()
        if len(d) < 6:
            continue
        diff = d[a] - d[b]
        try:
            p = stats.wilcoxon(d[a], d[b]).pvalue
        except ValueError:
            p = 1.0
        rows.append({"비교": f"{REGION_KO[a]} − {REGION_KO[b]}", "n": len(d), "중앙 차이": diff.median()})
        ps.append(p)
    if not rows:
        return ""
    adj = _holm(ps)
    for r, p, q in zip(rows, ps, adj):
        r["p"], r["p(Holm)"] = p, q
    return f"**{label}**\n\n" + md_table(pd.DataFrame(rows), {"중앙 차이": "{:+.3f}", "p": "{:.4f}", "p(Holm)": "{:.4f}"}) + "\n"


def build(outdir: str, out_path: str = "paper/results_tactical.md", main_dir: str = "results/main",
          hyb_dir: str = "results/main_hyb") -> str:
    reg_raw = pd.read_csv(os.path.join(outdir, "regional.csv"))
    reg = _agg_regional(reg_raw)
    dev = pd.read_csv(os.path.join(outdir, "deviation.csv"))
    share = pd.read_csv(os.path.join(outdir, "share.csv"))
    man = json.load(open(os.path.join(outdir, "manifest.json"), encoding="utf-8"))
    for d in (reg, dev, share):
        d["group"] = d["cond"].map(group_of)
    score = pd.concat([pd.read_csv(os.path.join(p, "merged.csv"))[["cond", "score"]] for p in (main_dir, hyb_dir)]) \
        .drop_duplicates("cond")

    out = ["# 전술 영역별 설명 비용 (자동 생성, R11)\n",
           f"조건 {len(man['conditions'])}개. 생성: `python -m dfxai.analysis.tactical_regions`. 영역은 Saldiran 외(2024) 식 (39)–(42)의 "
           "정의(정면 조우 ATA ≤ 45° 이고 AA ≥ 135°, 공격 ATA ≤ 90° 이면서 정면 조우 아님, 방어 ATA > 90° 이고 AA > 90°, "
           "중립 ATA > 90° 이고 AA ≤ 90°)를 그대로 썼다. 영역 안에서 표본이 적은 조건은 제외했다"
           f"(자기 로그: 영역 표본 ≥ {man['min_n_own']}, 시험 표본 ≥ {man['min_test_own']}; 공통 상태: ≥ {man['min_n_pool']}). "
           "셀은 중앙값(최소–최대)이고, 정책이 하나뿐인 행은 그 값이다. 대괄호 [k/n] 은 표본 부족으로 n개 중 k개만 계산됐다는 뜻이다.\n"]

    out.append("### 표 R11-1. 자기 로그에서 각 영역에 머문 시간 비율\n")
    out.append(_table(share.melt(id_vars=["cond", "group"], value_vars=list(REGIONS), var_name="region", value_name="share"), "share") + "\n")

    own = reg[(reg["source"] == "own") & (reg["split"] == "grouped")]
    own_r = reg[(reg["source"] == "own") & (reg["split"] == "random")]
    pool = reg[reg["source"] == "pool"]

    out.append("### 표 R11-2. 영역 안에서 깊이 4 결정트리의 충실도 F(4) — 자기 로그, 교전 단위 분할(주)\n")
    out.append("영역 안에서 정책이 낸 지령을 깊이 4 트리가 R² 얼마로 재현하는가. 1에 가까우면 그 영역의 행동이 몇 개의 규칙으로 요약된다.\n")
    out.append(_table(own, "F4", "{:.3f}") + "\n")
    out.append("### 표 R11-3. 같은 지표 — 공통 상태(모든 정책에 같은 관측을 질의)\n")
    out.append(_table(pool, "F4", "{:.3f}") + "\n")
    out.append("### 표 R11-4. 영역 안에서 충실도 0.90 에 처음 도달하는 깊이 D\\*(0.90) — 자기 로그(교전 단위 분할) / 공통 상태\n")
    out.append("자기 로그\n\n" + _table(own, "D90", "{:.0f}") + "\n\n공통 상태\n\n" + _table(pool, "D90", "{:.0f}") + "\n")
    out.append("(17 은 깊이 16 까지 도달하지 못한 절단값이다.)\n")

    out.append("### 표 R11-5. BT-v2 대비 지령 편차 — 뱅크 채널 원형 차이[도]\n")
    out.append("같은 관측에서 BT-v2 가 낼 지령과 정책 지령의 차이. BT-v2 자신은 0 이다(자체 점검).\n\n자기 로그 상태\n")
    out.append(_table(dev[dev["source"] == "own"], "dev_bank_deg", "{:.1f}") + "\n\n공통 상태\n\n")
    out.append(_table(dev[dev["source"] == "pool"], "dev_bank_deg", "{:.1f}") + "\n")
    out.append("**뱅크 반전 비율**(뱅크 차이가 90° 를 넘는 상태의 비율) — 자기 로그 상태\n\n")
    out.append(_table(dev[dev["source"] == "own"], "bank_flip", "{:.3f}") + "\n")
    out.append("**하중배수 지령 차이**(정규화 지령, 0–2) — 자기 로그 상태\n\n")
    out.append(_table(dev[dev["source"] == "own"], "dev_load", "{:.3f}") + "\n")

    out.append("### 표 R11-6. 학습 정책 20개에서 영역 사이 비교 (시드 대응 Wilcoxon, Holm 보정)\n")
    rl = own[own["group"].isin(["학습(치고 빠지기)", "학습(추격)"])]
    out.append(_pairwise(rl.pivot(index="cond", columns="region", values="F4"), "F(4), 자기 로그·교전 단위 분할 (차이가 음수면 앞 영역이 더 복잡)") + "\n")
    rlp = pool[pool["group"].isin(["학습(치고 빠지기)", "학습(추격)"])]
    out.append(_pairwise(rlp.pivot(index="cond", columns="region", values="F4"), "F(4), 공통 상태") + "\n")
    rld = dev[(dev["source"] == "own") & dev["group"].isin(["학습(치고 빠지기)", "학습(추격)"])]
    out.append(_pairwise(rld.pivot(index="cond", columns="region", values="dev_bank_deg"), "BT-v2 대비 뱅크 편차[도], 자기 로그 (차이가 양수면 앞 영역에서 더 크게 벗어남)") + "\n")

    out.append("### 표 R11-7. 영역별 지표와 전체 점수의 순위상관 (탐색적, 표의 모든 행에 Holm 보정)\n")
    sets = {"학습 정책 20개(300세대)": ["학습(치고 빠지기)", "학습(추격)"],
            "학습·혼합·Shield·잔차·게이팅 (n≈66)": ["학습(치고 빠지기)", "학습(추격)", "혼합 α=0.5", "Shield", "잔차형", "게이팅형"]}
    rows = []
    for nm, groups in sets.items():
        for rn in REGIONS:
            for lab, df_, col in (("자기 로그 F(4)", own, "F4"), ("공통 상태 F(4)", pool, "F4"), ("BT-v2 대비 뱅크 편차(자기 로그)", dev[dev["source"] == "own"], "dev_bank_deg")):
                d = df_[(df_["region"] == rn) & df_["group"].isin(groups)][["cond", col]].dropna().merge(score, on="cond")
                if len(d) >= 8:
                    r, p = stats.spearmanr(d[col], d["score"])
                    rows.append({"집합": nm, "영역": REGION_KO[rn], "지표": lab, "n": len(d), "ρ(점수)": r, "p": p})
    if rows:
        for r_, q in zip(rows, _holm([r_["p"] for r_ in rows])):
            r_["p(Holm)"] = q
    out.append(md_table(pd.DataFrame(rows), {"ρ(점수)": "{:+.2f}", "p": "{:.3f}", "p(Holm)": "{:.3f}"}) + "\n")

    ep = os.path.join(outdir, "error_share.csv")
    if os.path.exists(ep):
        es = pd.read_csv(ep)
        es = es.groupby(["cond", "depth", "region"]).agg(time_share=("time_share", "mean"), err_share=("err_share", "mean"),
                                                         unexplained=("unexplained", "mean")).reset_index()
        es["group"] = es["cond"].map(group_of)
        out.append("### 표 R11-8. 전역 트리의 미설명 분산이 나오는 영역 — 오차 몫 대 시간 비율\n")
        out.append("전역 결정트리(교전 단위 분할 5개의 평균)의 시험 교전 제곱오차 합 가운데 각 영역이 차지하는 몫과 그 영역에 머문 시험 표본 비율. "
                   "몫이 시간 비율보다 크면 그 영역이 행동 재현을 어렵게 한다. 셀은 `오차 몫 / 시간 비율`의 중앙값이다.\n")
        for d in (4, 8):
            rows = []
            for g in ROWS:
                sub = es[(es.group == g) & (es.depth == d)]
                if sub.empty:
                    continue
                r = {"정책 묶음": f"{g} (n={sub.cond.nunique()})", "미설명 분산 1−R²": float(sub.groupby("cond").unexplained.first().median())}
                for rn in REGIONS:
                    x = sub[sub.region == rn]
                    r[REGION_KO[rn]] = f"{x.err_share.median():.2f} / {x.time_share.median():.2f}"
                rows.append(r)
            out.append(f"**전역 트리 깊이 {d}**\n\n" + md_table(pd.DataFrame(rows), {"미설명 분산 1−R²": "{:.3f}"}) + "\n")
    out.append("### 표 R11-9. 스텝 무작위 분할로 다시 잰 F(4) — 자기 로그 (교전 단위 분할과의 차이 확인)\n")
    out.append(_table(own_r, "F4", "{:.3f}") + "\n")
    own_sd = own[["cond", "group", "region", "F4_sd"]].dropna()
    if len(own_sd):
        out.append("### 표 R11-10. 교전 단위 분할 난수 시드 5개에 따른 F(4) 표준편차의 중앙값 — 표 R11-2 셀의 불확실성 규모\n")
        out.append("조건마다 분할 시드 5개로 잰 F(4) 의 표준편차를 구해 묶음 안에서 중앙값을 냈다. 영역 안 표본이 적은 정면 조우가 가장 흔들린다.\n")
        out.append(_table(own_sd, "F4_sd", "{:.3f}") + "\n")
    doc = "\n".join(out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(doc)
    return out_path
