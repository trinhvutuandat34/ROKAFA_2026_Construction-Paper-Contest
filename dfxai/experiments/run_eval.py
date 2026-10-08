"""
본실험 배치 실행기.

실험 설계
---------
조건(condition) = (모델종류, alpha, 체크포인트, BT버전) 의 조합
상대(opponent)  = 고정 BT 2종. 그중 하나는 학습에 전혀 쓰지 않은 **보류 상대**.
각 (조건 x 상대) 마다 seed 를 공유하며 **진영을 바꿔 두 번** 싸웁니다.

진영 교대 짝지은 설계가 중요한 이유
-----------------------------------
초기조건이 아무리 무작위여도 특정 시드가 청군에 유리할 수 있습니다.
같은 시드로 청/홍을 바꿔 한 번 더 싸우면 그 우연이 정확히 상쇄됩니다.
실제로 이 설계를 넣기 전에는 동일 정책끼리의 자기대전 승률이 0.59가
나왔지만, 넣은 뒤에는 정확히 0.500 이 나옵니다(설계 검증 완료).
논문 방법 절에 이 검증 결과를 그대로 쓰면 좋습니다.

출력
----
results/<run>/episodes.csv   : 교전별 결과 (승패, 피해, WEZ, 위반율)
results/<run>/traj_<cond>.npz: 대리모델 적합용 (관측, 행동) 궤적
results/<run>/manifest.json  : 설정 스냅샷 (재현성)
"""
from __future__ import annotations
import argparse, json, os, re
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ..config import config_dump, EngagementConfig, AircraftConfig
from ..env import DogfightEnv, run_episode
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy, HybridPolicy


# 민감도 분석용 환경 옵션. run() 이 설정하면 워커 프로세스에 인자로 전달됩니다.
ENV_OPTS: dict = {}


def make_env(opts: dict | None = None) -> DogfightEnv:
    """opts: 교전 규칙(episode_time, timeout_rule)과 비행역학(R9) 민감도 옵션.

    비행역학 옵션은 **환경의 기체**만 바꿉니다. 정책이 안에 품은 기체 가정
    (BT 의 코너속도 계산 등)은 본실험 그대로이므로, 실제 기체와 정책의 가정이
    어긋날 때 결론이 유지되는지를 보는 시험이 됩니다.
      cl_max_scale  최대 양력계수 배율       thrust_scale  최대추력 배율
      lag_scale     뱅크·하중·스로틀 지연 시상수 배율
      roll_rate_max_deg  롤 속도 상한 [deg/s]
    """
    opts = opts or {}
    ec = EngagementConfig()
    if "episode_time" in opts:
        ec.episode_time = float(opts["episode_time"])
    if "timeout_rule" in opts:
        ec.timeout_rule = str(opts["timeout_rule"])
    ac = AircraftConfig()
    if "cl_max_scale" in opts:
        ac.cl_max *= float(opts["cl_max_scale"])
    if "thrust_scale" in opts:
        ac.thrust_max_sl *= float(opts["thrust_scale"])
    if "lag_scale" in opts:
        s = float(opts["lag_scale"])
        ac.tau_bank *= s; ac.tau_load *= s; ac.tau_throttle *= s
    if "roll_rate_max_deg" in opts:
        ac.roll_rate_max = float(np.radians(float(opts["roll_rate_max_deg"])))
    if opts.get("sixdof"):
        # 6자유도(JSBSim) 교차검증: 교전 규칙은 그대로, 비행역학만 JSBSim F-16.
        # JSBSim 모델 로드가 느리므로 작업자 프로세스마다 환경 하나를 재사용합니다.
        key = json.dumps(opts, sort_keys=True)
        if key not in _SIXDOF_CACHE:
            from ..sixdof.env6dof import SixDofEnv
            _SIXDOF_CACHE[key] = SixDofEnv(ac=ac, ec=ec)
        return _SIXDOF_CACHE[key]
    return DogfightEnv(ac=ac, ec=ec)


_SIXDOF_CACHE: dict = {}


def build_condition(spec: dict):
    """조건 명세 -> 정책 객체.

    kind: bt (행동트리) | rl (순수 학습) | hybrid (선형 혼합) |
          bto (상수 최적화 행동트리, bt_param.py) | shield (감독형 혼합, shield.py)
    """
    kind = spec["kind"]
    if kind == "bt":
        return BTPolicy(version=spec.get("bt_version", 2))
    if kind == "bto":
        from ..agents.bt_param import ParamBTPolicy
        return ParamBTPolicy.load(spec["ckpt"])
    if kind == "residual":
        from ..agents.residual import ResidualPolicy
        return ResidualPolicy.load(spec["ckpt"])
    if kind == "gating":
        from ..agents.gating import GatingPolicy
        return GatingPolicy.load(spec["ckpt"])
    rl = MLPPolicy.load(spec["ckpt"])
    if kind == "rl":
        return rl
    if kind == "shield":
        from ..agents.shield import ShieldPolicy
        return ShieldPolicy(rl, bt_version=spec.get("bt_version", 3))
    return HybridPolicy(BTPolicy(version=spec.get("bt_version", 2)),
                        rl, alpha=float(spec["alpha"]))


def _one_job(args):
    spec, opp_version, seed, side, record, *rest = args
    env_opts = rest[0] if rest else {}
    pol = build_condition(spec)
    opp = BTPolicy(version=opp_version)
    env = make_env(env_opts)
    alpha = float(spec.get("alpha", 1.0 if spec["kind"] in ("rl", "shield", "residual", "gating") else 0.0))

    if side == 0:                      # 평가 대상이 청군
        out = run_episode(pol, opp, seed=seed, alpha=alpha, alpha_red=0.0,
                          env=env, record=record)
        res, ob, ac = out if record else (out, None, None)
    else:                              # 진영 교대: 평가 대상이 홍군
        res = run_episode(opp, pol, seed=seed, alpha=0.0, alpha_red=alpha,
                          env=env, record=False)
        ob, ac = None, None

    win = res.winner if side == 0 else -res.winner
    score = 1.0 if win == 1 else (0.5 if win == 0 else 0.0)
    v = res.violations_blue if side == 0 else res.violations_red
    row = dict(
        cond=spec["name"], kind=spec["kind"], alpha=alpha,
        bt_version=spec.get("bt_version", -1), ckpt=spec.get("ckpt", ""),
        budget=spec.get("budget", -1), train_seed=spec.get("train_seed", -1),
        opponent=f"BT-v{opp_version}", seed=seed, side=side,
        score=score, win=int(win == 1), draw=int(win == 0), loss=int(win == -1),
        outcome=res.outcome, duration=res.duration,
        damage_dealt=res.damage_dealt if side == 0 else res.damage_taken,
        damage_taken=res.damage_taken if side == 0 else res.damage_dealt,
        wez_time=res.wez_time_blue if side == 0 else res.wez_time_red,
        surv_time=res.duration,
        viol_deck=v.get("deck", 0.0), viol_over_g=v.get("over_g", 0.0),
        viol_sep=v.get("separation", 0.0),
        mean_es=res.mean_es_blue if side == 0 else np.nan,
    )
    if hasattr(env, "trim_ok"):                # 6자유도: 초기 트림 성공 여부 기록
        row["trim_ok"] = bool(env.trim_ok)
    return row, (ob, ac)


def run(conditions, opponents=(2, 3), n_seeds=100, workers=1,
        outdir="results/main", record_episodes=30, seed0=10_000,
        env_opts: dict | None = None):
    """본실험 실행.

    record_episodes : 조건당 궤적을 기록할 교전 수. 궤적은 대리모델
                      적합에만 쓰이므로 전부 기록할 필요가 없습니다.
    env_opts        : 민감도 분석용 환경 옵션 (episode_time, timeout_rule).
    """
    os.makedirs(outdir, exist_ok=True)
    env_opts = dict(env_opts or {})
    jobs = []
    for spec in conditions:
        for ov in opponents:
            for i in range(n_seeds):
                s = seed0 + i
                rec = i < record_episodes
                jobs.append((spec, ov, s, 0, rec, env_opts))
                jobs.append((spec, ov, s, 1, False, env_opts))

    print(f"총 {len(jobs)}회 교전 실행 (조건 {len(conditions)} x 상대 "
          f"{len(opponents)} x 시드 {n_seeds} x 진영 2)", flush=True)

    if workers > 1:
        with Pool(workers) as p:
            results = p.map(_one_job, jobs, chunksize=8)
    else:
        results = [_one_job(j) for j in jobs]

    rows = [r for r, _ in results]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "episodes.csv"), index=False)

    # --- 궤적을 조건별로 합쳐 저장 ---
    traj = {}
    for (row, (ob, ac)) in results:
        if ob is None:
            continue
        traj.setdefault(row["cond"], ([], []))
        traj[row["cond"]][0].append(ob)
        traj[row["cond"]][1].append(ac)
    for cond, (obs_l, act_l) in traj.items():
        safe = cond.replace("/", "_").replace(" ", "_")
        np.savez_compressed(os.path.join(outdir, f"traj_{safe}.npz"),
                            obs=np.concatenate(obs_l, axis=0),
                            act=np.concatenate(act_l, axis=0))

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"conditions": conditions, "opponents": list(opponents),
                   "n_seeds": n_seeds, "env_opts": env_opts,
                   "config": config_dump()}, f,
                  indent=2, ensure_ascii=False)
    print(f"저장 완료: {outdir}")
    return df


def default_conditions(ckpts: list[str] | None = None,
                       alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                       bt_version: int = 2, rl_prefix: str = "RL") -> list[dict]:
    """기본 실험 조건 목록.

    BT 3종(교리 비교) + 학습예산별 체크포인트 x alpha 스윕.
    alpha=0 은 정의상 순수 BT 이므로 중복을 피해 체크포인트마다 반복하지 않습니다.
    """
    conds = [dict(name=f"BT-v{v}", kind="bt", bt_version=v) for v in (1, 2, 3)]
    for ck in (ckpts or []):
        base = os.path.basename(ck)
        # ckpt_seed<S>_gen<G>.npz  ->  학습 시드 S, 예산 G
        ms = re.search(r"seed(\d+)", base)
        mg = re.search(r"gen(\d+)", base)
        train_seed = int(ms.group(1)) if ms else 0
        budget = int(mg.group(1)) if mg else \
            int("".join(ch for ch in base if ch.isdigit())[-5:] or 0)
        for a in alphas:
            if a == 0.0:
                continue
            kind = "rl" if a == 1.0 else "hybrid"
            conds.append(dict(
                name=(f"{rl_prefix}-s{train_seed}-b{budget}" if a == 1.0 and rl_prefix != "RL"
                      else f"{'RL' if a==1.0 else 'HYB'}-s{train_seed}-b{budget}-a{a:.2f}"),
                kind=kind, alpha=float(a), ckpt=ck, train_seed=train_seed,
                bt_version=bt_version, budget=budget))
    return conds


def main():
    ap = argparse.ArgumentParser(description="본실험 배치 실행")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--opponents", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="results/main")
    ap.add_argument("--record-episodes", type=int, default=30)
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--bto-ckpts", nargs="*", default=[],
                    help="상수 최적화 BT 체크포인트 (bt_param.py). 조건명 BTO-s<S>-b<G>")
    ap.add_argument("--shield-ckpts", nargs="*", default=[],
                    help="감독형 혼합에 얹을 학습 정책 체크포인트. 조건명 SHD-s<S>-b<G>")
    ap.add_argument("--ppo-ckpts", nargs="*", default=[],
                    help="밑바닥 PPO 학습 정책 체크포인트 (train_ppo.py). 조건명 PPO-s<S>-b<step>")
    ap.add_argument("--residual-ckpts", nargs="*", default=[],
                    help="잔차형 혼합 체크포인트 (train_es_hybrid.py --kind residual). 조건명 RES-s<S>-b<G>")
    ap.add_argument("--gate-ckpts", nargs="*", default=[],
                    help="게이팅형 혼합 체크포인트 (train_es_hybrid.py --kind gating). 조건명 GATE-s<S>-b<G>")
    ap.add_argument("--no-bt", action="store_true", help="BT 3종 조건을 넣지 않음")
    ap.add_argument("--rl-prefix", default="RL",
                    help="순수 학습 정책 조건명의 접두어. RL 이 아니면 <접두어>-s<S>-b<G> (예: 수준 향상 실험 POOL)")
    ap.add_argument("--episode-time", type=float, default=None,
                    help="교전 제한시간 [s] (민감도 분석)")
    ap.add_argument("--timeout-rule", default=None, choices=(None, "hp", "draw"),
                    help="시간종료 규칙 (민감도 분석): hp=잔여 HP 비교, draw=무승부")
    ap.add_argument("--cl-max-scale", type=float, default=None, help="최대 양력계수 배율 (R9)")
    ap.add_argument("--thrust-scale", type=float, default=None, help="최대추력 배율 (R9)")
    ap.add_argument("--lag-scale", type=float, default=None, help="지령 지연 시상수 배율 (R9)")
    ap.add_argument("--roll-rate-max", type=float, default=None, help="롤 속도 상한 [deg/s] (R9)")
    ap.add_argument("--sixdof", action="store_true",
                    help="6자유도(JSBSim F-16) 평가 전용 교차검증 환경 (dfxai/sixdof)")
    a = ap.parse_args()
    conds = default_conditions(a.ckpts, tuple(a.alphas), a.bt_version, a.rl_prefix)
    if a.no_bt:
        conds = [c for c in conds if c["kind"] != "bt"]
    for ck in a.bto_ckpts:
        base = os.path.basename(ck)
        ms, mg = re.search(r"seed(\d+)", base), re.search(r"gen(\d+)", base)
        conds.append(dict(name=f"BTO-s{int(ms.group(1)) if ms else 0}-b{int(mg.group(1)) if mg else 0}",
                          kind="bto", ckpt=ck, train_seed=int(ms.group(1)) if ms else 0,
                          budget=int(mg.group(1)) if mg else 0, alpha=0.0))
    for ck in a.shield_ckpts:
        base = os.path.basename(ck)
        ms, mg = re.search(r"seed(\d+)", base), re.search(r"gen(\d+)", base)
        conds.append(dict(name=f"SHD-s{int(ms.group(1)) if ms else 0}-b{int(mg.group(1)) if mg else 0}",
                          kind="shield", ckpt=ck, train_seed=int(ms.group(1)) if ms else 0,
                          budget=int(mg.group(1)) if mg else 0, alpha=1.0, bt_version=3))
    for ck in a.ppo_ckpts:
        base = os.path.basename(ck)
        ms, mg = re.search(r"seed(\d+)", base), re.search(r"step(\d+)", base)
        conds.append(dict(name=f"PPO-s{int(ms.group(1)) if ms else 0}-b{int(mg.group(1)) if mg else 0}",
                          kind="rl", ckpt=ck, train_seed=int(ms.group(1)) if ms else 0,
                          budget=int(mg.group(1)) if mg else 0, alpha=1.0))
    for ck, prefix, kind in [(c, "RES", "residual") for c in a.residual_ckpts] + \
                            [(c, "GATE", "gating") for c in a.gate_ckpts]:
        base = os.path.basename(ck)
        ms, mg = re.search(r"seed(\d+)", base), re.search(r"gen(\d+)", base)
        conds.append(dict(name=f"{prefix}-s{int(ms.group(1)) if ms else 0}-b{int(mg.group(1)) if mg else 0}",
                          kind=kind, ckpt=ck, train_seed=int(ms.group(1)) if ms else 0,
                          budget=int(mg.group(1)) if mg else 0, alpha=1.0, bt_version=2))
    env_opts = {}
    if a.episode_time is not None:
        env_opts["episode_time"] = a.episode_time
    if a.timeout_rule is not None:
        env_opts["timeout_rule"] = a.timeout_rule
    for key, val in (("cl_max_scale", a.cl_max_scale), ("thrust_scale", a.thrust_scale),
                     ("lag_scale", a.lag_scale), ("roll_rate_max_deg", a.roll_rate_max)):
        if val is not None:
            env_opts[key] = val
    if a.sixdof:
        env_opts["sixdof"] = True
    run(conds, tuple(a.opponents), a.n_seeds, a.workers, a.outdir,
        a.record_episodes, env_opts=env_opts)


if __name__ == "__main__":
    main()
