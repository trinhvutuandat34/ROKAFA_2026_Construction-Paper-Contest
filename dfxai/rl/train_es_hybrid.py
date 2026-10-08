"""
잔차형(RES)·게이팅형(GATE) 혼합 정책의 ES 학습기.

train_es_bt.py(BTO) 와 같은 방식이다: ES 루프(미러 샘플링·순위 정규화·갱신 수용
검사)는 그대로 두고 **학습 대상 파라미터 블록만** 바꾼다.

    --kind residual : ResidualPolicy 의 δ-신경망 1,827개 (agents/residual.py)
    --kind gating   : GatingPolicy 의 게이트 177개 (agents/gating.py). 학습 정책 절반은
                      --rl-ckpt 로 준 기존 순수 학습 정책 체크포인트를 고정해서 쓴다.

적합도·상대(BT-v1/v2 교대)·개체군 32·후보당 8교전은 순수 학습 정책과 같다. 관측의
α 원소는 두 계열 모두 1.0 으로 고정한다(혼합 비율은 정책 안에서 정해지므로).

    python -m dfxai.rl.train_es_hybrid --kind residual --beta 0.35 --seed 0 --generations 200
    python -m dfxai.rl.train_es_hybrid --kind gating --rl-ckpt results/es/ckpt_seed11_gen00300.npz --seed 0

학습 중 진단: 매 세대 현재 정책을 같은 시드로 평가하면서 잔차형은 |δ| 평균을,
게이팅형은 g 의 평균·표준편차를 history 에 남긴다. 잔차형의 |δ| 가 0 에 붙거나
게이팅형의 g 표준편차가 0 에 붙으면 "학습은 되는데 의도한 대로 안 되는" 경우다
(하이브리드_확장_실험설계.md 7절).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from ..env import DogfightEnv
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy
from ..agents.residual import ResidualPolicy
from ..agents.gating import GatingPolicy
from .shaping import potential, episode_fitness
from .train_es import _rank_transform, _dump_history, load_resume


def _make(kind: str, theta: np.ndarray, cfg: dict):
    if kind == "residual":
        return ResidualPolicy(beta=cfg["beta"], bt_version=cfg["bt_version"],
                              hidden=cfg["hidden"], params=theta)
    rl = MLPPolicy(hidden=cfg["rl_hidden"], params=cfg["rl_flat"], out_act=cfg["rl_out_act"])
    return GatingPolicy(rl, bt_version=cfg["bt_version"], gate_hidden=cfg["gate_hidden"],
                        gate_params=theta)


def _rollout(args):
    kind, theta, cfg, seeds, opps, preset, diag = args
    pol = _make(kind, theta, cfg)
    env = DogfightEnv()
    total, stat = 0.0, []
    for seed, ov in zip(seeds, opps):
        red = BTPolicy(version=int(ov))
        ob, orr = env.reset(seed=int(seed), alpha=1.0)
        pot_sum, n = 0.0, 0
        for _ in range(env.max_steps + 1):
            a_b = pol.act(ob)
            if diag:
                stat.append(float(np.abs(pol.last_delta).mean()) if kind == "residual"
                            else float(pol.last_gate))
            ob, orr, done = env.step(a_b, red.act(orr))
            pot_sum += potential(ob); n += 1
            if done:
                break
        total += episode_fitness(env.result(), pot_sum / max(n, 1), preset)
    if diag:
        s = np.asarray(stat)
        return total / len(seeds), (float(s.mean()), float(s.std()))
    return total / len(seeds)


def _draw_generation(rng, episodes, opponents, pop, n_par):
    """한 세대의 난수. 학습 루프와 이어 학습의 난수 되감기가 함께 쓴다."""
    seeds = rng.integers(0, 10 ** 6, episodes)
    opps = np.array([opponents[i % len(opponents)] for i in range(episodes)])
    rng.shuffle(opps)
    eps = rng.normal(0.0, 1.0, (pop // 2, n_par))
    return seeds, opps, np.concatenate([eps, -eps], axis=0)


def train(kind: str, generations=200, pop=32, sigma=0.05, lr=0.01, episodes=8, seed=0,
          workers=1, opponents=(1, 2), outdir="results/es_hybrid", checkpoint_every=50,
          tag="seed0", fitness="default", beta=0.35, bt_version=2, rl_ckpt=None,
          gate_hidden=(8,), resume=None, resume_history=None):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if kind == "residual":
        proto = ResidualPolicy(beta=beta, bt_version=bt_version, seed=seed)
        cfg = dict(beta=beta, bt_version=bt_version, hidden=proto.mlp.hidden)
    else:
        assert rl_ckpt, "--kind gating 은 --rl-ckpt 가 필요합니다"
        rl = MLPPolicy.load(rl_ckpt)
        proto = GatingPolicy(rl, bt_version=bt_version, gate_hidden=gate_hidden, seed=seed)
        cfg = dict(bt_version=bt_version, gate_hidden=tuple(gate_hidden), rl_hidden=rl.hidden,
                   rl_flat=rl.flat, rl_out_act=rl.out_act)
    theta = proto.flat.copy()
    n_par = theta.size
    print(f"[{kind}] 학습 파라미터 {n_par}개, pop {pop}, 후보당 {episodes}교전, "
          f"σ {sigma}, lr {lr}" + (f", β {beta}" if kind == "residual" else f", rl {rl_ckpt}"),
          flush=True)
    lr_eff, n_accept, history, start_gen, t_offset = lr, 0, [], 0, 0.0
    if resume:
        # 이어 학습(train_es.py 와 같은 방식): 학습 대상 블록만 체크포인트에서 복원하고
        # 지난 세대의 난수는 평가 없이 다시 뽑아 버린다.
        if kind == "residual":
            from ..agents.residual import ResidualPolicy as _P
        else:
            from ..agents.gating import GatingPolicy as _P
        theta = _P.load(resume).flat.copy()
        assert theta.size == n_par, "체크포인트의 학습 파라미터 수가 다릅니다"
        start_gen, history = load_resume(resume, resume_history)
        lr_eff = float(history[-1]["lr_eff"])
        n_accept = int(sum(h["accepted"] for h in history))
        t_offset = float(history[-1].get("elapsed", 0.0))
        for _ in range(start_gen):
            _draw_generation(rng, episodes, opponents, pop, n_par)
        print(f"이어 학습: {resume} ({start_gen}세대, lr={lr_eff:.4f}, "
              f"accept={n_accept}/{start_gen}) -> {generations}세대", flush=True)
    pool = Pool(workers) if workers > 1 else None
    t0 = time.time() - t_offset
    for gen in range(start_gen + 1, generations + 1):
        seeds, opps, eps = _draw_generation(rng, episodes, opponents, pop, n_par)
        cands = theta[None, :] + sigma * eps
        jobs = [(kind, cands[i], cfg, seeds, opps, fitness, False) for i in range(pop)]
        fits = (np.array(pool.map(_rollout, jobs)) if pool
                else np.array([_rollout(j) for j in jobs]))
        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta_new = theta + lr_eff * grad
        chk = [(kind, theta, cfg, seeds, opps, fitness, True),
               (kind, theta_new, cfg, seeds, opps, fitness, True)]
        (f_old, d_old), (f_new, d_new) = (pool.map(_rollout, chk) if pool
                                          else [_rollout(j) for j in chk])
        accepted = f_new >= f_old
        if accepted:
            theta = theta_new; lr_eff = min(lr, lr_eff * 1.15)
        else:
            lr_eff = max(lr * 0.25, lr_eff * 0.7)
        n_accept += int(accepted)
        d_cur = d_new if accepted else d_old
        history.append(dict(gen=gen, fit_mean=float(fits.mean()), fit_max=float(fits.max()),
                            fit_theta=float(f_new if accepted else f_old),
                            accepted=bool(accepted), lr_eff=float(lr_eff),
                            elapsed=time.time() - t0, alpha_lo=1.0,
                            diag_mean=d_cur[0], diag_std=d_cur[1]))
        if gen % max(1, generations // 20) == 0 or gen == start_gen + 1:
            lab = "|δ|" if kind == "residual" else "g"
            print(f"[gen {gen:4d}] fit mean={fits.mean():8.3f} max={fits.max():8.3f} "
                  f"theta={history[-1]['fit_theta']:8.3f} accept={n_accept}/{gen} "
                  f"lr={lr_eff:.4f} {lab}={d_cur[0]:.3f}±{d_cur[1]:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if gen % checkpoint_every == 0 or gen == generations:
            _make(kind, theta, cfg).save(os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))
            _dump_history(outdir, tag, history)
    if pool:
        pool.close(); pool.join()
    history[-1]["final"] = True
    _dump_history(outdir, tag, history)
    return theta, history


def main():
    ap = argparse.ArgumentParser(description="잔차형·게이팅형 혼합 ES 학습기")
    ap.add_argument("--kind", required=True, choices=("residual", "gating"))
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fitness", default="default", choices=("default", "kill_first"))
    ap.add_argument("--beta", type=float, default=0.35, help="잔차형 보정 허용폭")
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--rl-ckpt", default=None, help="게이팅형이 고정해서 쓸 학습 정책 체크포인트")
    ap.add_argument("--gate-hidden", type=int, nargs="*", default=[8])
    ap.add_argument("--resume", default=None, help="이어 학습할 체크포인트 (ckpt_<tag>_gen<G>.npz)")
    ap.add_argument("--resume-history", default=None, help="그 학습의 history_<tag>.json")
    a = ap.parse_args()
    outdir = a.outdir or ("results/es_res" if a.kind == "residual" else "results/es_gate")
    train(a.kind, a.generations, a.pop, a.sigma, a.lr, a.episodes, a.seed, a.workers,
          outdir=outdir, checkpoint_every=a.checkpoint_every, tag=a.tag or f"seed{a.seed}",
          fitness=a.fitness, beta=a.beta, bt_version=a.bt_version, rl_ckpt=a.rl_ckpt,
          gate_hidden=tuple(a.gate_hidden), resume=a.resume, resume_history=a.resume_history)


if __name__ == "__main__":
    main()
