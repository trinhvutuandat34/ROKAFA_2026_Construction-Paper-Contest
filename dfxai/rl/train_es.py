"""
진화전략(OpenAI-ES) 학습기 — numpy만 있으면 돌아갑니다.

왜 이걸 같이 넣었나:
  - torch / SB3 설치 없이 **당장 오늘** 학습 정책을 뽑을 수 있습니다.
  - 하이퍼파라미터가 거의 없고, 병렬화가 자명하며, 발산하지 않습니다.
    PPO 튜닝에 며칠 날리는 위험이 없어 '플랜 B'로 확실합니다.
  - 설명가능성 축에서 문제가 되는 것은 "블랙박스 신경망 정책"이지
    "그것이 PPO로 학습되었는가"가 아닙니다. 논문에서는 학습 알고리즘을
    명시하고 모델을 '학습 기반 모델'로 부르면 논지가 그대로 유지됩니다.

PPO 경로는 rl/train_ppo.py 를 쓰세요. 두 경로 모두 같은 MLPPolicy 포맷으로
체크포인트를 저장하므로 평가·분석 파이프라인은 공유됩니다.

가장 중요한 설계: **여러 세대에서 체크포인트를 저장**합니다.
이 체크포인트들이 논문의 '학습 예산' 축이 되고,
RL이 BT를 못 이기더라도 성능-설명가능성 곡선을 그릴 수 있게 해 줍니다.
"""
from __future__ import annotations
import argparse, json, os, re, time
import numpy as np
from multiprocessing import Pool

from ..config import OBS_DIM, EngagementConfig
from ..env import DogfightEnv, run_episode
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy, HybridPolicy
from .shaping import potential, episode_fitness


# ----------------------------------------------------------------- 적합도
def _rollout_fitness(args):
    """후보 파라미터 하나에 대한 평균 적합도."""
    flat, hidden, seeds, alphas, opp_versions, bt_version, *rest = args
    preset = rest[0] if rest else "default"
    timeout_rule = rest[1] if len(rest) > 1 else "hp"
    rl = MLPPolicy(hidden=hidden, params=flat)
    bt = BTPolicy(version=bt_version)
    env = DogfightEnv(ec=EngagementConfig(timeout_rule=timeout_rule))
    total = 0.0
    for seed, alpha, ov in zip(seeds, alphas, opp_versions):
        blue = HybridPolicy(bt, rl, alpha=alpha)
        red = BTPolicy(version=ov)
        ob, orr = env.reset(seed=int(seed), alpha=float(alpha))
        pot_sum, n = 0.0, 0
        for _ in range(env.max_steps + 1):
            a_b, a_r = blue.act(ob), red.act(orr)
            ob, orr, done = env.step(a_b, a_r)
            pot_sum += potential(ob); n += 1
            if done:
                break
        total += episode_fitness(env.result(), pot_sum / max(n, 1), preset)
    return total / len(seeds)


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """순위 기반 정규화. 적합도 스케일 변화에 둔감해져 학습이 안정됩니다."""
    ranks = np.empty_like(x)
    ranks[np.argsort(x)] = np.arange(len(x))
    y = ranks / (len(x) - 1) - 0.5
    return y / (y.std() + 1e-8)


# ----------------------------------------------------------------- 학습 루프
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0)     # 평가 격자와 동일


def _draw_generation(rng, gen, generations, episodes, opponents, pop, n_par,
                     alpha_curriculum):
    """한 세대에 쓰는 난수를 모두 뽑습니다.

    학습 루프와 이어 학습(--resume)의 난수 되감기가 이 함수 하나를 공유하므로,
    300세대에서 이어 학습해도 처음부터 1,000세대를 돌린 것과 같은 난수열을 씁니다.
    """
    # --- alpha 배정 ---
    # 기본(grid): 매 세대 평가 격자 {0.25, 0.5, 0.75, 1.0} 를 한 판씩.
    #   정책 하나가 전 구간에서 동작해야 하므로 모든 alpha 를 끝까지 학습합니다.
    #   (예전 커리큘럼은 후반에 alpha>=0.9 만 뽑아 alpha=0.25 정책이 사실상
    #    미학습 상태로 평가되는 문제가 있었습니다.)
    # curriculum: 초반엔 BT 비중을 높여 탐색을 유도하는 옛 방식 (비교용).
    if alpha_curriculum:
        a_lo = min(0.9, 0.15 + 0.85 * gen / max(1, generations * 0.7))
        alphas = rng.uniform(a_lo, 1.0, episodes)
        alphas[-1] = 1.0
    else:
        a_lo = float(min(ALPHA_GRID))
        alphas = np.array([ALPHA_GRID[i % len(ALPHA_GRID)] for i in range(episodes)])
    # 세대 내 모든 후보가 **같은 시드**를 쓰도록 고정 (분산 감소, 매우 중요)
    seeds = rng.integers(0, 10 ** 6, episodes)
    opps = np.array([opponents[i % len(opponents)] for i in range(episodes)])
    rng.shuffle(opps)
    eps = rng.normal(0.0, 1.0, (pop // 2, n_par))
    eps = np.concatenate([eps, -eps], axis=0)      # 미러 샘플링
    return a_lo, alphas, seeds, opps, eps


def load_resume(ckpt: str, history_path: str):
    """이어 학습 준비: 체크포인트의 세대 번호까지 history 를 잘라 돌려줍니다."""
    m = re.search(r"gen(\d+)", os.path.basename(ckpt))
    assert m, f"체크포인트 이름에 gen 번호가 없습니다: {ckpt}"
    start_gen = int(m.group(1))
    with open(history_path, encoding="utf-8") as f:
        prev = json.load(f)
    history = [dict(h) for h in prev if h["gen"] <= start_gen]
    assert len(history) == start_gen and history[-1]["gen"] == start_gen, \
        f"history 가 {start_gen}세대까지 이어져 있지 않습니다: {history_path}"
    for h in history:
        h.pop("final", None)
    return start_gen, history


def train(generations=200, pop=40, sigma=0.08, lr=0.03, episodes=4,
          hidden=(32, 32), seed=0, workers=1, bt_version=2,
          opponents=(1, 2), outdir="results/es", checkpoint_every=20,
          alpha_curriculum=False, tag="seed0", init: str | None = None,
          accept_test: bool = True, accept_tol: float = 0.0,
          fitness: str = "default", timeout_rule: str = "hp",
          resume: str | None = None, resume_history: str | None = None):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    lr_eff, n_accept = lr, 0
    start_gen, history, t_offset = 0, [], 0.0
    if resume:
        # 이어 학습: 가중치·보폭·수용 횟수를 이전 기록에서 복원하고, 난수열은
        # 이미 지난 세대만큼 그대로 다시 뽑아 버립니다(평가는 하지 않음).
        proto = MLPPolicy.load(resume)
        hidden = proto.hidden
        theta = proto.flat.copy()
        start_gen, history = load_resume(resume, resume_history)
        lr_eff = float(history[-1]["lr_eff"])
        n_accept = int(sum(h["accepted"] for h in history))
        t_offset = float(history[-1].get("elapsed", 0.0))
        for g in range(1, start_gen + 1):
            _draw_generation(rng, g, generations, episodes, opponents, pop,
                             theta.size, alpha_curriculum)
        print(f"이어 학습: {resume} ({start_gen}세대, lr={lr_eff:.4f}, "
              f"accept={n_accept}/{start_gen}) -> {generations}세대", flush=True)
    elif init:
        # 행동 복제 초기화 (pretrain_bc.py). 무작위 초기화는 원거리 회피로 수렴합니다.
        proto = MLPPolicy.load(init)
        hidden = proto.hidden
        theta = proto.flat.copy()
        print(f"초기화: {init} (hidden={hidden}, {theta.size} params)", flush=True)
    else:
        proto = MLPPolicy(hidden=hidden, seed=seed)
        theta = proto.flat.copy() * 0.1      # 작은 초기값에서 출발
    n_par = theta.size
    pool = Pool(workers) if workers > 1 else None
    t0 = time.time() - t_offset

    for gen in range(start_gen + 1, generations + 1):
        a_lo, alphas, seeds, opps, eps = _draw_generation(
            rng, gen, generations, episodes, opponents, pop, n_par, alpha_curriculum)
        cands = theta[None, :] + sigma * eps

        jobs = [(cands[i], hidden, seeds, alphas, opps, bt_version, fitness, timeout_rule)
                for i in range(pop)]
        fits = (np.array(pool.map(_rollout_fitness, jobs)) if pool
                else np.array([_rollout_fitness(j) for j in jobs]))

        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta_new = theta + lr_eff * grad

        # --- 수용 검사 (백트래킹) ---
        # 순위 정규화 기울기는 신호가 약해도 크기가 일정해서, 그대로 누적하면
        # 가중치가 lr/(sigma*sqrt(pop)) 씩 무작위 행보를 합니다. 복제 초기화한
        # 정책이 100세대 만에 지워지는 것을 실제로 겪었습니다. 그래서 갱신 전후의
        # theta 를 **같은 시드**로 평가해(대응 비교라 잡음이 작음) 나빠지면 거부하고
        # 보폭을 줄입니다. 수용되면 보폭을 서서히 원래대로 되돌립니다.
        if accept_test:
            chk = [(theta, hidden, seeds, alphas, opps, bt_version, fitness, timeout_rule),
                   (theta_new, hidden, seeds, alphas, opps, bt_version, fitness, timeout_rule)]
            f_old, f_new = (pool.map(_rollout_fitness, chk) if pool
                            else [_rollout_fitness(j) for j in chk])
            accepted = f_new >= f_old - accept_tol
            if accepted:
                theta = theta_new
                lr_eff = min(lr, lr_eff * 1.15)
            else:
                lr_eff = max(lr * 0.25, lr_eff * 0.7)
            n_accept += int(accepted)
        else:
            f_old, f_new, accepted = float("nan"), float("nan"), True
            theta = theta_new

        history.append(dict(gen=gen, fit_mean=float(fits.mean()),
                            fit_max=float(fits.max()), alpha_lo=float(a_lo),
                            fit_theta=float(f_new if accepted else f_old),
                            accepted=bool(accepted), lr_eff=float(lr_eff),
                            elapsed=time.time() - t0))
        if gen % max(1, generations // 20) == 0 or gen == start_gen + 1:
            print(f"[gen {gen:4d}] fit mean={fits.mean():8.3f} "
                  f"max={fits.max():8.3f} theta={history[-1]['fit_theta']:8.3f} "
                  f"accept={n_accept}/{gen} lr={lr_eff:.4f} alpha>={a_lo:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if gen % checkpoint_every == 0 or gen == generations:
            p = MLPPolicy(hidden=hidden, params=theta)
            p.save(os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))
            _dump_history(outdir, tag, history)        # 학습 중에도 뷰어가 읽도록

    if pool:
        pool.close(); pool.join()
    history[-1]["final"] = True
    _dump_history(outdir, tag, history)
    return theta, history


def _dump_history(outdir: str, tag: str, history: list) -> None:
    """원자적 저장: 뷰어가 쓰는 도중의 파일을 읽지 않도록 임시 파일 후 교체."""
    path = os.path.join(outdir, f"history_{tag}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="ES 학습기 (numpy 전용)")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="results/es")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--alpha-curriculum", action="store_true",
                    help="옛 방식(alpha 하한 상승 커리큘럼). 기본은 고정 격자.")
    ap.add_argument("--init", type=str, default=None,
                    help="초기 가중치 체크포인트 (pretrain_bc.py 출력). 권장.")
    ap.add_argument("--fitness", default="default", choices=("default", "kill_first"),
                    help="적합도 프리셋 (shaping.PRESETS). kill_first 는 부록 강건성용.")
    ap.add_argument("--no-accept-test", action="store_true",
                    help="수용 검사(백트래킹) 끄기. 순수 OpenAI-ES 갱신 (비교용).")
    ap.add_argument("--timeout-rule", default="hp", choices=("hp", "draw"),
                    help="학습 환경의 시간종료 규칙. draw = 격추만 승리 (재학습 실험).")
    ap.add_argument("--resume", default=None,
                    help="이어 학습할 체크포인트 (ckpt_<tag>_gen<G>.npz). --resume-history 필요.")
    ap.add_argument("--resume-history", default=None,
                    help="그 체크포인트를 만든 학습의 history_<tag>.json")
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    train(generations=args.generations, pop=args.pop, sigma=args.sigma,
          lr=args.lr, episodes=args.episodes, seed=args.seed,
          workers=args.workers, bt_version=args.bt_version,
          outdir=args.outdir, checkpoint_every=args.checkpoint_every, tag=tag,
          alpha_curriculum=args.alpha_curriculum, init=args.init,
          accept_test=not args.no_accept_test, fitness=args.fitness,
          timeout_rule=args.timeout_rule, resume=args.resume,
          resume_history=args.resume_history)


if __name__ == "__main__":
    main()
