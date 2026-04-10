import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/babybench_mplconfig")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
from stable_baselines3 import PPO

from babybench.selftouch_author import (
    ensure_eval_dirs,
    load_vecnormalize_for_eval,
    make_eval_runner,
    make_stage_env,
    manual_contact_stats,
    resolve_device,
    str2bool,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/config_selftouch.yml", type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--vecnormalize-path", required=True, type=str)
    parser.add_argument("--save-dir", default=None, type=str)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--duration", default=1000, type=int)
    parser.add_argument("--render", default=True, type=str2bool)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.vecnormalize_path):
        raise FileNotFoundError(f"VecNormalize file not found: {args.vecnormalize_path}")

    resolved_device = resolve_device(args.device)
    if args.save_dir:
        config["save_dir"] = args.save_dir
    ensure_eval_dirs(config["save_dir"])

    raw_env = make_stage_env(config=config, stage="base", training=False)
    venv = load_vecnormalize_for_eval(raw_env, args.vecnormalize_path)
    model = PPO.load(args.model_path, env=venv, device=resolved_device)

    evaluation = make_eval_runner(
        env=raw_env,
        duration=args.duration,
        render=args.render,
        save_dir=config["save_dir"],
    )
    evaluation.eval_logs()

    all_left_touches = set()
    all_right_touches = set()

    for ep_idx in range(args.episodes):
        print(f"Running evaluation episode {ep_idx + 1}/{args.episodes}")

        obs = venv.reset()
        evaluation.reset()

        for _ in range(args.duration):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, infos = venv.step(action)

            info = infos[0]
            evaluation.eval_step(info)

            left_hit, right_hit = manual_contact_stats(raw_env)
            all_left_touches.update(left_hit)
            all_right_touches.update(right_hit)

        evaluation.end(episode=ep_idx)

        score = (len(all_left_touches) + len(all_right_touches)) / 68.0
        print(f"Cumulative left unique geoms: {len(all_left_touches)}")
        print(f"Cumulative right unique geoms: {len(all_right_touches)}")
        print(f"Cumulative manual score: {score:.4f}")

    venv.close()
    raw_env.close()


if __name__ == "__main__":
    main()
