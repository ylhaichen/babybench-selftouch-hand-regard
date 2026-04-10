import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/babybench_mplconfig")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
from stable_baselines3.common.callbacks import CallbackList

from babybench.selftouch_author import (
    LiveMetricsPlotCallback,
    ProgressCallback,
    StdDecayCallback,
    load_or_create_stage_model,
    make_monitored_stage_env,
    make_vec_env_from_single_env,
    make_vecnormalize,
    resolve_device,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/config_selftouch.yml", type=str)
    parser.add_argument("--train_for", default=2_000_000, type=int)
    parser.add_argument("--save_dir", default="results/resume_random_init", type=str)
    parser.add_argument("--resume_model", default=None, type=str)
    parser.add_argument("--rand_start_prob", type=float, default=0.0)
    parser.add_argument("--rand_end_prob", type=float, default=1.0)
    parser.add_argument("--rand_start_noise", type=float, default=0.1)
    parser.add_argument("--rand_end_noise", type=float, default=1.0)
    parser.add_argument("--rand_ramp_episodes", type=int, default=1000)
    parser.add_argument("--rand_start_after_episodes", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save_freq", default=25_000, type=int)
    parser.add_argument("--plot_freq", default=2_048, type=int)
    parser.add_argument("--plot_window", default=200, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["save_dir"] = args.save_dir
    os.makedirs(cfg["save_dir"], exist_ok=True)
    resolved_device = resolve_device(args.device)

    env = make_monitored_stage_env(
        config=cfg,
        stage="difficult",
        training=True,
        enable_speed_penalty=False,
        rand_start_prob=args.rand_start_prob,
        rand_end_prob=args.rand_end_prob,
        rand_start_noise=args.rand_start_noise,
        rand_end_noise=args.rand_end_noise,
        rand_ramp_episodes=args.rand_ramp_episodes,
        rand_start_after_episodes=args.rand_start_after_episodes,
    )
    venv = make_vec_env_from_single_env(env)
    venv = make_vecnormalize(venv, training=True)

    print(f"Save dir: {args.save_dir}")
    print(f"Train for: {args.train_for}")
    print(f"Resume model: {args.resume_model}")
    print(f"Device: {resolved_device}")

    model = load_or_create_stage_model(
        stage="difficult",
        venv=venv,
        save_dir=cfg["save_dir"],
        resume_model=args.resume_model,
        device=resolved_device,
    )

    callbacks = CallbackList(
        [
            ProgressCallback(cfg["save_dir"], save_freq=args.save_freq),
            StdDecayCallback(
                total_timesteps=int(args.train_for),
                init_log_std=0.0,
                final_log_std=-1.1,
            ),
            LiveMetricsPlotCallback(
                save_dir=cfg["save_dir"],
                plot_freq=args.plot_freq,
                rolling_window=args.plot_window,
            ),
        ]
    )

    print(f"Training for {args.train_for} timesteps (resuming model: {bool(args.resume_model)}) ...")
    model.learn(total_timesteps=args.train_for, callback=callbacks)

    final_path = os.path.join(cfg["save_dir"], "stelios_x_giannis_ppo_final.zip")
    model.save(final_path)
    venv.save(os.path.join(cfg["save_dir"], "vecnormalize.pkl"))
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
