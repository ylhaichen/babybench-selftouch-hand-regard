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
    default_save_dir,
    default_train_for,
    load_or_create_stage_model,
    make_monitored_stage_env,
    make_vec_env_from_single_env,
    make_vecnormalize,
    resolve_device,
    stage_log_std_schedule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("base", "difficult", "after"), required=True)
    parser.add_argument("--config", default="examples/config_selftouch.yml", type=str)
    parser.add_argument("--train_for", default=None, type=int)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--resume-model", default=None, type=str)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-freq", default=25_000, type=int)
    parser.add_argument("--plot-freq", default=2_048, type=int)
    parser.add_argument("--plot-window", default=200, type=int)
    parser.add_argument("--rand_start_prob", type=float, default=0.0)
    parser.add_argument("--rand_end_prob", type=float, default=1.0)
    parser.add_argument("--rand_start_noise", type=float, default=0.1)
    parser.add_argument("--rand_end_noise", type=float, default=1.0)
    parser.add_argument("--rand_ramp_episodes", type=int, default=1000)
    parser.add_argument("--rand_start_after_episodes", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_for = args.train_for or default_train_for(args.stage)
    save_dir = args.save_dir or default_save_dir(args.stage)
    resolved_device = resolve_device(args.device)

    if args.stage == "after" and not args.resume_model:
        raise ValueError("The 'after' stage requires --resume-model")

    config["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Stage: {args.stage}")
    print(f"Config: {args.config}")
    print(f"Save dir: {save_dir}")
    print(f"Train for: {train_for}")
    print(f"Resume model: {args.resume_model}")
    print(f"Device: {resolved_device}")

    env = make_monitored_stage_env(
        config=config,
        stage=args.stage,
        training=True,
        enable_speed_penalty=(args.stage == "base"),
        rand_start_prob=args.rand_start_prob,
        rand_end_prob=args.rand_end_prob,
        rand_start_noise=args.rand_start_noise,
        rand_end_noise=args.rand_end_noise,
        rand_ramp_episodes=args.rand_ramp_episodes,
        rand_start_after_episodes=args.rand_start_after_episodes,
    )
    venv = make_vec_env_from_single_env(env)
    venv = make_vecnormalize(venv, training=True)

    model = load_or_create_stage_model(
        stage=args.stage,
        venv=venv,
        save_dir=save_dir,
        resume_model=args.resume_model,
        device=resolved_device,
    )

    init_log_std, final_log_std, decay_steps = stage_log_std_schedule(args.stage, train_for)
    callbacks = CallbackList(
        [
            ProgressCallback(save_dir=save_dir, save_freq=args.save_freq),
            StdDecayCallback(
                total_timesteps=decay_steps,
                init_log_std=init_log_std,
                final_log_std=final_log_std,
            ),
            LiveMetricsPlotCallback(
                save_dir=save_dir,
                plot_freq=args.plot_freq,
                rolling_window=args.plot_window,
            ),
        ]
    )

    model.learn(total_timesteps=train_for, callback=callbacks)

    final_model_path = os.path.join(save_dir, "stelios_x_giannis_ppo_final.zip")
    model.save(final_model_path)
    venv.save(os.path.join(save_dir, "vecnormalize.pkl"))
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
