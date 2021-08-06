import rlkit.torch.pytorch_util as ptu

# Set to True to use GPU accelerator
ptu.set_gpu_mode(False)

import argparse

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.risk_sac import RiskSACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from data_generator.maze.maze_env import MazeEnv, RiskAwareMaze


def experiment(variant):
    env_name = "OneObstacle"  # Choose one of the environments from maze/maze_env.py.
    resize_factor = 1
    start = np.array([2, 5], dtype=np.float32)
    goal = np.array([9, 5], dtype=np.float32)

    expl_env = RiskAwareMaze(
        walls=env_name,
        resize_factor=resize_factor,
        dynamics_noise=np.array([0.5, 0.5]),
        start=start,
        goal=goal,
        noise_sample_size=500,
        random_seed=0,
    )
    eval_env = RiskAwareMaze(
        walls=env_name,
        resize_factor=resize_factor,
        dynamics_noise=np.array([0.5, 0.5]),
        start=start,
        goal=goal,
        noise_sample_size=500,
        random_seed=0,
    )

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    # Set up SAC model.
    M = variant["layer_size"]
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    rf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    rf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_rf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_rf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        env_info_sizes={"collision": 1, "risk": 1},
    )
    trainer = RiskSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        rf1=rf1,
        rf2=rf2,
        target_rf1=target_rf1,
        target_rf2=target_rf2,
        **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=0.2, help="Upper risk bound.")
    parser.add_argument(
        "--risk-coeff", type=float, default=10, help="Coefficient of risk term."
    )
    args = parser.parse_args()

    variant = dict(
        algorithm="Risk-Bounded-SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=200,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            delta=args.delta,
            risk_coeff=args.risk_coeff,
        ),
    )
    log_dir = setup_logger(
        "Risk-SAC-OneObstacle-delta-{}-risk-coeff-{}".format(
            args.delta, args.risk_coeff
        ),
        variant=variant,
    )

    print("Begin training...")
    experiment(variant)
    print("Save training log to directory: {}".format(log_dir))
