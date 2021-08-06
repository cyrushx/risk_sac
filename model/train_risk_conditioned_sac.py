import rlkit.torch.pytorch_util as ptu

## Use GPU accelerator
ptu.set_gpu_mode(False)

import argparse

import numpy as np
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.path_collector import RiskConditionedPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.risk_conditioned_sac import RiskConditionedSACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from data_generator.maze.maze_env import RiskConditionedDubinsMaze, RiskConditionedMaze


def experiment(variant):
    env_name = variant["env_dict"]["name"]
    start = np.array(variant["env_dict"]["start"], dtype=np.float32)
    goal = np.array(variant["env_dict"]["end"], dtype=np.float32)
    resize_factor = 1

    if variant["dubins"]:
        maze = RiskConditionedDubinsMaze
    else:
        maze = RiskConditionedMaze

    expl_env = maze(
        walls=env_name,
        resize_factor=resize_factor,
        dynamics_noise=np.array([0.5, 0.5]),
        start=start,
        goal=goal,
        noise_sample_size=variant["sample_size"],
        random_seed=0,
        risk_bound_init=variant["delta"],
        # risk_bound_range=[0.1, 0.4],
        risk_bound_range=[0.05, 0.4],
    )
    eval_env = maze(
        walls=env_name,
        resize_factor=resize_factor,
        dynamics_noise=np.array([0.5, 0.5]),
        start=start,
        goal=goal,
        noise_sample_size=variant["sample_size"],
        random_seed=0,
        risk_bound_init=variant["delta"],
        # risk_bound_range=[0.1, 0.4],
        risk_bound_range=[0.05, 0.4],
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

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
        obs_dim=obs_dim + 2,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = RiskConditionedPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = RiskConditionedPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        env_info_sizes={
            "collision": 1,
            "risk": 1,
            "allocated_risk": 1,
            "risk_bound": 1,
        },
    )
    trainer = RiskConditionedSACTrainer(
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
    parser.add_argument("--name", type=str, default="none", help="Model note.")
    parser.add_argument(
        "--delta", type=float, default=0.2, help="Upper risk bound for initialization."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--risk-coeff", type=float, default=10.0, help="Coefficient of risk term."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of samples for evaluating risk.",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["OneObstacle", "FlyTrapBig", "TwoRooms"],
        default="OneObstacle",
        help="Environment.",
    )
    parser.add_argument(
        "--dubins", action="store_true", help="Whether to use Dubins car model."
    )
    args = parser.parse_args()

    if args.env == "FlyTrapBig":
        env_dict = {"name": args.env, "start": [4, 4], "end": [4, 18]}
    elif args.env == "OneObstacle":
        env_dict = {"name": args.env, "start": [2, 5], "end": [9, 5]}
    elif args.env == "TwoRooms":
        env_dict = {"name": args.env, "start": [1, 5], "end": [9, 5]}
    else:
        raise Exception("Undefined environment.")

    if args.dubins:
        algorithm = "Risk-Bound-Conditioned-SAC-Dubins"
    else:
        algorithm = "Risk-Bound-Conditioned-SAC"

    # noinspection PyTypeChecker
    variant = dict(
        algorithm=algorithm,
        env_dict=env_dict,
        version="normal",
        dubins=args.dubins,
        layer_size=256,
        replay_buffer_size=int(1e6),
        delta=args.delta,
        sample_size=args.sample_size,
        algorithm_kwargs=dict(
            num_epochs=args.epochs,
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
            risk_coeff=args.risk_coeff,
        ),
    )

    log_name = "{}-{}-{}-delta-{}-risk-coeff-{}-epochs-{}".format(
        algorithm, args.env, args.name, args.delta, args.risk_coeff, args.epochs
    )
    log_dir = setup_logger(
        log_name, variant=variant, snapshot_mode="gap_and_last", snapshot_gap=20
    )

    print("Begin training...")
    experiment(variant)
    print("Save training log to directory: {}".format(log_dir))
