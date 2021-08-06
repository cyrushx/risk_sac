# @title Implement the 2D navigation environment and helper functions.
from __future__ import absolute_import, division, print_function

import argparse

import gym
import gym.spaces
import matplotlib

matplotlib.use("TkAgg")
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from data_generator.maze.maze_definition import WALLS
from data_generator.maze.utils import (
    compute_dense_reward,
    plot_all_environments,
    plot_problem,
    resize_walls,
)


class MazeEnv(gym.Env):
    """Abstract class for 2D navigation environments."""

    def __init__(
        self,
        walls: str = None,
        resize_factor: int = 1,
        action_noise: float = 1.0,
        percent_random_change: float = 0.0,
        start: np.array = None,
        goal: np.array = None,
        noise_sample_size: int = None,
        random_seed: int = 0,
        dense_reward: bool = True,
    ):
        """
        Initialize Maze environment.

        Parameters
        ----------
        walls : str
            Map name, defined in data_generator/maze/maze_definition.py
        resize_factor : int
            Scale the map by this factor.
        action_noise : float
            Standard deviation of noise to add to actions. Use 0 to add no noise.
        percent_random_change : float
            Percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
        start : np.array
            Start location of the agent.
        goal : np.array
            Goal location of the agent.
        noise_sample_size : int
            Number of noisy maps to simulate uncertainty in perception.
        random_seed : int
            Random seed to support consistency in tests.
        dense_reward : bool
            Whether to compute dense reward.
        """
        # Resize environment if necessary.
        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        print("Walls shape", self._walls.shape)

        self._apsp = self._compute_apsp(self._walls)
        self._original_walls = deepcopy(self._walls)
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._action_noise = action_noise
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self._height, self._width]),
            dtype=np.float32,
        )
        self.start = start
        self.goal = goal
        self.percent_random_change = percent_random_change
        self.dense_reward = dense_reward
        self.env_name = walls
        self.noise_sample_size = noise_sample_size

        # Use fixed random seed for test purposes.
        np.random.seed(random_seed)
        self.reset()

    def create_noisy_environments(self, percent_random_change: float):
        """Create randomly created environments from the existing one"""
        if self.noise_sample_size is not None:
            self._walls_samples = []
            self._apsp_samples = []
            for _ in range(self.noise_sample_size):
                self.randomly_change_walls(percent_random_change)
                self._walls_samples.append(self._walls)
                self._apsp_samples.append(self._apsp)

            # Shape [num_samples, wall_size_x, wall_size_y].
            self._walls_samples = np.stack(self._walls_samples)
            # Compute probability of being occupied.
            self._walls_samples_p = (
                np.sum(self._walls_samples, axis=0) / self.noise_sample_size
            )
            # Reset self._walls
            self._walls = deepcopy(self._original_walls)

    def randomly_change_walls(
        self, percent_random_change: float = None, verbose: bool = False
    ):
        """Randomly change environment."""
        if percent_random_change is None:
            percent_random_change = self.percent_random_change
        num_change = np.floor(self._height * self._width * percent_random_change)
        num_change = int(num_change)
        if verbose:
            print("changing amount: {}".format(num_change))
        self._walls = deepcopy(self._original_walls)
        pos_x = np.random.randint(low=0, high=self._walls.shape[0], size=num_change)
        pos_y = np.random.randint(low=0, high=self._walls.shape[1], size=num_change)
        for x, y in zip(pos_x, pos_y):
            self._walls[x, y] = 1 - self._walls[x, y]
        self._apsp = self._compute_apsp(self._walls)

    def _sample_empty_state(self):
        """Sample random state from maze."""
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array(
            [candidate_states[0][state_index], candidate_states[1][state_index]],
            dtype=np.float,
        )
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state

    def set_start_state(self, start):
        """Set start state."""
        self.start = start

    def reset(
        self,
        risk_bound: float = None,
        random: bool = False,
        noise: [] = None,
        test_mode=False,
    ):
        """Reset state."""
        # TODO(cyrushx): Add noise to start state.
        if self.start is not None and not random:
            self.state = self.start.copy()
        elif test_mode:
            self.state = self.start.copy()
        else:
            self.state = self._sample_empty_state()
        return self.state.copy()

    def _get_distance(self, obs, goal):
        """Compute the shortest path distance.

        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return self._apsp[i1, j1, i2, j2]

    def _is_done(self, obs, goal, threshold_distance=0.5):
        """Determines whether observation equals goal."""
        return np.linalg.norm(obs - goal) < threshold_distance

    def _discretize_state(self, state, resolution=1.0):
        """Discretize a state."""
        (i, j) = np.floor(resolution * state).astype(np.int)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)

    def _is_blocked(self, state):
        """Check if a given state is blocked by an obstacle."""
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return self._walls[i, j] == 1

    def render(self):
        pass

    def step(self, action):
        """Perform an action, but it may be blocked by a wall"""
        old_state = self.state
        # Add random noise.
        if np.sum(np.abs(self._action_noise)) > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        # Perform action into sub-steps for collision avoidance in a finer resolution.
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state

        done = self._is_done(self.state, self.goal, 1.2)

        ##NOTE: reward is overriden in the goal condition wrapper
        rew = -1.0 * np.linalg.norm(self.state)
        if self.dense_reward:
            rew = compute_dense_reward(self.state.copy(), self.goal.copy())

        # Compute travelled distance to the new state.
        env_info = {}
        env_info["Step distance"] = np.sqrt(np.sum((self.state - old_state) ** 2, -1))
        return self.state.copy(), rew, done, env_info

    @property
    def walls(self):
        return self._walls

    def _compute_apsp(self, walls):
        """Compute all pair shortest path."""
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float("inf"))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist


class DubinsMaze(MazeEnv):
    """
    Class for navigation environment with Dubins car model.
    """

    def __init__(
        self,
        walls: str = None,
        resize_factor: int = 1,
        action_noise: float = 1.0,
        percent_random_change: float = 0.0,
        start: np.array = None,
        goal: np.array = None,
        noise_sample_size: int = None,
        random_seed: int = 0,
        dense_reward: bool = True,
    ):
        """
        Initialize Maze environment.

        Parameters
        ----------
        walls : str
            Map name, defined in data_generator/maze/maze_definition.py
        resize_factor : int
            Scale the map by this factor.
        action_noise : float
            Standard deviation of noise to add to actions. Use 0 to add no noise.
        percent_random_change : float
            Percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
        start : np.array
            Start location of the agent.
        goal : np.array
            Goal location of the agent.
        noise_sample_size : int
            Number of noisy maps to simulate uncertainty in perception.
        random_seed : int
            Random seed to support consistency in tests.
        dense_reward : bool
            Whether to compute dense reward.
        """
        super().__init__(
            walls=walls,
            resize_factor=resize_factor,
            action_noise=action_noise,
            percent_random_change=percent_random_change,
            start=start,
            goal=goal,
            noise_sample_size=noise_sample_size,
            random_seed=random_seed,
            dense_reward=dense_reward,
        )
        # Define action space for angular vel and acceleration.
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi / 4, -0.5]),
            high=np.array([np.pi / 4, 0.5]),
            dtype=np.float32,
        )
        # Define limits for heading angle and scalar speed.
        self.speed_ub = 1.0
        self.speed_lb = 0.0
        self.theta_ub = np.pi / 4.0
        self.theta_lb = -np.pi / 4.0

        self.dubins_state = np.array([0.0, 0.5])

    def reset(
        self,
        risk_bound: float = None,
        random: bool = False,
        noise: [] = None,
        test_mode=False,
    ):
        """
        Reset state.
        """
        self.dubins_state = np.array([0.0, 0.5])
        if self.start is not None:
            self.state = self.start.copy()
            return self.state.copy()
        self.state = self._sample_empty_state()
        return self.state.copy()

    def step(self, action):
        """Perform a Dubins path action."""
        old_state = self.state
        if np.sum(np.abs(self._action_noise)) > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)

        dubin_action = np.copy(action)
        self.dubins_state[0] = self.dubins_state[0] + dubin_action[0]
        self.dubins_state[1] = np.clip(
            self.dubins_state[1] + dubin_action[1],
            a_min=self.speed_lb,
            a_max=self.speed_ub,
        )
        theta, v = self.dubins_state

        # Compute state changes in (x, y) space.
        action_xy = np.array([v * np.cos(theta), v * np.sin(theta)])

        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action_xy[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state

        done = self._is_done(self.state, self.goal, 1.0)
        ##NOTE: reward is overriden in the goal condition wrapper
        rew = -1.0 * np.linalg.norm(self.state)
        if self.dense_reward:
            rew = compute_dense_reward(self.state.copy(), self.goal.copy())
        # Compute collisions at the new state.
        env_info = {}
        env_info["Step distance"] = np.sqrt(np.sum((self.state - old_state) ** 2, -1))
        return self.state.copy(), rew, done, env_info


class RiskAwareMaze(MazeEnv):
    """
    Class for risk aware maze.
    """

    def __init__(
        self,
        walls: str = None,
        resize_factor: int = 1,
        dynamics_noise: list = [1.0, 1.0],
        percent_random_change: float = 0.0,
        start: np.array = None,
        goal: np.array = None,
        noise_sample_size: int = None,
        random_seed: int = 0,
        dense_reward: bool = True,
    ):
        """
        Initialize Maze environment.

        Parameters
        ----------
        walls : str
            Map name, defined in data_generator/maze/maze_definition.py
        resize_factor : int
            Scale the map by this factor.
        dynamics_noise : list
            List of dynamics noise.
        percent_random_change : float
            Percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
        start : np.array
            Start location of the agent.
        goal : np.array
            Goal location of the agent.
        noise_sample_size : int
            Number of noisy maps to simulate uncertainty in perception.
        random_seed : int
            Random seed to support consistency in tests.
        dense_reward : bool
            Whether to compute dense reward.
        """
        super().__init__(
            walls=walls,
            resize_factor=resize_factor,
            action_noise=0.0,
            percent_random_change=percent_random_change,
            start=start,
            goal=goal,
            noise_sample_size=noise_sample_size,
            random_seed=random_seed,
            dense_reward=dense_reward,
        )
        self._dynamics_noise = dynamics_noise
        self._noise_sample_size = noise_sample_size

    def step(self, action, test_mode=None):
        """Perform a step."""
        sn, reward, done, info = super().step(action)

        # Compate step-wise risk using Monte-Carlo.
        samples = np.random.normal(
            loc=sn, scale=self._dynamics_noise, size=[self._noise_sample_size, len(sn)]
        )
        count_blocked_samples = 0
        for sn_sample in samples:
            if self._is_blocked(sn_sample):
                count_blocked_samples += 1

        stepwise_risk = count_blocked_samples * 1.0 / self._noise_sample_size
        info["collision"] = stepwise_risk
        info["risk"] = stepwise_risk
        return sn, reward, done, info


class RiskConditionedMaze(MazeEnv):
    """
    Class for risk bound conditioned maze.
    """

    def __init__(
        self,
        walls: str = None,
        resize_factor: int = 1,
        dynamics_noise: list = [1.0, 1.0],
        percent_random_change: float = 0.0,
        start: np.array = None,
        goal: np.array = None,
        noise_sample_size: int = None,
        random_seed: int = 0,
        dense_reward: bool = True,
        risk_bound_init: float = 0.2,
        risk_bound_range: list = [0.0, 1.0],
    ):
        """
        Initialize Maze environment.

        Parameters
        ----------
        walls : str
            Map name, defined in data_generator/maze/maze_definition.py
        resize_factor : int
            Scale the map by this factor.
        dynamics_noise : list
            List of dynamics noise.
        percent_random_change : float
            Percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
        start : np.array
            Start location of the agent.
        goal : np.array
            Goal location of the agent.
        noise_sample_size : int
            Number of noisy maps to simulate uncertainty in perception.
        random_seed : int
            Random seed to support consistency in tests.
        dense_reward : bool
            Whether to compute dense reward.
        risk_bound_init : float
            Upper risk bound for initialization.
        risk_bound_range: list
            Range of risk bound to train.
        """
        self._dynamics_noise = dynamics_noise
        self._noise_sample_size = noise_sample_size
        self._risk_bound = risk_bound_init
        self._allocated_risk = 0.0
        self._risk_bound_range = risk_bound_range

        super().__init__(
            walls=walls,
            resize_factor=resize_factor,
            action_noise=0.0,
            percent_random_change=percent_random_change,
            start=start,
            goal=goal,
            noise_sample_size=noise_sample_size,
            random_seed=random_seed,
            dense_reward=dense_reward,
        )
        maze_obs_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array(self._walls.shape),
            dtype=np.float32,
        )
        budget_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": maze_obs_space,
                "risk_budget": budget_space,
                "risk_bound": budget_space,
            }
        )
        self.observation_space = maze_obs_space

    def step(self, action, test_mode=False):
        """Perform a step."""
        # Fix random seed at test time for consistent results.
        if test_mode:
            np.random.seed(0)

        sn, reward, done, info = super().step(action)

        if not test_mode:
            # Compute step-wise risk through Monte-Carlo sampling.
            samples = np.random.normal(
                loc=sn,
                scale=self._dynamics_noise,
                size=[self._noise_sample_size, len(sn)],
            )
            count_blocked_samples = 0
            for sn_sample in samples:
                if self._is_blocked(sn_sample):
                    count_blocked_samples += 1

            stepwise_risk = count_blocked_samples * 1.0 / self._noise_sample_size
        else:
            # Skip expensive risk computing at test time.
            stepwise_risk = 0

        # Compute accumulated risk.
        self._allocated_risk = (
            self._allocated_risk + (1.0 - self._allocated_risk) * stepwise_risk
        )

        info["collision"] = stepwise_risk
        info["risk"] = stepwise_risk
        info["risk_bound"] = self._risk_bound
        # This field is deprecated and not used.
        info["allocated_risk"] = 0.0

        sn_dict = {
            "observation": sn,
            "risk_bound": self._risk_bound,
            "allocated_risk": self._allocated_risk,
        }
        return sn_dict, reward, done, info

    def set_risk_bound(self, risk_bound):
        """Update risk bound."""
        self._risk_bound = risk_bound

    def compute_collision(self, state, noise_sample_size=500):
        """Compute probability of collision given a state, using Monte Carlo."""
        samples = np.random.normal(
            loc=state, scale=self._dynamics_noise, size=[noise_sample_size, len(state)]
        )
        count_blocked_samples = 0
        for sn_sample in samples:
            if self._is_blocked(sn_sample):
                count_blocked_samples += 1

        stepwise_risk = count_blocked_samples * 1.0 / noise_sample_size
        return stepwise_risk

    def reset(
        self,
        risk_bound: float = None,
        random: bool = False,
        noise: [] = None,
        test_mode=False,
    ):
        # Reset allocated risk.
        self._allocated_risk = 0.0

        # Reset risk bound randomly so that the model can be trained to handle different risk bounds.
        if risk_bound is None:
            # Reset risk bound randomly.
            self._risk_bound = np.random.uniform(
                self._risk_bound_range[0], self._risk_bound_range[1]
            )
        else:
            self._risk_bound = risk_bound

        sn_dict = {
            "observation": super().reset(
                noise=[2, 2], random=True, test_mode=test_mode
            ),
            "risk_bound": self._risk_bound,
            "allocated_risk": self._allocated_risk,
        }
        return sn_dict


class RiskConditionedDubinsMaze(DubinsMaze):
    """
    Class for risk bound conditioned Dubins maze.
    """

    def __init__(
        self,
        walls: str = None,
        resize_factor: int = 1,
        dynamics_noise: list = [1.0, 1.0],
        percent_random_change: float = 0.0,
        start: np.array = None,
        goal: np.array = None,
        noise_sample_size: int = None,
        random_seed: int = 0,
        dense_reward: bool = True,
        risk_bound_init: float = 0.2,
        risk_bound_range: list = [0.0, 1.0],
    ):
        """
        Initialize Maze environment.

        Parameters
        ----------
        walls : str
            Map name, defined in data_generator/maze/maze_definition.py
        resize_factor : int
            Scale the map by this factor.
        dynamics_noise : list
            List of dynamics noise.
        percent_random_change : float
            Percentage of map pixels to be flipped (free -> obstacle, obstacle -> free).
        start : np.array
            Start location of the agent.
        goal : np.array
            Goal location of the agent.
        noise_sample_size : int
            Number of noisy maps to simulate uncertainty in perception.
        random_seed : int
            Random seed to support consistency in tests.
        dense_reward : bool
            Whether to compute dense reward.
        risk_bound_init : float
            Upper risk bound for initialization.
        risk_bound_range: list
            Range of risk bound to train.
        """
        self._dynamics_noise = dynamics_noise
        self._noise_sample_size = noise_sample_size
        self._risk_bound = risk_bound_init
        self._risk_bound_range = risk_bound_range
        self._allocated_risk = 0.0

        super().__init__(
            walls=walls,
            resize_factor=resize_factor,
            action_noise=0.0,
            percent_random_change=percent_random_change,
            start=start,
            goal=goal,
            noise_sample_size=noise_sample_size,
            random_seed=random_seed,
            dense_reward=dense_reward,
        )
        maze_obs_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array(self._walls.shape),
            dtype=np.float32,
        )
        budget_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": maze_obs_space,
                "risk_budget": budget_space,
                "risk_bound": budget_space,
            }
        )
        self.observation_space = maze_obs_space

    def step(self, action, test_mode=False):
        """Perform a step."""
        # Fix random seed at test time for consistent results.
        if test_mode:
            np.random.seed(0)

        sn, reward, done, info = super().step(action)
        if not test_mode:
            # Compute step-wise risk through Monte-Carlo.
            samples = np.random.normal(
                loc=sn,
                scale=self._dynamics_noise,
                size=[self._noise_sample_size, len(sn)],
            )
            count_blocked_samples = 0
            for sn_sample in samples:
                if self._is_blocked(sn_sample):
                    count_blocked_samples += 1

            stepwise_risk = count_blocked_samples * 1.0 / self._noise_sample_size
        else:
            # Skip expensive risk computing at test time.
            stepwise_risk = 0

        # Compute accumulated risk.
        self._allocated_risk = (
            self._allocated_risk + (1.0 - self._allocated_risk) * stepwise_risk
        )

        info["collision"] = stepwise_risk
        info["risk"] = stepwise_risk
        info["risk_bound"] = self._risk_bound
        info["allocated_risk"] = 0.0

        sn_dict = {
            "observation": sn,
            "risk_bound": self._risk_bound,
            "allocated_risk": self._allocated_risk,
        }
        return sn_dict, reward, done, info

    def set_risk_bound(self, risk_bound):
        """Update risk bound."""
        self._risk_bound = risk_bound

    def compute_collision(self, state, noise_sample_size=500):
        """Compute probability of collision given a state, using Monte Carlo."""
        samples = np.random.normal(
            loc=state, scale=self._dynamics_noise, size=[noise_sample_size, len(state)]
        )
        count_blocked_samples = 0
        for sn_sample in samples:
            if self._is_blocked(sn_sample):
                count_blocked_samples += 1

        stepwise_risk = count_blocked_samples * 1.0 / noise_sample_size
        return stepwise_risk

    def reset(
        self,
        risk_bound: float = None,
        random: bool = False,
        noise: [] = None,
        test_mode=False,
    ):
        # Reset allocated risk.
        self._allocated_risk = 0.0

        # Reset risk bound randomly so that the model can be trained to handle different risk bounds.
        if risk_bound is None:
            # Reset risk bound randomly.
            self._risk_bound = np.random.uniform(
                self._risk_bound_range[0], self._risk_bound_range[1]
            )
        else:
            self._risk_bound = risk_bound

        sn_dict = {
            "observation": super().reset(),
            "risk_bound": self._risk_bound,
            "allocated_risk": self._allocated_risk,
        }
        return sn_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize all environments."
    )
    parser.add_argument(
        "-e", "--env-name", default="TwoRooms", help="Select an environment."
    )
    args = parser.parse_args()

    if args.visualize:
        plot_all_environments(WALLS)

    max_episode_steps = 20
    env_name = args.env_name
    resize_factor = 1  # Inflate the environment to increase the difficulty.

    if args.env_name == "FlyTrapBig":
        start = np.array([4, 3], dtype=np.float32)
        goal = np.array([4, 18], dtype=np.float32)
    else:
        start = np.array([2, 5], dtype=np.float32)
        goal = np.array([8, 5], dtype=np.float32)

    env = MazeEnv(env_name, resize_factor, start=start, goal=goal)
    env.reset()
    env.step(np.array([0.1, 0.1]))
    plot_problem(env)
    plt.show()
    print("DONE")
