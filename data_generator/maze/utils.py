import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def compute_execution_risk(collisions: []):
    """
    Compute execution risk given a sequence of collision probabilities through dynamic programming.

    Parameters
    ----------
    collisions : []
        A list of step collision probabilities.

    Returns
    -------
    overall_risk: float
        Overall Execution risk.
    """
    overall_risk = 0.0
    for rb in collisions[::-1]:
        overall_risk = rb + (1 - rb) * overall_risk
    return overall_risk


def resize_walls(walls: np.array, factor: int):
    """
    Increase the environment size by rescaling.

    Parameters
    ----------
    walls : np.array
        0/1 array indicating obstacle locations.
    factor : int
        Factor by which to rescale the environment

    Returns
    -------
    walls : np.array
        Updated obstacle locations.
    """
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


def compute_dense_reward(state, goal):
    """Compute negative euclidean distance."""
    batch_mode = True
    if len(state.shape) == 1:
        state = state[None]
        batch_mode = False
    if len(goal.shape) == 1:
        goal = goal[None]
    dist = np.linalg.norm(state - goal, axis=1)
    if batch_mode:
        return -dist[:, np.newaxis]  # adhere to imagination format
    return -dist[0]


def plot_walls(walls, fig=None, ax=None):
    """Plot walls."""
    walls = walls.T
    (height, width) = walls.shape
    if fig is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
    scaling = np.max([height, width])
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j + 1]) / float(scaling)
        y0 = np.array([i, i]) / float(scaling)
        y1 = np.array([i + 1, i + 1]) / float(scaling)
        ax.fill_between(x, y0, y1, color="grey")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_walls_helper(walls):
    """Helper to plot walls."""
    walls = walls.T
    (height, width) = walls.shape
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j + 1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i + 1, i + 1]) / float(height)
        plt.fill_between(x, y0, y1, color="grey")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([])
    plt.yticks([])


def plot_all_environments(walls_all):
    """Plot a set of environments."""
    plt.figure(figsize=(16, 7))
    walls = list(walls_all.items())
    for index, (name, walls) in enumerate(walls[:21]):
        plt.subplot(3, 7, index + 1)
        plt.title(name)
        plot_walls_helper(walls)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.suptitle("Navigation Environments", fontsize=20)
    plt.show()


def plot_problem(env):
    """Plot an environment with start and goal."""
    goal = env.goal
    start = env.start
    walls = env._walls
    (height, width) = walls.shape
    scaling = np.max([height, width])

    start[0] = start[0] / scaling
    start[1] = start[1] / scaling
    goal[0] = goal[0] / scaling
    goal[1] = goal[1] / scaling

    plot_walls(env._walls)
    plt.scatter([start[0]], [start[1]], marker="*", color="green", s=200, label="start")
    plt.scatter([goal[0]], [goal[1]], marker="+", color="red", s=200, label="goal")
    plt.legend()
    plt.title(env.env_name)


def plot_problem_path(env, path, show_fig=False):
    """Plot a single path."""
    goal = env.goal
    start = env.start
    walls = env._walls
    (height, width) = walls.shape

    start[0] = start[0] / width
    start[1] = start[1] / height
    goal[0] = goal[0] / width
    goal[1] = goal[1] / height

    plot_walls(env._walls)
    if isinstance(path, list):
        path = np.vstack(path)

    path_x = path[:, 0] / width
    path_y = path[:, 1] / height
    plt.plot(path_x, path_y, "b-o", alpha=0.3)
    plt.scatter([start[0]], [start[1]], marker="*", color="green", s=200, label="start")
    plt.scatter([goal[0]], [goal[1]], marker="+", color="red", s=200, label="goal")
    plt.legend()
    plt.title(env.env_name)
    if show_fig:
        plt.show()


def plot_problem_paths(
    env,
    paths: [],
    risk_bounds: [],
    times: [],
    fig_dir: str,
    show_fig: bool = True,
    show_baseline: bool = False,
    pos_offset: float = 0.0,
):
    """
    Plot a set of planned paths in the same environment.

    Parameters
    ----------
    env : data_generator.maze.maze_env.RiskConditionedMaze
        Planning environment.
    paths : []
        A list of planned paths.
    risk_bounds : []
        A list of risk bounds.
    times : []
        A list of running times.
    fig_dir : str
        Path to save figure.
    show_fig : bool
        Whether to pop up figure.
    show_baseline : bool
        Whether to plot pre-generated baseline paths.
    pos_offset : float
        Offset of positions due to discretization of grid.
    """
    # Visualize environment.
    goal = env.goal
    start = env.start
    walls = env._walls
    (height, width) = walls.shape
    scaling = np.max([height, width])

    start[0] = start[0] / scaling
    start[1] = (start[1] + pos_offset) / scaling
    goal[0] = goal[0] / scaling
    goal[1] = (goal[1] + pos_offset) / scaling

    fig, ax = plt.subplots()
    plot_walls(env._walls, fig, ax)
    ax.scatter([start[0]], [start[1]], marker="*", color="green", s=200, label="start")
    ax.scatter([goal[0]], [goal[1]], marker="+", color="red", s=200, label="goal")

    # Obtain baseline paths from IRA (pSulu).
    psulu_ers = []
    if show_baseline:
        # Obtain pSulu paths generated from risk bounds of [0.1, 0.2, 0.3].
        psulu_paths = [
            [
                [2.0, 5.0],
                [2.969107, 4.0],
                [3.969107, 3.020893],
                [4.969107, 3.020893],
                [5.969107, 3.020893],
                [6.969107, 3.020893],
                [7.6608406, 4.020893],
                [8.6608406, 4.6608406],
            ],
            [
                [2.0, 5.0],
                [2.8406971, 4.0],
                [3.8406971, 3.1493029],
                [4.8406971, 3.1493029],
                [5.8406971, 3.1493029],
                [6.8406971, 3.1493029],
                [7.6608406, 4.1493029],
                [8.6608406, 4.6608406],
            ],
            [
                [2.0, 5.0],
                [2.7585149, 4.0],
                [3.7585149, 3.2314851],
                [4.7585149, 3.2314851],
                [5.7585149, 3.2314851],
                [6.7585149, 3.2314851],
                [7.7585149, 4.2314851],
                [8.6608406, 4.6608406],
            ],
        ]
        psulu_times = [0.155, 0.153, 0.155]
        psulu_paths = np.array(psulu_paths)

        # Compute risk for IRA paths.
        for psulu_path in psulu_paths:
            psulu_path_risk = [env.compute_collision(state) for state in psulu_path]
            psulu_er = compute_execution_risk(psulu_path_risk)
            psulu_ers.append(psulu_er)

    # Visualize paths.
    colors = ["cyan", "magenta", "orange"]
    for i, path_dict in enumerate(paths):
        path = path_dict["observations"]
        next_path = path_dict["next_observations"]
        if isinstance(path, list):
            path = np.vstack(path)
            next_path = np.vstack(next_path)
        path = np.vstack((path, next_path[-1:]))
        collisions = [env.compute_collision(path[i]) for i in range(path.shape[0])]
        er = compute_execution_risk(collisions)

        dist = np.sum(np.sqrt(np.sum((path[1:] - path[:-1]) ** 2, -1)))
        path_x = path[:, 0] / scaling
        path_y = (path[:, 1] + pos_offset) / scaling
        ax.plot(
            path_x,
            path_y,
            "o-",
            color=colors[i],
            alpha=0.3,
            label="Delta: {}, Ours.".format(risk_bounds[i]),
        )

        print(
            "Ours: Delta: {}, Risk: {:.3f}, Distance: {:.6f}, time: {:.5f}".format(
                risk_bounds[i], er, dist, times[i]
            )
        )

        if show_baseline:
            psulu_dist = np.sum(
                np.sqrt(np.sum((psulu_paths[i, 1:] - psulu_paths[i, :-1]) ** 2, -1))
            )
            path_x_psulu = psulu_paths[i, :, 0] / scaling
            path_y_psulu = psulu_paths[i, :, 1] / scaling
            ax.plot(
                path_x_psulu,
                path_y_psulu,
                "s-",
                color=colors[i],
                alpha=0.3,
                label="Delta: {}, IRA.".format(risk_bounds[i]),
            )
            print(
                "IRA: Delta: {}, Risk: {:.3f}, Distance: {:.3f}, time: {:.3f}".format(
                    risk_bounds[i], psulu_ers[i], psulu_dist, psulu_times[i]
                )
            )

    ax.legend()
    ax.set_title(env.env_name)
    fig.savefig(fig_dir + "/policy_vis.png", dpi=320)
    if show_fig:
        plt.show()


def plot_paths_from_different_start(
    env,
    paths: [],
    fig_dir: str,
    show_fig: bool = True,
    pos_offset: float = 0.0,
):
    """
    Plot a set of planned paths in the same environment.

    Parameters
    ----------
    env : data_generator.maze.maze_env.RiskConditionedMaze
        Planning environment.
    paths : []
        A list of planned paths.
    fig_dir : str
        Path to save figure.
    show_fig : bool
        Whether to pop up figure.
    pos_offset : float
        Offset of positions due to discretization of grid.
    """
    # Visualize environment.
    goal = env.goal
    start = env.start
    walls = env._walls
    (height, width) = walls.shape
    scaling = np.max([height, width])

    start[0] = start[0] / scaling
    start[1] = (start[1] + pos_offset) / scaling
    goal[0] = goal[0] / scaling
    goal[1] = (goal[1] + pos_offset) / scaling

    fig, ax = plt.subplots()
    plot_walls(env._walls, fig, ax)
    ax.scatter([goal[0]], [goal[1]], marker="+", color="red", s=200, label="goal")

    # Visualize paths.
    for i, path_dict in enumerate(paths):
        path = path_dict["observations"]
        next_path = path_dict["next_observations"]
        if isinstance(path, list):
            path = np.vstack(path)
            next_path = np.vstack(next_path)
        path = np.vstack((path, next_path[-1:]))
        path_x = path[:, 0] / scaling
        path_y = (path[:, 1] + pos_offset) / scaling
        if i == 0:
            ax.plot(
                path_x, path_y, "o-", color="grey", alpha=0.3, label="Delta: 0.2, Ours."
            )
        else:
            ax.plot(path_x, path_y, "o-", color="grey", alpha=0.3)

    ax.legend(prop={"size": 15})
    ax.set_title(env.env_name)
    fig.savefig(fig_dir + "/policy_vis_multiple_start.png", dpi=320)
    if show_fig:
        plt.show()
