from __future__ import annotations

import math

import numpy as np
import safety_gymnasium
from typing import Any

from custom_env.point_gather.register_env import register_point_gather_environments


def make_env() -> Any:
    register_point_gather_environments()
    return safety_gymnasium.make('SafetyPointGather1-v0')


def test_point_gather_env_resets_and_matches_observation_space() -> None:
    env = make_env()
    obs, info = env.reset(seed=1)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    env.close()


def test_point_gather_reward_cost_and_goal_semantics() -> None:
    env = make_env()
    env.reset(seed=1)
    task = env.unwrapped.task

    robot_x, robot_y = task._agent_xy()
    task._object_x[:3] = np.array([robot_x, robot_x, robot_x + 10.0], dtype=np.float64)
    task._object_y[:3] = np.array([robot_y, robot_y, robot_y + 10.0], dtype=np.float64)
    task._object_type[:3] = np.array([0, 1, 0], dtype=np.int8)
    task._num_objects = 3

    reward = task.calculate_reward()
    cost_info = task.calculate_cost()

    assert reward == 9.0
    assert cost_info['cost_sum'] == 1.0
    assert cost_info['apples'] == 1.0
    assert cost_info['bombs'] == 1.0
    assert task._num_objects == 1
    assert not task.goal_achieved

    task._object_x[0] = robot_x
    task._object_y[0] = robot_y
    task._object_type[0] = 0
    final_reward = task.calculate_reward()

    assert final_reward == 10.0
    assert task.goal_achieved

    env.close()


def test_point_gather_sensor_bins_keep_nearest_object() -> None:
    env = make_env()
    env.reset(seed=1)
    task = env.unwrapped.task

    robot_x, robot_y = task._agent_xy()
    yaw = task._agent_yaw()
    angle = yaw
    far_dist = 5.0
    near_dist = 2.0

    task._object_x[:2] = np.array(
        [
            robot_x + far_dist * math.cos(angle),
            robot_x + near_dist * math.cos(angle),
        ],
        dtype=np.float64,
    )
    task._object_y[:2] = np.array(
        [
            robot_y + far_dist * math.sin(angle),
            robot_y + near_dist * math.sin(angle),
        ],
        dtype=np.float64,
    )
    task._object_type[:2] = np.array([0, 0], dtype=np.int8)
    task._num_objects = 2

    apple_readings, bomb_readings = task._get_readings()

    assert bomb_readings.max() == 0.0
    assert apple_readings.max() == 1.0 - near_dist / task.sensor_range

    env.close()
