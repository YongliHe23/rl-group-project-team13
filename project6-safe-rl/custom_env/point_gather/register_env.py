"""Registration of custom Point Gather environments with Safety-Gymnasium."""

from __future__ import annotations

import safety_gymnasium.tasks as safety_tasks
from safety_gymnasium import register

from .point_gather_task import GatherLevel0, GatherLevel1, GatherLevel2


def register_point_gather_environments() -> None:
    """Register Point Gather variants for Safety-Gymnasium's builder."""
    setattr(safety_tasks, 'GatherLevel0', GatherLevel0)
    setattr(safety_tasks, 'GatherLevel1', GatherLevel1)
    setattr(safety_tasks, 'GatherLevel2', GatherLevel2)

    register(
        id='SafetyPointGather0-v0',
        entry_point='safety_gymnasium.builder:Builder',
        kwargs={
            'task_id': 'SafetyPointGather0-v0',
            'config': {
                'agent_name': 'Point',
            },
        },
        max_episode_steps=15,
    )

    register(
        id='SafetyPointGather1-v0',
        entry_point='safety_gymnasium.builder:Builder',
        kwargs={
            'task_id': 'SafetyPointGather1-v0',
            'config': {
                'agent_name': 'Point',
            },
        },
        max_episode_steps=15,
    )

    register(
        id='SafetyPointGather2-v0',
        entry_point='safety_gymnasium.builder:Builder',
        kwargs={
            'task_id': 'SafetyPointGather2-v0',
            'config': {
                'agent_name': 'Point',
            },
        },
        max_episode_steps=15,
    )
