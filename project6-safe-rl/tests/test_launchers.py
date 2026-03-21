from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_point_gather_grid, train_point_gather_cpo


def test_train_register_with_omnisafe_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    support_envs = ['ExistingEnv-v0']
    fake_env_cls = SimpleNamespace(_support_envs=support_envs)
    monkeypatch.setattr(train_point_gather_cpo, 'SafetyGymnasiumEnv', fake_env_cls)

    train_point_gather_cpo.register_with_omnisafe()
    train_point_gather_cpo.register_with_omnisafe()

    assert support_envs == [
        'ExistingEnv-v0',
        'SafetyPointGather0-v0',
        'SafetyPointGather1-v0',
        'SafetyPointGather2-v0',
    ]


def test_build_train_cfg_reads_yaml() -> None:
    algo, env_id, config = train_point_gather_cpo.build_train_cfg()

    assert algo == 'CPO'
    assert env_id == 'SafetyPointGather1-v0'
    assert config['train_cfgs']['vector_env_nums'] == 1
    assert config['algo_cfgs']['steps_per_epoch'] == 50000


def test_train_main_uses_yaml_config(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: dict[str, Any] = {}

    class FakeAgent:
        def __init__(self, *, algo: str, env_id: str, custom_cfgs: dict[str, Any]) -> None:
            calls['algo'] = algo
            calls['env_id'] = env_id
            calls['custom_cfgs'] = custom_cfgs
            self.agent = SimpleNamespace(logger=SimpleNamespace(log_dir='/tmp/fake-log'))

        def learn(self) -> tuple[float, float, float]:
            return (1.0, 2.0, 3.0)

    monkeypatch.setattr(train_point_gather_cpo, 'register_point_gather_environments', lambda: calls.setdefault('registered', True))
    monkeypatch.setattr(train_point_gather_cpo, 'register_with_omnisafe', lambda: calls.setdefault('omnisafe_registered', True))
    monkeypatch.setattr(train_point_gather_cpo.mp, 'set_start_method', lambda method: calls.setdefault('start_method', method))
    monkeypatch.setattr(train_point_gather_cpo.omnisafe, 'Agent', FakeAgent)

    train_point_gather_cpo.main()
    out = capsys.readouterr().out

    assert calls['registered'] is True
    assert calls['omnisafe_registered'] is True
    assert calls['start_method'] == 'fork'
    assert calls['algo'] == 'CPO'
    assert calls['env_id'] == 'SafetyPointGather1-v0'
    assert calls['custom_cfgs']['train_cfgs']['vector_env_nums'] == 1
    assert 'Starting full Point Gather training' in out
    assert 'Training completed.' in out


def test_grid_register_with_omnisafe_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    support_envs = []
    fake_env_cls = SimpleNamespace(_support_envs=support_envs)
    monkeypatch.setattr(run_point_gather_grid, 'SafetyGymnasiumEnv', fake_env_cls)

    run_point_gather_grid.register_with_omnisafe()
    run_point_gather_grid.register_with_omnisafe()

    assert support_envs == [
        'SafetyPointGather0-v0',
        'SafetyPointGather1-v0',
        'SafetyPointGather2-v0',
    ]


def test_grid_builds_from_yaml_values(monkeypatch: pytest.MonkeyPatch) -> None:
    adds: list[tuple[str, list[Any], bool]] = []

    class FakeGrid:
        def __init__(self, exp_name: str) -> None:
            self.exp_name = exp_name

        def add(self, key: str, vals: list[Any], in_name: bool = False, shorthand: str | None = None) -> None:
            adds.append((key, vals, in_name))

    monkeypatch.setattr(run_point_gather_grid, 'ExperimentGrid', FakeGrid)

    grid = run_point_gather_grid.build_grid()

    assert isinstance(grid, FakeGrid)
    assert ('env_id', ['SafetyPointGather1-v0'], True) in adds
    assert ('seed', [0], True) in adds
    assert ('train_cfgs:vector_env_nums', [1, 1, 1, 1], False) in adds
    assert ('algo_cfgs:steps_per_epoch', [50000, 20000, 20000, 20000], False) in adds


def test_grid_main_runs_and_analyzes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: dict[str, Any] = {}

    class FakeGrid:
        def __init__(self, exp_name: str) -> None:
            self.exp_name = exp_name

        def run(self, thunk: Any, num_pool: int, parent_dir: str | None, gpu_id: Any) -> None:
            calls['run'] = {
                'thunk': thunk,
                'num_pool': num_pool,
                'parent_dir': parent_dir,
                'gpu_id': gpu_id,
            }

        def analyze(
            self,
            parameter: str,
            values: list[Any] | None = None,
            compare_num: int | None = None,
            cost_limit: float | None = None,
            show_image: bool = False,
        ) -> None:
            calls['analyze'] = {
                'parameter': parameter,
                'values': values,
                'compare_num': compare_num,
                'cost_limit': cost_limit,
            }

    monkeypatch.setattr(run_point_gather_grid, 'register_point_gather_environments', lambda: calls.setdefault('registered', True))
    monkeypatch.setattr(run_point_gather_grid, 'register_with_omnisafe', lambda: calls.setdefault('omnisafe_registered', True))
    monkeypatch.setattr(run_point_gather_grid.mp, 'set_start_method', lambda method: calls.setdefault('start_method', method))
    monkeypatch.setattr(run_point_gather_grid, 'build_grid', lambda: FakeGrid('PointGather_Compare'))

    run_point_gather_grid.main()
    out = capsys.readouterr().out

    assert calls['registered'] is True
    assert calls['omnisafe_registered'] is True
    assert calls['start_method'] == 'fork'
    assert calls['run']['num_pool'] == run_point_gather_grid.DEFAULT_NUM_POOL
    assert Path(calls['run']['parent_dir']) == run_point_gather_grid.DEFAULT_PARENT_DIR
    assert calls['analyze']['parameter'] == 'algo'
    assert calls['analyze']['values'] == list(run_point_gather_grid.POINT_GATHER_ALGOS)
    assert 'Running Point Gather grid' in out
