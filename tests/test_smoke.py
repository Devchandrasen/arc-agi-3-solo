"""End-to-end smoke test: random agent runs N steps without crashing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_FILES_DIR = ROOT / "data" / "environment_files"


def _data_available() -> bool:
    return ENV_FILES_DIR.is_dir() and any(ENV_FILES_DIR.iterdir())


@pytest.mark.skipif(not _data_available(), reason="competition data not downloaded yet")
def test_discover_games_finds_at_least_one() -> None:
    from arc_agi3_solo.eval.runner import discover_games

    games = discover_games(ENV_FILES_DIR)
    assert len(games) > 0, f"no games discovered under {ENV_FILES_DIR}"


@pytest.mark.skipif(not _data_available(), reason="competition data not downloaded yet")
def test_random_agent_runs_one_game_offline() -> None:
    """Run RandomAgent against one game for a small budget; assert a scorecard comes back."""
    from arc_agi3_solo.agents.random_agent import RandomAgent
    from arc_agi3_solo.eval.runner import discover_games, run_many

    games = discover_games(ENV_FILES_DIR)
    target = games[0]

    # Cap the agent's exploration so the test is fast.
    RandomAgent.MAX_ACTIONS = 50

    scorecard = run_many(
        RandomAgent,
        game_ids=[target],
        env_files_dir=ENV_FILES_DIR,
        tags=["smoke"],
    )
    assert scorecard is not None
    dumped = scorecard.model_dump()  # pydantic
    # Scorecard contains all 25 environments (whether played or not).
    # The played one should have non-zero actions across its runs.
    target_env = next(
        (e for e in dumped["environments"] if e["id"].startswith(target)), None
    )
    assert target_env is not None, f"{target} not in scorecard environments"
    assert target_env["actions"] > 0, f"agent took 0 actions on {target}: {target_env}"
    assert dumped["competition_mode"] is True, "expected competition (offline) mode"


def test_runner_module_imports() -> None:
    """The runner module loads even without competition data; just an import-time sanity."""
    import arc_agi3_solo.eval.runner as r  # noqa: F401
