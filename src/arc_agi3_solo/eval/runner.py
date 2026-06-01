"""Offline-mode runner for ARC-AGI-3 agents.

Drives an `agents.agent.Agent` subclass against the local `arc_agi.Arcade` in
OFFLINE mode. Designed to work the same locally and on Kaggle (no internet, no
HTTP API server).

Usage (programmatic):

    from arc_agi3_solo.eval.runner import run_one, run_many
    from arc_agi3_solo.agents.random_agent import RandomAgent

    card = run_many(RandomAgent, game_ids=["ar25", "bp35"], env_files_dir="/kaggle/input/arc-prize-2026-arc-agi-3/environment_files")

CLI:

    python -m arc_agi3_solo.eval.runner --agent random --games ar25,bp35
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Type

logger = logging.getLogger(__name__)


def _configure_offline(env_files_dir: str | os.PathLike[str] | None) -> None:
    """Force OFFLINE mode (no socket at all) for local testing.

    Respects an existing OPERATION_MODE if the caller has set one
    explicitly -- on Kaggle the notebook uses 'competition', which still
    talks to a local API server.
    """
    os.environ.setdefault("OPERATION_MODE", "offline")
    if env_files_dir is not None:
        os.environ["ENVIRONMENTS_DIR"] = str(env_files_dir)


def discover_games(env_files_dir: str | os.PathLike[str]) -> list[str]:
    """Return sorted game_ids found under env_files_dir/<game_id>/<hash>/<game_id>.py."""
    root = Path(env_files_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"env_files_dir does not exist: {root}")
    games: set[str] = set()
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for level in sub.iterdir():
            if not level.is_dir():
                continue
            for f in level.glob("*.py"):
                if f.stem == sub.name:
                    games.add(sub.name)
                    break
    return sorted(games)


def run_many(
    agent_cls: Type,
    game_ids: Iterable[str],
    env_files_dir: str | os.PathLike[str] | None = None,
    tags: list[str] | None = None,
    record: bool = False,
) -> object:
    """Run `agent_cls` against each game in sequence and return the scorecard.

    Sequential, not threaded — the Kaggle worker is single-GPU and Swarm
    threading just adds noise to debugging. Re-introduce threading later
    only if it actually helps.
    """
    _configure_offline(env_files_dir)

    from arc_agi import Arcade  # type: ignore[import-not-found]

    arc = Arcade()
    card_id: str = arc.open_scorecard(tags=tags or [])
    logger.info("Opened scorecard %s", card_id)

    for gid in game_ids:
        logger.info("Starting game %s", gid)
        env = arc.make(gid, scorecard_id=card_id, save_recording=record)
        if env is None:
            logger.warning("env make returned None for %s; skipping", gid)
            continue
        agent = agent_cls(
            card_id=card_id,
            game_id=gid,
            agent_name=agent_cls.__name__.lower(),
            arc_env=env,
            tags=tags or [],
        )
        try:
            agent.main()
        except Exception:
            logger.exception("Agent crashed on %s; continuing", gid)

    scorecard = arc.close_scorecard(card_id)
    logger.info("Closed scorecard %s", card_id)
    return scorecard


def _resolve_agent(name: str) -> Type:
    """Map an agent name to its class. Extend as we add agents."""
    name = name.lower()
    if name == "random":
        from arc_agi3_solo.agents.random_agent import RandomAgent
        return RandomAgent
    if name == "graph":
        from arc_agi3_solo.agents.graph_agent import GraphAgent
        return GraphAgent
    if name == "cnn-graph":
        from arc_agi3_solo.agents.cnn_graph_agent import CNNGraphAgent
        return CNNGraphAgent
    raise ValueError(f"unknown agent: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="ARC-AGI-3 offline runner")
    p.add_argument("--agent", required=True, help="agent name, e.g. 'random'")
    p.add_argument(
        "--games",
        default="",
        help="comma-separated game_ids; if empty, all games under --env-files-dir",
    )
    p.add_argument(
        "--env-files-dir",
        default=os.environ.get(
            "ENVIRONMENTS_DIR",
            "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files",
        ),
    )
    p.add_argument("--tags", default="")
    p.add_argument("--record", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    games = (
        [g.strip() for g in args.games.split(",") if g.strip()]
        if args.games
        else discover_games(args.env_files_dir)
    )
    if not games:
        raise SystemExit(f"No games found under {args.env_files_dir}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    agent_cls = _resolve_agent(args.agent)

    scorecard = run_many(agent_cls, games, args.env_files_dir, tags, args.record)

    # Best-effort summary print
    try:
        import json as _json
        print(_json.dumps(scorecard.model_dump(), indent=2))  # type: ignore[attr-defined]
    except Exception:
        print(scorecard)


if __name__ == "__main__":
    main()
