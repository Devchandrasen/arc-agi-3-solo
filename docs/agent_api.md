# Agent API (arc-agi 0.9.8 / arcengine 0.9.3)

Reference: `vendor/arc-agi-3-agents/agents/agent.py`. Distilled here so you don't have to re-read the whole vendor tree.

## What you implement

Subclass `agents.agent.Agent` and override two methods:

```python
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

class MyAgent(Agent):
    MAX_ACTIONS: int = 80  # safety cap to stop the loop

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        ...
```

Everything else (env stepping, recording, scorecard, threading) is provided by `Agent.main()` and `Swarm`.

## FrameData

```python
class FrameData(BaseModel):
    game_id: str
    frame: list[list[list[int]]]   # shape (N, 64, 64); usually N=1, sometimes >1
    state: GameState               # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int          # NEW name (was `score` pre-0.9.3)
    win_levels: int                # NEW name (was `win_score` pre-0.9.3)
    guid: Optional[str]
    full_reset: bool
    available_actions: list[int]   # subset of [0..7]; 0=RESET, 6=ACTION6
```

Cell values are 0..15 (4-bit). The `frame` is a list of grids — most games return one grid per step, but a few return multiple (e.g. animation frames).

## GameAction

```python
class GameAction(Enum):
    RESET   = (0, SimpleAction)
    ACTION1 = (1, SimpleAction)   # arrow / button
    ACTION2 = (2, SimpleAction)
    ACTION3 = (3, SimpleAction)
    ACTION4 = (4, SimpleAction)
    ACTION5 = (5, SimpleAction)
    ACTION6 = (6, ComplexAction)  # click at (x, y), x/y in [0, 63]
    ACTION7 = (7, SimpleAction)   # added in 0.9.2
```

For ACTION6 you must call `action.set_data({"x": x, "y": y})` before returning it. Optionally set `action.reasoning` (≤16 KB JSON-serializable) for traceability.

## Loop (provided by Agent.main)

```
while not is_done(frames, frames[-1]) and action_counter <= MAX_ACTIONS:
    action = choose_action(frames, latest_frame)
    frame  = take_action(action)        # calls self.arc_env.step(...)
    if frame: append_frame(frame)
    action_counter += 1
cleanup()
```

`action_counter` is bumped even when the env returns no frame, so set MAX_ACTIONS generously if you want long exploration.

## Environment access

`self.arc_env: EnvironmentWrapper` is injected by the Swarm or runner. `Arcade()` picks an `OperationMode` from the `OPERATION_MODE` env var:

- `normal` (default) — local games + an anonymous API key fetched from `https://three.arcprize.org`. Handy for dev with internet.
- `online` — API only (uses `ARC_API_KEY`). Not useful on Kaggle.
- `offline` — local games, no network at all.
- `competition` — offline + emits a `competition_mode: True` flag in the scorecard. **This is the Kaggle mode.**

`ENVIRONMENTS_DIR` controls where the per-game `.py` files are read from. On Kaggle that's `/kaggle/input/arc-prize-2026-arc-agi-3/environment_files`; locally it's `data/environment_files` after `kaggle competitions download`.

Each game lives at `<ENVIRONMENTS_DIR>/<game_id>/<level_hash>/<game_id>.py` plus a `metadata.json`.

## Scorecard

`Arcade.close_scorecard(card_id) -> EnvironmentScorecard`. The pydantic dump looks like:

```python
{
  "card_id": "...", "competition_mode": True,
  "total_environments": 25, "total_environments_completed": 0,
  "total_levels": 183, "total_levels_completed": 0, "total_actions": 51,
  "environments": [
    {
      "id": "ar25-0c556536",            # game_id-level_hash
      "level_count": 8, "levels_completed": 0,
      "actions": 51, "resets": 1, "completed": False,
      "runs": [{
        "guid": "...", "score": 0.0, "actions": 51, "resets": 1,
        "state": "NOT_FINISHED", "levels_completed": 0,
        "level_scores": [0.0]*8,
        "level_actions": [51, 0, 0, 0, 0, 0, 0, 0],
        "level_baseline_actions": [32, 50, 75, 37, 89, 159, 233, 73],   # human reference
      }],
    },
    ...
  ],
  "tags_scores": [
    {"id": "keyboard_click", "score": 0.0, "actions": 51, "number_of_levels": 93, "number_of_environments": 13},
    ...
  ],
}
```

Per-game key takeaways:
- `level_count` is the total levels in this environment.
- `levels_completed` is what we've actually beaten — that's the leaderboard signal.
- `level_baseline_actions` is the **human-action count per level**. The competition's "5× human cap" guidance is your action budget per level.
- `tags_scores` aggregates by interaction style (`keyboard_click`, `click`, `keyboard`) — useful when deciding what kinds of games to optimize for first.

The scorecard is generated server-side from per-step events; we just have to keep playing and the engine does the bookkeeping. Write the dump to `/kaggle/working/scorecard.json` at the end of the notebook.

## Things to remember

- **`score` is gone.** Use `levels_completed`. The `Card` schema in `arcengine` still calls aggregated values `score`, but per-frame it's `levels_completed`.
- **Frame can have multiple grids** (`len(frame) > 1`). Heuristic agents in vendor take `frame[-1]` as "the current grid"; an animation flicker is a signal in itself.
- **`available_actions` shrinks** depending on game state (e.g. menus expose only ACTION1; mid-level may expose ACTION6 only). Always intersect your action set with this.
- **`x, y` are in [0, 63]** for ACTION6 — that's 4096 candidate clicks. Naive enumeration explodes the search; use segmentation (see `vendor/just-explore/agents/heuristic_agent.py::FrameProcessor.segment_frame`).
- **`MAX_ACTIONS=80`** is the default and a hard limit on small-budget local debugging. The competition runtime uses much higher caps.
- **Recordings are JSONL** under `RECORDINGS_DIR`; useful for replay-based unit tests of an agent without booting the engine.
