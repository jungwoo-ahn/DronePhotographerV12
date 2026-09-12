"""A one-room noticeboard for metrics that one callback computes and another needs.

`HeldOutLossProbe` computes the held-out loss; `CheckpointRetention` decides which checkpoint
is "best". Nothing connects them: the probe logs to wandb, and retention reads
`output_batch`, which the probe does not write to. Without a channel, `keep_best` silently
degrades to keep-newest — retention says so in its docstring, and that is what has been
happening.

This is that channel, and it is deliberately the smallest thing that works: a module-level
dict, written by whoever computes a number and read by whoever needs it. No ordering
requirement beyond "publish before you read", which the callback order already gives (the
probe is registered ahead of retention, and both fire on the same iterations).

Not a general metrics system. If a second consumer ever appears, that is the moment to ask
for one — until then a dict is honest about how much coupling actually exists.
"""

from __future__ import annotations

_LATEST: dict[str, tuple[int, float]] = {}


def publish(payload: dict[str, float], iteration: int) -> None:
    """Record every finite scalar in `payload`, stamped with the iteration."""
    for key, value in payload.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == v and abs(v) != float("inf"):      # drop NaN / inf rather than rank on them
            _LATEST[key] = (int(iteration), v)


def latest(key: str) -> tuple[int, float] | None:
    """(iteration, value) most recently published under `key`, or None."""
    return _LATEST.get(key)


def keys() -> list[str]:
    return sorted(_LATEST)


def clear() -> None:
    """Tests only — a process trains one run."""
    _LATEST.clear()


__all__ = ["publish", "latest", "keys", "clear"]
