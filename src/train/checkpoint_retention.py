"""Prune old checkpoints — keep the newest N and the best K.

Cosmos never deletes a checkpoint: there is no retention logic anywhere in the
framework, so every save is kept forever. A training checkpoint here is larger
than the 30 GB params-only base (it also carries optimizer state and EMA), so a
long run at a tight `save_iter` fills a filesystem rather than finishing.

Layout this operates on (`checkpoint/dcp.py`)::

    <ckpt_dir>/iter_000000100/…          one directory per save
    <ckpt_dir>/latest_checkpoint.txt     pointer to the most recent one

Safety rules, in order of how much damage getting them wrong would do:

1. **Never delete what `latest_checkpoint.txt` points at.** That is the resume
   target; losing it costs the run.
2. **Never delete the newest directory on disk** even if the pointer disagrees —
   a save that is still in flight has not updated the pointer yet.
3. **Never delete anything younger than `min_age_s`.** Saving can be
   asynchronous (`dcp_async_mode_enabled`), so recency is the cheap proxy for
   "possibly still being written".
4. Rank 0 only, and every failure is swallowed with a warning — a janitor must
   never be the thing that kills a training run.

"Best" needs a metric, and which metric is honest depends on the run: during a
smoke there is only the training loss, which is noisy. Pass `metric_key` once a
validation metric exists (e.g. mean-of-K action MSE) and it will be tracked
instead. With no metric observed, retention degrades to keep-newest-N, which is
the safe default rather than a silent guess.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from cosmos_framework.utils.callback import Callback

_ITER_DIR = re.compile(r"^iter_(\d+)$")


class CheckpointRetention(Callback):
    """Keep the newest `keep_last` checkpoints plus the best `keep_best` by metric."""

    def __init__(
        self,
        keep_last: int = 5,
        keep_best: int = 3,
        metric_key: str | None = None,
        metric_mode: str = "min",
        min_age_s: float = 900.0,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.keep_last = int(keep_last)
        self.keep_best = int(keep_best)
        self.metric_key = metric_key
        self.metric_mode = metric_mode
        self.min_age_s = float(min_age_s)
        self.dry_run = bool(dry_run)
        self._metric_by_iter: dict[int, float] = {}

    # ---- metric tracking --------------------------------------------------
    def on_training_step_end(
        self, model: Any, data_batch: dict, output_batch: dict, loss: Any, iteration: int = 0
    ) -> None:
        if not self.metric_key:
            return
        value = output_batch.get(self.metric_key) if isinstance(output_batch, dict) else None
        if value is None:
            return
        try:
            self._metric_by_iter[int(iteration)] = float(
                value.item() if hasattr(value, "item") else value
            )
        except Exception:  # noqa: BLE001
            pass

    # ---- pruning ----------------------------------------------------------
    def on_save_checkpoint_end(self, model: Any, iteration: int = 0) -> None:
        if os.environ.get("RANK", "0") != "0":
            return
        try:
            self._prune()
        except Exception as exc:  # noqa: BLE001 — never take the run down
            print(f"[ckpt_retention] skipped: {type(exc).__name__}: {exc}", flush=True)

    def _checkpoint_dir(self) -> Path | None:
        """Locate the checkpoint directory under the run's output root.

        Prefers `latest_checkpoint.txt`, but does NOT depend on it: if that
        pointer is missing or renamed we still find the directory by looking for
        `iter_*` children. Depending on the pointer alone would make retention
        silently do nothing — which is the exact failure (a filling disk) this
        callback exists to prevent.
        """
        root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT")
        if not root:
            return None
        base = Path(root)
        pointers = sorted(base.rglob("latest_checkpoint.txt"))
        if pointers:
            return pointers[-1].parent
        with_iters = [
            d for d in base.rglob("iter_*")
            if d.is_dir() and _ITER_DIR.match(d.name) and d.parent != base
        ]
        return max((d.parent for d in with_iters), default=None,
                   key=lambda p: len(list(p.glob("iter_*")))) if with_iters else None

    def _prune(self) -> None:
        ckpt_dir = self._checkpoint_dir()
        if ckpt_dir is None or not ckpt_dir.is_dir():
            return

        entries: list[tuple[int, Path]] = []
        for child in ckpt_dir.iterdir():
            m = _ITER_DIR.match(child.name)
            if m and child.is_dir():
                entries.append((int(m.group(1)), child))
        if len(entries) <= self.keep_last:
            return
        entries.sort(key=lambda e: e[0])

        keep: set[Path] = set()

        # rule 1 — the resume target
        pointer = ckpt_dir / "latest_checkpoint.txt"
        if pointer.exists():
            target = ckpt_dir / pointer.read_text().strip()
            if target.exists():
                keep.add(target)
        # rule 2 — newest on disk regardless of the pointer
        keep.add(entries[-1][1])
        # newest N
        for _, path in entries[-self.keep_last:]:
            keep.add(path)
        # best K, only if a metric was actually observed
        if self.keep_best > 0 and self._metric_by_iter:
            scored = [(it, p) for it, p in entries if it in self._metric_by_iter]
            scored.sort(key=lambda e: self._metric_by_iter[e[0]],
                        reverse=(self.metric_mode == "max"))
            for _, path in scored[: self.keep_best]:
                keep.add(path)

        now = time.time()
        removed, freed = 0, 0
        for _, path in entries:
            if path in keep:
                continue
            # rule 3 — a young directory may still be being written
            if now - path.stat().st_mtime < self.min_age_s:
                continue
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            if self.dry_run:
                print(f"[ckpt_retention] would remove {path.name} ({size/1e9:.1f} GB)", flush=True)
                continue
            shutil.rmtree(path)
            removed += 1
            freed += size
        if removed:
            print(f"[ckpt_retention] removed {removed} checkpoint(s), freed {freed/1e9:.1f} GB; "
                  f"kept {len(keep)}", flush=True)


__all__ = ["CheckpointRetention"]
