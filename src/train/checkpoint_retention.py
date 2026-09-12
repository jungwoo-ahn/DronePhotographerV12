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
smoke there is only the training loss, which is noisy. `metric_key` is looked up
first in `output_batch` and then on `src.train.metric_bus`, which is how the
held-out loss reaches here — `HeldOutLossProbe` computes it and logs it to wandb,
and nothing else could see it, so `keep_best` had been degrading to keep-newest
for every run so far. With no metric observed it still degrades to keep-newest-N,
which is the safe default rather than a silent guess.

`keep_every` keeps the milestones a long run is judged on (every 10k here)
independently of both rules above: those checkpoints are the record of the run and
must survive however the loss moves and however many saves come after them.

Two rules exist because this deleted `iter_000040000` — the checkpoint every
published v5 number was measured on, and the best model the project had.

5. **A `KEEP` file inside a checkpoint directory pins it forever.** `keep_best`
   can only rank what THIS process measured, so a checkpoint inherited from an
   earlier run is not a "best" candidate at all — it is just an old directory,
   and it gets deleted while a worse checkpoint is kept under the name "best".
   Importance that comes from outside training (this is the one we evaluated,
   this is the one we shipped) cannot be inferred from a loss curve, so it is
   declared by hand: `touch <ckpt>/iter_000040000/KEEP`.
6. **Never delete the checkpoint this process resumed from, while it runs.** If
   this link crashes, that is what the next one falls back to. The pin lasts only
   as long as the process, so a long chain does not accumulate pins.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from cosmos_framework.utils.callback import Callback
except ImportError:  # pragma: no cover - lets the pruning logic be tested without the framework
    class Callback:  # type: ignore[no-redef]
        pass

from src.train import metric_bus

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
        keep_every: int = 0,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.keep_last = int(keep_last)
        self.keep_best = int(keep_best)
        self.metric_key = metric_key
        self.metric_mode = metric_mode
        self.min_age_s = float(min_age_s)
        self.keep_every = int(keep_every)
        self.dry_run = bool(dry_run)
        self._metric_by_iter: dict[int, float] = {}
        self._resumed_from: Path | None = None
        self._looked_for_origin = False

    # ---- metric tracking --------------------------------------------------
    def _note_resume_origin(self) -> None:
        """What `latest_checkpoint.txt` pointed at before this process saved anything.

        Read on the first training step, not on the first prune: pruning happens after a
        save, by which time the pointer has already moved to the new checkpoint and the
        thing we resumed from looks like any other old directory.
        """
        if self._looked_for_origin:
            return
        self._looked_for_origin = True
        try:
            d = self._checkpoint_dir()
            if d is None:
                return
            pointer = d / "latest_checkpoint.txt"
            if pointer.exists():
                target = d / pointer.read_text().strip()
                if target.is_dir():
                    self._resumed_from = target
        except Exception:  # noqa: BLE001
            pass

    def on_training_step_end(
        self, model: Any, data_batch: dict, output_batch: dict, loss: Any, iteration: int = 0
    ) -> None:
        self._note_resume_origin()
        if not self.metric_key:
            return
        value = output_batch.get(self.metric_key) if isinstance(output_batch, dict) else None
        if value is None:
            # Not in the step output: try the noticeboard, where a probe that computes on its
            # own schedule leaves its number. Take it only if it was published at THIS
            # iteration, so a stale value cannot be credited to a later checkpoint.
            posted = metric_bus.latest(self.metric_key)
            if posted is None or posted[0] != int(iteration):
                return
            value = posted[1]
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
        """THIS run's checkpoint directory.

        Ask the config, which knows: `job.path_local` is
        `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>`, the same path the checkpointer
        writes to.

        It used to guess instead — glob every `latest_checkpoint.txt` under the output root
        and take `sorted(...)[-1]`. With `camera_policy_nano_v5` and
        `camera_policy_nano_v5_smoke` both present, `/` sorts before `_`, so the smoke run
        came last and retention spent every save tidying a directory with nothing in it
        while the real run grew to seven checkpoints — 960 GB — without one line of output.
        A janitor that cleans the wrong room is worse than no janitor: it looks like it works.

        Falls back to the old search only when there is no config (direct construction in a
        test), and then insists on a UNIQUE candidate rather than picking one.
        """
        cfg = getattr(self, "config", None)
        job = getattr(cfg, "job", None) if cfg is not None else None
        if job is not None and getattr(job, "path_local", None):
            return Path(job.path_local) / "checkpoints"
        root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT")
        if not root:
            return None
        pointers = sorted(Path(root).rglob("latest_checkpoint.txt"))
        if len(pointers) == 1:
            return pointers[0].parent
        if pointers:
            print(f"[ckpt_retention] {len(pointers)} runs under {root} and no config to say "
                  "which is mine — pruning nothing rather than the wrong one", flush=True)
        return None

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
        # milestones — the record of the run, kept regardless of the loss or of how many
        # saves come after them
        if self.keep_every > 0:
            for it, path in entries:
                if it > 0 and it % self.keep_every == 0:
                    keep.add(path)
        # rule 6 — what this process resumed from, for as long as it runs
        if self._resumed_from is not None and self._resumed_from.is_dir():
            keep.add(self._resumed_from)
        # rule 5 — a hand-placed pin, which outranks everything and never expires
        for _, path in entries:
            if (path / "KEEP").exists():
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
            best = ""
            if self._metric_by_iter:
                b = min(self._metric_by_iter.items(), key=lambda kv: kv[1]) \
                    if self.metric_mode == "min" else \
                    max(self._metric_by_iter.items(), key=lambda kv: kv[1])
                best = f"; best {self.metric_key}={b[1]:.6g} at iter {b[0]}"
            print(f"[ckpt_retention] removed {removed} checkpoint(s), freed {freed/1e9:.1f} GB; "
                  f"kept {len(keep)}{best}", flush=True)


__all__ = ["CheckpointRetention"]
