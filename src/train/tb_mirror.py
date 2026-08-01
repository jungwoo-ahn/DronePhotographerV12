"""Mirror every metric Cosmos logs into TensorBoard.

Cosmos ships no TensorBoard writer — its 41 metric call sites all go through
``wandb.log(...)``, each behind an ``if wandb.run:`` gate. So rather than
re-deriving metrics in a callback (which would only ever see what that callback
itself computes), this wraps that single choke point: whatever any callback logs
now or adds later shows up in TensorBoard automatically.

Consequences of that design worth knowing:

* **wandb must be live.** With ``job.wandb_mode=disabled`` the gates are false and
  nothing is logged at all, so nothing can be mirrored. Use ``offline`` — it writes
  to a local run directory and uploads nothing unless you later run ``wandb sync``.
* Only scalars are mirrored. Tables, HTML and media (``DeviceMonitor/prof_data``,
  ``table_data_stats/html``) have no TensorBoard scalar equivalent and are skipped.
* Rank-0 only, since that is where the framework's own logging is gated.

Wire it by importing and calling :func:`install` before the trainer starts; the
launcher does this via ``COSMOS_TB_DIR``.

View with::

    tensorboard --logdir <IMAGINAIRE_OUTPUT_ROOT>/tb --port 6006
    # then from your laptop:  ssh -L 6006:localhost:6006 <login-host>
"""

from __future__ import annotations

import numbers
import os
from pathlib import Path
from typing import Any

_writer = None
_installed = False


def _as_scalar(value: Any) -> float | None:
    """Return a float if this value is a plottable scalar, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        f = float(value)
        return f if f == f and abs(f) != float("inf") else None   # drop NaN/inf
    # 0-d tensors / numpy scalars
    item = getattr(value, "item", None)
    if callable(item):
        try:
            got = item()
        except Exception:  # noqa: BLE001  (multi-element tensor, etc.)
            return None
        return _as_scalar(got)
    return None


def install(log_dir: str | Path | None = None, force: bool = False) -> bool:
    """Wrap ``wandb.log`` so scalars are also written to TensorBoard.

    TIMING MATTERS: before ``wandb.init()``, ``wandb.log`` is a placeholder
    (``PreInitCallable``) that init REPLACES — so a wrapper installed at process
    start is silently discarded and nothing is ever mirrored. Install from
    :class:`TensorBoardMirror`'s ``on_train_start``, which runs after init.

    Returns True if the mirror is active. Safe to call repeatedly, and safe when
    wandb or tensorboard is missing — it degrades to a no-op rather than taking
    the training run down with it.
    """
    global _writer, _installed
    if _installed and not force:
        return _writer is not None

    _installed = True
    directory = log_dir or os.environ.get("COSMOS_TB_DIR")
    if not directory:
        return False

    try:
        import wandb
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        print(f"[tb_mirror] disabled ({type(exc).__name__}: {exc})", flush=True)
        return False

    # Rank check via torchrun's env, since this runs before distributed init.
    if os.environ.get("RANK", "0") != "0":
        return False

    Path(directory).mkdir(parents=True, exist_ok=True)
    _writer = SummaryWriter(log_dir=str(directory))
    original_log = wandb.log

    def log_and_mirror(data=None, step=None, **kwargs):
        result = original_log(data, step=step, **kwargs)
        if _writer is not None and isinstance(data, dict):
            # Cosmos passes the x-axis both as `step=` and as an "iteration" entry.
            it = step if step is not None else _as_scalar(data.get("iteration"))
            for key, value in data.items():
                if key == "iteration":
                    continue
                scalar = _as_scalar(value)
                if scalar is not None:
                    _writer.add_scalar(key, scalar, global_step=int(it) if it is not None else None)
            _writer.flush()
        return result

    wandb.log = log_and_mirror
    print(f"[tb_mirror] mirroring wandb.log -> TensorBoard at {directory}", flush=True)
    return True


def close() -> None:
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None


from cosmos_framework.utils.callback import Callback  # noqa: E402


class TensorBoardMirror(Callback):
    """Installs the mirror at ``on_train_start`` — i.e. after ``wandb.init()``.

    Deliberately a callback rather than a process-start hook: wandb replaces
    ``wandb.log`` during init, so anything wrapped earlier is thrown away.
    """

    def __init__(self, log_dir: str | None = None) -> None:
        super().__init__()
        self.log_dir = log_dir

    def on_train_start(self, model: Any = None, iteration: int = 0) -> None:
        install(self.log_dir, force=True)

    def on_train_end(self, model: Any = None, iteration: int = 0) -> None:
        close()


__all__ = ["install", "close", "TensorBoardMirror"]
