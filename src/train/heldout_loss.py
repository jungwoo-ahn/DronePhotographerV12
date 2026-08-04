"""Held-out loss, measured through `training_step` rather than the validation path.

Why this exists: the framework's validation path is a stub. `OmniMoTModel.validation_step`
is a bare `pass` returning None, so setting `run_validation=True` does not produce a
number — it takes the run down. (`--dryrun` cannot catch that: the config resolves
fine and the stub is only reached at runtime.) So the config keeps validation off and
the held-out number is computed here instead, by calling `training_step` under
`no_grad` on batches the model never trains on.

Three design choices make the resulting curve readable rather than noise:

1. FIXED BATCHES. The held-out batches are drawn once at train start and reused at
   every evaluation. Re-drawing would stack batch-sampling noise on top of (2).

2. FIXED NOISE. The flow-matching loss samples a random sigma and a random noise
   tensor per call, and that variance dwarfs the training signal — the same effect
   makes `goal_dep/ratio` swing between 0.9 and 3.7 on a model that is genuinely
   using its goal. Seeding per batch index makes every evaluation ask the identical
   question ("loss on these batches at these noise levels"), so a change in the
   number is a change in the model.

3. A TRAIN CONTROL, measured the same way. `train/loss` is a running average over
   fresh batches at random sigma; comparing it against a fixed-batch fixed-noise val
   estimate would show a "generalization gap" that is mostly a difference of
   estimators. So a cached set of TRAIN batches goes through the identical procedure,
   and `heldout/gap` is the difference between two comparable numbers.

The split itself is a scene-level holdout: the export writes episodes grouped by
placement and `CameraPoseLeRobotDataset` takes a contiguous tail, so held-out
placements are scenes the model has never rendered — not a reshuffle of the same
scenes, which would measure memorisation.

Caveat worth keeping in mind when reading the output: on this project a flow-matching
loss is only loosely tied to action quality. v11 found its sampled action MSE was
ANTI-correlated with closed-loop success. Treat this as a fit/overfit tripwire;
closed-loop rollout remains the metric that decides whether a checkpoint is good.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import wandb
except Exception:  # noqa: BLE001
    wandb = None  # type: ignore[assignment]

from cosmos_framework.utils.callback import Callback

_LOSS_KEYS = (
    "flow_matching_loss_action",
    "flow_matching_loss_vision",
)


class HeldOutLossProbe(Callback):
    """Logs `heldout/*` every `every_n` iterations."""

    def __init__(
        self,
        every_n: int = 500,
        num_batches: int = 2,
        seed: int = 1234,
        loss_key: str = "flow_matching_loss_action",
    ) -> None:
        # `num_batches` default 2, not 4: the val split is 200 episodes and sequence
        # packing turns that into exactly 2 batches per epoch. Asking for more used to
        # raise StopIteration inside `on_train_start`, which the guard below would
        # swallow — the probe would disable itself and log nothing, silently.
        super().__init__()
        self.every_n = int(every_n)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.loss_key = loss_key
        self._val_batches: list[dict] | None = None
        self._train_batches: list[dict] = []
        self._failed = False

    # ---- setup ------------------------------------------------------------
    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        """Cache the held-out batches once.

        Deliberately tolerant: a probe must never be the reason a training run dies.
        If the val dataloader cannot be built, the probe disables itself and says so.
        """
        try:
            from cosmos_framework.utils.lazy_config.instantiate import instantiate

            cfg = getattr(self, "config", None)
            loader_cfg = getattr(cfg, "dataloader_val", None) if cfg is not None else None
            if loader_cfg is None:
                raise RuntimeError("config.dataloader_val is not defined")
            loader = instantiate(loader_cfg)
            batches: list[dict] = []
            for _pass in range(2):               # a second pass re-iterates the epoch
                for batch in loader:
                    batches.append(batch)
                    if len(batches) >= self.num_batches:
                        break
                if len(batches) >= self.num_batches:
                    break
            if not batches:
                raise RuntimeError("val dataloader yielded no batches")
            self._val_batches = batches
            del loader
            note = "" if len(batches) == self.num_batches else \
                f" (asked for {self.num_batches}; the split has no more)"
            print(f"[heldout] cached {len(batches)} held-out batches{note}", flush=True)
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            print(f"[heldout] disabled: {type(exc).__name__}: {exc}", flush=True)

    def on_training_step_start(self, model: Any, data: dict, iteration: int = 0) -> None:
        """Keep as many *training* batches as we have val ones — the matched control.

        Same count on both sides so `heldout/gap` is not also a difference in how many
        batches each average was taken over.
        """
        if self._failed or self._val_batches is None:
            return
        if len(self._train_batches) >= len(self._val_batches):
            return
        self._train_batches.append(data)

    # ---- measurement ------------------------------------------------------
    def _mean_losses(self, model: Any, batches: list[dict], iteration: int) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for i, batch in enumerate(batches):
            torch.manual_seed(self.seed + i)     # same noise for this batch, every time
            output, _ = model.training_step(batch, iteration)
            if not isinstance(output, dict):
                continue
            for key in _LOSS_KEYS:
                value = output.get(key)
                if value is None:
                    continue
                totals.setdefault(key, []).append(float(value))
        return {k: sum(v) / len(v) for k, v in totals.items() if v}

    def on_training_step_end(
        self, model: Any, data_batch: dict, output_batch: dict,
        loss: torch.Tensor, iteration: int = 0,
    ) -> None:
        if self._failed or self.every_n <= 0 or iteration % self.every_n != 0:
            return
        if not self._val_batches or not self._train_batches:
            return

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                val = self._mean_losses(model, self._val_batches, iteration)
                ref = self._mean_losses(model, self._train_batches, iteration)
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            print(f"[heldout] disabled at iter {iteration}: {type(exc).__name__}: {exc}",
                  flush=True)
            return
        finally:
            if was_training:
                model.train()
            torch.set_rng_state(cpu_state)       # leave training's stream untouched
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        payload: dict[str, float] = {}
        for key, value in val.items():
            payload[f"heldout/val_{key}"] = value
        for key, value in ref.items():
            payload[f"heldout/train_{key}"] = value
        v, t = val.get(self.loss_key), ref.get(self.loss_key)
        if v is not None and t is not None:
            payload["heldout/gap"] = v - t
            payload["heldout/ratio"] = v / t if t > 1e-12 else float("nan")
            print(f"[heldout] iter {iteration}: val {v:.5f} | train {t:.5f} | "
                  f"gap {v - t:+.5f}", flush=True)
        if payload and wandb is not None and getattr(wandb, "run", None):
            wandb.log(payload, step=iteration)   # the TB mirror picks this up


__all__ = ["HeldOutLossProbe"]
