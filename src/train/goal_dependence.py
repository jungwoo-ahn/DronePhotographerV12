"""Does the policy actually USE the goal?

This is the crux of the project and the question v11 could not answer from its
training curves. A falling action loss proves the model is learning *something*;
it does not prove the goal is what it is learning from. A policy that ignored the
prompt entirely and predicted the average camera motion would also show a falling
loss.

The probe is a counterfactual, not a metric: run the same batch twice, once with
the real prompts and once with the prompts SHUFFLED across the batch, and compare
the flow-matching action loss.

    ratio = loss(shuffled) / loss(real)

    ratio ≈ 1   the goal is being ignored — the prediction is the same whether or
                not the prompt matches the observation
    ratio > 1   the goal is being used, and the size of the gap is how much

Shuffling rather than blanking is deliberate: it keeps the prompt distribution,
the token count and the sequence structure identical, so the only thing that
changes is whether the goal MATCHES the observation. Blanking would also change
sequence length, and the model could react to that rather than to the goal.

Two details that decide whether the number means anything:

* **Same noise on both passes.** Flow matching samples a random sigma per call,
  and that variance is far larger than the effect being measured — two calls with
  the same prompts would already differ. Both passes run from an identical seed.
* **Training's RNG stream is restored afterwards.** Otherwise the probe would
  perturb the very run it is observing, and the measurement would change the
  result.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import wandb
except Exception:  # noqa: BLE001
    wandb = None  # type: ignore[assignment]

from cosmos_framework.utils.callback import Callback

_TEXT_KEY = "text_token_ids"


class GoalDependenceProbe(Callback):
    """Logs loss(real) vs loss(shuffled-goal) every `every_n` iterations."""

    def __init__(self, every_n: int = 200, seed: int = 1234,
                 loss_key: str = "flow_matching_loss_action") -> None:
        super().__init__()
        self.every_n = int(every_n)
        self.seed = int(seed)
        self.loss_key = loss_key
        self._warned = False

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _shuffle_prompts(batch: dict[str, Any], generator: torch.Generator) -> dict | None:
        """A shallow copy of `batch` with the prompts permuted across samples.

        Returns None when there is nothing to shuffle — a single-sample batch has
        no counterfactual available, and reporting a ratio of exactly 1 from one
        would look like "the goal is ignored" when it only means "not measurable".
        """
        prompts = batch.get(_TEXT_KEY)
        if not isinstance(prompts, (list, tuple)) or len(prompts) < 2:
            return None
        n = len(prompts)
        # a derangement-ish permutation: roll by a random non-zero offset, so no
        # sample keeps its own prompt
        offset = int(torch.randint(1, n, (1,), generator=generator).item())
        shuffled = list(prompts[offset:]) + list(prompts[:offset])
        out = dict(batch)
        out[_TEXT_KEY] = shuffled
        return out

    def _loss(self, model: Any, batch: dict, iteration: int) -> float | None:
        output, _ = model.training_step(batch, iteration)
        value = output.get(self.loss_key) if isinstance(output, dict) else None
        if value is None:
            return None
        return float(value.detach().item() if hasattr(value, "detach") else value)

    def _log(self, payload: dict[str, float], iteration: int) -> None:
        if wandb is not None and getattr(wandb, "run", None):
            wandb.log(payload, step=iteration)

    # ---- hook -------------------------------------------------------------
    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.every_n <= 0 or iteration == 0 or iteration % self.every_n != 0:
            return

        generator = torch.Generator().manual_seed(self.seed + iteration)
        shuffled_batch = self._shuffle_prompts(data_batch, generator)
        if shuffled_batch is None:
            if not self._warned:
                self._warned = True
                print(f"[goal_dep] no '{_TEXT_KEY}' list with >=2 samples in the batch — "
                      "probe disabled", flush=True)
            return

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                torch.manual_seed(self.seed + iteration)      # identical noise ...
                real = self._loss(model, data_batch, iteration)
                torch.manual_seed(self.seed + iteration)      # ... on both passes
                shuffled = self._loss(model, shuffled_batch, iteration)
        except Exception as exc:  # noqa: BLE001 — never take the run down
            print(f"[goal_dep] skipped at iter {iteration}: {type(exc).__name__}: {exc}", flush=True)
            return
        finally:
            torch.set_rng_state(cpu_state)                    # leave training's stream untouched
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        if real is None or shuffled is None or real <= 0:
            return
        self._log({
            "goal_dep/loss_real": real,
            "goal_dep/loss_shuffled": shuffled,
            "goal_dep/ratio": shuffled / real,
            "goal_dep/gap": shuffled - real,
        }, iteration)
        print(f"[goal_dep] iter {iteration}: real {real:.5f} | shuffled {shuffled:.5f} | "
              f"ratio {shuffled / real:.3f}", flush=True)


__all__ = ["GoalDependenceProbe"]
