"""Task-level training diagnostics for the camera policy.

Cosmos's own 41 metric sites are almost entirely INFRASTRUCTURE — MFU, gradient
norms, token counts, step timing. They tell you the run is healthy; they cannot
tell you the policy is learning the right thing. v11 logged only
``train/<loss>``, ``train/lr`` and a single-sample ``val/action_mse`` — and that
last one actively misled us: it is dominated by multimodal sampling variance and
ANTI-correlated with closed-loop performance (a checkpoint with better sampled
MSE was worse in rollout).

This callback logs what those runs were missing. Everything here is derived from
the training batch, so it costs no extra forward pass:

  action/*      per-block and per-dimension magnitude of the action target.
                Translation and rot6d are fed RAW and deliberately unnormalized;
                if their scales drift apart, rotation — the DOF that aims the
                camera — is being silently down-weighted in the flow loss. This
                is the metric that would catch it.
  action/rotation_deg_*   the relative rotation as a geodesic angle, which is
                interpretable in a way the six rot6d numbers are not.
  goal/*        the goal actually reaching the model: bearing distribution and
                per-sector counts. Given only ~4% of goals are front views, a
                dataloader change that quietly drops them further would
                otherwise be invisible.
  data/*        chunk shape, batch size and prompt length — cheap tripwires for
                a silently broken export.

Not here yet, and needing a validation pass with sampling: mean-of-K action MSE
(the fix for v11's noisy single-sample metric), goal-dependence (real vs shuffled
vs blank prompt — the crux of this project), predicted roll before upright
projection, and per-sector validation error.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import torch

try:
    import wandb
except Exception:  # noqa: BLE001
    wandb = None  # type: ignore[assignment]

from cosmos_framework.utils.callback import Callback

from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM

SECTORS = ("front", "front-right", "right", "back-right",
           "back", "back-left", "left", "front-left")


def _sector(bearing_deg: float) -> str:
    return SECTORS[int(((bearing_deg + 22.5) % 360.0) // 45.0)]


def _geodesic_deg(rot6d: np.ndarray) -> float:
    """Relative-rotation magnitude in degrees from the 6D encoding."""
    c0, c1 = rot6d[:3].astype(np.float64), rot6d[3:].astype(np.float64)
    matrix = np.stack([c0, c1, np.cross(c0, c1)], axis=1)
    u, _, vt = np.linalg.svd(matrix)
    if np.linalg.det(u @ vt) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
    trace = float(np.trace(u @ vt))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def _collect_action_rows(value: Any, _depth: int = 0) -> np.ndarray | None:
    """Every (…, >=9) tensor/array reachable from `value`, flattened to [-1, 9].

    The packed batch nests differently depending on how the packer groups samples —
    `action_raw` has been seen as a plain tensor and as a list — and two earlier
    attempts to pin the layout down each cost a training run before anyone noticed
    the metrics were missing. So recurse instead of predicting: descend lists,
    tuples and dicts, take anything whose last axis can hold a 9-D action, and
    concatenate.
    """
    if _depth > 3 or value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim >= 2 and value.shape[-1] >= 9:
            flat = value.detach().float().cpu().reshape(-1, value.shape[-1])
            return flat[:, :9].numpy()
        return None
    if isinstance(value, np.ndarray):
        if value.ndim >= 2 and value.shape[-1] >= 9:
            return value.reshape(-1, value.shape[-1])[:, :9].astype(np.float32)
        return None
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        parts = [p for p in (_collect_action_rows(v, _depth + 1) for v in value) if p is not None]
        if parts:
            return np.concatenate(parts, axis=0)
    return None


class CameraPolicyDiagnostics(Callback):
    """Logs action-target and goal-distribution statistics every `every_n` steps."""

    def __init__(self, every_n: int = 50, rotation_sample: int = 64) -> None:
        super().__init__()
        self.every_n = int(every_n)
        self.rotation_sample = int(rotation_sample)
        self._sector_counts: Counter = Counter()
        self._dumped_keys = False

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _actions(data_batch: dict[str, Any]) -> np.ndarray | None:
        """The RAW action target as [-1, 9].

        Prefer `action_raw` over `action`: the transform pipeline zero-pads the
        latter out to `max_action_dim` (9 -> 64), so its statistics would be
        dominated by padding.

        Shape-agnostic on purpose. `action_raw` is (chunk, 9) on a single sample
        and the packer may hand over (B, chunk, 9) or a flat (total, 9) — an
        earlier version required ndim == 3 and silently logged nothing at all.
        Anything whose last axis is at least 9 is flattened to rows.
        """
        for key in ("action_raw", "action", "actions", "action_target"):
            rows = _collect_action_rows(data_batch.get(key))
            if rows is not None:
                return rows
        return None

    def _log(self, payload: dict[str, float], iteration: int) -> None:
        if wandb is not None and getattr(wandb, "run", None):
            wandb.log(payload, step=iteration)   # the TB mirror picks this up

    # ---- hooks ------------------------------------------------------------
    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.every_n <= 0 or iteration % self.every_n != 0:
            return

        # The packed batch's schema is not documented, and `action/*` silently went
        # missing on the first real run because `action_raw` is not where we assumed.
        # Dump the keys once rather than guess again and burn another run.
        if not self._dumped_keys:
            self._dumped_keys = True
            described = []
            for key in sorted(data_batch):
                value = data_batch[key]
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                described.append(f"{key}{shape}")
            print(f"[diagnostics] data_batch keys: {', '.join(described)}", flush=True)

        payload: dict[str, float] = {}
        actions = self._actions(data_batch)
        if actions is not None:
            # 10 wide now (9 pose + shoot). Hardcoding 9 here silently reshaped the
            # batch wrong the moment the shoot channel landed, folding one sample's
            # shoot value into the next sample's translation.
            flat = actions.reshape(-1, CAMERA_ACTION_DIM)
            translation, rot6d = flat[:, :3], flat[:, 3:9]
            shoot = flat[:, 9]

            # The shoot channel is the whole point of the 10th dim, so it has to be
            # visible from iteration one rather than only at eval time. These are the
            # LABELS reaching the model; whether the policy reproduces them is measured
            # in closed_loop_eval.
            payload["shoot/label_positive_frac"] = float((shoot > 0.5).mean())
            payload["shoot/label_mean"] = float(shoot.mean())

            # Scale balance — the reason the action is fed raw at all.
            t_std, r_std = float(translation.std()), float(rot6d[:, [1, 2, 3, 5]].std())
            payload["action/translation_std"] = t_std
            payload["action/rot6d_offdiag_std"] = r_std
            payload["action/scale_ratio_t_over_r"] = t_std / r_std if r_std > 1e-9 else float("nan")
            payload["action/translation_abs_p99"] = float(np.percentile(np.abs(translation), 99))

            for i, name in enumerate(("d_right", "d_up", "d_fwd")):
                payload[f"action/dim_std/{name}"] = float(translation[:, i].std())
                payload[f"action/dim_mean/{name}"] = float(translation[:, i].mean())
            for i in range(6):
                payload[f"action/rot6d_mean/r{i}"] = float(rot6d[:, i].mean())

            # Interpretable rotation magnitude (subsampled: SVD per row is not free).
            idx = np.linspace(0, len(rot6d) - 1, min(self.rotation_sample, len(rot6d))).astype(int)
            angles = np.array([_geodesic_deg(rot6d[i]) for i in idx])
            payload["action/rotation_deg_mean"] = float(angles.mean())
            payload["action/rotation_deg_p99"] = float(np.percentile(angles, 99))
            payload["action/rotation_deg_max"] = float(angles.max())

            # `actions` is flattened to rows, so this counts steps in the batch
            # rather than assuming a (B, chunk, 9) layout that the packer may not use.
            payload["data/action_steps_in_batch"] = float(actions.shape[0])

        # Goal reach-through: the prompt is the only channel the goal has.
        captions = data_batch.get("ai_caption")
        if isinstance(captions, (list, tuple)) and captions:
            lengths = [len(str(c)) for c in captions]
            payload["data/prompt_chars_mean"] = float(np.mean(lengths))
            payload["data/prompt_blank_frac"] = float(np.mean([l == 0 for l in lengths]))
            for caption in captions:
                bearing = self._bearing_from_caption(str(caption))
                if bearing is not None:
                    self._sector_counts[_sector(bearing)] += 1
            total = sum(self._sector_counts.values())
            if total:
                for sector in SECTORS:
                    payload[f"goal/sector_frac/{sector}"] = self._sector_counts[sector] / total
                front = self._sector_counts["front"] + self._sector_counts["front-left"] \
                    + self._sector_counts["front-right"]
                payload["goal/front_family_frac"] = front / total

        if payload:
            self._log(payload, iteration)

    @staticmethod
    def _bearing_from_caption(caption: str) -> float | None:
        """Pull the bearing back out of the prompt we generated.

        The prompt is the only place the goal exists once the sample reaches the
        model, so reading it back is the honest check that the goal survived the
        transform pipeline (including any caption dropout).
        """
        marker = "bearing "
        i = caption.find(marker)
        if i < 0:
            return None
        j = i + len(marker)
        k = j
        while k < len(caption) and (caption[k].isdigit() or caption[k] in ".-"):
            k += 1
        try:
            return float(caption[j:k])
        except ValueError:
            return None


__all__ = ["CameraPolicyDiagnostics"]
