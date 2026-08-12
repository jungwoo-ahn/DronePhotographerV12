"""`CameraPolicyDiagnostics` must survive every layout the packer hands it.

Written after the collector and its consumer disagreed about width: the consumer moved to
CAMERA_ACTION_DIM=10 while `_collect_action_rows` still did `[:, :9]`, so it sliced off the
shoot channel and the run died on `cannot reshape array of size 9216 into shape (10)` —
nine minutes in, on a GPU. These cases replay the shapes the smoke actually produced, so
the same disagreement costs a second instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM
from src.train.diagnostics import CameraPolicyDiagnostics, _collect_action_rows

D = CAMERA_ACTION_DIM


@pytest.mark.parametrize("batch", [
    {"action_raw": torch.randn(16, 8, D)},                        # (B, chunk, D)
    {"action_raw": [torch.randn(8, D) for _ in range(16)]},       # list of samples
    {"action_raw": torch.randn(128, D)},                          # already flat
    {"action": torch.randn(18, 8, 64)},                           # only the padded one
    {"action_raw": np.random.randn(16, 8, D).astype(np.float32)}, # numpy
])
def test_every_layout_yields_full_width_rows(batch):
    rows = CameraPolicyDiagnostics(every_n=1)._actions(batch)
    assert rows is not None
    assert rows.shape[-1] == D
    flat = rows.reshape(-1, D)          # the exact line that died in the run
    assert flat[:, 9].shape[0] == flat.shape[0]


def test_a_too_narrow_source_is_rejected_not_truncated():
    """Silently accepting a 9-wide tensor is how the shoot channel disappeared."""
    assert _collect_action_rows(torch.randn(128, 9)) is None
    assert _collect_action_rows(torch.randn(4, 8, 9)) is None


def test_padded_action_is_cut_to_the_real_width():
    rows = _collect_action_rows(torch.randn(4, 8, 64))
    assert rows.shape == (32, D)
