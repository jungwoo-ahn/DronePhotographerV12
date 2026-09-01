"""Goal bank — the goal distribution the policy was actually trained on, as data.

Every training prompt is a full 8-key goal plus a crop phrase and a `visible` number, and the
44,560 unique task strings of an export ARE that distribution: parsing them back
(`serialize.parse_training_prompt`) gives one row per goal the policy has seen. The bank is
what lets an authored goal be COMPLETED with values the policy has seen together (see
`completion.BankCompleter`), and what defines "in support" for a requested category — the
vocabulary's own centroids are not: "close-up" (occupancy 88 %) sits above the trained maximum
of 80, "extreme wide shot" (4 %) below the minimum of 10, "low angle" (+33°) above +19.

The bank stores NUMBERS only (the words are recomputed from them through `vocab`, exactly as
the exporter does), so `prompt(i)` must reproduce the source string byte for byte; the build
script asserts that for every row. Ships as a ~1 MB `assets/goal_bank_v5.npz` so authoring does
not need the export on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.common.facing import SECTOR8
from src.common.goal_space import (
    CYCLIC_GOAL_KEYS,
    DEFAULT_GOAL_KEYS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    SUBJECT_BEARING_KEY,
)
from src.goal_authoring import vocab
from src.goal_authoring.serialize import parse_training_prompt, profile_to_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK_PATH = REPO_ROOT / "assets" / "goal_bank_v5.npz"

CROP_LABELS: tuple[str, ...] = tuple(vocab.CROP_SIDE)          # int code = position
NUMERIC_KEYS: tuple[str, ...] = tuple(DEFAULT_GOAL_KEYS) + ("visible_frac",)

# One unit of normalized distance = the key's full range (bearing: half a turn, cyclic).
SCALE: dict[str, float] = {
    "occupancy": 100.0, "body_in_frame_ratio": 100.0,
    SUBJECT_BEARING_KEY: 180.0, "cam_to_obj_elevation_deg": 90.0,
    "object_center_x": RENDER_WIDTH, "object_center_y": RENDER_HEIGHT,
    "bbox_x_offset": RENDER_WIDTH, "bbox_y_offset": RENDER_HEIGHT,
    "visible_frac": 1.0,
}

# key -> (table, divisor) for the keys that have a category band. Bearing is handled by sector8.
_BAND_TABLES: dict[str, tuple[dict, float]] = {
    "occupancy": (vocab.SHOT_SIZE, 1.0),
    "body_in_frame_ratio": (vocab.BODY_FRAMING, 1.0),
    "cam_to_obj_elevation_deg": (vocab.ELEVATION, 1.0),
    "object_center_x": (vocab.PLACE_X, RENDER_WIDTH),
    "object_center_y": (vocab.PLACE_Y, RENDER_HEIGHT),
}
BAND_KEYS: tuple[str, ...] = tuple(_BAND_TABLES) + (SUBJECT_BEARING_KEY,)


def band_labels(key: str) -> tuple[str, ...]:
    if key == SUBJECT_BEARING_KEY:
        return SECTOR8
    return tuple(_BAND_TABLES[key][0])


def band_codes(key: str, values) -> np.ndarray:
    """Vectorized `vocab._classify` / `sector8`: band index per value, clamped to the edge bands."""
    v = np.asarray(values, dtype=np.float64)
    if key == SUBJECT_BEARING_KEY:
        return (((v + 22.5) % 360.0) // 45.0).astype(np.int64)
    table, div = _BAND_TABLES[key]
    edges = [lo for (lo, _hi, _c) in table.values()][1:]        # interior lower edges
    return np.digitize(v / div, edges, right=False).astype(np.int64)


def band_of(key: str, value: float) -> str | None:
    """Category word for one value, or None for keys without a band (bbox offsets, visible)."""
    if key not in BAND_KEYS:
        return None
    return band_labels(key)[int(band_codes(key, [value])[0])]


@dataclass
class GoalBank:
    values: dict[str, np.ndarray]           # NUMERIC_KEYS -> float32 (n,)
    crop: np.ndarray                        # int8 codes into CROP_LABELS
    n_train: np.ndarray                     # episodes carrying this goal in TRAIN scenes
    n_val: np.ndarray                       # ... in the held-out scenes
    meta: dict = field(default_factory=dict)
    _bands: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        n = len(self.crop)
        for k in NUMERIC_KEYS:
            if k not in self.values or len(self.values[k]) != n:
                raise ValueError(f"goal bank column {k!r} missing or misaligned")
        self._bands = {k: band_codes(k, self.values[k]) for k in BAND_KEYS}

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_prompts(cls, prompts: Iterable[str], n_train: Sequence[int] | None = None,
                     n_val: Sequence[int] | None = None, meta: Mapping | None = None) -> "GoalBank":
        """Parse training prompts into rows. Prompts that do not have the training shape are
        skipped and counted in `meta["skipped"]` — a silently mis-parsed row would be a goal
        the policy never saw, which is the one thing the bank exists to rule out."""
        prompts = list(prompts)
        n_train = list(n_train) if n_train is not None else [1] * len(prompts)
        n_val = list(n_val) if n_val is not None else [0] * len(prompts)
        cols: dict[str, list[float]] = {k: [] for k in NUMERIC_KEYS}
        crop, tr, va, skipped = [], [], [], 0
        for p, a, b in zip(prompts, n_train, n_val):
            row = parse_training_prompt(p)
            if row is None:
                skipped += 1
                continue
            for k in NUMERIC_KEYS:
                cols[k].append(row[k])
            crop.append(CROP_LABELS.index(row["crop_label"]))
            tr.append(int(a))
            va.append(int(b))
        m = dict(meta or {})
        m["skipped"] = skipped
        return cls({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()},
                   np.asarray(crop, dtype=np.int8), np.asarray(tr, dtype=np.int32),
                   np.asarray(va, dtype=np.int32), m)

    @classmethod
    def from_lerobot(cls, root: str | Path, val_scenes: Iterable[str] | None = None) -> "GoalBank":
        """Build from a LeRobot export: `meta/tasks.parquet` (the strings) + the episodes
        table (`tasks` + `scene` per episode) for train / val episode counts per goal."""
        import pandas as pd

        root = Path(root)
        tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
        # LeRobot v3.0 keeps the task STRING as the index and only task_index as a column.
        strings = list(tasks["task"].astype(str)) if "task" in tasks.columns else list(tasks.index.astype(str))
        ep_dir = root / "meta" / "episodes"
        counts_tr: dict[str, int] = {}
        counts_va: dict[str, int] = {}
        vs = frozenset(val_scenes or ())
        for f in sorted(ep_dir.glob("chunk-*/file-*.parquet")):
            ep = pd.read_parquet(f, columns=["tasks", "scene"])
            for t, scene in zip(ep["tasks"], ep["scene"]):
                s = t[0] if isinstance(t, (list, tuple, np.ndarray)) else str(t)
                d = counts_va if scene in vs else counts_tr
                d[s] = d.get(s, 0) + 1
        meta = {"source": str(root), "n_prompts": len(strings),
                "val_scenes": sorted(vs), "render": [RENDER_WIDTH, RENDER_HEIGHT]}
        return cls.from_prompts(strings, [counts_tr.get(s, 0) for s in strings],
                                [counts_va.get(s, 0) for s in strings], meta)

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_BANK_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, crop=self.crop, n_train=self.n_train, n_val=self.n_val,
                            meta=np.array(json.dumps(self.meta)), **self.values)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GoalBank":
        p = Path(path) if path is not None else DEFAULT_BANK_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"no goal bank at {p}. Build it with scripts/build_goal_bank.py (needs the "
                "LeRobot export), or author with method='rules', which needs no bank.")
        with np.load(p, allow_pickle=False) as z:
            values = {k: np.asarray(z[k], dtype=np.float32) for k in NUMERIC_KEYS}
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            return cls(values, np.asarray(z["crop"], dtype=np.int8),
                       np.asarray(z["n_train"], dtype=np.int32),
                       np.asarray(z["n_val"], dtype=np.int32), meta)

    # ---- rows --------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.crop))

    def crop_label(self, i: int) -> str:
        return CROP_LABELS[int(self.crop[i])]

    def row(self, i: int) -> dict[str, float | str]:
        """One training goal as a complete profile dict. The crop FRACTIONS are not in the
        prompt, so they come from `vocab.CROP_SIDE` — the representative pair that
        `crop_phrase` maps back to the same word."""
        out: dict[str, float | str] = {k: float(self.values[k][i]) for k in NUMERIC_KEYS}
        label = self.crop_label(i)
        t, b = vocab.CROP_SIDE[label]
        out["top_cut_frac"], out["bot_cut_frac"] = float(t), float(b)
        out["head_in_frame"] = 0.0 if t > 0.02 else 1.0
        out["crop_label"] = label
        return out

    def prompt(self, i: int) -> str:
        """The training prompt this row came from, rebuilt from the numbers."""
        r = self.row(i)
        return profile_to_prompt({k: v for k, v in r.items() if k != "crop_label"})

    # ---- queries -----------------------------------------------------------------------
    def train_mask(self) -> np.ndarray:
        return self.n_train > 0

    def band_mask(self, key: str, value: float) -> np.ndarray:
        """Rows whose category word for `key` equals the word `value` would get. All-True for
        keys without a band (bbox offsets, visible_frac) — those constrain by distance only."""
        if key not in BAND_KEYS:
            return np.ones(len(self), dtype=bool)
        return self._bands[key] == int(band_codes(key, [value])[0])

    def crop_mask(self, label: str) -> np.ndarray:
        return self.crop == CROP_LABELS.index(label)

    def support(self, key: str, value: float, mask: np.ndarray | None = None) -> int:
        m = self.band_mask(key, value)
        if mask is not None:
            m = m & mask
        return int(m.sum())

    def distances(self, query: Mapping[str, float], keys: Sequence[str],
                  idx: np.ndarray | None = None) -> np.ndarray:
        """RMS normalized distance from `query` to the rows `idx`, over `keys` (cyclic bearing)."""
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        acc = np.zeros(len(idx), dtype=np.float64)
        m = 0
        for k in keys:
            if k not in self.values or k not in query:
                continue
            v = self.values[k][idx].astype(np.float64)
            q = float(query[k])
            if k in CYCLIC_GOAL_KEYS:
                diff = np.abs(((v - q + 180.0) % 360.0) - 180.0) / SCALE[k]
            else:
                diff = np.abs(v - q) / SCALE[k]
            acc += diff * diff
            m += 1
        return np.sqrt(acc / max(m, 1))

    def quantiles(self, key: str, q: Sequence[float], mask: np.ndarray | None = None) -> list[float]:
        v = self.values[key] if mask is None else self.values[key][mask]
        return [float(x) for x in np.quantile(v.astype(np.float64), q)]

    def support_table(self, mask: np.ndarray | None = None) -> dict[str, dict[str, int]]:
        """Per band, how many training goals carry that word — the honest version of the
        vocabulary: a band with a handful of rows is a request the policy cannot follow."""
        out: dict[str, dict[str, int]] = {}
        for k in BAND_KEYS:
            codes = self._bands[k] if mask is None else self._bands[k][mask]
            labels = band_labels(k)
            out[k] = {lab: int((codes == i).sum()) for i, lab in enumerate(labels)}
        crop = self.crop if mask is None else self.crop[mask]
        out["crop"] = {lab: int((crop == i).sum()) for i, lab in enumerate(CROP_LABELS)}
        return out


__all__ = ["GoalBank", "DEFAULT_BANK_PATH", "CROP_LABELS", "NUMERIC_KEYS", "BAND_KEYS",
           "SCALE", "band_of", "band_codes", "band_labels"]
