"""Checkpoint retention — the janitor that deletes 137 GB directories.

Cosmos never prunes, so without this a long run fills a shared filesystem instead of
finishing. That makes the failure modes asymmetric: keeping one checkpoint too many costs
disk, deleting the wrong one costs the run. So the rules are tested rather than trusted —
in particular that the resume target, the newest directory, anything still being written,
and the milestones are all untouchable.

No Blender, no framework, no GPU: the callback falls back to a stub `Callback` when
`cosmos_framework` is absent, which is exactly so this can run.
"""
from __future__ import annotations

import os
import time

import pytest

from src.train import metric_bus
from src.train.checkpoint_retention import CheckpointRetention


@pytest.fixture(autouse=True)
def _clean_bus():
    metric_bus.clear()
    yield
    metric_bus.clear()


def make_run(tmp_path, iters, *, pointer=None, age_s=10_000.0):
    """A checkpoint dir with one directory per iteration, aged past `min_age_s`."""
    ckpt = tmp_path / "run" / "checkpoints"
    ckpt.mkdir(parents=True)
    old = time.time() - age_s
    for it in iters:
        d = ckpt / f"iter_{it:09d}"
        d.mkdir()
        (d / "model").write_text("x" * 16)
        os.utime(d, (old, old))
    tgt = pointer if pointer is not None else max(iters)
    (ckpt / "latest_checkpoint.txt").write_text(f"iter_{tgt:09d}")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(tmp_path / "run")
    return ckpt


def survivors(ckpt) -> set[int]:
    return {int(p.name.split("_")[1]) for p in ckpt.iterdir() if p.name.startswith("iter_")}


def test_keeps_the_newest_n(tmp_path):
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000, 5000])
    CheckpointRetention(keep_last=3, keep_best=0)._prune()
    assert survivors(ckpt) == {3000, 4000, 5000}


def test_milestones_survive_however_many_saves_follow(tmp_path):
    """The record of the run. 100k/200k/... must outlive the keep-last window."""
    iters = [100_000, 150_000, 200_000, 250_000, 300_000, 301_000, 302_000, 303_000]
    ckpt = make_run(tmp_path, iters)
    CheckpointRetention(keep_last=3, keep_best=0, keep_every=100_000)._prune()
    assert survivors(ckpt) == {100_000, 200_000, 300_000, 301_000, 302_000, 303_000}


def test_best_is_kept_and_comes_from_the_noticeboard(tmp_path):
    """`keep_best` only ever worked if something wrote the metric where retention looks.
    The held-out probe publishes to `metric_bus`; without that hop it silently degraded
    to keep-newest."""
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000, 5000])
    r = CheckpointRetention(keep_last=2, keep_best=1, metric_key="heldout/val_loss")
    losses = {1000: 0.9, 2000: 0.4, 3000: 0.7, 4000: 0.8, 5000: 0.85}   # best at 2000
    for it, v in losses.items():
        metric_bus.publish({"heldout/val_loss": v}, it)
        r.on_training_step_end(None, {}, {}, None, iteration=it)
    r._prune()
    assert survivors(ckpt) == {2000, 4000, 5000}


def test_a_stale_metric_is_not_credited_to_a_later_checkpoint(tmp_path):
    """The probe runs on its own schedule. If it did not publish at THIS iteration, the
    last value it did publish belongs to an earlier one."""
    r = CheckpointRetention(keep_last=1, keep_best=1, metric_key="heldout/val_loss")
    metric_bus.publish({"heldout/val_loss": 0.5}, 1000)
    r.on_training_step_end(None, {}, {}, None, iteration=1000)
    r.on_training_step_end(None, {}, {}, None, iteration=2000)      # nothing new published
    assert r._metric_by_iter == {1000: 0.5}


def test_output_batch_still_wins_when_it_carries_the_metric(tmp_path):
    r = CheckpointRetention(keep_last=1, keep_best=1, metric_key="m")
    metric_bus.publish({"m": 9.0}, 1000)
    r.on_training_step_end(None, {}, {"m": 1.0}, None, iteration=1000)
    assert r._metric_by_iter == {1000: 1.0}


def test_never_deletes_the_resume_target(tmp_path):
    """Rule 1. Losing what `latest_checkpoint.txt` points at costs the run."""
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000, 5000], pointer=1000)
    CheckpointRetention(keep_last=2, keep_best=0)._prune()
    assert 1000 in survivors(ckpt)


def test_never_deletes_the_newest_even_if_the_pointer_disagrees(tmp_path):
    """Rule 2. A save in flight has not updated the pointer yet."""
    ckpt = make_run(tmp_path, [1000, 2000, 3000], pointer=1000)
    CheckpointRetention(keep_last=1, keep_best=0)._prune()
    assert {1000, 3000} <= survivors(ckpt)


def test_never_deletes_a_young_directory(tmp_path):
    """Rule 3. Saving can be asynchronous, so recency is the proxy for 'still being written'."""
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000, 5000], age_s=0.0)
    CheckpointRetention(keep_last=1, keep_best=0, min_age_s=900.0)._prune()
    assert survivors(ckpt) == {1000, 2000, 3000, 4000, 5000}


def test_does_nothing_when_there_is_nothing_to_prune(tmp_path):
    ckpt = make_run(tmp_path, [1000, 2000])
    CheckpointRetention(keep_last=3, keep_best=0)._prune()
    assert survivors(ckpt) == {1000, 2000}


def test_dry_run_deletes_nothing(tmp_path):
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000])
    CheckpointRetention(keep_last=1, keep_best=0, dry_run=True)._prune()
    assert survivors(ckpt) == {1000, 2000, 3000, 4000}


def test_the_configuration_this_run_uses(tmp_path):
    """What a 400k continuation is asked to keep: best, every 10k, and the last three."""
    iters = list(range(100_000, 400_001, 1000))
    ckpt = make_run(tmp_path, iters)
    r = CheckpointRetention(keep_last=3, keep_best=1, keep_every=10_000,
                            metric_key="heldout/val_loss")
    metric_bus.publish({"heldout/val_loss": 0.3}, 137_000)
    r.on_training_step_end(None, {}, {}, None, iteration=137_000)
    r._prune()
    milestones = set(range(100_000, 400_001, 10_000))
    assert survivors(ckpt) == milestones | {137_000, 398_000, 399_000}


def test_a_checkpoint_from_an_earlier_run_is_not_deleted_as_not_best(tmp_path):
    """The one that cost `iter_000040000`.

    `keep_best` can only rank iterations THIS process measured. A checkpoint inherited from
    an earlier run has no entry in `_metric_by_iter`, so it is not a best candidate — it is
    just an old directory. Here 40000 is genuinely the best model (0.00073 against 0.00109)
    and retention cannot know that, so the pin has to be declarable by hand.
    """
    ckpt = make_run(tmp_path, [40_000, 46_000, 48_000, 49_000, 50_000])
    r = CheckpointRetention(keep_last=2, keep_best=1, metric_key="heldout/val_loss")
    for it, v in {46_000: 0.0018, 48_000: 0.0011, 49_000: 0.00109}.items():
        metric_bus.publish({"heldout/val_loss": v}, it)
        r.on_training_step_end(None, {}, {}, None, iteration=it)

    r._prune()
    assert 40_000 not in survivors(ckpt), "documents the loss: unpinned, it goes"

    ckpt2 = make_run(tmp_path / "pinned", [40_000, 46_000, 48_000, 49_000, 50_000])
    (ckpt2 / "iter_000040000" / "KEEP").write_text("evaluated; every published v5 number")
    r2 = CheckpointRetention(keep_last=2, keep_best=1, metric_key="heldout/val_loss")
    for it, v in {46_000: 0.0018, 48_000: 0.0011, 49_000: 0.00109}.items():
        metric_bus.publish({"heldout/val_loss": v}, it)
        r2.on_training_step_end(None, {}, {}, None, iteration=it)
    r2._prune()
    assert 40_000 in survivors(ckpt2)


def test_keep_file_outranks_every_other_rule(tmp_path):
    ckpt = make_run(tmp_path, [1000, 2000, 3000, 4000, 5000])
    (ckpt / "iter_000001000" / "KEEP").touch()
    (ckpt / "iter_000002000" / "KEEP").touch()
    CheckpointRetention(keep_last=1, keep_best=0)._prune()
    assert {1000, 2000} <= survivors(ckpt)


def test_never_deletes_what_this_process_resumed_from(tmp_path):
    """Rule 6. The pointer has moved on by the time pruning runs, so the origin is noted on
    the first training step — while it is still the pointer."""
    ckpt = make_run(tmp_path, [10_000])
    r = CheckpointRetention(keep_last=1, keep_best=0)
    r.on_training_step_end(None, {}, {}, None, iteration=10_001)     # notes 10000

    old = time.time() - 10_000
    for it in (11_000, 12_000, 13_000):                              # this link's own saves
        d = ckpt / f"iter_{it:09d}"
        d.mkdir()
        os.utime(d, (old, old))
    (ckpt / "latest_checkpoint.txt").write_text("iter_000013000")

    r._prune()
    assert 10_000 in survivors(ckpt)


def test_the_resume_pin_does_not_outlive_the_process(tmp_path):
    """Otherwise a 40-link chain pins 40 checkpoints — 5.5 TB of them."""
    ckpt = make_run(tmp_path, [10_000, 11_000, 12_000, 13_000])
    fresh = CheckpointRetention(keep_last=1, keep_best=0)
    fresh.on_training_step_end(None, {}, {}, None, iteration=13_001)   # notes 13000, not 10000
    fresh._prune()
    assert 10_000 not in survivors(ckpt)
    assert 13_000 in survivors(ckpt)


def test_prune_never_raises_out(tmp_path, monkeypatch):
    """A janitor must not be the thing that kills a training run."""
    make_run(tmp_path, [1000, 2000, 3000])
    r = CheckpointRetention(keep_last=1, keep_best=0)
    monkeypatch.setattr(r, "_prune", lambda: (_ for _ in ()).throw(OSError("gpfs hiccup")))
    monkeypatch.setenv("RANK", "0")
    r.on_save_checkpoint_end(None, iteration=3000)          # must not raise


class _Job:
    def __init__(self, path_local): self.path_local = str(path_local)


class _Cfg:
    def __init__(self, path_local): self.job = _Job(path_local)


def test_prunes_the_run_the_config_names(tmp_path):
    """The regression that cost 960 GB.

    Retention used to find its directory by globbing every `latest_checkpoint.txt` under the
    output root and taking `sorted(...)[-1]`. With `camera_policy_nano_v5` and
    `camera_policy_nano_v5_smoke` side by side, `/` sorts before `_`, so it picked the SMOKE
    run — tidied an empty directory on every save, printed nothing, and let the real run grow
    to seven checkpoints. The config knows which run this is; ask it.
    """
    root = tmp_path / "out"
    real, smoke = root / "p/g/run", root / "p/g/run_smoke"
    for d, iters in ((real, [1000, 2000, 3000, 4000, 5000]), (smoke, [100])):
        ck = d / "checkpoints"
        ck.mkdir(parents=True)
        old = time.time() - 10_000
        for it in iters:
            (ck / f"iter_{it:09d}").mkdir()
            os.utime(ck / f"iter_{it:09d}", (old, old))
        (ck / "latest_checkpoint.txt").write_text(f"iter_{max(iters):09d}")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(root)

    r = CheckpointRetention(keep_last=2, keep_best=0)
    r.config = _Cfg(real)
    r._prune()

    assert survivors(real / "checkpoints") == {4000, 5000}      # the real run was pruned
    assert survivors(smoke / "checkpoints") == {100}            # the smoke run untouched


def test_refuses_to_guess_between_two_runs(tmp_path):
    """Without a config it must prune NOTHING rather than pick a run by sort order."""
    root = tmp_path / "out"
    for name in ("a", "b"):
        ck = root / f"p/g/{name}/checkpoints"
        ck.mkdir(parents=True)
        old = time.time() - 10_000
        for it in (1000, 2000, 3000, 4000):
            (ck / f"iter_{it:09d}").mkdir()
            os.utime(ck / f"iter_{it:09d}", (old, old))
        (ck / "latest_checkpoint.txt").write_text("iter_000004000")
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(root)

    CheckpointRetention(keep_last=1, keep_best=0)._prune()
    for name in ("a", "b"):
        assert survivors(root / f"p/g/{name}/checkpoints") == {1000, 2000, 3000, 4000}
