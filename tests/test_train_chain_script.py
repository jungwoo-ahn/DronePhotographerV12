"""Guards on scripts/sbatch_train_camera.sh — the chain that keeps a 7-day run alive.

This script is not importable and not unit-testable in the usual sense, but three of its bugs
are one-line regressions that a regex can hold down, and the fourth (its TOML field parser) can
simply be executed and compared against a real parser.

Each guard here corresponds to a failure that actually happened or was one submission away:
the chain reading another run's checkpoint directory, a shared stop-file killing an unrelated
run, and a successor silently resubmitting itself at a different GPU count.
"""
from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sbatch_train_camera.sh"
CONFIGS = sorted((REPO / "configs" / "policy").glob("camera_policy_nano*.toml"))


@pytest.fixture(scope="module")
def source() -> str:
    return SCRIPT.read_text()


def test_checkpoint_dir_is_not_hardcoded_to_one_run(source):
    """It named `camera_policy_nano_v5` outright.

    Pointed at any other config, the chain then read a directory the run does not write: the
    banked iteration never advances, so `stalled` climbs on every link and MAX_STALLED_LINKS
    kills a healthy chain after six — while `banked >= max_iter` never fires, so a finished
    run never stops either.
    """
    line = next(ln for ln in source.splitlines() if ln.startswith("_CKPT_DIR="))
    assert "$_RUN_PROJECT" in line and "$_RUN_GROUP" in line and "$_RUN_NAME" in line
    assert "camera_policy_nano_v5" not in line


def test_stop_and_state_files_are_per_run(source):
    """A shared `.camtrain_stop` from a stopped run would end the next one on its first link,
    printing a message about a deliberate stop nobody made."""
    for var in ("_CHAIN_STATE=", "_STOP_FILE="):
        line = next(ln for ln in source.splitlines() if ln.startswith(var))
        assert "$_RUN_NAME" in line, f"{var} is shared across runs"


def test_successor_inherits_the_allocation(source):
    """`sbatch --gres=gpu:8 ...` applies to the submitted job only. Without these flags the
    successor re-read the #SBATCH defaults and came back at a different GPU count — which
    changes the global batch halfway through training and shows up nowhere but a log header.
    """
    body = source.split("_queue_successor()", 1)[1].split("\n}", 1)[0]
    assert "--gres=gpu:${NGPU}" in body
    assert "SLURM_MEM_PER_NODE" in body
    assert "--time=" in body
    submit = next(ln for ln in body.splitlines() if "sbatch --dependency" in ln)
    assert "$alloc" in submit


def test_the_chain_reads_trailing_overrides_not_just_the_toml(source):
    """The one that made it to the queue.

    `cosmos_framework/scripts/train.py`: "Trailing key.path=value positionals are applied last
    (so they win over TOML)". A smoke submitted as
    `--sft-toml <100k config> job.name=..._smoke trainer.max_iter=100` therefore trains a
    DIFFERENT run than the TOML names. Deriving the chain's paths from the TOML alone put the
    trainer in the smoke directory and the chain on the real run's, reading banked 0 against
    max_iter 100000 — so the smoke queued itself a successor, forever.
    """
    for var, path in (("_RUN_NAME=", "job.name"),
                      ("_RUN_PROJECT=", "job.project"),
                      ("_RUN_GROUP=", "job.group")):
        line = next(ln for ln in source.splitlines() if ln.startswith(var))
        assert f"_setting {path}" in line, f"{var} ignores a trailing {path}= override"
    body = source.split("_max_iter()", 1)[1].split("\n}", 1)[0]
    assert "_setting trainer.max_iter" in body


def test_a_smoke_can_run_without_starting_a_chain(source):
    """A smoke queues its successor before it has trained a step, so `banked >= max_iter`
    cannot stop it at submission time."""
    body = source.split("_queue_successor()", 1)[1].split("\n}", 1)[0]
    assert "NO_CHAIN" in body
    guard = next(ln for ln in body.splitlines() if "NO_CHAIN" in ln and "return" in ln)
    assert guard.strip().startswith("[ -n"), "NO_CHAIN must short-circuit before queueing"


def _toml_via_shell(cfg: Path, key: str) -> str:
    """Run the script's own `_toml` helper, lifted verbatim from the source."""
    src = SCRIPT.read_text()
    fn = "_toml() {" + src.split("_toml() {", 1)[1].split("\n}", 1)[0] + "\n}"
    out = subprocess.run(
        ["bash", "-c", f'SFT_TOML="{cfg}"\n{fn}\n_toml {key}'],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.name)
@pytest.mark.parametrize("key", ["name", "project", "group"])
def test_the_shell_toml_parser_agrees_with_a_real_one(cfg, key):
    """The chain derives its paths with sed. If that disagrees with the parser the trainer
    uses, the chain and the run operate on different directories — which is the bug above,
    reintroduced by a quoting style rather than a hardcoded string."""
    expected = tomllib.loads(cfg.read_text())["job"][key]
    assert _toml_via_shell(cfg, key) == expected


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.name)
def test_every_config_has_a_valid_schedule(cfg):
    """`LambdaLinearScheduler.find_in_interval` returns None past the end of `cycle_lengths`
    and the run dies indexing a ListConfig with it — at exactly the boundary iteration, hours
    or days in.

    The constraint is one-sided: the cycles must COVER max_iter. Overshooting is safe and is
    what the smoke config does (cycles [40000], max_iter 100) — it simply stops partway through
    cycle 1. Undershooting is the one that kills a run, so that is what this asserts.
    """
    c = tomllib.loads(cfg.read_text())
    sched, cycles = c.get("scheduler"), None
    if not sched or "cycle_lengths" not in sched:
        pytest.skip("no scheduler table")
    cycles = sched["cycle_lengths"]
    assert sum(cycles) >= c["trainer"]["max_iter"], (
        f"cycle_lengths sum to {sum(cycles)} but max_iter is {c['trainer']['max_iter']}: "
        "the scheduler walks off the end and the run dies at the boundary")
    for key in ("warm_up_steps", "f_max", "f_min", "f_start"):
        if key in sched:
            assert len(sched[key]) == len(cycles), f"{key} has {len(sched[key])} entries"


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.name)
def test_save_iter_lands_on_the_heldout_probe_cadence(cfg):
    """CheckpointRetention credits a checkpoint with a metric only when the probe published at
    that exact iteration. A save_iter that misses the probe's cadence silently turns keep_best
    into nothing — and keep_best is what is supposed to protect the best model from the janitor.
    """
    save_iter = tomllib.loads(cfg.read_text()).get("checkpoint", {}).get("save_iter")
    if save_iter is None:
        pytest.skip("inherits save_iter from the recipe")
    recipe = (REPO / "configs" / "policy" / "action_policy_camera_nano.py").read_text()
    every_n = int(recipe.split("HeldOutLossProbe)(every_n=", 1)[1].split(",", 1)[0])
    assert save_iter % every_n == 0, f"save_iter {save_iter} misses the probe's {every_n}"
