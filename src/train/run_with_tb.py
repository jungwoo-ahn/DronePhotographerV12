"""Run Cosmos training with the TensorBoard mirror installed.

A drop-in replacement for ``-m cosmos_framework.scripts.train``: it wraps
``wandb.log`` first (see :mod:`src.train.tb_mirror`) and then hands over to the
real entrypoint unchanged, so no framework file has to be patched.

    torchrun --nproc_per_node=N -m src.train.run_with_tb --sft-toml <toml> [overrides]

``COSMOS_TB_DIR`` selects the TensorBoard log directory; without it the mirror is
a no-op and this is exactly the stock trainer.
"""

from __future__ import annotations

import runpy
import sys

from src.train.tb_mirror import install

TARGET = "cosmos_framework.scripts.train"


def main() -> None:
    install()
    # argv[0] must look like the real module so the trainer's own arg parsing and
    # launch-info recording behave identically.
    sys.argv[0] = TARGET
    runpy.run_module(TARGET, run_name="__main__")


if __name__ == "__main__":
    main()
