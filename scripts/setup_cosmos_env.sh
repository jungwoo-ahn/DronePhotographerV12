#!/bin/bash
# Idempotent patches to the VENDORED cosmos-framework checkout (repos/ is gitignored, so
# these would otherwise be invisible and unreproducible). Safe to re-run; run after any
# `uv sync` or re-clone.
#
#   bash scripts/setup_cosmos_env.sh
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
CF="$V12/repos/cosmos-framework"
SP="$CF/.venv/lib/python3.13/site-packages"

# 1. transformer_engine globs site-packages/nvidia/cuda_<name>/lib, i.e. `cuda_cudart/`,
#    but torch cu128 installs the runtime as `cuda_runtime/`. Package-naming mismatch, not
#    a missing library — without this every entrypoint dies with
#    "RuntimeError: cudart shared object not found".
if [ ! -e "$SP/nvidia/cuda_cudart" ]; then
    ln -s cuda_runtime "$SP/nvidia/cuda_cudart"
    echo "[setup] linked nvidia/cuda_cudart -> cuda_runtime"
else
    echo "[setup] nvidia/cuda_cudart already present"
fi

# 2. Experiments are registered by EXPLICIT imports in configs/base/config.py, so our
#    out-of-tree recipe has to be imported there too or `experiment=action_policy_camera_nano`
#    resolves to nothing. Our config dir is put on sys.path by the launcher (PYTHONPATH).
CONFIG_PY="$CF/cosmos_framework/configs/base/config.py"
MARKER="configs.policy.action_policy_camera_nano"
if ! grep -q "$MARKER" "$CONFIG_PY"; then
    python3 - "$CONFIG_PY" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = "    import cosmos_framework.configs.base.experiment.sft.vision_sft_edge  # noqa: F401\n"
add = ("\n    # DronePhotographerV12: out-of-tree camera-policy recipe (see\n"
       "    # DronePhotographerV12/scripts/setup_cosmos_env.sh). Optional so a plain\n"
       "    # cosmos-framework checkout without v12 on PYTHONPATH still imports.\n"
       "    try:\n"
       "        import configs.policy.action_policy_camera_nano  # noqa: F401\n"
       "    except ImportError:\n"
       "        pass\n")
assert anchor in s, "anchor import line not found — framework layout changed"
p.write_text(s.replace(anchor, anchor + add))
PY
    echo "[setup] registered action_policy_camera_nano in configs/base/config.py"
else
    echo "[setup] action_policy_camera_nano already registered"
fi

echo "[setup] done"
