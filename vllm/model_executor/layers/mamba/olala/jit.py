# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of the Olala CUDA extensions (csrc/*.cu).

``torch.utils.cpp_extension`` refuses an nvcc whose CUDA major differs from
torch's, so a host whose default toolkit is CUDA 13 under a cu12x torch (as
olala-env's serve.sh sets up for tilelang) could not build the kernels. The
toolkit is therefore chosen here: ``OLALA_JIT_CUDA_HOME`` if set, else torch's
own ``CUDA_HOME`` when its major matches, else the newest
``/usr/local/cuda-<major>.*`` that does.
"""

import glob
import hashlib
import os
import re
import subprocess
from functools import lru_cache

import torch


def _nvcc_major(cuda_home: str) -> int | None:
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc):
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else None


@lru_cache(maxsize=1)
def jit_cuda_home() -> str | None:
    from torch.utils import cpp_extension

    override = os.environ.get("OLALA_JIT_CUDA_HOME")
    if override:
        return override
    if torch.version.cuda is None:
        return cpp_extension.CUDA_HOME
    want = int(torch.version.cuda.split(".")[0])
    default = cpp_extension.CUDA_HOME
    if default and _nvcc_major(default) == want:
        return default
    for d in sorted(
        glob.glob(f"/usr/local/cuda-{want}.*"),
        key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))],
        reverse=True,
    ):
        if _nvcc_major(d) == want:
            return d
    return default


def load_extension(name: str, source: str, cuda_cflags: list[str], ldflags=None):
    """Build (once) and import a csrc extension.

    Each extension gets its own build directory, keyed by its source, flags,
    torch and toolkit, and a finished build there is imported without running
    ninja: with one shared directory every load rebuilt (each ninja run prunes
    the other extension's deps log), and a cache volume restores files with
    fresh mtimes, which ninja also takes for stale deps.
    """
    from torch.utils import cpp_extension

    base = os.environ.get(
        "OLALA_TPA_BUILD_DIR",
        # One directory per torch build: extensions compiled against another
        # torch ABI must never be picked up (two vLLM venvs on one machine).
        os.path.join(
            os.path.expanduser("~"), ".cache", "olala_tpa_factor",
            f"torch{torch.__version__.replace('+', '_')}",
        ),
    )
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", source)
    ldflags = ldflags or []
    cuda_home = jit_cuda_home()
    with open(src, "rb") as f:
        key = hashlib.sha256(
            b"\0".join(
                [f.read(), " ".join(cuda_cflags + ldflags).encode()]
                + [str(x).encode() for x in (torch.__version__, cuda_home)]
            )
        ).hexdigest()[:16]
    build_dir = os.path.join(base, f"{name}-{key}")
    os.makedirs(build_dir, exist_ok=True)
    if os.path.exists(os.path.join(build_dir, f"{name}.so")):
        try:
            return cpp_extension._import_module_from_library(name, build_dir, True)
        except (ImportError, OSError):
            pass  # e.g. a build cut short: rebuild below
    saved = cpp_extension.CUDA_HOME
    cpp_extension.CUDA_HOME = cuda_home
    try:
        return cpp_extension.load(
            name=name,
            sources=[src],
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=ldflags,
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        cpp_extension.CUDA_HOME = saved
