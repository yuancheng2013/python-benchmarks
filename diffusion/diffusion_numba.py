# Authors: Yuancheng Peng
# License: MIT

from numba import autojit
from diffusion_python import diffusePurePython


benchmarks = (
    ("diffuse_numba_for_loops",
        autojit(diffusePurePython)),
)
