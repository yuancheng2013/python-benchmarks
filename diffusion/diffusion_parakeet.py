# Authors: Yuancheng Peng
# License: MIT

from parakeet import jit
from diffusion_python import diffusePurePython


benchmarks = (
    ("diffuse_parakeet_for_loops",
        jit(diffusePurePython)),
)
