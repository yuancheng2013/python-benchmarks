# Authors: Yuancheng Peng
# License: MIT
"""Solve diffusion equation by using a Forward-Euler Approximation

The code uses a lot of extended slice with 1-stride, which are vectorizable.
The whole expression is parallel thanks to the double buffer.

See also http://continuum.io/blog/the-python-and-the-complied-python
"""

import numpy as np


def make_env(power=7, iterNum=200):
    lx, ly = (2 ** power, 2 ** power)
    u_in = np.zeros([lx, ly], dtype=np.double)
    u_in[lx / 2, ly / 2] = 1000.0
    tempU = np.zeros([lx, ly], dtype=np.double)

    return (u_in, tempU, iterNum), {}
