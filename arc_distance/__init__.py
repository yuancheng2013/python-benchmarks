# Authors: Yuancheng Peng
# License: MIT

import numpy as np

def make_env(n=100):
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    return (a, b), {}
