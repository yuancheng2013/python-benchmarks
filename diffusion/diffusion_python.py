# Authors: Benjamin Zaitlen, Yuancheng Peng
# License: MIT
# Reference: http://continuum.io/blog/the-python-and-the-complied-python

import numpy as np


def diffusePurePython(u, tempU, iterNum):
    """
    Apply nested iteration for the Forward-Euler Approximation
    """
    mu = .1
    row = u.shape[0]
    col = u.shape[1]

    #"omp parallel for private(i, j)"
    for n in range(iterNum):
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                tempU[i, j] = u[i, j] + mu * (
                    u[i + 1, j] - 2 * u[i, j] + u[i - 1, j] +
                    u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                u[i, j] = tempU[i, j]
                tempU[i, j] = 0.0


def diffuseNumpy(u, tempU, iterNum):
    """
    Apply Numpy matrix for the Forward-Euler Approximation
    """
    mu = .1

    for n in range(iterNum):
        tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (
            u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1] +
            u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
        u[:, :] = tempU[:, :]
        tempU[:, :] = 0.0


benchmarks = (
    ("diffuse_python_for_loops", diffusePurePython),
    ("diffuse_python_numpy", diffuseNumpy),
)
