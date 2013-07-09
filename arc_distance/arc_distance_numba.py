# Authors: Yuancheng Peng
# License: MIT

from arc_distance import arc_distance_python
from numba import autojit

benchmarks = (
        ("arc_distance_numba_for_loops",
        autojit(arc_distance_python.arc_distance_python_nested_for_loops)),
)
