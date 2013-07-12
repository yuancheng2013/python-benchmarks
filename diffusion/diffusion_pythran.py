# Authors: Yuancheng Peng
# License: MIT

from diffusion_python import diffusePurePython, diffuseNumpy
from pythran import compile_pythrancode
from inspect import getsource
import re
import imp


# grab imports
imports = '''
import numpy as np
'''

exports = '''
#pythran export diffusePurePython(float [][], float [][], int)
#pythran export diffuseNumpy(float [][], float [][], int)
'''

modname = 'diffusion_pythran'

# grab the source from original functions
funs = (diffusePurePython, diffuseNumpy)
sources = map(getsource, funs)
source = '\n'.join(sources)
source = re.sub(r'#"omp', '"omp', source)

# compile to native module
source = '\n'.join([exports, imports, source])

native = compile_pythrancode(modname, source)

# load
native = imp.load_dynamic(modname, native)

benchmarks = (
    ("diffuse_pythran_for_loop",
        native.diffusePurePython),
    ("diffuse_pythran_numpy",
        native.diffuseNumpy)
)
