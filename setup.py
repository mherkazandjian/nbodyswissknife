from distutils.core import setup, Extension
import numpy

module1 = Extension(
    "nbodyswissknife.potential_cpu",
    include_dirs=[numpy.get_include()],
    libraries=['pthread', 'gomp'],
    extra_compile_args=['-fno-strict-aliasing', '-fopenmp'],
    sources=['nbodyswissknife/backends/cpu/potential.c']
)

setup(
    name="nbodyswissknife",
    version="1.0",
    description='Package with various handy tools for nbody calculations',
    author='Mher Kazandjian',
    url='https://github.com/mherkazandjian/nbodyswissknife',
    packages=['nbodyswissknife'],
    ext_modules=[module1]
)
