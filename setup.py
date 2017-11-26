from distutils.core import setup, Extension

module1 = Extension(
    "nbodyswissknife.potential_cpu",
    include_dirs=[],
    libraries=['pthread', 'gomp'],
    extra_compile_args=['-fno-strict-aliasing', '-fopenmp'],
    sources=['nbodyswissknife/backends/cpu/exmodmodule.c']
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
