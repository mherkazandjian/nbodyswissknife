"""
<keywords>
test, python, potential, nbody
</keywords>
<description>
example script for computing the potential for a bunch of particles
</description>
<seealso>
</seealso>
"""
import ctypes
import numpy
from astropy import units as u
from astropy import constants as const

from nbodyswissknife import potential_cpu

#
# sample a plummer sphere with core radius 'a' and mass 'M'
#
n = 10
a = 2.0 * u.pc
M = 1000.0 * const.M_sun


phi = numpy.random.rand(n)*2.0*numpy.pi
theta = numpy.arccos((numpy.random.rand(n) - 0.5)*2.0)
r = a / numpy.sqrt(numpy.random.rand(n)**(-2.0/3.0) - 1)

m = numpy.ones(n) * M / n
x = r*numpy.sin(theta)*numpy.cos(phi)
y = r*numpy.sin(theta)*numpy.sin(phi)
z = r*numpy.cos(theta)

#
# compute the potential of the cluster at a certain distance
#
pot = potential_cpu.potential(
    x.value,
    y.value,
    z.value,
    m.value,
    0.0
)

print('done')

