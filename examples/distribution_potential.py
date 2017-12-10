"""
<keywords>
test, python, potential, nbody, plummer
</keywords>
<description>
example script for computing the potential for a sampled plummer sphere
</description>
<seealso>
</seealso>
"""
import numpy

from astropy import units as u
from astropy import constants as const

import nbodyswissknife

#
# sample a plummer sphere with core radius 'b' and mass 'M'
#
n = 1000                   # number of particles
b = 5.3 * u.pc             # scale length of the plummer model
M = 132.0 * const.M_sun    # the mass of the cluster
G = const.G                # gravitational constant
r_loc = numpy.array([1.0, 2.0, 3.0]) * u.pc

#
# sample a plummer sphere with core radius 'b' and mass 'M'
#
phi = numpy.random.rand(n)*2.0*numpy.pi
theta = numpy.arccos((numpy.random.rand(n) - 0.5)*2.0)
r = b / numpy.sqrt(numpy.random.rand(n)**(-2.0/3.0) - 1)

m = numpy.ones(n) * M / n
x = r*numpy.sin(theta)*numpy.cos(phi)
y = r*numpy.sin(theta)*numpy.sin(phi)
z = r*numpy.cos(theta)

#
# compute the potential of the cluster at a certain distance
#
pot = nbodyswissknife.potential.potential_native(
    numpy.vstack((x, y, z)),
    m,
    soft=0.0,
    gauss=G,
    location=r_loc,
)

pot_theoretical_plummer_at_r_loc = - G * M / numpy.sqrt(
    numpy.linalg.norm(r_loc)**2 + b.value**2
)

print('computed potential at r_loc    = ', pot)
print('theoretical potential at r_loc = ', pot_theoretical_plummer_at_r_loc)
print(
    'relative difference          = ',
    1.0 - (pot / pot_theoretical_plummer_at_r_loc).value
)

print(pot)

print('done')

