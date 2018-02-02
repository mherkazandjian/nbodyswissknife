"""
<keywords>
test, python, potential, energy, nbody
</keywords>
<description>
example script for computing the potential energy of a sampled plummer sphere
</description>
<seealso>
</seealso>
"""
import numpy

from astropy import units as u
from astropy import constants as const

from nbodyswissknife import potential

#
# set the parameters of the model
#
n = 1000                   # number of particles
b = 5.3 * u.pc            # scale length of the plummer model
M = 132.0 * const.M_sun   # the mass of the cluster
G = const.G               # gravitational constant

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
# compute the potential energy of the cluster at a certain distance
#
pot_energy = potential.potential_energy_native(
    numpy.vstack((x.to('m'), y.to('m'), z.to('m'))).value,
    m.to('kg').value,
    soft=0.0,
    gauss=G.value
)

pot_energy_theoretical_plummer = (
    - 3.0*numpy.pi*G*M**2 / (32.0*b)
).to('kg m2 / s2').value

print('computed potential energy    = ', pot_energy)
print('theoretical potential energy = ', pot_energy_theoretical_plummer)
print(
    'relative difference          = ',
    1.0 - (pot_energy / pot_energy_theoretical_plummer)
)

print('done')

