"""
<keywords>
test, python, potential, energy, nbody, grid, mesh
</keywords>
<description>
example script for computing the potential energy of a sampled plummer sphere
by computing the density grid and then using the density grid to compute
the potential energy
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
# construct the mass grid (bin masses in the grid)
#
r_max = 20*b
nbins = 50
xgrid, ygrid, zgrid = numpy.meshgrid(
    numpy.linspace(-r_max, r_max, nbins),
    numpy.linspace(-r_max, r_max, nbins),
    numpy.linspace(-r_max, r_max, nbins)
)
assert xgrid.shape == ygrid.shape == zgrid.shape
grid_shape = xgrid.shape

bsz = xgrid[0, 1, 0] - xgrid[0, 0, 0]  # all dims have equal bin sizes
mass_grid = numpy.zeros_like(xgrid.value).flatten()
mass_grid_cntrd_x = numpy.zeros_like(xgrid.value).flatten()
mass_grid_cntrd_y = numpy.zeros_like(xgrid.value).flatten()
mass_grid_cntrd_z = numpy.zeros_like(xgrid.value).flatten()

for i, (_x, _y, _z) in enumerate(
    zip(xgrid.flatten(), ygrid.flatten(), zgrid.flatten())):
    inds = numpy.where(
        (x >= _x) * (x < _x + bsz) *
        (y >= _y) * (y < _y + bsz) *
        (z >= _z) * (z < _z + bsz)
    )
    mass_grid[i] = m[inds].sum().value
    mass_grid_cntrd_x[i] = (_x + bsz / 2.0).value
    mass_grid_cntrd_y[i] = (_y + bsz / 2.0).value
    mass_grid_cntrd_z[i] = (_z + bsz / 2.0).value
    # print(i, int(i % (int(nbins**3) / 100)))
    if (i % int((nbins**3) / 100)) == 0:
        print('{:.2f}'.format((i / nbins**3)*100))

mass_grid = mass_grid.reshape(xgrid.shape) * m.unit
mass_grid_cntrd_x = mass_grid_cntrd_x.reshape(xgrid.shape) * x.unit
mass_grid_cntrd_y = mass_grid_cntrd_y.reshape(xgrid.shape) * y.unit
mass_grid_cntrd_z = mass_grid_cntrd_z.reshape(xgrid.shape) * z.unit

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

