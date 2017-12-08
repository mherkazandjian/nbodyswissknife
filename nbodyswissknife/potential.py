# coding=utf-8
"""module for computing potentials"""
import numpy

__all__ = [
    'potential_native',
    'potential_cpu'
]


def potential_native(coords, mass, soft, gauss, location):
    """
    Compute the potential of particles at a certain location

    .. code-block:: python

       pot = potential.potential_native(
           coords=numpy.zeros((3, n_part), 'f8'),
           mass=numpy.zeros(n_part, 'f8'),
           soft=0.1,
           gauss=2.1,
           location=[1.0, 2.0, 3.0]
       )

    :param ndarray coords: the coordinates of the particles. The shape
     of this array should be 3 x N, where N is the number of particles.
     x = coords[0, :], y = coords[1, :], z = coords[2, :] are the x, y, z
     coordinates respectively.
    :param ndarray mass: the masses of the particles of size N
    :param float soft: the softening
    :param float gauss: the gravitational constant
    :param ndarray location: the x,y,z location of the point where the
     the potential will be calculated.
    :return: float: The potential at the location r
    """
    r_vec = coords - numpy.array(location).reshape(3, 1)
    r_mag = numpy.sqrt((r_vec**2).sum(axis=0) + soft**2)
    pot = - mass * gauss / r_mag
    return pot.sum()


def potential_cpu():
    pass


class Potential(object):
    """

    """
    def __init__(self):
        """

        """
        pass

    @property
    def coordinates(self):
        """

        :return:
        """
        pass

    @property
    def masses(self):
        """
        """
        pass

    @property
    def softening(self):
        """

        :return:
        """
        pass

    @property
    def gauss(self):
        """

        :return:
        """
        pass

    def prepare(self):
        """

        :return:
        """
        pass

    def use(self):
        """

        :return:
        """
        pass

    def compute(self, using='native'):
        """

        :param using:
        :return:
        """
        pass
