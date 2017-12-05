# coding=utf-8
"""test suite for testing the potential module"""
import numpy
from nbodyswissknife import potential


def test_single_particle_potential_at_the_origin():

    pot = potential.potential_native(
        coords=numpy.array([0.0, 0.0, 0.0]).reshape(3, 1),
        mass=1.0,
        soft=0.0,
        gauss=1.0,
        location=numpy.array([1.0, 0.0, 0.0])
    )

    pot_expected = - 1.0
    numpy.testing.assert_allclose(pot, pot_expected, rtol=1.0e-16)


def test_single_particle_potential_with_softening():

    pot = potential.potential_native(
        coords=numpy.array([0.0, 0.0, 0.0]).reshape(3, 1),
        mass=1.0,
        soft=1.0,
        gauss=1.0,
        location=numpy.array([1.0, 0.0, 0.0])
    )

    pot_expected = - 1.0 / numpy.sqrt(2.0)
    numpy.testing.assert_allclose(pot, pot_expected, rtol=1.0e-16)


def test_potential_many_paricles_at_the_origin():
    n_part = 50

    origin = [3.1, -4.1, 0.23]
    mass = 0.2
    soft = 0.1
    gauss = 2.1
    r_loc = [1.0, 2.0, 3.0]

    pot = potential.potential_native(
        coords=numpy.repeat(origin, n_part).reshape(3, n_part),
        mass=numpy.ones(n_part, 'f8')*mass,
        soft=soft,
        gauss=gauss,
        location=r_loc
    )

    pot_expected = - n_part * gauss * mass / numpy.sqrt(
        (origin[0] - r_loc[0]) ** 2 +
        (origin[1] - r_loc[1]) ** 2 +
        (origin[2] - r_loc[2]) ** 2 +
        soft**2
    )

    numpy.testing.assert_allclose(
        pot,
        pot_expected,
        rtol=1.0e-6,
        atol=0.0,
        verbose=True
    )
