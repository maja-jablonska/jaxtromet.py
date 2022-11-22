import astromet
import jax.numpy as jnp
import numpy as np
import pytest

from ..fits import (downweight, agis_2d_prior, fit, mock_obs)
from ..lensing import (define_lens)
from ..tracks import barycentricPosition


class TestBlend:
    def test_downweight(self):
        astromet_weights = astromet.downweight(np.array([1., 1.]),
                                               np.array([1e-2, 1.2e-2]),
                                               np.array([1e-1, 2e-2]))

        jaxtromet_weights = downweight(jnp.array([1., 1.]),
                                       jnp.array([1e-2, 1.2e-2]),
                                       jnp.array([1e-1, 2e-2]))

        assert np.all(
            np.isclose(astromet_weights, jaxtromet_weights, 1e-6)
        )

    def test_agis_2d_prior(self):
        astromet_prior = astromet.agis_2d_prior(10., 20., 15.)
        jaxtromet_prior = agis_2d_prior(10., 20., 15.)

        assert np.all(np.isclose(astromet_prior, jaxtromet_prior, 1e-6))

    def test_fit(self):
        bs = barycentricPosition(np.array([2014.5, 2014.6]))
        astromet_fit = astromet.fit(np.array([2014.5, 2014.6]),
                                    np.array([1e-2, 1.05e-2]),
                                    np.array([.9e-4, 5.e-5]),
                                    10.,
                                    -5.,
                                    15.)

        jaxtromet_fit = fit(jnp.array([2014.5, 2014.6]),
                            bs,
                            jnp.array([1e-2, 1.05e-2]),
                            jnp.array([.9e-4, 5.e-5]),
                            10.,
                            -5.,
                            15.)

        for k in astromet_fit.keys():
            assert (
                    (np.isnan(astromet_fit[k]) and jnp.isnan(jaxtromet_fit[k])) or
                    np.isclose(astromet_fit[k], jaxtromet_fit[k], 1e-6)
            )

    def test_mock_obs(self):
        astromet_mock_obs = astromet.mock_obs(np.array([2014.5, 2014.75]),
                                              np.array([0.75, 0.9]),
                                              np.array([1., 2.]),
                                              np.array([0.9, 0.6]),
                                              jnp.repeat(np.array([0., 0.]), 9))
        jaxtromet_mock_obs = mock_obs(jnp.array([2014.5, 2014.75]),
                                      jnp.array([0.75, 0.9]),
                                      jnp.array([1., 2.]),
                                      jnp.array([0.9, 0.6]),
                                      jnp.repeat(jnp.array([0., 0.]), 9))

        assert np.all(np.isclose(astromet_mock_obs, jaxtromet_mock_obs, 1e-3))
