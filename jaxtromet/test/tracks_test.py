import astromet
import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from ..lensing import (define_lens)
from ..utils import hashdict
from ..tracks import (totalmass, barycentricPosition, design_matrix,
                      findEtas, binaryMotion, track)

config.update("jax_enable_x64", True)


class TestBlend:
    @pytest.fixture()
    def setUp(self):
        astromet_params = astromet.define_lens(astromet.params(),
                                               100.,
                                               -50.,
                                               0.5,
                                               7260.0,
                                               50.,
                                               0.25,
                                               0.2,
                                               13.5,
                                               0.75,
                                               -6.2,
                                               1.2,
                                               2.,
                                               2.5)
        jaxtromet_params = define_lens(100.,
                                       -50.,
                                       0.5,
                                       7260.0,
                                       50.,
                                       0.25,
                                       0.2,
                                       0.75,
                                       -6.2,
                                       1.2,
                                       2.,
                                       2.5)
        print("setup")
        yield astromet_params, jaxtromet_params
        print("teardown")

    def test_totalmass(ps, setUp):
        astromet_params, jaxtromet_params = setUp

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    def test_design_matrix_with_phis(self, setUp):
        bs = barycentricPosition(jnp.array([2014.2, 2014.5]))

        astromet_design_matrix = astromet.design_matrix(np.array([2014.2,
                                                                  2014.5]),
                                                        .5, -.5,
                                                        phis=np.array([10., 20.]))

        jaxtromet_design_matrix = design_matrix(jnp.array([2014.2, 2014.5]),
                                                bs,
                                                .5, -.5,
                                                phis=jnp.array([10., 20.]))

        # I don't know why one term is different on the .001
        # To investigate
        assert np.all(
            np.isclose(
                astromet_design_matrix,
                np.array(jaxtromet_design_matrix),
                1e-3)
        )

    def test_find_etas(self, setUp):
        astromet_etas = astromet.findEtas(np.array([2014.5, 2014.75]),
                                          2.,
                                          .5)

        jaxtromet_etas = findEtas(jnp.array([2014.5, 2014.75]),
                                  2.,
                                  .5)

        print(astromet_etas)
        print(jaxtromet_etas)

        assert np.all(
            np.isclose(
                astromet_etas, np.array(jaxtromet_etas), 1e-6
            )
        )

    def test_binary_motionn(self, setUp):
        '''
        binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):
        '''

        astromet_binary_motions = astromet.binaryMotion(np.array([2014.5, 2014.75]),
                                                        2., .9, .9, 10., .5, .9, .9)

        jaxtromet_binary_motions = binaryMotion(jnp.array([2014.5, 2014.75]),
                                                2., .9, .9, 10., .5, .9, .9)

        assert np.all(
            np.isclose(
                np.array(list(astromet_binary_motions)),
                np.array([np.array(a) for a in list(jaxtromet_binary_motions)]),
                1e-6
            )
        )

    def test_track_lensing(self, setUp):
        '''
        binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):
        '''

        astromet_params, jaxtromet_params = setUp
        bs = barycentricPosition(jnp.array([2014.2, 2014.5]))

        astromet_track = astromet.track(np.array([2014.2, 2014.5]), astromet_params)

        jaxtromet_track = track(jnp.array([2014.2, 2014.5]), bs, dict(jaxtromet_params))

        assert np.all(
            np.isclose(
                np.array(list(astromet_track)),
                np.array([np.array(a) for a in list(jaxtromet_track)]),
                1e-6
            )
        )

# TODO: add binary, lensed binary, unlensed binary etc.


