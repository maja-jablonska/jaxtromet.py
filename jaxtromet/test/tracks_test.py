import astromet
import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from ..lensing import (define_lens)
from ..utils import hashdict
from ..tracks import (totalmass, barycentricPosition, design_matrix,
                      findEtas, binaryMotion, track)

from .utils import *

config.update("jax_enable_x64", True)

class TestBlend:

    @pytest.mark.parametrize("ra,dec", test_ra_dec_data)
    def test_totalmass_different_positions(self, ra, dec):
        astromet_params, jaxtromet_params = generate_lens_params(
            ra, dec, 0.5, 7200., 40., 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("u0", test_u0_data)
    def test_totalmass_different_u0(self, u0):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., u0, 7200., 40., 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("t0", test_t0_data)
    def test_totalmass_different_t0(self, t0):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, t0, 40., 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("tE", test_tE_data)
    def test_totalmass_different_tE(self, tE):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, tE, 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("piEN,piEE", test_pi_data)
    def test_totalmass_different_pi(self, piEN, piEE):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., piEN, piEE, 15., 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)


    @pytest.mark.parametrize("m0", test_m0_data)
    def test_totalmass_different_m0(self, m0):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., 0.5, 0.5, m0, 1., 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("fbl", test_fbl_data)
    def test_totalmass_different_fbl(self, fbl):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., 0.5, 0.5, 15., fbl, 5., 5., 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)


    @pytest.mark.parametrize("pm_drac,pm_ddec", test_pm_source_data)
    def test_totalmass_different_pm_source(self, pm_drac, pm_ddec):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., 0.5, 0.5, 15., 1., pm_drac, pm_ddec, 1., 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("d_source", test_d_source_data)
    def test_totalmass_different_d_source(self, d_source):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., 0.5, 0.5, 15., 1., 5., 5., d_source, 1.)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    @pytest.mark.parametrize("thetaE", test_thetaE_data)
    def test_totalmass_different_thetaE(self, thetaE):
        astromet_params, jaxtromet_params = generate_lens_params(
            10., 10., 0.5, 7200, 40., 0.5, 0.5, 15., 1., 5., 5., 1., thetaE)

        astromet_totalmass = astromet.totalmass(astromet_params)
        jaxtromet_totalmass = totalmass(jaxtromet_params)

        assert np.isclose(astromet_totalmass, jaxtromet_totalmass, 1e-6)

    def test_design_matrix_with_phis(self):
        bs = barycentricPosition(jnp.array([2014.2, 2014.5]))

        astromet_design_matrix = astromet.design_matrix(np.array([2014.2,
                                                                  2014.5]),
                                                        .5, -.5,
                                                        phis=np.array([10., 20.]))

        jaxtromet_design_matrix = design_matrix(jnp.array([2014.2, 2014.5]),
                                                bs,
                                                .5, -.5,
                                                phis=jnp.array([10., 20.]))

        assert np.all(
            np.isclose(
                astromet_design_matrix,
                np.array(jaxtromet_design_matrix),
                1e-6)
        )

    def test_find_etas(self):
        astromet_etas = astromet.findEtas(np.array([2014.5, 2014.75], dtype=float),
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

#     def test_binary_motionn(self, setUp):
#         '''
#         binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):
#         '''

#         astromet_binary_motions = astromet.binaryMotion(np.array([2014.5, 2014.75]),
#                                                         2., .9, .9, 10., .5, .9, .9)

#         jaxtromet_binary_motions = binaryMotion(jnp.array([2014.5, 2014.75]),
#                                                 2., .9, .9, 10., .5, .9, .9)

#         print(list(astromet_binary_motions))
#         print(list(jaxtromet_binary_motions))

#         assert np.all(
#             np.isclose(
#                 np.array(list(astromet_binary_motions)),
#                 np.array([np.array(a) for a in list(jaxtromet_binary_motions)]),
#                 1e-6
#             )
#         )

#     def test_track_lensing(self, setUp):
#         '''
#         binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):
#         '''

#         astromet_params, jaxtromet_params = setUp
#         bs = barycentricPosition(jnp.array([2014.2, 2014.5]))

#         astromet_track = astromet.track(np.array([2014.2, 2014.5]), astromet_params)

#         jaxtromet_track = track(jnp.array([2014.2, 2014.5]), bs, dict(jaxtromet_params))

#         assert np.all(
#             np.isclose(
#                 np.array(list(astromet_track)),
#                 np.array([np.array(a) for a in list(jaxtromet_track)]),
#                 1e-6
#             )
#         )

# # TODO: add binary, lensed binary, unlensed binary etc.


