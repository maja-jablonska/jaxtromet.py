import astromet
import jax.numpy as jnp
import numpy as np
import pytest

from ..lensing import (blend, get_dirs, define_lens,
                       lensed_binary, ulens, get_offset,
                       onsky_lens)

from .utils import *


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

    def test_blend_proper_blend(self, setUp):
        astromet_blend_result_drac, astromet_blend_result_ddec = astromet.blend(np.array([1.e-1, 1.5e-1]),
                                                                                np.array([.5e-1, .75e-1]),
                                                                                np.array([1.1e-1, 1.2e-1]),
                                                                                np.array([1.e-1, .9e-1]),
                                                                                .75)
        jaxtromet_blend_result_drac, jaxtromet_blend_result_ddec = blend(jnp.array([1.e-1, 1.5e-1]),
                                                                         jnp.array([.5e-1, .75e-1]),
                                                                         jnp.array([1.1e-1, 1.2e-1]),
                                                                         jnp.array([1.e-1, .9e-1]),
                                                                         .75)

        assert np.all(np.isclose(astromet_blend_result_drac,
                                 np.array(jaxtromet_blend_result_drac),
                                 1e-6))
        assert np.all(np.isclose(astromet_blend_result_ddec,
                                 np.array(jaxtromet_blend_result_ddec),
                                 1e-6))

    def test_blend_proper_no_blend(self, setUp):
        astromet_blend_result_drac, astromet_blend_result_ddec = astromet.blend(np.array([1.e-1, 1.5e-1]),
                                                                                np.array([.5e-1, .75e-1]),
                                                                                np.array([1.1e-1, 1.2e-1]),
                                                                                np.array([1.e-1, .9e-1]),
                                                                                1.)
        jaxtromet_blend_result_drac, jaxtromet_blend_result_ddec = blend(jnp.array([1.e-1, 1.5e-1]),
                                                                         jnp.array([.5e-1, .75e-1]),
                                                                         jnp.array([1.1e-1, 1.2e-1]),
                                                                         jnp.array([1.e-1, .9e-1]),
                                                                         1.)

        assert np.all(np.isclose(astromet_blend_result_drac,
                                 np.array(jaxtromet_blend_result_drac),
                                 1e-6))
        assert np.all(np.isclose(astromet_blend_result_ddec,
                                 np.array(jaxtromet_blend_result_ddec),
                                 1e-6))

    def test_get_dirs(self, setUp):
        astromet_east, astromet_north = astromet.get_dirs(100., -50.)
        jaxtromet_east, jaxtromet_north = get_dirs(100., -50.)

        assert np.all(np.isclose(astromet_east, np.array(jaxtromet_east), 1e-6))
        assert np.all(np.isclose(astromet_north, np.array(jaxtromet_north), 1e-6))

    def test_lens_params(self, setUp):
        astromet_params, jaxtromet_params = setUp

        assert np.all(np.isclose(astromet_params.pmrac,
                                 jaxtromet_params.pmrac,
                                 1e-6))
        assert np.all(np.isclose(astromet_params.pmdec,
                                 jaxtromet_params.pmdec,
                                 1e-6))
        assert np.all(np.isclose(astromet_params.blendparallax,
                                 jaxtromet_params.blendparallax,
                                 1e-6))
        assert np.all(np.isclose(astromet_params.blendpmrac,
                                 jaxtromet_params.blendpmrac,
                                 1e-6))
        assert np.all(np.isclose(astromet_params.blendpmdec,
                                 jaxtromet_params.blendpmdec,
                                 1e-6))

    def test_lensed_binary(self, setUp):
        astromet_params, jaxtromet_params = setUp

        # binary parameters
        astromet_params.period = 1  # year
        astromet_params.a = 100  # AU
        astromet_params.e = 0.1
        astromet_params.q = .5
        astromet_params.l = .5  # assumed < 1 (though may not matter)
        astromet_params.vtheta = np.pi / 4
        astromet_params.vphi = np.pi / 4
        astromet_params.vomega = .5
        astromet_params.tperi = .5  # jyear

        jaxtromet_params.period = 1  # year
        jaxtromet_params.a = 100  # AU
        jaxtromet_params.e = 0.1
        jaxtromet_params.q = .5
        jaxtromet_params.l = .5  # assumed < 1 (though may not matter)
        jaxtromet_params.vtheta = jnp.pi / 4
        jaxtromet_params.vphi = jnp.pi / 4
        jaxtromet_params.vomega = .5
        jaxtromet_params.tperi = .5  # jyear

        astromet_lensed_binary = astromet.lensed_binary(astromet_params,
                                                        np.array([1.e-1, .5e-1]),
                                                        np.array([1.e-1, .5e-1]),
                                                        np.array([9.e-1, .95e-1]),
                                                        np.array([1.12e-1, 1.11e-1]),
                                                        np.array([.85e-1, .85e-1]),
                                                        np.array([.9e-1, 1.e-1]))
        jaxtromet_lensed_binary = lensed_binary(jaxtromet_params,
                                                jnp.array([1.e-1, .5e-1]),
                                                jnp.array([1.e-1, .5e-1]),
                                                jnp.array([9.e-1, .95e-1]),
                                                jnp.array([1.12e-1, 1.11e-1]),
                                                jnp.array([.85e-1, .85e-1]),
                                                jnp.array([.9e-1, 1.e-1]))

        assert np.all(
            np.isclose(
                np.array(list(astromet_lensed_binary)).flatten(),
                np.array([np.array(a) for a in list(jaxtromet_lensed_binary)]).flatten(),
                1e-5
            )
        )

    def test_ulens(self, setUp):
        astromet_ulens = astromet.ulens(np.array([1.e-1, 1.5e-1]),
                                        np.array([1.2e-1, .9e-1]),
                                        2.6)
        jaxtromet_ulens = ulens(jnp.array([1.e-1, 1.5e-1]),
                                jnp.array([1.2e-1, .9e-1]),
                                2.6)

        assert np.all(
            np.isclose(
                np.array(list(astromet_ulens)),
                np.array([np.array(a) for a in list(jaxtromet_ulens)]),
                1e-6
            )
        )

    @pytest.mark.parametrize("ra,dec", test_ra_dec_data)
    def test_offset_different_ra_dec(self, ra, dec):
        astromet_params, jaxtromet_params = generate_lens_params(
            ra, dec, 0.5, 7200., 40., 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_params_w_offset = astromet.get_offset(astromet_params,
                                                       0.5,
                                                       7620.)

        jaxtromet_params_w_offset = get_offset(jaxtromet_params,
                                               0.5,
                                               7620.)

        assert np.isclose(
            astromet_params_w_offset.blenddrac,
            jaxtromet_params_w_offset.blenddrac,
            1e-6
        )

        assert np.isclose(
            astromet_params_w_offset.blendddec,
            jaxtromet_params_w_offset.blendddec,
            1e-6
        )


    @pytest.mark.parametrize("u0", test_u0_data)
    def test_offset_different_y0(self, u0):
        astromet_params, jaxtromet_params = generate_lens_params(
            10, 10, u0, 7200., 40., 0.5, 0.5, 15., 1., 5., 5., 1., 1.)

        astromet_params_w_offset = astromet.get_offset(astromet_params,
                                                       0.5,
                                                       7620.)

        jaxtromet_params_w_offset = get_offset(jaxtromet_params,
                                               0.5,
                                               7620.)

        assert np.isclose(
            astromet_params_w_offset.blenddrac,
            jaxtromet_params_w_offset.blenddrac,
            1e-6
        )

        assert np.isclose(
            astromet_params_w_offset.blendddec,
            jaxtromet_params_w_offset.blendddec,
            1e-6
        )


    def test_onsky_lens(self, setUp):
        astromet_lens = astromet.onsky_lens(np.array([1e-1, .5e-1]),
                                            np.array([.5e-1, .75e-1]),
                                            np.array([.95e-1, 1.1e-1]),
                                            np.array([.8e-1, .85e-1]),
                                            2.5,
                                            .75)

        jaxtromet_lens = onsky_lens(jnp.array([1e-1, .5e-1]),
                                    jnp.array([.5e-1, .75e-1]),
                                    jnp.array([.95e-1, 1.1e-1]),
                                    jnp.array([.8e-1, .85e-1]),
                                    2.5,
                                    .75)

        assert np.all(
            np.isclose(
                np.array(list(astromet_lens)).flatten(),
                np.array([np.array(a) for a in list(jaxtromet_lens)]).flatten(),
                1e-5
            )
        )

    def test_onsky_lens_no_blending(self, setUp):
        astromet_lens = astromet.onsky_lens(np.array([1e-1, .5e-1]),
                                            np.array([.5e-1, .75e-1]),
                                            np.array([.95e-1, 1.1e-1]),
                                            np.array([.8e-1, .85e-1]),
                                            2.5,
                                            0.)

        jaxtromet_lens = onsky_lens(jnp.array([1e-1, .5e-1]),
                                    jnp.array([.5e-1, .75e-1]),
                                    jnp.array([.95e-1, 1.1e-1]),
                                    jnp.array([.8e-1, .85e-1]),
                                    2.5,
                                    0.)

        assert np.all(
            np.isclose(
                np.array(list(astromet_lens)).flatten(),
                np.array([np.array(a) for a in list(jaxtromet_lens)]).flatten(),
                1e-5
            )
        )
