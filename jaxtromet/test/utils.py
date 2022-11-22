import astromet
import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from jaxtromet.tracks import T

from ..lensing import (define_lens)

test_ra_dec_data = np.array(np.meshgrid(np.linspace(0, 360., 20), np.linspace(-180, 180., 20))).T.reshape((-1, 2))
test_u0_data = np.linspace(-10., 10., 20)
test_t0_data = np.linspace(6000., 8000., 20)
test_tE_data = np.linspace(0.1, 1000., 20)
test_pi_data = np.array(np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))).T.reshape((-1, 2))
test_m0_data = np.linspace(10., 20, 20)
test_fbl_data = np.linspace(0.01, 1., 20)
test_pm_source_data = np.array(np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))).T.reshape((-1, 2))
test_d_source_data = np.linspace(0.01, 20., 20)
test_thetaE_data = np.linspace(0.01, 50., 20)


def generate_lens_params(ra: float, dec: float, u0: float,
                        t0: float, tE: float, piEN: float,
                        piEE: float, m0: float, fbl: float,
                        pmrac_source: float, pmdec_source: float,
                        d_source: float, thetaE: float):
    astromet_params = astromet.define_lens(
        astromet.params(),
        ra=ra,
        dec=dec,
        u0=u0,
        t0=t0,
        tE=tE,
        piEN=piEN,
        piEE=piEE,
        m0=m0,
        fbl=fbl,
        pmrac_source=pmrac_source,
        pmdec_source=pmdec_source,
        d_source=d_source,
        thetaE=thetaE)
    jaxtromet_params = define_lens(
        ra=ra,
        dec=dec,
        u0=u0,
        t0=t0,
        tE=tE,
        piEN=piEN,
        piEE=piEE,
        fbl=fbl,
        pmrac_source=pmrac_source,
        pmdec_source=pmdec_source,
        d_source=d_source,
        thetaE=thetaE)

    return astromet_params, jaxtromet_params
