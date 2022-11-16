import astromet
import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from jaxtromet.tracks import T

from ..lensing import (define_lens)


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
