from functools import partial

import astropy.coordinates
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax import random

from .tracks import design_matrix_with_phis

key = random.PRNGKey(420)
T = (1.0 * u.year).to(u.day).value

mas = (1.0 * u.mas).to(u.deg).value


@jit
def downweight(R: jnp.array,
               err: jnp.array,
               aen: jnp.array) -> jnp.array:
    """
    Downweighting function used in AGIS. Lindegren 2012.
    Args:
        - R,    jnp.array - residual of observed source position from astrometric solution.
        - err,  jnp.array - astrometric uncertainty of each observation
        - aen,  jnp.array - source astrometric excess noise ?why not scalar?
    Returns:
        - w,    jnp.array - observation weights
    """
    z = jnp.sqrt(R ** 2 / (err ** 2 + aen ** 2))
    w = jnp.where(z < 2, 1, 1 - 1.773735 * (z - 2) ** 2 + 1.141615 * (z - 2) ** 3)
    w = jnp.where(z < 3, w, jnp.exp(-z / 3))
    # ? w = 0 or 1 only ?
    return w


@jit
def en_fit(R: jnp.array,
           err: jnp.array,
           w: jnp.array) -> jnp.array:
    """
    Iterative optimization to fit excess noise in AGIS (inner iteration). Lindegren 2012.
    Args:
        - R,    jnp.array - residual of observed source position from astrometric solution.
        - err,  jnp.array - astrometric uncertainty of each observation
        - w,    jnp.array - observation weights
    Returns:
        - aen,  jnp.array - astrometric_excess_noise
    """
    i = 0
    y = 0
    nu = jnp.sum(w >= 0.2) - 5

    W = w / (err ** 2 + y)
    Q = jnp.sum(R ** 2 * W)

    W_prime = -w / (err ** 2 + y) ** 2
    Q_prime = jnp.sum(R ** 2 * W_prime)
    
    @jit
    def __iterative_y(state):
        y, nu, W, Q, W_prime, Q_prime = state
        def __loop_body(x, state):
            y, nu, W, Q, W_prime, Q_prime = state
            W = w / (err ** 2 + y)
            Q = jnp.sum(R ** 2 * W)
            W_prime = -w / (err ** 2 + y) ** 2
            Q_prime = jnp.sum(R ** 2 * W_prime)

            y = y + (1 - Q / nu) * Q / Q_prime

            return (y, nu, W, Q, W_prime, Q_prime)
        
        (y, nu, W, Q, W_prime, Q_prime) = lax.fori_loop(0, 4,
                                                        __loop_body,
                                                        (y, nu, W, Q, W_prime, Q_prime))
        
        return jnp.sqrt(y)
    
    return lax.cond(jnp.sum(R**2 * (w/(err**2 + y))) <= nu,
                    lambda _: 0.,
                    __iterative_y,
                    (y, nu, W, Q, W_prime, Q_prime))


def agis_2d_prior(ra, dec, G):
    coord = astropy.coordinates.SkyCoord(ra, dec, unit='deg', frame='icrs')
    _l = coord.galactic.l.rad
    _b = coord.galactic.b.rad

    # Prior
    s0 = 2.187 - 0.2547 * G + 0.006382 * G ** 2
    s1 = 0.114 - 0.0579 * G + 0.01369 * G ** 2 - 0.000506 * G ** 3
    s2 = 0.031 - 0.0062 * G
    sigma_pi_f90 = 10 ** (s0 + s1 * jnp.abs(jnp.sin(_b)) + s2 * jnp.cos(_b) * jnp.cos(_l))

    prior_cov = jnp.eye(5)
    prior_cov = prior_cov.at[:2, :2].set(prior_cov[:2, :2] * 1000 ** 2)
    prior_cov = prior_cov.at[2:4, 2:4].set(prior_cov[2:4, 2:4] * sigma_pi_f90 ** 2)
    prior_cov = prior_cov.at[4, 4].set(prior_cov[4, 4] * (10 * sigma_pi_f90) ** 2)

    prior_prec = jnp.linalg.inv(prior_cov)

    return prior_prec


@jit
def fit_model(x_obs, x_err, M_matrix, prior):
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration). Lindegren 2012.
    Args:
        - x_obs,    jnp.array - observed along-scan source position at each epoch.
        - x_err,    jnp.array - astrometric measurement uncertainty for each observation.
        - M_matrix, jnp.array - Design matrix.
    Returns:
        - r5d_mean
        - r5d_cov
        - R
        - aen_i
        - W_i
    """

    prior = jax.lax.cond(prior is None, lambda _: np.zeros((5, 5)),
                         lambda x: x, prior)

    # Initialise - get initial astrometry estimate with weights=1
    aen = 0
    weights = jnp.ones(len(x_obs))

    W = jnp.eye(len(x_obs)) * weights / (x_err ** 2 + aen ** 2)
    r5d_cov = jnp.linalg.inv(jnp.matmul(M_matrix.T, jnp.matmul(W, M_matrix)) + prior)
    r5d_mean = jnp.matmul(r5d_cov, jnp.matmul(M_matrix.T, jnp.matmul(W, x_obs)))
    R = x_obs - jnp.matmul(M_matrix, r5d_mean)

    # Step 1: initial Weights
    # Intersextile range
    ISR = jnp.diff(jnp.percentile(R, jnp.array([100. * 1. / 6, 100. * 5. / 6.])))[0]
    # Evaluate weights
    weights = downweight(R, ISR / 2., 0.)

    @jit
    def __loop_body(x, state):
        W, r5d_cov, r5d_mean, R, aen, weights = state
        W = jnp.eye(len(x_obs)) * weights / (x_err ** 2 + aen ** 2)
        r5d_cov = jnp.linalg.inv(jnp.matmul(M_matrix.T, jnp.matmul(W, M_matrix)) + prior)
        r5d_mean = jnp.matmul(r5d_cov, jnp.matmul(M_matrix.T, jnp.matmul(W, x_obs)))
        R = x_obs - jnp.matmul(M_matrix, r5d_mean)

        # Step 3 - astrometric_excess_noise
        aen = en_fit(R, x_err, weights)

        # Step 4 - Observation Weights
        weights = downweight(R, x_err, aen)

        # Step 3 - astrometric_excess_noise
        aen = en_fit(R, x_err, weights)

        return (W, r5d_cov, r5d_mean, R, aen, weights)

    W, r5d_cov, r5d_mean, R, aen, weights = lax.fori_loop(0, 10,
                                                          __loop_body,
                                                          (W, r5d_cov, r5d_mean, R, aen, weights))

    # Final Astrometry Linear Regression fit
    r5d_cov = jnp.linalg.inv(jnp.matmul(M_matrix.T, jnp.matmul(W, M_matrix)) + prior)
    r5d_mean = jnp.matmul(r5d_cov, jnp.matmul(M_matrix.T, jnp.matmul(W, x_obs)))
    R = x_obs - jnp.matmul(M_matrix, r5d_mean)

    return r5d_mean, r5d_cov, R, aen, weights


@partial(jit, static_argnums=(5, 6, 7))
def fit(ts, bs, xs, phis, xerr, ra, dec, G=12, epoch=2016.0) -> dict:
    """
    Iterative optimization to fit astrometric solution in AGIS (outer iteration).
    Lindegren 2012.
    Args:
        - ts,          jnp.array - source observation times, jyear
        - xs,          jnp.array - source 1d positions relative to ra,dec , mas
        - phis,        jnp.array - source observation scan angles, deg
        - errs,        jnp.array - scan measurement error, mas
        - ra,          float - RA for design_1d, deg
        - dec,         float - Dec for design_1d, deg
        - G,           float - Apparent magnitude, 2 parameter prior only, mag
        - epoch        float - Epoch at which results are calculated, jyear
            Returns:
        - results      dict - output data Gaia would produce
    """

    results = {}
    results['vis_periods'] = jnp.sum(jnp.sort(ts)[1:] * T - jnp.sort(ts)[:-1] * T > 4)
    results['n_obs'] = len(ts)

    prior = jax.lax.cond(results['vis_periods'] < 6,
                         lambda _: agis_2d_prior(ra, dec, G),
                         lambda _: jnp.zeros((5, 5)),
                         results)
    results['params_solved'] = jax.lax.cond(results['vis_periods'] < 6,
                                            lambda _: 2,
                                            lambda _: 5,
                                            results)

    # Design matrix
    design = design_matrix_with_phis(ts, bs, jnp.deg2rad(ra), jnp.deg2rad(dec), phis=phis, epoch=epoch)

    r5d_mean, r5d_cov, R, aen, weights = fit_model(xs, xerr, design, prior=prior)

    coords = ['drac', 'ddec', 'parallax', 'pmrac', 'pmdec']
    for i in range(5):
        results[coords[i]] = r5d_mean[i]
        results[coords[i] + '_error'] = jnp.sqrt(r5d_cov[i, i])
        for j in range(i):
            results[coords[j] + '_' + coords[i] + '_corr'] = \
                r5d_cov[i, j] / jnp.sqrt(r5d_cov[i, i] * r5d_cov[j, j])

    results['excess_noise'] = aen
    results['chi2'] = jnp.sum(R ** 2 / xerr ** 2)
    results['n_good_obs'] = jnp.sum(weights > 0.2)
    nparam = 5
    results['uwe'] = jnp.sqrt(jnp.sum(R ** 2 / xerr ** 2) / (jnp.sum(weights > 0.2) - nparam))
    results['ra_ref'] = ra
    results['dec_ref'] = dec

    return results


@jit
def mock_obs(ts: jnp.array,
             phis: jnp.array,
             racs: jnp.array,
             decs: jnp.array,
             key,
             err=0,
             nmeasure=9) -> tuple:
    """
    Converts positions to comparable observables to real astrometric measurements
    (i.e. 1D positions along some scan angle, optionlly with errors added)
    Args:
        - ts,       jnp.array - Observation times, jyear.
        - phis,     jnp.array - Scanning angles (0 north, 90 east), degrees.
        - racs,     jnp.array - RAcosDec at each scan, mas
        - decs,     jnp.array - Dec at each scan, mas
        - err,      float or jnp.array - optional normal distributed error to be added (default 0)
        - nmeasure, int - optinal, number of measurements per transit (default 9)
    Returns:
        - copies of all entered parameters measured nmeasure times with errors
        - xs        ndarray - 1D projected displacements
    """
    ts = jnp.repeat(ts, nmeasure)
    phis = jnp.repeat(phis, nmeasure)
    errs = err * jax.random.normal(key, shape=ts.shape)
    racs = jnp.repeat(racs, nmeasure) + errs * jnp.sin(jnp.deg2rad(phis))
    decs = jnp.repeat(decs, nmeasure) + errs * jnp.cos(jnp.deg2rad(phis))
    xs = racs * jnp.sin(jnp.deg2rad(phis)) + decs * jnp.cos(jnp.deg2rad(phis))
    return ts, xs, phis, racs, decs


def gaia_results(results):
    # translates results from full fit into a Gaia specific dictionary
    gresults = {}
    # TODO: fix
    # gresults['astrometric_matched_transits'] = results['n_obs'] / 9.0
    gresults['visibility_periods_used'] = results['vis_periods']
    gresults['astrometric_n_obs_al'] = results['n_obs']

    # TODO: fix
    # if results['params_solved'] == 2:
    #     gresults['astrometric_params_solved'] = 3
    # else:
    #     gresults['astrometric_params_solved'] = 31

    coords = ['drac', 'ddec', 'parallax', 'pmrac', 'pmdec']
    gcoords = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    for i in range(5):
        gresults[gcoords[i]] = results[coords[i]]
        gresults[gcoords[i] + '_error'] = results[coords[i] + '_error']
        for j in range(i):
            gresults[gcoords[j] + '_' + gcoords[i] + '_corr'] = \
                results[coords[j] + '_' + coords[i] + '_corr']
    gresults['ra'] = results['ra_ref'] + results['drac'] * mas / jnp.cos(results['dec_ref'])
    gresults['dec'] = results['dec_ref'] + results['ddec'] * mas

    gresults['astrometric_excess_noise'] = results['excess_noise']
    gresults['astrometric_chi2_al'] = results['chi2']
    gresults['astrometric_n_good_obs_al'] = results['n_good_obs']
    gresults['uwe'] = results['uwe']
    return gresults
