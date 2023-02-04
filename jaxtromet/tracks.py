from cmath import cos
import astropy.coordinates
import erfa
import jax.numpy as jnp
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.time import Time
from jax import jit, lax
import jax

from .lensing import blend, onsky_lens

mSun = constants.M_sun.to(u.kg).value
lSun = constants.L_sun.to(u.W).value
kpc = constants.kpc.to(u.m).value
Gyr = (1.0 * u.Gyr).to(u.s).value
day = (1.0 * u.day).to(u.s).value
G = constants.G.to(u.m ** 3 / u.kg / u.s ** 2).value
AU = (1.0 * u.AU).to(u.m).value
c = constants.c.to(u.m / u.s).value
T = (1.0 * u.year).to(u.day).value
year = (1.0 * u.year).to(u.s).value
AU_c = (1.0 * u.AU / constants.c).to(u.day).value
Galt = constants.G.to(u.AU ** 3 / u.M_sun / u.year ** 2).value
mas2rad = (1.0 * u.mas).to(u.rad).value
mas = (1.0 * u.mas).to(u.deg).value
earth_sun_mass_ratio = (constants.M_earth / constants.M_sun).value
tbegin = 2014.6670  # time (in years) of Gaia's first observations

use_backup = True  # If set to true use simpler backup Kepler equation solver.


def totalmass(ps):
    ps.totalmass = (4 * (jnp.pi ** 2) / Galt) * ((ps.a ** 3) / (ps.period ** 2))
    return ps.totalmass


def get_jd12(time, scale):
    """
    Gets ``jd1`` and ``jd2`` from a time object in a particular scale.
    Parameters
    ----------
    time : `~astropy.time.Time`
        The time to get the jds for
    scale : str
        The time scale to get the jds for
    Returns
    -------
    jd1 : float
    jd2 : float
    """
    newtime = getattr(time, scale)
    return newtime.jd1, newtime.jd2


def barycentricPosition(time):
    jd1, jd2 = get_jd12(astropy.time.Time(time, format='jyear'), 'tdb')
    _, earth_pv_bary = erfa.epv00(jd1, jd2)
    pos = jnp.array(earth_pv_bary['p'])
    # gaia satellite is at Earth-Sun L2
    l2corr = 1 + jnp.power(earth_sun_mass_ratio / 3, 1 / 3)
    return jnp.array(l2corr * pos)


class HashableArrayWrapper:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper) and
                jnp.all(jnp.equal(self.val, other.val)))


# TODO: add design_matrix with or without phis

@jit
def design_matrix_no_phis(ts: jnp.array,
                          bs: jnp.array,
                          ra: jnp.array,
                          dec: jnp.array,
                          epoch=2016.0) -> jnp.array:
    """
    design_matrix - Design matrix for ra,dec source track
    Args:
        - ts,       jnp.array - Observation times, jyear.
        - bs,       jnp.array - Barycentric coordinates of Gaia at times ts
        - ra, dec,  float - reference right ascension and declination of source, radians
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design,   jnp.array - Design matrix
    """
    # unit vector in direction of increasing ra - the local west unit vector
    p0 = jnp.array([-jnp.sin(ra), jnp.cos(ra), 0])
    # unit vector in direction of increasing dec - the local north unit vector
    q0 = jnp.array([-jnp.cos(ra) * jnp.sin(dec), -jnp.sin(ra) * jnp.sin(dec), jnp.cos(dec)])

    # Construct design matrix for ra and dec positions
    design = jnp.zeros((2, ts.shape[0], 5))
    design = design.at[0, :, 0].set(1)  # ra*cos(dec)
    design = design.at[1, :, 1].set(1)  # dec
    design = design.at[0, :, 2].set(-jnp.dot(p0, bs.T))  # parallax (ra component)
    design = design.at[1, :, 2].set(-jnp.dot(q0, bs.T))  # parallax (dec component)
    design = design.at[0, :, 3].set(ts - epoch)  # pmra
    design = design.at[1, :, 4].set(ts - epoch)  # pmdec

    return design


@jit
def design_matrix_with_phis(ts, bs, ra, dec, phis, epoch=2016.0):
    """
    design_matrix - Design matrix for ra,dec source track
    Args:
        - ts,       jnp.array - Observation times, jyear.
        - bs,       jnp.array - Barycentric coordinates of Gaia at times ts
        - phis,     jnp.array - scan angles.
        - ra, dec,  float - reference right ascension and declination of source, radians
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design, ndarry - Design matrix
    """
    # unit vector in direction of increasing ra - the local west unit vector
    p0 = jnp.array([-jnp.sin(ra), jnp.cos(ra), 0])
    # unit vector in direction of increasing dec - the local north unit vector
    q0 = jnp.array([-jnp.cos(ra) * jnp.sin(dec), -jnp.sin(ra) * jnp.sin(dec), jnp.cos(dec)])

    # Construct design matrix for ra and dec positions
    design = jnp.zeros((2, ts.shape[0], 5))
    design = design.at[0, :, 0].set(1)  # ra*cos(dec)
    design = design.at[1, :, 1].set(1)  # dec
    design = design.at[0, :, 2].set(-jnp.dot(p0, bs.T))  # parallax (ra component)
    design = design.at[1, :, 2].set(-jnp.dot(q0, bs.T))  # parallax (dec component)
    design = design.at[0, :, 3].set(ts - epoch)  # pmra
    design = design.at[1, :, 4].set(ts - epoch)  # pmdec

    angles = jnp.deg2rad(phis)
    sina = jnp.sin(angles)
    cosa = jnp.cos(angles)

    # Construct design matrix
    design = design[0] * sina[:, None] + design[1] * cosa[:, None]

    return design


@jit
def design_matrix(ts, bs, ra, dec, phis=None, epoch=2016.0):
    """
    design_matrix - Design matrix for ra,dec source track
    Args:
        - ts,       jnp.array - Observation times, jyear.
        - bs,       jnp.array - Barycentric coordinates of Gaia at times ts
        - phis,     jnp.array - scan angles.
        - ra, dec,  float - reference right ascension and declination of source, radians
        - epoch     float - time at which position and pm are measured, years CE
    Returns:
        - design, ndarry - Design matrix
    """
    # return jax.lax.cond(phis is None, lambda phis: design_matrix_no_phis(ts, bs, ra, dec, epoch=epoch),
    #                     lambda phis: design_matrix_with_phis(ts, bs, ra, dec, phis, epoch=epoch), phis)
    return design_matrix_with_phis(ts, bs, ra, dec, phis, epoch=epoch)


# TODO: add findEtas or findEtas_full as condition
@jit
def findEtas(ts: jnp.array,
             period: jnp.float32,
             eccentricity: jnp.float32,
             tPeri=0,
             N_it=10) -> jnp.array:
    """Finds eccentric anomaly with iterative Halley's method
    Args:
        ts (np.ndarray): Times.
        period (float): Period of orbit.
        eccentricity (float): Eccentricity. Must be in the range 0<e<1.
        tPeri (float): Pericentre time.
        N_it (float): Number of grid-points.
    Returns:
        np.ndarray: Array of eccentric anomalies, E.
    Slightly edited version of code taken from https://github.com/oliverphilcox/Keplers-Goat-Herd
    """

    phase = 2*jnp.pi*(((ts-tPeri)/period)%1)
    sph = jnp.sin(phase)
    cph = jnp.cos(phase)
    eta = phase + eccentricity*sph + (eccentricity**2)*sph*cph + 0.5*(eccentricity**3)*sph*(3*(cph**2)-1)
    deltaeta = jnp.ones_like(eta)

    def iteration(carry):
        it, deltaeta, eta = carry
        it += 1
        sineta = jnp.sin(eta)
        coseta = jnp.cos(eta)
        f = eta - eccentricity*sineta - phase
        df = 1. - eccentricity*coseta
        d2f = eccentricity*sineta
        deltaeta = -f*df / (df*df - 0.5*f*d2f)
        eta += deltaeta
        return it, deltaeta, eta

        # ((jnp.max(jnp.abs(arg[1]))>1e-5)) &

    _, _, eta = lax.while_loop(lambda arg: (jnp.max(jnp.abs(arg[1]))>1e-5) & arg[0]<N_it, iteration, (0, deltaeta, eta))
    return eta


@jit
def bodyPos(pxs, pys, l, q):  # given the displacements transform to c.o.m. frame
    px1s = -pxs * q / (1 + q)
    px2s = pxs / (1 + q)
    py1s = -pys * q / (1 + q)
    py2s = pys / (1 + q)
    pxls = pxs * (l - q) / ((1 + l) * (1 + q))
    pyls = pys * (l - q) / ((1 + l) * (1 + q))
    return px1s, py1s, px2s, py2s, pxls, pyls


@jit
def binaryMotion(ts, P, q, l, a, e, vTheta, vPhi, tPeri=0):  # binary position (in projected AU)
    etas = findEtas(ts, P, e, tPeri=tPeri)
    phis = 2 * jnp.arctan(jnp.sqrt((1 + e) / (1 - e)) * jnp.tan(etas / 2)) % (2 * jnp.pi)
    vPsis = vPhi - phis
    rs = a * (1 - e * jnp.cos(etas))
    g = jnp.power(1 - (jnp.cos(vPhi) ** 2) * (jnp.sin(vTheta) ** 2), -0.5)
    # projected positions in the c.o.m frame (in AU)
    pxs = rs * g * (jnp.cos(phis) - jnp.cos(vPsis) * jnp.cos(vPhi) * (jnp.sin(vTheta) ** 2))
    pys = rs * g * jnp.sin(phis) * jnp.cos(vTheta)
    # positions of sources 1 and 2 and the center of light
    px1s, py1s, px2s, py2s, pxls, pyls = bodyPos(pxs, pys, l, q)
    # x, y posn of each body and c.o.l.
    # in on-sky coords such that x is projected onto i dirn and y has no i component
    return px1s, py1s, px2s, py2s, pxls, pyls


@jit
def track(ts, bs, ps, comOnly=False):
    """
    Astrometric track in RAcos(Dec) and Dec [mas] for a given binary (or lensing event)
    Args:
        - ts,       jnp.array - Observation times, jyear.
        - bs,       jnp.array - Barycentric coordinates of Gaia at times ts
        - ps,       params object - Astrometric, binary and lensing parameters.
    Returns:
        - racs      ndarry - RAcosDec at each time, mas
        - decs      ndarry - Dec at each time, mas
        (optionally) - mag_diff     ndarray - difference from baseline magnitude at each time (for lensing events)
    """

    def add_lensing():
        r5d_blend = jnp.array(
            [ps["blenddrac"], ps["blendddec"], ps["blendparallax"], ps["blendpmrac"], ps["blendpmdec"]])
        dracs_blend, ddecs_blend = xij @ r5d_blend  # all in mas
        dracs_lensed, ddecs_lensed, mag_diff = onsky_lens(dracs, ddecs, dracs_blend, ddecs_blend, ps["thetaE"],
                                                          ps["blendl"])
        return dracs_lensed, ddecs_lensed, mag_diff

    def add_blend():
        r5d_blend = jnp.array(
            [ps["blenddrac"], ps["blendddec"], ps["blendparallax"], ps["blendpmrac"], ps["blendpmdec"]])
        dracs_blend, ddecs_blend = xij @ r5d_blend  # all in mas
        dracs_blended, ddecs_blended = blend(dracs, ddecs, dracs_blend, ddecs_blend, ps["blendl"])
        return dracs_blended, ddecs_blended, dracs_blended * 0.

    def comCorrection():
        # extra c.o.l. correction due to binary
        px1s, py1s, px2s, py2s, pxls, pyls = binaryMotion(
            ts - ps["tperi"], ps["period"], ps["q"], ps["l"], ps["a"], ps["e"], ps["vtheta"], ps["vphi"])
        rls = ps["parallax"] * (pxls * jnp.cos(ps["vomega"]) + pyls * jnp.sin(ps["vomega"]))
        dls = ps["parallax"] * (pyls * jnp.cos(ps["vomega"]) - pxls * jnp.sin(ps["vomega"]))
        return dracs + rls, ddecs + dls

    xij = design_matrix_no_phis(ts, bs, jnp.deg2rad(ps["ra"]), jnp.deg2rad(ps["dec"]), epoch=ps["epoch"])

    r5d = jnp.array([ps["drac"], ps["ddec"], ps["parallax"], ps["pmrac"], ps["pmdec"]])
    dracs, ddecs = xij @ r5d  # all in mas

    dracs, ddecs = lax.cond(comOnly == True,
                            comCorrection,
                            lambda: (dracs, ddecs))

    return lax.cond(ps["thetaE"] > 0, add_lensing,
                    lambda: lax.cond(ps["blendl"] > 0,
                                     add_blend,
                                     lambda: (dracs, ddecs, dracs * 0.)))
