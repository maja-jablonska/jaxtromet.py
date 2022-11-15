# jaxtromet

astromet.py (https://github.com/zpenoyre/astromet.py) rewritten to JAX.

## Install

To install requirements.txt with conda:

```
conda install -c conda-forge -y --file requirements.txt
```

In case of problems with astromet and/or jax installation, refer to https://github.com/google/jax and install with pip. Beware of the caveats of using conda and pip install simultaneously: https://www.anaconda.com/blog/using-pip-in-a-conda-environment.

If you wish to use the conda enivronment's pip, use:

```
<path/to/conda>/envs/<env-name>/bin/pip install ...
```

## Main changes from astromet

I tried to make the API as  consistent as possible, and the only changes I have made were due to differences coming from JAX and some computational overhead or complexity they would cause.

### fit

Barycentric positions (```bs```) are now calculated for the passed times outside the loop. This can be done using the ```barycentricPosition``` function in jaxtromet. It uses external libraries and therefore cannot be jitted (easily).

```def fit(ts, bs, xs, phis, xerr, ra, dec, G=12, epoch=2016.0)```

### track

So far, return all components isn't yet supported -- sorry! I will add that soon.

## astromet.py

Please take a look at the wonderful job done by Zephyr Penoyre and their team.

A simple python package for generating astrometric tracks of single stars and the center of light of unresolved binary, blended and lensed systems. Includes a close emulation of Gaia's astrometric fitting pipeline.

https://astrometpy.readthedocs.io/en/latest/

**pip install astromet**

Still in development, functional but may occasional bugs or future changes. Get in touch with issues/suggestions.

Requires
- numpy
- astropy
- scipy
- matplotlib (for notebooks)
