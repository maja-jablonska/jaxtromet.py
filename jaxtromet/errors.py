import os

import jax.numpy as jnp
import numpy as np

# loads data needed to find astrometric error as functon of magnitude
# data digitized from Lindegren+2020 fig A.1
local_dir = os.path.dirname(__file__)
rel_path = '/data/scatteral_edr3.csv'
abs_file_path = local_dir + rel_path  # os.path.join(local_dir, rel_path)
sigma_al_data = np.genfromtxt(abs_file_path, skip_header=1, delimiter=',', unpack=True)
mags = jnp.array(sigma_al_data[0])
sigma_als = jnp.array(sigma_al_data[1])

def sigma_ast(mags_to_evaluate_at: jnp.array) -> jnp.array:
    return jnp.interp(mags_to_evaluate_at, mags, sigma_als)
