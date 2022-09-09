# to run: python setup.py install
try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup
#from distutils.extension import Extension

setup(name="jaxtromet",
      version='0.8',
      description='One and two body astrometry. Created by Zephyr Penoyre and his team, rewritten to JAX by Maja Jabłońska',
      author='Zephyr Penoyre/rewritten by Maja Jabłońska',
      author_email='majajjablonska@gmail.com',
      url='https://github.com/maja-jablonska/jaxtromet.py',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=['numpy','astropy','scipy','astromet','jax','jaxlib'],
      include_package_data=True,
      package_data={'': ['data/*.csv']},
      )
