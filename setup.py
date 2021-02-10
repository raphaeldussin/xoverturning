""" setup for xoverturning """
from setuptools import setup, find_packages
import os


is_travis = "TRAVIS" in os.environ

setup(
    name="xoverturning",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("compute overturning in xarray"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/xoverturning",
    packages=find_packages(),
)
