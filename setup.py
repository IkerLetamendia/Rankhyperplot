#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: GNU General Public License
import os
from setuptools import setup, find_packages

__version__ = "0.0.0"
NAME = 'hyperplot'
AUTHOR = "Renzo Comolatti & Daniele Marinazzo"
MAINTAINER = "Renzo Comolatti"
EMAIL = 'renzo.com@gmail.com'
KEYWORDS = "hypergraph plot highorder network"
DESCRIPTION = "High-order interaction visualizations"
URL = 'https://github.com/renzocom/hyperplot'
DOWNLOAD_URL = ("https://github.com/renzocom/hyperplot/archive/v" +
                __version__ + ".tar.gz")
# Data path :
PACKAGE_DATA = {}

def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    package_dir={'hyperplot': 'hyperplot'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    description=DESCRIPTION,
    long_description=read('README.md'),
    platforms='any',
    setup_requires=['numpy'],
    install_requires=requirements,
    dependency_links=[],
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license="GNU General Public License",
    keywords=KEYWORDS,
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 ])