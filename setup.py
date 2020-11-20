#!/usr/bin/env python

import os
from setuptools import setup

setup(
    setup_requires=['pbr'],
    install_requires=[
        'psychrolib',
        'pint',
        'scipy',
        'matplotlib',
        'seaborn'
    ],
    pbr=True
)