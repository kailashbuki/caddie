#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages # type: ignore


with open('README.rst', 'r') as fp:
    README = fp.read()

with open('LICENSE', 'r') as fp:
    LICENSE = fp.read()

INSTALL_REQUIRES = [
    'tqdm',
    'pandas',
    'numpy',
    'scipy',
    'pytest'
]

setup(
    name='caddie',
    version='0.1.0',
    description='Information-Theoretic Causal Inference on Discrete Data',
    long_description=README,
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES,
    author='Kailash Budhathoki',
    author_email='kailash.buki@gmail.com',
    url='https://github.com/kailashbuki/caddie',
    license=LICENSE,
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
