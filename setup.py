#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages # type: ignore


with open('README.md', 'r') as fp:
    README = fp.read()

INSTALL_REQUIRES = [
    'tqdm',
    'pandas',
    'numpy',
    'scipy',
    'pytest'
]

setup(
    name='caddie',
    version='0.1.5',
    description='Information-Theoretic Causal Inference on Discrete Data',
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRES,
    author='Kailash Budhathoki',
    author_email='kailash.buki@gmail.com',
    url='https://github.com/kailashbuki/caddie',
    license='MIT License',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
