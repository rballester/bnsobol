#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

setup(
    name='bnsobol',
    version='0.1.0',
    description="Variance-based Sensitivity Analysis of Bayesian Networks",
    long_description="bnsobol computes the main Sobol indices of a function defined by a Bayesian network.",
    url='https://github.com/rballester/bnsobol',
    author="Rafael Ballester-Ripoll",
    author_email='rafael.ballester@ie.edu',
    packages=[
        'bnsobol',
    ],
    include_package_data=True,
    install_requires=[
        'pgmpy',
        'numpy',
    ],
    license="BSD",
    zip_safe=False,
    keywords='bnsobol',
    classifiers=[
        'License :: OSI Approved',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require='pytest'
)
