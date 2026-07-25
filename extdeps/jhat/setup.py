"""Setup for the st123 custom JHAT build (extdeps/jhat)."""

from setuptools import find_packages, setup

# Distinct from PyPI jhat so installs are clearly the repo-local build.
VERSION = '0.3.7+st123'

setup(
    name='jhat',
    version=VERSION,
    description=(
        'Custom JHAT (JWST/HST Alignment Tool) build vendored for st123. '
        'Not the unmodified PyPI package.'
    ),
    long_description=(
        'Repository-local JHAT for st123 relative alignment / MIRI pipelines. '
        'Upstream: https://github.com/arminrest/jhat'
    ),
    author='Armin Rest & Justin Pierel (upstream); st123 maintainers (custom build)',
    author_email='arest@stsci.edu',
    url='https://github.com/arminrest/jhat',
    packages=find_packages(include=['jhat', 'jhat.*']),
    scripts=[
        'bin/run_st_wcs_align.py',
        'bin/run_st_wcs_align_batch.py',
    ],
    # Runtime libs are pinned by the parent st123 requirements.txt.
    # Keep this list minimal so `pip install -e ./extdeps/jhat` does not
    # fight the parent environment.
    install_requires=[],
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
