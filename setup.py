from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ODEyssey',
    version='0.0',
    author="Jorrit Stada",
    author_email="jorrit.stada@scilifelab.se",
    description="A Python library for solving and visualizing ordinary differential equations (ODEs), with examples and tools inspired by biological systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JorritJS/ODEyssey",
    project_urls={
        "Bug Tracker": "https://github.com/JorritJS/ODEyssey/issues",
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research/Student",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12.9+",
        "Operating System :: OS Independent",
    ],
    packages=['ODEyssey'],
    include_package_data=True,
    install_requires=[
            'pandas',
            'numpy',
            'typing',
            'scipy',
            'matplotlib',
        ],
)
