import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iqa-jax",
    version="0.0.1",
    author="dslisleedh",
    author_email="dslisleedh@gmail.com",
    description="IQA library for Jax",
    long_description=long_description,
    url="https://github.com/dslisleedh/IQA-jax",
    projects_urls={
        "Bug Tracker": "https://github.com/dslisleedh/IQA-jax/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    packages=setuptools.find_packages(),
)
