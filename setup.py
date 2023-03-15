import setuptools


install_requires = [
    "numpy>=1.12",
    "jax>=0.4.2",
    "tensorflow>=2.0.0",
    "flax>=0.6.0"
]
testing_requires = [
    "basicsr>=1.4.2"
]

setuptools.setup(
    name="iqa-jax",
    version="0.0.1",
    author="dslisleedh",
    author_email="dslisleedh@gmail.com",
    description="IQA library for Jax",
    long_description=open('README.md', 'rt').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dslisleedh/IQA-jax",
    project_urls={
        "Bug Tracker": "https://github.com/dslisleedh/IQA-jax/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "testing": testing_requires
    },
    python_requires=">=3.6",
)
