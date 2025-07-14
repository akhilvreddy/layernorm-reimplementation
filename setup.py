from setuptools import setup, find_packages

setup(
    name="layernorm_reimplementation",
    version="0.1.0",
    author="Akhil Reddy",
    description="A from-scratch reimplementation of LayerNorm (Ba et al. 2016) in PyTorch.",
    packages=find_packages(),  # this will find `layernorm/` automatically
    install_requires=[
        "torch>=1.10",
        "numpy>=1.21"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


# to install:
# pip install -e .