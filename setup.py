import setuptools
from requirements import REQUIRED_PACKAGES

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="few-shot-learning-nlp",
    version="0.0.19",
    author="Pedro Silva",
    author_email="pedrolmssilva@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peulsilva/few-shot-learning-nlp",
    packages=setuptools.find_packages(),
    install_requires = REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)