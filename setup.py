# setup.py

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RackioAI",
    version="0.0.1",
    author="Carlos Rivero",
    author_email="cdrr.rivero@gmail.com",
    description="A modern Python Framework for AI models development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/crivero7/RackioAI",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'Rackio==0.9.4',
        'pandas==1.0.1',
        'numpy==1.16.4',
        'tqdm==4.43.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
