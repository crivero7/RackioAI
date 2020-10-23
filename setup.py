# setup.py

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RackioAI",
    version="0.0.4",
    author="Carlos Rivero",
    author_email="cdrr.rivero@gmail.com",
    description="A Rackio extension for AI models development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/crivero7/RackioAI",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.18.5',
        'scipy==1.4.1',
        'scikit-learn==0.23.2',
        'tensorflow==2.3.0',
        'matplotlib==3.3.2',
        'pandas==1.1.3',
        'tqdm==4.50.2',
        'Pillow==8.0.0',
        'Rackio==0.9.7',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
