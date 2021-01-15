# setup.py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RackioAI",
    version="0.1.8",
    author="Carlos Rivero",
    author_email="cdrr.rivero@gmail.com",
    description="A Rackio extension for AI models development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/crivero7/RackioAI",
    package_data={'': ['Leak/*.tpl', 'pkl_files/*.pkl']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'seaborn==0.11.0',
        'scikit-learn',
        'tensorflow',
        'Pillow==8.0.0',
        'Rackio==1.0.2',
        'easy-deco==0.0.7'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
