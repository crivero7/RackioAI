# setup.py
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RackioAI",
    version="0.2.6",
    author="Carlos Rivero",
    author_email="cdrr.rivero@gmail.com",
    description="A Rackio extension for AI models development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/crivero7/RackioAI",
    package_data={'': ['Leak/*.tpl', 
    'pkl_files/*.pkl', 
    'csv/Hysys/*.csv', 
    'csv/standard/*.csv',
    'csv/VMGSim/*.csv',
    'excel/*.xlsx',
    'excel/*.xls',
    'json/*.json']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
        'Pillow==8.0.0',
        'Rackio==1.0.2',
        'easy-deco==0.1.1',
        'seaborn',
        'xlrd==1.2.0',
        'openpyxl==3.0.6',
        'odfpy==1.4.1',
        'pyxlsb==1.0.8',
        'statsmodels==0.12.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
