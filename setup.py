'''
# PyPi
python setup.py sdist bdist_wheel
python -m twine upload dist/*
    (To use this API token:
    Set your username to __token__
    Set your password to the token value, including the pypi- prefix
    pypi-AgEIcHlwaS5vcmcCJGVlN2Q1NDIwLTNmODQtNGEzOC1hNTIwLWU4ZTRhMGE2OGQzMgACEVsxLFsicHltaW5lcnZhIl1dAAIsWzIsWyI4MWRjZDdlNC00NWEzLTQ1NGItOTU2ZC01YmNiN2UxZGNlOTciXV0AAAYgAIf29-tmuIBNgj52KFSZA33c8vGz7xCtelz5WwZV5IE
    )
python -m pip install pyminerva
# https://pypi.org/project/pyminerva/0.0.X/
pip install pyminerva --upgrade

# test PyPi
python setup.py sdist bdist_wheel
python -m twine upload --repository testpypi dist/*
    (To use this API token:
    Set your username to __token__
    Set your password to the token value, including the pypi- prefix)
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps minerv
https://test.pypi.org/project/minerv/

'''

from setuptools import setup, find_packages
from os.path import abspath, dirname, join

# Fetches the content from README.md
# This will be used for the "long_description" field.
README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pyminerva',
    version='0.0.146',    # version.directory.file
    description='To get an insight from Financial Data Anlaysis',
    url='',
    author='Jeongmin Kang',
    author_email='jarvisNim@gmail.com',
    license='MIT',
    # packages=['minerv'],
    # install_requires=required(filename='requirements.txt'),
    include_package_data=True,
    # url="https://github.com/driscollis/arithmetic",
    packages=find_packages(exclude="tests"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=', '.join([
        'minerva', 'minerva-api', 'historical-data',
        'financial-data', 'stocks', 'funds', 'etfs',
        'indices', 'currency crosses', 'bonds', 'commodities',
        'crypto currencies'
    ]),
    project_urls={
        'Bug Reports': 'https://github.com/jarvisNim/minerva/issues',
        'Source': 'https://github.com/jarvisNim/minerva',
        'Documentation': 'https://miraelabs.com/'
    },
    python_requires='>=3.6',
)