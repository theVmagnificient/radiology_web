"""
RadIO is a framework for batch-processing of computational tomography (CT)-scans
for deep learning experiments.
Documentation - https://analysiscenter.github.io/radio/
"""
import os
import re
import shutil
from setuptools import setup, find_packages

dir_name = os.path.dirname(__file__)
shutil.copyfile(os.path.join(dir_name, 'config.yml'),
                os.path.join(dir_name, 'radio/config/config.yml'))


with open('radio/__init__.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)



with open('./requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name='radio',
    packages=find_packages(exclude=['examples']),
    version=version,
    url='https://github.com/analysiscenter/radio',
    license='Apache License 2.0',
    author='Data Analysis Center team',
    author_email='radio@analysiscenter.ru',
    description='A framework for deep research of CT scans',
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=requirements,
    extras_require={
        'tensorflow': ['tensorflow>=1.4'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.4'],
        'keras': ['keras>=2.0.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)
