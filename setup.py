from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='logistic',
    version='0.1.0',
    description='A dead-simple logistic regression library.',
    long_description=long_description,
    url='https://github.com/useanalias/logistic',
    author='Chandler',
    author_email='use.an.alias@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='logistic regression machine learning',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['peppercorn'],
)
