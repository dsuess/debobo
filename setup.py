import sys
import setuptools
from pathlib import Path


with open('README.md', 'r') as fh:
    long_description = fh.read()


try:
    sys.path.append(Path(__name__).parent)
    from debobo import __version__ as version
except ImportError:
    version = '__unknown__'


setuptools.setup(
    name='debobo',
    version=version,
    author='Daniel Suess',
    author_email='daniel@dsuess.me',
    description='Package for evaluating object detection models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dseuss/debobo.git',
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
    ],
    install_requires=['numpy>=1.12'],
    tests_require=['pytest', 'pycocotools', 'ignite'])
