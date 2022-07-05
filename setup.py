import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='rep_eval',
    version='1.0.0',
    packages=find_packages(),
    description='A benchmark suite for evaluating representations on control tasks',
    long_description=read('README.md'),
    url='https://github.com/aravindr93/rep_eval.git',
    author='Robot Learning Lab, Berkeley'
)