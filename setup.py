from setuptools import setup
from setuptools import find_packages

setup(name='skinisic',
      version='0.0.1',
      description='ISIC 2017 Skin Challenge - Part 2 - Detect Dermoscopic Criteria',
      author='Jeremy Kawahara',
      url='https://github.com/jeremykawahara/skinisic',
      install_requires=['numpy',
                        'matplotlib',
                        'keras',
                        'Pillow',
                        'scipy'
                        'json'],
      packages=find_packages())