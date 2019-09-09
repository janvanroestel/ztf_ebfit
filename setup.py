from setuptools import setup

setup(name='ztf_ebfit',
      version='0.1',
      description='Fitting tools for ztf eclipsing binaries',
      url='http://github.com/storborg/funniest',
      author='J. van Roestel',
      author_email='jcjvanroestel@gmail.com',
      license='GNU GPLv3',
      packages=['ztf_ebfit'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ]
      zip_safe=False)
