from setuptools import setup

setup(
   name='thresholder',
   version='1.0',
   description='A module to give optimal probability cutoffs using ROC curve',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['thresholder'],  #same as name
   install_requires=[
       "numpy==1.17.2",
       "scikit-learn==0.20.3",
       "pandas==0.25.1"
   ], #external packages as dependencies
)