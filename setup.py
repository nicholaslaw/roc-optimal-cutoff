from setuptools import setup

setup(
   name='thresholder',
   version='1.0.2',
   description='A module to give optimal probability cutoffs using ROC or PR curve',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['thresholder'],  #same as name
   install_requires=[
       "numpy==1.18.1",
       "scikit-learn==0.22.2.post1",
       "pandas==1.0.1",
       "matplotlib==3.3.0"
   ], #external packages as dependencies
)