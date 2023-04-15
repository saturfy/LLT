from setuptools import setup
  
with open('requirements.txt') as f:
    requirements = f.readlines()
  
long_description = 'Python package which implements the LLT method \
      (Linear Law Transformation) for time series classification. \
       The package contains sample preprocessing routines and feature \
       generation tools for the linear law transformation.'
  
setup(
        name ='LLT',
        version ='1.0.0',
        author ='Peter Posfay',
        author_email ='posfay.peter@wigner.hu',
        url ='https://github.com/saturfy/LLT',
        description ='python package which implements the LLT method',
        long_description = long_description,
        long_description_content_type ="text/markdown",
        license ='MIT',
        packages = ['LLT'],
        classifiers =(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ),
        keywords ='LLT, classification, feature_generation, linear_law',
        install_requires = requirements,
        zip_safe = False
)