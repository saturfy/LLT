![](https://raw.githubusercontent.com/saturfy/LLT/main/LLT.png)

![](https://img.shields.io/badge/LLT-v1.0.0-brightgreen)
![](https://img.shields.io/badge/python->=3.9-brightgreen)
![](https://img.shields.io/badge/scipy-1.10.1-orange)
![](https://img.shields.io/badge/numpy-1.24.2-orange)

# Description
## The LLT method
Thit python package implements the Linear Law Transformation algorithm (LLT) for time series type data. This new technique automatically generates features from time series-type data samples which can be used for machine learning tasks for example classification. 

The mathematical description and the explanation of this method can be found in the publications below:

- [Solving mechanical differential equations using LLT and Extreme Learning Machine](https://iopscience.iop.org/article/10.1088/1367-2630/ac7c2d)
- [Categorizing human movement type using LLT transformed multi dimensional sensor data](https://www.nature.com/articles/s41598-022-22829-2)
- [Detecting anomalies in bitcoin prices using LLT](https://arxiv.org/abs/2201.09790)
- [Detailed mathematical background](https://arxiv.org/abs/2104.10970)
- [Slides which explains how the method works and present the ECG signal classification task as a case study](https://github.com/saturfy/LLT/blob/main/docs/Peter_Posfay_ECG_linear_law_1_0.pdf)

## The LLT package
The package contains two subpackages
- __preprocessing__: conatins commonly used function duging evaluation and data formatting for the transformation.
- __linear_law__ : the implementation of the LLT technique and can generate linear law features from the dataset.

----

# Installation
It is recommended to use a virtual environment for ensuring maximal compatibility.

## Requirements
- pyhton  >=v3.9
- scipy v1.10.1
- numpy v1.24.2

note: the correct versions of these packages are automatically installed using the requirements.txt files. If you have newer versions they won't be downgraded but compatibility is not guaranteed. 

## How to install:

1. ___OFFLINE___: Download the dist/LLT-1.0.0-py3-none-any.whl file from the release folder and install it using pip:

        python -m pip install LLT-1.0.0-py3-none-any.whl
        
2. ___ONLINE___
        
        python -m pip install git+https://github.com/saturfy/LLT
        
# USAGE
After installation it can be used as normal pyhton package. The usual way to import it is:
    
    from LLT import preprocessing as pp
    from LLT import linear_law as ll
    
## Demo
There is an example ipython notebook in the archive under demo\LLT_QRS_peak_classifier.ipynb which demonstrates how to use the package. The notebook shows an example of ECG signal classification using the LLT technique. To run this demo you need to install additional packages which are listed in the demo\demo_requirements.txt file. Use the following command to install the exact versions of these packages with the help of the file: 

    python pip install -r demo_requirements.txt
    
You also have to download the sample ECG data (ecg_test.mat and ecg_train.mat) from [here](https://git.silicon-austria.com/pub/sparseestimation/vpnet/-/tree/master/data). Put these files in the same directory as the ipyython notebook. 

Note: you have to make sure that your virtual environment is avalibale for jupyter and you can select is as kernel for the notbeook. The simplest way to do this if you are using venv
    
    python -m pip install ipykernel
    python -m ipykernel install --name=<name of your virtual environment>
    
# Documentation
The package contains an API like html documentation in [\docs\LLT\index.html](https://rawcdn.githack.com/saturfy/LLT/3900885c310b19bc94333fd5079a3587e84cad8f/docs/LLT/index.html)
