 # Code Location
 github: https://github.com/Inquisitive-ME/Random-Optimization-Project-Code
 Backup Google Drive: https://drive.google.com/file/d/1fa9y3VQv30ECV1RvW4mFbRv3JLonzDs6/view?usp=sharing

 # Part 1: 4 Random Search Algorithms
 The Code is broken into 3 jupyter notebooks for each problem. The jupyter notebooks use common.py
 for common functions shared between the problems

 1. Continous Peaks Problem
    ContinousPeaks.ipynb - Contains code to create all graphs for the Continous Peaks Problem Section of the Report

2. Knapsack Problem
   Knapsack.ipynb - Contains code to create all graphs for the Knapsack Problem section of the report

3. Max K Colors Problem
   MaxKColor.ipynb - contains code to create all graphs for the Max K Colors Problem in the report

For this section all graphs were saved from the Jupyter Notebooks

# Part 2: Finiding Weights for Neural Network

## Data
The data from Assaignment 1 is in the data folder

The Noisy Non-Linear data set is generated using the functions in data/generated/generated_data.py
Analysis of the Noisy Non-Linear Data Set can be found in data/generated/Generated_Noisy_Nonlinear_Data_Analysis.ipynb

All the code for This assaignment is contained in NeuralNetworks/mlrose_train_test.py
This generates all the plots and saves them as .png files, but takes a very long time to run so it also generates and saves
temporary pickled files of the results.

In order for this file to run properly the mlrose directory must be in the system path. This should be taken care of by
the file but could cause problems. In Pycharm this can also be done by adding the mlrose directory as a sources folder
in the project structure

I also forked the mlrose library to implement timing and counting function calls. This code is in
the mlrose folder and is a submodule to the forked repo

## Python packages
I think the python packages are pretty standard but a requirements.txt file is provided in the repo listing all packages
that were installed when running the code

# References
All specific code references are in the code directly

## mlrose
https://mlrose.readthedocs.io/en/stable/

## mlrose-hiive
https://github.com/hiive/mlrose

## Scikit-Learn
https://scikit-learn.org/stable/
https://scikit-learn.org/stable/auto_examples/index.html

## Class Lectures:
https://classroom.udacity.com/courses/ud262


