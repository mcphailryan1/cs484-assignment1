
"""
Course: CS 484 Intro to Machine Learning
Name: Ryan McPhail
A Number: 20441791
"""

import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
import scipy
from numpy import linalg as LA
from scipy import linalg as LA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors as kNN

# base work for question 3
fraud = pd.read_csv('Fraud.csv', delimiter=',')


def answerPrint():
    print("The data FRAUD.csv contains the results of fraud investigations of 5,960 cases.  The binary variable FRAUD indicates the result of a fraud investigation: 1 = Fraud, 0 = Not Fraud.  The other quantitative variables contain information about the cases. We will train the Nearest Neighbors algorithm to predict the likelihood of fraud. ")

# work for question 3A


def twoA():
    fraud_count = fraud[fraud['FRAUD'] == 1].count()['FRAUD']
    total_count = fraud['FRAUD'].count()
    fraudPercent = (fraud_count / total_count) * 100
    # formatting the fraud percentage
    fraudPercent = float("{0:.4f}".format(fraudPercent))
    print("\nQuestion 3A: What percent of investigations are found to be frauds?  This is the empirical fraud rate.  Please round your answers to the fourth decimal place.\n")
    print("Question 3A Response: \n")
    print("This is the percentage of fraud: ")
    print("\n{0}%\n".format(fraudPercent))


def twoB():
    print("\nPlaceholder")


twoA()
