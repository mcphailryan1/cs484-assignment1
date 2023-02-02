"""
Course: CS 484 Intro to Machine Learning
Name: Ryan McPhail
A Number: 20441791
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

Fraud = pd.read_csv('Fraud.csv')
Fraud.describe()

#question3A
def threeA():
    isFraud = Fraud.where(Fraud['FRAUD'] == 1)
    isFraudCount = isFraud.count()
    FraudPercentage = isFraud['CASE_ID'].count() / Fraud['CASE_ID'].count()
    print(f'Empirical Fraud Percentage: {FraudPercentage}')
    
def threeC():
    ## KNeighborClassifier
    
    for i in range(2,8):
        Fraud_wIndex = Fraud.set_index('CASE_ID')
        trainData = Fraud_wIndex[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'MEMBER_DURATION']].dropna().reset_index()

        X = trainData[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'MEMBER_DURATION']]
        y = trainData['CASE_ID']

        kNC = KNeighborsClassifier(n_neighbors = i, metric = 'euclidean')
        nbrs = kNC.fit(X, y)

        class_proba = nbrs.predict_proba(X)
        print('Score with', i ,'neighbors: ', kNC.score(X, y))
        
threeA()
threeC()