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


def answerPrint2():
    print("\nQuestion 3: The data FRAUD.csv contains the results of fraud investigations of 5,960 cases.  The binary variable FRAUD indicates the result of a fraud investigation: 1 = Fraud, 0 = Not Fraud.  The other quantitative variables contain information about the cases.")


def threeA():
    print("\nQuestion 3A: What percent of investigations are found to be frauds?  This is the empirical fraud rate.  Please round your answers to the fourth decimal place.\n")
    isFraud = Fraud.where(Fraud['FRAUD'] == 1)
    isFraudCount = isFraud.count()
    FraudPercentage = isFraud['CASE_ID'].count() / Fraud['CASE_ID'].count()
    print('Question 3A Answer: Empirical Fraud Percentage: ', FraudPercentage)


def threeC():
    # KNeighborClassifier
    print("\nQuestion 3C Answer: ")

    for i in range(2, 8):
        Fraud_wIndex = Fraud.set_index('CASE_ID')
        trainData = Fraud_wIndex[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS',
                                  'NUM_MEMBERS', 'OPTOM_PRESC', 'MEMBER_DURATION']].dropna().reset_index()

        X = trainData[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS',
                       'NUM_MEMBERS', 'OPTOM_PRESC', 'MEMBER_DURATION']]
        y = trainData['CASE_ID']

        kNC = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
        nbrs = kNC.fit(X, y)

        class_proba = nbrs.predict_proba(X)
        print('Score with', i, 'neighbors: ', kNC.score(X, y))


answerPrint2()
threeA()
threeC()
