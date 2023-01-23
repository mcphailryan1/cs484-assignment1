
"""
Course: CS 484 Intro to Machine Learning
Name: Ryan McPhail
A Number: 20441791
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#THIS IS FOR ALL OF QUESTION 1

#read in gamma file for use
gammaData = pd.read_csv('Gamma4804.csv', delimiter=',', usecols=['x'])

def calculate(Y, delta) :
    #this function makes the calculations for oneB()
    #finding inital values using numpy methods
    maxY = np.max(Y)
    minY = np.min(Y)
    meanY = np.mean(Y)
    
    middleY = delta * np.round(meanY / delta)
    #find bins on sides of new mean
    numberRight = np.ceil((maxY - middleY) / delta)
    numberLeft = np.ceil((middleY - minY) / delta)
    lowY = middleY - numberLeft * delta
    
    #NEED TO FINSIH THIS FUNCTION BEFORE FINISHING 1B
    

def answerPrint():
    print("Queston 1: Please use the field x in the Gamma4804.csv file. ")

def oneA ():
    gammaDataX = gammaData['x']
    # describe() is built in method in pandas
    gammaDataDescribe = gammaData.describe()
    #printing out answer
    print("Question 1A: (What are the count, the mean, the standard deviation, the minimum, the 25th percentile, the median, the 75th percentile, and the maximum of the feature x? Please round your answers to the seventh decimal place.")
    print("Question 1A Answer: ")
    print(gammaDataDescribe)
    
def oneB() :
    result = pd.DataFrame()
    dList = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]
    
    #looping through d list
    for x in dList:
        newBin, middleY, lowY, CDelta, uBin, binFrequency = 'yes'


answerPrint()
oneA()