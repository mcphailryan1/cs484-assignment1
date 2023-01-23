
"""
Course: CS 484 Intro to Machine Learning
Name: Ryan McPhail
A Number: 20441791
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# THIS IS FOR ALL OF QUESTION 1

# read in gamma file for use
gammaData = pd.read_csv('Gamma4804.csv', delimiter=',', usecols=['x'])
gammaDataX = gammaData['x']


def calculate(Y, delta):
    # this function makes the calculations for oneB()
    # finding inital values using numpy methods
    maxY = np.max(Y)
    minY = np.min(Y)
    meanY = np.mean(Y)
    middleY = delta * np.round(meanY / delta)
    # find bins on sides of new mean
    numberRight = np.ceil((maxY - middleY) / delta)
    numberLeft = np.ceil((middleY - minY) / delta)
    lowY = middleY - numberLeft * delta
    # observations -> bins starting at 0
    num = numberRight + numberLeft
    bin = 0
    boundryY = lowY
    for i in np.arange(num):
        boundryY = boundryY + delta
        bin = np.where(Y > boundryY, i+1, bin)
    # using np.unique to count oberservations / bin
    binObservations, frequency = np.unique(bin, return_counts=True)
    meanFrequency = np.sum(frequency) / num
    stdFrequency = np.sum((Y - meanFrequency)**2) / num
    CDelta = ((2.0 * meanFrequency) - stdFrequency) / (delta * delta)
    return (num, middleY, lowY, CDelta, binObservations, frequency)


def answerPrint():
    print("Queston 1: Please use the field x in the Gamma4804.csv file. ")


def oneA():
    # describe() is built in method in pandas
    gammaDataDescribe = gammaData.describe()
    # printing out answer
    print("\nQuestion 1A: (What are the count, the mean, the standard deviation, the minimum, the 25th percentile, the median, the 75th percentile, and the maximum of the feature x? Please round your answers to the seventh decimal place. \n")
    print("Question 1A Answer: \n")
    print(gammaDataDescribe)


def oneB():
    result = pd.DataFrame()
    dList = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0,
             2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]
    # start the print out of the results
    print("\nQuestion 1B: Use the Shimazaki and Shinomoto (2007) method to recommend a bin width.  We will try d = 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, and 100.  What bin width would you recommend if we want the number of bins to be between 10 and 100 inclusively? You need to show your calculations to receive full credit.\n")
    print("Question 1B Answer: \n")
    # looping through d list given from the assignment
    for x in dList:
        newBin, middleY, lowY, CDelta, binObservations, frequency = calculate(
            gammaDataX, x)
        highY = lowY + newBin * x
        # appending result
        result = result.append(
            [[x, CDelta, lowY, middleY, highY, newBin, binObservations, frequency]], ignore_index=True)
        middleBin = lowY + 0.5 * x + np.arange(newBin) * x
        # plotting graphs
        plt.hist(gammaDataX, bins=middleBin, align='mid')
        plt.title('Delta = ' + str(x))
        plt.ylabel("Number of Observations")
        plt.grid(axis='y')
        #plt.show()

    result = result.rename(columns={0: 'Delta', 1: 'C(Delta)', 2: 'Low Y', 3: 'Middle Y',
                           4: 'High Y', 5: 'New Bin', 6: 'Bin Observations', 7: 'Frequency'})
    # print(result)
    sortedResult = result.sort_values(by=['C(Delta)']).reset_index(drop=True)
    print(sortedResult)
    widthReccomendation = sortedResult['Delta'][0]
    print("\nRecomended bin-width is, d = {0}\n".format(widthReccomendation))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Box Plot')
    ax1.boxplot(gammaDataX, labels=['x'])
    ax1.grid(linestyle='--', linewidth=1)
    plt.show()


answerPrint()
oneA()
oneB()
