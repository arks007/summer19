#Author: Sujoy Purkayastha 
#Description: A library that will generate artifical signal dips to train a neural net that will identify the Q-factor of an optical microresonator
import matplotlib
import numpy as np
import math
import random
from matplotlib import pyplot as plt

#Starting and ending wavelength in nm
startWaveLength = 1280
endWaveLength  = 1320

#Maximum and minimum amplitudes 
maxAmp = .98
minAmp = .2

#Lambda resolution in nm
#for consistent window of observation 
lambdaRes = .0000026


#modeled by 1- cauchyDist(x, gamma, center)
def dipDist(x, amp, gamma, center):
    return 1 - amp/(1 + math.pow((x - center)/gamma, 2))

def diffDist(x, amp, gamma, center, xShift):
    return dipDist(x, amp, gamma, center) - dipDist(x, amp, gamma, center - xShift)

    

#simulate a dip over a given range with n samples, noise factor introduces random noise into the simulated dip
#return a list with the simulation results 
def dipSim(amp, gamma, center, numSamples, xStart, xStop, rFactor):
    xStep = (xStop - xStart)/numSamples
    dipSimList = []
    for i in range(0, numSamples):
        noise = random.uniform(-1, 1) * rFactor
        dipSimList.append(noise + dipDist(xStart + i*xStep, amp, gamma, center))
    return dipSimList

#an alternate version of dim sim that records points at every lambdaRes step
def dipSimConst(amp, gamma, center, numSamples, rFactor):
    #create an empty list of size 1000
    dipSimList = [0] * numSamples
    currentWavelength = center - gamma
    index = 0
    while(currentWavelength < center + gamma):
        noise = random.uniform(-1, 1) * rFactor
        dipSimList[index] = (noise + dipDist(currentWavelength, amp, gamma, center))
        currentWavelength = currentWavelength + lambdaRes
        index = index + 1
    return dipSimList

def differenceDistSim(amp, gamma, center, numSamples, xStart, xStop, xShift, rFactor):
    xStep = (xStop - xStart)/numSamples
    diffSimList = []
    for i in range(0, numSamples):
        noise = random.uniform(-1, 1) * rFactor
        diffSimList.append(noise + diffDist(xStart + xStep * i, amp, gamma, center, xShift))
    return diffSimList






#classification library that will generate categories of possible q-factors 
def genClassificationArray(qStep):
    qFactorArray = []
    qFactor = 6000000

    while(qFactor < 31000000):
        qFactorArray.append(qFactor)
        qFactor = qFactor + qStep * 1000000

    return qFactorArray


#This function generates a tuple that contains a 2D numpy array of simulated trials and a corresponding 1D array of respective classifications 
#For the sake of completeness, the center will be modulated, but will not effect the output data values 
def genData(qStep, numCenterSamples, numAmpSamples, numLambdaSamples, xStart, xStop, xWindow, rFactor):
    #create the requested classification array
    qFactorArray = genClassificationArray(qStep)
    
    #initialize the training array
    #dataArray = np.zeros((len(qFactorDict) * numCenterSamples * numAmpSamples, numLambdaSamples)) 
    dataArray = np.zeros((len(qFactorArray) * numCenterSamples * numAmpSamples, numLambdaSamples))

    #initailize the classification list 
    classificationArray = np.zeros((len(qFactorArray) * numCenterSamples * numAmpSamples))
    
    index = 0
    #iterate through all of the Q-factors in qFactorArray
    for element in qFactorArray:
        #iterate through all appropriate center values 
        for i in range(0, numCenterSamples):
            center = startWaveLength + (endWaveLength - startWaveLength)/numCenterSamples * i
            #iterate through amplitudes between minAmp and maxAmp
            for j in range(0, numAmpSamples):
                amp = minAmp + ((maxAmp - minAmp)/numAmpSamples) * j
                gamma = float(center) / float((2 * element))

                #generate a dip simulation and store in the data array 
                dataArray[index] = dipSim(amp, gamma, center, numLambdaSamples, center - xWindow, center + xWindow, rFactor)
                '''
                dataArray[index] = dipSimConst(amp, gamma, center, 0.001)
                '''
                #dataArray[index] = dipSim(amp, gamma, center, numLambdaSamples, center - 0.01, center + 0.01, 0.001)
                #store corresponding classification/desired value in the classification array 
                classificationArray[index] = element 
                index = index + 1

    data = (dataArray, classificationArray)
    return data
            

#This function generates the "difference" signal of dipDist(x - shift) and dipDist(x)
#This function will return a tuple that contains a 2D array that will represent a difference signal and a 1D array of respective wgm shifts in nm
#Center value for the main reference dip will always be centered at 1300
def genWGMShiftData(qStep, numShifts, shiftDist, numAmpSamples, numLambdaSamples, rFactor):
    #create the requested classification array
    qFactorArray = genClassificationArray(qStep)
    
    #create the training array 
    dataArray = np.zeros((len(qFactorArray) * numShifts * numAmpSamples, numLambdaSamples))

    #initailize the classification list 
    classificationArray = np.zeros((len(qFactorArray) * numShifts * numAmpSamples))

    index = 0
    center = 1300
    #iterate through all off the Q-factors in qFactorArray
    for element in qFactorArray:
        for i in range(0, numAmpSamples):
            amp = minAmp + ((maxAmp - minAmp)/numAmpSamples) * i
            gamma = float(center) / float((2 * element))
            for j in range(0, numShifts):
                shiftedCenterDistance = shiftDist * j 
                #center and window values are determined visually by the experimenter 
                #dataArray[index] = differenceDistSim(amp, gamma, 0, numLambdaSamples, -0.12, 0.02, shiftedCenterDistance, rFactor)
                dataArray[index] = differenceDistSim(amp, gamma, center, numLambdaSamples, center-0.008, center+0.008, shiftedCenterDistance, rFactor)
                classificationArray[index] = shiftedCenterDistance
                index = index + 1

    data = (dataArray, classificationArray)
    return data







    



            


        




#1.25microns - 1.35microns Lambda range 
#1250nm - 1350nm
#1.28 - 1.32
#gamma = center / 2Q

#Test Functions#
'''
dataTuple = genData(1, 5, 5, 1000, startWaveLength, endWaveLength, 0.0004, .001)
arr = dataTuple[0]
qFactorArr = dataTuple[1]
print(arr.shape)
print(qFactorArr[0])
for element in arr[0]:
    print(element)


dataTuple = genWGMShiftData(1, 5, .01, 5, 1000, 0)
arr = dataTuple[0]
shiftArr = dataTuple[1]
print(shiftArr[624])
for element in arr[624]:
    print(element)

x = 0
plt.plot(arr[x], color = 'red', label = 'dist0', marker = '.')
plt.plot(arr[1], color = 'orange', label = 'dist1', marker = '.')
#plt.plot(arr[x + 1], color = 'orange', label = 'dist1', marker = '.')
#plt.plot(arr[x + 2], color = 'yellow', label = 'dist2', marker = '.')
#plt.plot(arr[x + 3], color = 'green', label = 'dist3', marker = '.')
#plt.plot(arr[x + 4], color = 'blue', label = 'dist4', marker = '.')
plt.legend()
plt.show()


x = 0
#plt.plot(arr[x], color = 'red', label = 'dist0', marker = '.')
#plt.plot(arr[x + 1], color = 'orange', label = 'dist1', marker = '.')
#plt.plot(arr[x + 2], color = 'yellow', label = 'dist2', marker = '.')
#plt.plot(arr[x + 3], color = 'green', label = 'dist3', marker = '.')
#plt.plot(arr[x + 4], color = 'blue', label = 'dist4', marker = '.')

plt.plot(dipSim(.5, 1300/6000000, 0, 1500, -.12, .02, 0), marker = '.')
plt.plot(dipSim(.5, 1300/6000000, 0.002, 1500, -.06, .02, 0), marker = '.')
plt.plot(dipSim(.5, 1300/6000000, 0.005, 1500, -.06, .02, 0), marker = '.')
plt.title('Distribution Test')
plt.legend()
plt.show()


testList = differenceDistSim(.5, 1300/6000000, 1300, 1000, -0.12, 0.02, 0, 0.001)
for element in (testList):
    print(str(element))
'''