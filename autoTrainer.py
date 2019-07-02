import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from matplotlib.widgets import Slider, Button, RadioButtons
from dipGen import dipSim
from dipGen import differenceDistSim
from numpy import array
from numpy import *
from dipGen import genWGMShiftData


# load json and create model
json_file = open('wgmShiftModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("wgmShiftModel.h5")
print("Loaded model from disk")


xWindow = .008
center = 1300
xRes = 500
numShifts = 120

#dealing with positive shifts only 
xStart = center
xStop = center - xWindow
xStep = .00005

#create a 2D-list of difference distribution lists
data = genWGMShiftData(1, numShifts, xStep, 15, xRes, 0.001)
#data = genWGMShiftData(1, 60, .0001, 15, 500, 0.001)
diffDistsArr = data[0]
shiftArr = data[1]

#error threshold (in %)
threshold = 10
#a list that keeps track of faulty wgm shift distances (error exceeds a certain percent threshold)
errShiftArr = []

sc = StandardScaler()
diffDistsArr = sc.fit_transform(diffDistsArr)


wgmShiftPreds = loaded_model.predict(diffDistsArr)
#print(wgmShiftPreds)
for index, element in enumerate(wgmShiftPreds, 0):
    #prevent divide by 0 errors 
    if(shiftArr[index] == 0):
        #shift value should be as close to 0 as possible 
        if(element[0] > 0.000001):
            if shiftArr[index] not in errShiftArr:
                errShiftArr.append(shiftArr[index])

    elif(abs(100*(element[0] - shiftArr[index])/shiftArr[index]) > threshold):
        if shiftArr[index] not in errShiftArr:
            errShiftArr.append(shiftArr[index])

#errShiftArr now contains wgm shift values that the model predicts an unacceptable value for 
#further train with these values at varying q factor and amp values 



print(len(shiftArr))
print(errShiftArr)








