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
numShifts = 60

#dealing with positive shifts only 
xStart = center
xStop = center - xWindow
xStep = .0001

#create a 2D-list of difference distribution lists
data = genWGMShiftData(1, numShifts, xStep, 15, xRes, 0.001)
#data = genWGMShiftData(1, 60, .0001, 15, 500, 0.001)
diffDistsArr = data[0]

#error threshold of 5%
threshold = 5
#a list that keeps track of faulty wgm shift distances (error exceeds a certain percent threshold)
errShiftArr = []

sc = StandardScaler()
diffDistsArr = sc.fit_transform(diffDistsArr)

wgmShiftPreds = loaded_model.predict(diffDistsArr)
#print(wgmShiftPreds)
for element in 







