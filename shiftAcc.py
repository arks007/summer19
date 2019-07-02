# import libraries 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
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

# create the shifting signal
# data is an arr that encapsulates the entire time series 
data = []
theoreticalShifts = []


numShifts = 12
shiftDist = .0035 #nm

# parameters for the first reference dip 
amp = 0.5
gamma = 1300/6000000
center = 1300
xRes = 500
xWindow = 0.008
rFactor = 0.001

# linear shift w.r.t 
for i in range (0, numShifts):
    refDip = np.array(dipSim(amp, gamma, center, xRes, center - xWindow, center + xWindow, rFactor))
    shiftedDip = np.array(dipSim(amp, gamma, center - shiftDist, xRes, center - xWindow, center + xWindow, rFactor))
    data.append(refDip - shiftedDip)
    theoreticalShifts.append(i * shiftDist)
    center = center + shiftDist


sc = StandardScaler()
scaledData = sc.fit_transform(data)

acc = 0
accShift = []
for i in range (0, numShifts):
    pred = loaded_model.predict(scaledData)
    acc = acc + pred[0][0]
    accShift.append(acc)
    





plt.plot(theoreticalShifts, color = 'red', label = 'time vs. theoretical WGM shift', marker = 'o')
plt.plot(accShift, color = 'blue', label = 'time vs. predicted WGM shift', marker = 'o')
plt.title('test')
plt.legend()
plt.show()




