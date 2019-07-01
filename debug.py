import numpy as np 
from keras.layers import Dense, Activation 
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
from dipGen import genWGMShiftData

import time 


# load json and create model
json_file = open('wgmShiftModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("wgmShiftModel.h5")
print("Loaded model from disk")

data = genWGMShiftData(1, 60, .0001, 15, 500, 0.001)

differenceDistData = data[0]
wgmShiftClassification = data[1]

x_train, x_test, y_train, y_test = train_test_split(differenceDistData, wgmShiftClassification, test_size = 0.02, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

qFactorPrediction = loaded_model.predict(x_test)

plt.plot(y_test, color = 'red', label = 'Theorectical WGM Shift (in nm)', marker = 'o')
plt.plot(qFactorPrediction, color = 'blue', label = 'Predicted WGM Shift (in nm)', marker = 'o')
plt.title('Model Prediction')
plt.legend()
plt.show()