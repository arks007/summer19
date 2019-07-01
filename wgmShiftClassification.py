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

'''

data = genWGMShiftData(1, 25, .001, 15, 500, 0.02)

differenceDistData = data[0]
wgmShiftClassification = data[1]

x_train, x_test, y_train, y_test = train_test_split(differenceDistData, wgmShiftClassification, test_size = 0.01, random_state = 0)
'''
#Generate data for positive wgm shifts
data = genWGMShiftData(1, 60, .0001, 15, 500, 0.001)
print(data[1])
'''
#Generate data for negative wgm shifts
dataNeg = genWGMShiftData(1, 60, -.0001, 15, 500, 0.001)
#concatenate
differenceDistData = np.concatenate((data[0], dataNeg[0]), axis = 0)
wgmShiftClassification = np.concatenate((data[1], dataNeg[1]), axis = 0)
'''
differenceDistData = data[0]
wgmShiftClassification = data[1]

x_train, x_test, y_train, y_test = train_test_split(differenceDistData, wgmShiftClassification, test_size = 0.002, random_state = 0)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Defining a model that takes in a 1D numpy array of 500 inputs
model = Sequential()
model.add(Dense(units=125, input_dim=500, activation = 'relu'))
model.add(Dense(units=125, activation = 'relu'))
model.add(Dense(units=125, activation = 'relu'))
model.add(Dense(units=1))

#Compile
model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

#Fitting the NN to training data
model.fit(x_train, y_train, epochs = 500)


qFactorPrediction = model.predict(x_test)


plt.plot(y_test, color = 'red', label = 'Theorectical WGM Shift (in nm)', marker = 'o')
plt.plot(qFactorPrediction, color = 'blue', label = 'Predicted WGM Shift (in nm)', marker = 'o')
plt.title('Model Prediction')
plt.legend()
plt.show()


#save model for later use 
# serialize model to JSON
model_json = model.to_json()
with open("wgmShiftModel.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("wgmShiftModel.h5")
print("Saved model to disk")