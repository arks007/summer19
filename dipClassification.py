import numpy as np 
from keras.layers import Dense, Activation 
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
from dipGen import genData

import time 

#Generate training and testing data
data = genData(1, 50, 50, 1200, 1280, 1320, 0.004, .005)

dipData = data[0]
qFactorClassification = data[1]


x_train, x_test, y_train, y_test = train_test_split(dipData, qFactorClassification, test_size = 0.01, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Defining a model that takes in a 1D numpy array of 1000 inputs
model = Sequential()
model.add(Dense(units=600, input_dim=1200, activation = 'relu'))
model.add(Dense(units=600, activation = 'relu'))
model.add(Dense(units=600, activation = 'relu'))
model.add(Dense(units=1))

#Compile
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(x_train, y_train, epochs = 5000)

start = time.time()
qFactorPrediction = model.predict(x_test)
end = time.time()

print(end - start)

plt.plot(y_test, color = 'red', label = 'Theorectical Q-Factor')
plt.plot(qFactorPrediction, color = 'blue', label = 'Predicted Q-Factor')
plt.title('Model Prediction')
plt.legend()
plt.show()


#save model for later use 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")