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

#Load a generated neural network

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

data = genData(1, 50, 50, 1200, 1280, 1320, 0.004, .001)

dipData = data[0]
qFactorClassification = data[1]

x_train, x_test, y_train, y_test = train_test_split(dipData, qFactorClassification, test_size = 0.01, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

start = time.time()
qFactorPrediction = loaded_model.predict(x_test)
end = time.time()

print(str((end - start)))

plt.plot(y_test, color = 'red', label = 'Theorectical Q-Factor', marker = '.')
plt.plot(qFactorPrediction, color = 'blue', label = 'Predicted Q-Factor', marker = '.')
plt.title('Model Prediction')
plt.legend()
plt.show()