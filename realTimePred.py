# import libraries 
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

data = genWGMShiftData(1, 60, .0001, 15, 500, 0.001)

differenceDistData = data[0]
wgmShiftClassification = data[1]

x_train, x_test, y_train, y_test = train_test_split(differenceDistData, wgmShiftClassification, test_size = 0.02, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

#Make a new fig object
fig = plt.figure()

#Make three plots, one for the two WGM Signals, other for the diff dist, and one for the xcorr
ax1 = fig.add_subplot(131)
wgmShiftPred = 0.0
#ax1.text(1300 - 3 * .008, .8, "Predicted WGM Shift: %5.2f" % (wgmShiftPred))
ax1.annotate('Predicted WGM Shift: %5.2f' % (wgmShiftPred), xy = (10, 10), xycoords='figure pixels')
ax2 = fig.add_subplot(132)
ax2.set_ylim([-1, 1]) 

ax3 = fig.add_subplot(133)


#Add space to the bottom of the plots to display slider
fig.subplots_adjust(bottom = 0.25)

#Establish the x-values (in nm)
xWindow = .008
resolution = 500
rFactor = 0.001
center = 1300
lambdaValues = np.arange(1300 - xWindow, 1300 + xWindow, ((1300 + xWindow) - (1300 - xWindow))/resolution)

#Plot the reference signal 
referenceDip_y = dipSim(.5, 1300/6000000, 1300, resolution, 1300 - xWindow, 1300 + xWindow, rFactor)
[referenceDip] = ax1.plot(lambdaValues, referenceDip_y, linewidth=1, color = 'blue', marker = '.')

#Plot the shifted signal and the difference signal
wgmShift = 0.0
shiftedDip_y = dipSim(.5, 1300/6000000, 1300 + wgmShift, resolution, 1300 - xWindow, 1300 + xWindow, rFactor)
[shiftedDip] = ax1.plot(lambdaValues, shiftedDip_y, linewidth=1, color = 'red', marker = '.')

#Plot the xcorr of the two signals 
xcorr_y = np.correlate(shiftedDip_y, referenceDip_y, 'full')
[xcorr] = ax3.plot(xcorr_y, linewidth=1, color = 'green', marker = '.')


#Function to generate a difference distribution from a ref and shifted signal
def getDiffDist(shiftVal):
    diffDist = differenceDistSim(0.5, 1300/6000000, center, 500, center - 0.008, center + 0.008, shiftVal, rFactor)
    '''
    diffDist = []
    for i in range (0, len(rDip)):
        diffDist.append(rDip[i] - sDip[i])
    '''
    return diffDist

#differenceDist_y = getDiffDist(referenceDip_y, shiftedDip_y)
differenceDist_y = getDiffDist(wgmShift)
[differenceDist] = ax2.plot(lambdaValues, differenceDist_y, linewidth=1, color = 'purple', marker = '.')

#Implement the wgm shift slider 
center_slider_ax1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
#center_slider = Slider(center_slider_ax1, 'WGM Shift (nm)', -0.006, 0.006, valinit=0, valfmt='%5.4f')
center_slider = Slider(center_slider_ax1, 'WGM Shift (nm)', 0, 0.006, valinit=0, valfmt='%5.4f')

def sliders_on_changed(val):
    shiftedDip_y = dipSim(.5, 1300/6000000, 1300 + center_slider.val, resolution, 1300 - xWindow, 1300 + xWindow, rFactor)
    shiftedDip.set_ydata(shiftedDip_y)
    #differenceDist_y = getDiffDist(referenceDip_y, shiftedDip_y)
    differenceDist_y = getDiffDist(center_slider.val)
    differenceDist.set_ydata(differenceDist_y)

    xcorr_y = np.correlate(shiftedDip_y, referenceDip_y, "full")
    xcorr.set_ydata(xcorr_y)

    values = sc.transform(array([differenceDist_y]))

    wgmShiftPred = loaded_model.predict(array(values))
    #ax1.annotate('Predicted WGM Shift: %5.2f' % (wgmShiftPred), xy = (10, 10), xycoords='figure pixels')
    print("prediction: %10.7f | actual: %10.7f | percent error: %10.7f %%" % (wgmShiftPred, center_slider.val, ((wgmShiftPred-center_slider.val)/center_slider.val) * 100))
    fig.canvas.draw_idle()

center_slider.on_changed(sliders_on_changed)


ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

#plt.subplots_adjust(left=0.25)
plt.show()