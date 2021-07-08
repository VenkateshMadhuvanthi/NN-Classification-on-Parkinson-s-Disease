import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from interpolateNaNs import interpolateNaNs
#from scipy.special import softmax
#input 
signatureFile = "signatureStrengths_psdWelch.csv"
dataLabelFile = "CIS-PD_Training_Data_IDs_Labels.csv"

signatureDf = pd.read_csv(signatureFile).set_index("Row")
signatureDf.index.name = "measurement_id"
dataLabelDf = pd.read_csv(dataLabelFile).set_index(['subject_id','m
easurement_id'])
interpolateDf = interpolateNaNs(dataLabelDf)
dataLabelDf=interpolateDf.round()

combinedDf = dataLabelDf.join(signatureDf)
#combinedDf = combinedDf.loc[1004,:]

signatures = list(signatureDf.columns)
numSignatures = len(signatureDf.columns)
observations = list(dataLabelDf.columns)
inData = combinedDf[signatures]
outData = combinedDf[observations]
testSize = 0.5
lossfn = 'binary_crossentropy'
yNp=outData.to_numpy()
y_normed = yNp / yNp.max(axis=0)
y_normed=y_normed.round()


#training
inTrain, inTest, outTrain, outTest = train_test_split(inData, 
y_normed, test_size=testSize, random_state=0)

model = keras.Sequential([
    keras.layers.Dense((numSignatures+3)/2,input_shape=(numSignatu
    res,),activation=tf.nn.sigmoid),
    keras.layers.Dense(3,activation=tf.nn.sigmoid),
    ])

optRMSprop=keras.optimizers.RMSprop(learning_rate = 0.01)
optAdam=keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer=optAdam, loss=lossfn,metrics=['accuracy'])
history = model.fit(inTrain, outTrain, epochs=1000, batch_size=500,
validation_split=0.5)
# Plotting block
fig = plt.figure(figsize = (15,5), constrained_layout=True)
spec = gs.GridSpec(ncols=1, nrows=1, figure=fig)
axAcc = fig.add_subplot(spec[0,0])
axLoss= axAcc.twinx()
axAcc.plot(history.history['accuracy'], c='b', label="Training 
accuracy")
axAcc.plot(history.history['val_accuracy'], c='r', 
label="Validation accuracy")
#axAcc.axhline(1,color='k', ls='dashed')
axAcc.set_ylim(0,1.1)
axAcc.set_ylabel('Accuracy')
axAcc.set_xlabel('Epochs')
#axAcc.legend()
axLoss.semilogy(history.history['loss'], c='b',ls='dotted', 
label="Training loss")
axLoss.semilogy(history.history['val_loss'], c='r', ls='dotted', 
label="Validation loss")
axLoss.set_ylabel(lossfn.capitalize())
#axLoss.legend()
fig.legend(loc='lower left')
plt.show()


autoPredictions = model.predict(inData)
autoPredictionDf = pd.DataFrame(autoPredictions, 
index=outData.index, columns=outData.columns).round()
labelDf= (combinedDf[observations]*4).round()
difDf = (labelDf-autoPredictionDf).abs()
difDf.hist()
print(difDf.median(axis=0))
