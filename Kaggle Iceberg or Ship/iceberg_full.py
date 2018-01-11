import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import KFold
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os
import gc
import keras as k
from keras.layers import Conv2D, MaxPooling2D,Concatenate,concatenate
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,MaxPooling1D,AveragePooling1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import keras

#Load the data.
train = pd.read_json("/home/santanu/Downloads/Kaggle Iceberg/train.json")
test = pd.read_json("/home/santanu/Downloads/Kaggle Iceberg/test.json")

#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1-X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)


target_train=train['is_iceberg']

k = 0
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_index, test_index in kf.split(X_train):
    k += 1 
    X_train1,X_test1 = X_train[train_index],X_train[test_index]
    y_train1, y_test1 = target_train[train_index],target_train[test_index]
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (75,75, 3))
    for layer in model.layers[:4]:
        layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    ############
    # added 7:43 morning 10th July
    # x = Dropout(0.5)(x)
    ############

    output = (Dense(1, activation='sigmoid')(x))
    model_final = Model(input = model.input, output = output)

    #datagen = ImageDataGenerator(
    #    featurewise_center=False,
    #    featurewise_std_normalization=False,
    #    rotation_range=20,
    #    zoom_range=0.15,
    #    width_shift_range=0.2,
    #    height_shift_range=0.2,
    #    horizontal_flip=True)

    #datagen.fit(x_train)

    #sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model_final.compile(optimizer=sgd, loss=["binary_crossentropy"])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    

    model_final.fit(X_train1,y_train1,
              batch_size=64,
              epochs=50,
              callbacks=[early,reduce_lr],
              verbose=1,
              validation_data=(X_test1,y_test1))
    model_name = '/home/santanu/Downloads/Kaggle Iceberg/' + 'nn_model_kk' + str(k) 
    model_final.save(model_name)

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1-X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
#out = np.zeros((8424,1))
df = pd.DataFrame()

for k in xrange(1,6):
    print(k)
    model_name = '/home/santanu/Downloads/Kaggle Iceberg/' + 'nn_model_kk' + str(k) 
    model = load_model(model_name)
    predicted_test=model.predict(X_test,batch_size=10)[:,0].tolist()
    df[str(k)] = predicted_test

out = np.mean(df[['1','2','3','4','5']].values,axis=1)
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=out
submission.to_csv('/home/santanu/Downloads/Kaggle Iceberg/sub6.csv', index=False)

