_author__ = 'Santanu Pattanayak'

import numpy as np
np.random.seed(1000)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
import keras
from keras import __version__ as keras_version
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.applications.resnet50 import preprocess_input
import h5py

# Read the Image and resize to the suitable dimension size
def get_im_cv2(path,dim=224):
    img = cv2.imread(path)
    resized = cv2.resize(img, (dim,dim), cv2.INTER_LINEAR)
    return resized

# Pre Process the Images based on the ImageNet pre-trained model Image transformation
def pre_process(img):
    img[:,:,0] = img[:,:,0] - 103.939
    img[:,:,1] = img[:,:,0] - 116.779
    img[:,:,2] = img[:,:,0] - 123.68
    return img

   
# Function to build X, y in numpy format based on the train/validation datasets
def read_data_train(class_folders,path,num_class,dim,train_val='train'):
    print train_val
    train_X,train_y = [],[] 
    for c,i in zip(class_folders,range(12)):
        path_class = path + str(train_val) + '/' + str(c)
        file_list = os.listdir(path_class)
        file_list = [ x for x in file_list if not (x.startswith('.'))]
       # file_list= file_list[:-1]
        print c 
        for f in file_list:
          #  print (path_class + '/' + f)
            img = get_im_cv2(path_class + '/' + f,dim)
            img = pre_process(img)
            train_X.append(img)
            train_y.append(i)
    train_y = keras.utils.np_utils.to_categorical(np.array(train_y),num_class) 
    return np.array(train_X),train_y



def read_data_test(path,dim,train_val='train'):
    print train_val
    test_X = []
    #for c,i in zip(class_folders,range(10):
    path_class = path + str(train_val)  
    file_list = os.listdir(path_class)
    file_list = [ x for x in file_list if not (x.startswith('.'))]

    for f in file_list:
        img = get_im_cv2(path_class + '/' + f,dim)
        img = pre_process(img)
        test_X.append(img)
      
    #train_y = keras.utils.np_utils.to_categorical(np.array(train_y),num_class)
    return np.array(test_X),file_list


# Inception V3 Model for transfer Learning 
def inception_pseudo(dim=224,freeze_layers=311,full_freeze='N'):
    model = InceptionV3(weights='imagenet',include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(12,activation='softmax')(x)
    model_final = Model(input = model.input,outputs=out)
    print len(model.layers)  
    if full_freeze != 'N':
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False
    return model_final

 
# ResNet50 Model for transfer Learning 
def resnet_pseudo(dim=224,freeze_layers=10,full_freeze='N'):
    model = ResNet50(weights='imagenet',include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(12,activation='softmax')(x)
    model_final = Model(input = model.input,outputs=out)
    if full_freeze != 'N':
        for layer in model.layers[0:0]:
            layer.trainable = False
    return model_final

# VGG16 Model for transfer Learning 

def VGG16_pseudo(dim=224,freeze_layers=10,full_freeze='N'):
    model = VGG19(weights='imagenet',include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(12,activation='softmax')(x)
    model_final = Model(input = model.input,outputs=out)
    
    if full_freeze != 'N':
        for layer in model.layers[0:5]:
            layer.trainable = False
    return model_final

    return model_final
'''
def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(224,224,3)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
'''
def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224,224,3), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

#    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

'''
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), )
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12,activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model
'''

# Training function
def train_model(train_X,train_y,n_fold=5,batch_size=16,dim=512,lr=1e-5,model='ResNet50'):
    model_save_dest = {}
    k = 0
    kf = KFold(n_splits=n_fold, random_state=0, shuffle=True)

    for train_index, test_index in kf.split(train_X):

        k += 1 
        X_train,X_test = train_X[train_index],train_X[test_index]
        y_train, y_test = train_y[train_index],train_y[test_index]
        print 'val',np.sum(y_test,axis=0)
    
        if model == 'Resnet50':
            model_final = resnet_pseudo(dim=512,freeze_layers=30,full_freeze='N')
        if model == 'VGG16':
            model_final = VGG16_pseudo(dim=512,freeze_layers=10,full_freeze='N') 
        if model == 'InceptionV3':
            model_final = inception_pseudo(dim=512,freeze_layers=311,full_freeze='N')
        if model == 'basic':
            model_final = create_model()
    
        datagen = ImageDataGenerator(
             horizontal_flip = False,
                             vertical_flip = False,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             channel_shift_range=0,
                             zoom_range = 0.2,
                             rotation_range = 20)

        
        
        adam = optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_final.compile(optimizer=adam, loss=["categorical_crossentropy"],metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,
                              patience=3, min_lr=0.000001)
    
        callbacks = [
                EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1),
             CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False),reduce_lr,
                ModelCheckpoint(
                        'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
                        monitor='val_acc', mode='max', # mode must be set to max or Keras will be confused
                        save_best_only=True,
                        verbose=1)
            ]
        class_weight = {0:0.0041493776,1:0.0049751244,2:0.0046296296,3:0.001283697,4:0.002173913,5:0.0024390244,6:0.005,7:0.005,8:0.0024154589,9:0.0014409222,10:0.0011614402,11:0.001321004} 
        model_final.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                steps_per_epoch=X_train.shape[0]/batch_size,epochs=20,verbose=1,validation_data=datagen.flow(X_test,y_test,batch_size),validation_steps=X_test.shape[0]/batch_size,callbacks=callbacks)
        del model_final
        model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'

        f = h5py.File(model_name, 'r+')
        del f['optimizer_weights'] 
        f.close()
        model_final = keras.models.load_model(model_name)
        for layer in model_final.layers[:249]:
            layer.trainable  = False
        for layer in model_final.layers[249:]:
            layer.trainable = True

        adam = optimizers.Adam(lr=1e-5,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_final.compile(optimizer=adam, loss=["categorical_crossentropy"],metrics=['accuracy'])
        model_final.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                steps_per_epoch=X_train.shape[0]/batch_size,epochs=30,verbose=1,validation_data=datagen.flow(X_test,y_test,batch_size),validation_steps=X_test.shape[0]/batch_size,callbacks=callbacks)
         
 
        model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'
        del model_final
        f = h5py.File(model_name, 'r+')
        del f['optimizer_weights']
        f.close()
        model_final = keras.models.load_model(model_name)
        model_name1 = '/home/santanu/Downloads/Superheroes/' + str(model) + 'y512y' + str(k) 
        model_final.save(model_name1)
        model_save_dest[k] = model_name1
        
    return model_save_dest

# Hold out dataset validation function

def inference_validation(test_X,model_save_dest,n_class=12,folds=5):
    path = '/home/santanu/Downloads/Superheroes/'
    pred = np.zeros((len(test_X),n_class))

    for k in xrange(1,6):
        print k
        model = keras.models.load_model(path + model_save_dest[k])
        pred = pred + model.predict(test_X,batch_size=16)

#    for k in xrange(1,folds + 1):
#        print k
#        model = keras.models.load_model(path + model_save_dest1[k])
#        pred = pred + model.predict(test_X,batch_size=64)
#
#    
#    for k in xrange(1,folds + 1):
#        print k
#        model = keras.models.load_model(path + model_save_dest2[k])
#        pred = pred + model.predict(test_X,batch_size=64)
# 
#'''
    pred = pred/(1.0*1) 
    pred_class = np.argmax(pred,axis=1) 
    #act_class = np.argmax(test_y,axis=1)
    #accuracy = np.sum([pred_class == act_class])*1.0/len(test_X)
    #kappa = cohen_kappa_score(pred_class,act_class,weights='quadratic')
    return pred_class
    
# Invoke the main function to trigger the training process
if __name__ == "__main__":
    start_time = time.time()
    path = '/home/santanu/Downloads/Superheroes/'
    class_folders = ['Ant-Man','Aquaman','Avengers','Batman','Black Panther','Captain America','Catwoman','Ghost Rider','Hulk','Iron Man','Spiderman','Superman']
    class_folder1 = ['ant_man','aqua_man','avengers','bat_man','black_panther','captain_america','cat_woman','ghostrider','hulk','iron_man','spider_man','super_man']
    num_class = len(class_folders)
    dim = 512
    lr = 1e-5
    print 'Starting time:',start_time

    train_X,train_y = read_data_train(class_folders,path,num_class,dim,train_val='CAX_Superhero_Train/CAX_Superhero_Train')
    print np.shape(train_X),np.shape(train_y)
    print np.sum(train_y,axis=0)
     
    model_save_dest = train_model(train_X,train_y,n_fold=5,batch_size=16,lr=1e-5,model='InceptionV3')

   # model_save_dest1 = {1:'VGG16____1',2:'VGG16____2',3:'VGG16____3',4:'VGG16____4',5:'VGG16____5'} 
#   model_save_dest2 = {1:'Resnet50yy1',2:'Resnet50yy2',3:'Resnet50yy3',4:'Resnet50yy4',5:'Resnet50yy5'}
#   model_save_dest = {1:'InceptionV3yy1',2:'InceptionV3yy2',3:'InceptionV3yy3',4:'InceptionV3yy4',5:'InceptionV3yy5'}  
    #model_save_dest ={1:'InceptionV3y512y1'}
   # model_save_dest ={1:'VGG16y512y1',2:'VGG16y512y2',3:'VGG16y512y3',4:'VGG16y512y4',5:'VGG16y512y5'} 
    test_X,files_out = read_data_test(path,dim,train_val='CAX_Superhero_Test/CAX_Superhero_Test')
    print test_X.shape
    pred_class =  inference_validation(test_X,model_save_dest,n_class=12,folds=5)
    np.save(path + "dict_model",model_save_dest)
    sub = pd.DataFrame()
    sub['filename'] = files_out
    sub['Superhero'] = [class_folder1[i] for i in pred_class]
    def func1(x):
        x = x.split('.')[0]
        x = x.split('_')[2]
        print x
        return int(x)
          
    sub['index']  = sub['filename'].apply(func1) 
    sub = sub.sort_values(['index'],ascending=True)
    sub['filename'] = sub['filename'].apply(lambda x:x.split('.')[0]) 
    del sub['index']
    sub.to_csv('/home/santanu/Downloads/Superheroes/submission_11.csv',index=False)


