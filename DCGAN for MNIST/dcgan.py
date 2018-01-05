# Deep Convolutional GAN with Keras 
# GANs Components -> Generator G  Converts noise z to fake images G(z) similar to real images x 
#                    Discriminator D -> Tries to distinguish between real (x) and fake images generated(G(z)) 
#                    Learning Mechanism -> Both the Generator and Discriminator learns by playing a zero sum min-max game.
#                    Solution is the Saddle point solution of the classification logloss of Real and Fake images.
#                    The saddle point would be a minima point with respect to the Discriminator and maxima with respect to the Generator
#                    This saddle point in terms of Game theory is the famous Nash Equilibrium. At the Nash equilibrium the probability distribution 
#                    of the Generator Images should match the probability distribution of the Real Images i.e.
#                    P(G(z)) ~  P(x)
           

# Load the libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

# Define the Generator Network

def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7,128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    return model

#Define the Discriminator Network


def discriminator():
    model = Sequential()
    model.add(
            Conv2D(64, (3, 3),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# Define a combination of Generator and Discriminator 

def generator_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

# Steps to display the results intermitently 

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


# Training the Discriminator and Generator 

def train(batch_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #Image pixels are normalized between -1 to +1 so that one can use the tanh activation function
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator()
    g = generator()
    g_d = generator_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g_d.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/batch_size))
        for index in range(int(X_train.shape[0]/batch_size)):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                # Images converted back to be within 0 to 255  
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            # Train the Discriminator on both real and fake images 
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            # Train the generator on fake images from Noise   
            g_loss = g_d.train_on_batch(noise, [1] * batch_size)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

# this is a generator with already trained models
# If "quality" is set the generator images are evaluated on how 
# good they are based on the discriminator performance on them
# Based on the discriminator score the images are sorted with the 
# less realistic images coming before the quality images.
def generate(batch_size, quality=False):
    g = generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if quality:
        d = discriminator()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (batch_size*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size*20)
        index.resize((batch_size*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

batch_size=32
train(batch_size)
