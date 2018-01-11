#import the required library
# Variational Auto Encoders for Generative Modelling
 
import tensorflow.contrib.layers as lays
import numpy as np
from skimage import transform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
n_hid =8 

batch_size = 64

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(inputs,keep_prob):
    with tf.variable_scope("encoder", reuse=None):

    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
       net = lays.conv2d(inputs, 64, [4,4], stride=2, padding='SAME',activation_fn=lrelu)
       net = tf.nn.dropout(net,keep_prob)
       net = lays.conv2d(net, 64, [4, 4], stride=2, padding='SAME',activation_fn=lrelu)
       net = tf.nn.dropout(net,keep_prob)
       net = lays.conv2d(net, 64, [4, 4], stride=1, padding='SAME',activation_fn=lrelu)
       net = tf.contrib.layers.flatten(net)
       m_ = tf.layers.dense(net,units=n_hid)
       s_  = 0.5*tf.layers.dense(net,units=n_hid)
       epsilon = tf.random_normal(tf.stack([tf.shape(net)[0],n_hid])) 
       z_  = m_ + tf.multiply(epsilon,s_)


       return z_,m_,s_

def decoder(input_,keep_prob):
    with tf.variable_scope("decoder", reuse=None):
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
       inp_ = tf.layers.dense(input_,units=24,activation=lrelu)
       inp_ = tf.layers.dense(inp_,units=49,activation=lrelu)
       inp_ = tf.reshape(inp_,[-1,7,7,1])
       net = lays.conv2d_transpose(inp_, 64, [4, 4], stride=2, padding='SAME',activation_fn=tf.nn.relu)
       net = tf.nn.dropout(net,keep_prob)  
       net = lays.conv2d_transpose(net, 64, [4, 4], stride=1, padding='SAME',activation_fn=tf.nn.relu)
       net = tf.nn.dropout(net,keep_prob) 
       net = lays.conv2d_transpose(net, 64, [4,4], stride=1, padding='SAME', activation_fn=tf.nn.relu)
       net = tf.contrib.layers.flatten(net)
       net = tf.layers.dense(net,units=28*28,activation=tf.nn.sigmoid)
       net = tf.reshape(net,[-1,28,28,1])

       return net


def KL_Div(mean_,sd_):
    latent_loss = 0.5 * tf.reduce_sum(tf.square(mean_) + tf.square(sd_) - tf.log(tf.square(sd_)) - 1,1) 
    #latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0*sd_ - tf.square(mean_) -  tf.exp(2.0*sd_), 1) 
    return latent_loss

def resize_batch(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return imgs



ae_inputs = tf.placeholder(tf.float32, (None, 28, 28, 1))  # input to the network (MNIST images)
keep_prob = tf.placeholder(tf.float32)
hidden_,mean_,sd_  = encoder(ae_inputs,keep_prob)
ae_outputs =   decoder(hidden_,keep_prob)
ae_outputs_cost = tf.reshape(ae_outputs,[-1,28*28*1])
ae_inputs_cost =  tf.reshape(ae_inputs,[-1,28*28*1])

#hidden_test = tf.placeholder(tf.float32, (None,n_hid)) 
#ae_outputs_test  = decoder(hidden_test,keep_prob)
#mean_ = tf.reduce_mean(hidden1,0)
latent_loss = KL_Div(mean_,sd_)
# calculate the loss and optimize the network
epsilon = 1e-8
#loss1 = -tf.reduce_sum(ae_inputs_cost*tf.log(epsilon + ae_outputs_cost) + (1 - ae_inputs_cost)*tf.log(1 + epsilon -  ae_outputs_cost),1)
loss1 = tf.reduce_sum( tf.square( ae_inputs_cost - ae_outputs_cost),1)
loss_1 = tf.reduce_mean(loss1)
latent_loss_1 = tf.reduce_mean(latent_loss)
loss = (loss_1 + latent_loss_1)
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()
batch_size = 128  # Number of samples in each batch
epoch_num = 10 #Number of epochs to train the network
lr = 0.0001        # Learning rate

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
            _, c,c_1,c_2 = sess.run([train_op, loss,loss_1,latent_loss_1],feed_dict={ae_inputs: batch_img,keep_prob:0.8})
            print('Epoch:',(ep + 1),' Batch:',batch_n,'loss:',c)
            print('Epoch:',(ep + 1),' Batch:',batch_n,'Reconstruction loss:',c_1)
            print('Epoch:',(ep + 1),' Batch:',batch_n,'KL div loss:',c_2)
            
                      

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = batch_img.reshape(-1,28,28,1)
   
            
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img,keep_prob:1})[0]
    
    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    noise_z = np.random.normal(size=[50,n_hid])
    #oise_z = np.reshape(noise_z,(50,2,2,8))
    out_img = sess.run(ae_outputs,feed_dict={hidden_:noise_z,keep_prob:1})
    plt.figure(3)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(out_img[i, ..., 0], cmap='gray')
    
    plt.show()
