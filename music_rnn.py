#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: jie
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#import function from local file
from util.util import print_progress
from util.create_dataset import create_dataset, get_batch
from util.midi_manipulation import noteStateMatrixToMidi

#################
###Dataset#######
min_song_length  = 128
encoded_songs    = create_dataset(min_song_length)

NUM_SONGS = len(encoded_songs)
print(str(NUM_SONGS) + " total songs to learn from")


###############################
## Neural Network Parameters ##
input_size       = encoded_songs[0].shape[1]   # The number of possible MIDI Notes
output_size      = input_size                  # Same as input size
hidden_size      = 128                         # Number of neurons in hidden layer, the minimum length

learning_rate    = 0.001 # Learning rate of the model
training_steps   = 200   # Number of batches during training
batch_size       = 100   # Number of songs per batch
timesteps        = 64    # Length of song snippet -- this is what is fed into the model

# assert timesteps < min_song_length



##########################
## Model Initialization ##
input_placeholder_shape = [None, timesteps, input_size] #[None, 64, 78]
output_placeholder_shape = [None, output_size] #[None, 78]

input_vec  = tf.placeholder("float", input_placeholder_shape)  
output_vec = tf.placeholder("float", output_placeholder_shape)  

# Define weights
weights = tf.Variable(tf.random_normal([hidden_size, output_size])) 

biases = tf.Variable(tf.random_normal([output_size])) 




##########################
## Model Computation #####
def RNN(input_vec, weights, biases):
    """
    @param input_vec: (tf.placeholder) The input vector's placeholder
    @param weights: (tf.Variable) The weights variable
    @param biases: (tf.Variable) The bias variable
    @return: The RNN graph that will take in a tensor list of shape (batch_size, timesteps, input_size)
    and output tensors of shape (batch_size, output_size)
    """

    input_vec = tf.unstack(input_vec, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(hidden_size)  

    outputs, states = rnn.static_rnn(lstm_cell, input_vec, dtype=tf.float32)
    
    recurrent_net = tf.matmul(outputs[-1], weights) + biases 
    
    prediction = tf.nn.softmax(recurrent_net) 
    
    return recurrent_net, prediction



###########################################
##### loss, Optimization and Accuracy #####

# LOSS
logits, prediction = RNN(input_vec, weights, biases)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=output_vec))  

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
train_op = optimizer.minimize(loss_op)

# Accuracy
true_note = tf.argmax(output_vec,1)
pred_note = tf.argmax(prediction, 1) 
correct_pred = tf.equal(pred_note, true_note)

accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# INITIALIZER:
init = tf.global_variables_initializer()



###################
#### Train RNN ####
# 1) Launch the session
sess = tf.InteractiveSession()

# 2) Initialize the variables
sess.run(init)

# 3) Train!
display_step = 5 

for step in range(training_steps):
    # GET BATCH
    batch_x, batch_y = get_batch(encoded_songs, batch_size, timesteps, input_size, output_size) 
    
    feed_dict = {
                    input_vec: batch_x, 
                    output_vec: batch_y 
                }
    
    sess.run(train_op, feed_dict=feed_dict)
    
    # DISPLAY METRICS
    if step % display_step == 0 or step == 1:

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict=feed_dict)     
        suffix = "\nStep " + str(step) + ", Minibatch Loss= " + \
                 "{:.4f}".format(loss) + ", Training Accuracy= " + \
                 "{:.3f}".format(acc)

        print_progress(step, training_steps, barLength=50, suffix=suffix)




