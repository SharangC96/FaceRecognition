import tensorflow as tf
import numpy as np
from InceptionResNetv2 import inception_resnet_v2
from triplet_loss import triplet_loss
import os

dir_name = os.path.dirname(__file__)
data_dir = os.path.join(dir_name,'data')

batch_size = 128

###############implement input pipeline
tf.train.string_input_producer('',shuffle=True,capacity=)
###############

embedding = inception_resnet_v2(x, reuse=False, scope='InceptionResnetV2')

# check this stuff
##############################################################
anchor = tf.slice(embedding,[0,0],[batch_size,128])
positive = tf.slice(embedding,[0,128],[batch_size,256])
negative = tf.slice(embedding,[0,256],[batch_size,384])
##############################################################

loss = triplet_loss(anchor,positive,negative)

global_step = tf.Variable(0,'global_step')
lr = tf.train.exponential_decay(0.001, global_step, decay_steps=10000, decay_rate=0.96, staircase=True, name='decayLR')
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

train_step = optimizer.minimize(loss)

load_model = True
iterations = 1000

with tf.Session() as sess:
    saver = tf.train.Saver()
    if(load_model):
        saver.restore(sess,data_dir)

    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        sess.run(train_step)
        global_step = tf.assign(global_step,iterations,'update_global_step')

    saver.save(sess,data_dir,global_step = iterations , latest_filename= 'latest')
