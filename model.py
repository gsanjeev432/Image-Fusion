from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
# import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('CT_input'):
        # CT image patch
        self.images_ct = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ct')
        self.labels_ct = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ct')
    with tf.name_scope('MRI_input'):
        # MRI Imagepatch
        self.images_mri = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_mri')
        self.labels_mri = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_mri')
    # Connect the CT and MRI in the channel direction, the first channel is the CT image, the second channel is the MRI image
    with tf.name_scope('input'):
        self.input_image=tf.concat([self.images_ct,self.images_mri],axis=-1)
    # Fusion image
    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_image)
    with tf.name_scope('d_loss'):
        pos=self.discriminator(self.labels_mri,reuse=False)
        neg=self.discriminator(self.fusion_image,reuse=True,update_collection='NO_OPS')
        pos_loss=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
        self.d_loss=neg_loss+pos_loss
        tf.summary.scalar('loss_d',self.d_loss)
    with tf.name_scope('g_loss'):
        self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_1',self.g_loss_1)
        self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ct))+5*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_mri)))
        tf.summary.scalar('g_loss_2',self.g_loss_2)
        self.g_loss_total=self.g_loss_1+100*self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=50)
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_CT")
      input_setup(self.sess,config,"Train_MRI")
    else:
      nx_ct, ny_ct = input_setup(self.sess, config,"Test_CT")
      nx_mri,ny_mri=input_setup(self.sess, config,"Test_MRI")

    if config.is_train:     
      data_dir_ct = os.path.join('./{}'.format(config.checkpoint_dir), "Train_CT","train.h5")
      data_dir_mri = os.path.join('./{}'.format(config.checkpoint_dir), "Train_MRI","train.h5")
    else:
      data_dir_ct = os.path.join('./{}'.format(config.checkpoint_dir),"Test_CT", "test.h5")
      data_dir_mri = os.path.join('./{}'.format(config.checkpoint_dir),"Test_MRI", "test.h5")

    train_data_ct, train_label_ct = read_data(data_dir_ct)
    train_data_mri, train_label_mri = read_data(data_dir_mri)
    # Find the variable group that is updated during training (discriminator and generator are trained separately, so you need to find the corresponding variable)
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    print(self.d_vars)
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)
    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    # Put all the statistics together
    self.summary_op = tf.summary.merge_all()
    # Generate log files
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ct) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images_ct = train_data_ct[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ct = train_label_ct[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_mri = train_data_mri[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_mri = train_label_mri[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ct: batch_images_ct, self.images_mri: batch_images_mri, self.labels_mri: batch_labels_mri,self.labels_ct:batch_labels_ct})
            
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ct: batch_images_ct, self.images_mri: batch_images_mri, self.labels_ct: batch_labels_ct,self.labels_mri:batch_labels_mri})
          # Write the statistics to the log file
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d,err_g))

        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ct: train_data_ct, self.labels_ct: train_label_ct,self.images_mri: train_data_mri, self.labels_mri: train_label_mri})
      result=result*127.5+127.5
      result = merge(result, [nx_ct, ny_ct])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def fusion_model(self,img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[5,5,2,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1",[256],initializer=tf.constant_initializer(0.0))
            conv1_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ct = lrelu(conv1_ct)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[5,5,256,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2",[128],initializer=tf.constant_initializer(0.0))
            conv2_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ct = lrelu(conv2_ct)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3",[64],initializer=tf.constant_initializer(0.0))
            conv3_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ct = lrelu(conv3_ct)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4",[32],initializer=tf.constant_initializer(0.0))
            conv4_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ct = lrelu(conv4_ct)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,32,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_ct= tf.nn.conv2d(conv4_ct, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ct=tf.nn.tanh(conv5_ct)
    return conv5_ct
    
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_mri=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_mri = lrelu(conv1_mri)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_mri= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_mri, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_mri = lrelu(conv2_mri)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_mri= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_mri, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_mri=lrelu(conv3_mri)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_mri= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_mri, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_mri=lrelu(conv4_mri)
            conv4_mri = tf.reshape(conv4_mri,[self.batch_size,6*6*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[6*6*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_mri, weights) + bias
    return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
