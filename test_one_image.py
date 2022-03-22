import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True Read as a grayscale image 
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ct = lrelu(conv1_ct)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ct = lrelu(conv2_ct)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ct = lrelu(conv3_ct)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ct= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ct, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ct = lrelu(conv4_ct)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ct= tf.nn.conv2d(conv4_ct, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ct=tf.nn.tanh(conv5_ct)
    return conv5_ct
    

def input_setup(index):
    padding=6
    sub_ct_sequence = []
    sub_mri_sequence = []
    input_ct=(imread(data_ct[index])-127.5)/127.5
    input_ct=np.lib.pad(input_ct,((padding,padding),(padding,padding)),'edge')
    w,h=input_ct.shape
    input_ct=input_ct.reshape([w,h,1])
    input_mri=(imread(data_mri[index])-127.5)/127.5
    input_mri=np.lib.pad(input_mri,((padding,padding),(padding,padding)),'edge')
    w,h=input_mri.shape
    input_mri=input_mri.reshape([w,h,1])
    sub_ct_sequence.append(input_ct)
    sub_mri_sequence.append(input_mri)
    train_data_ct= np.asarray(sub_ct_sequence)
    train_data_mri= np.asarray(sub_mri_sequence)
    return train_data_ct,train_data_mri


num_epoch=3
while(num_epoch==3):

    reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_120/CGAN.model-'+ str(num_epoch))

    with tf.name_scope('CT_input'):
        # CT image patch
        images_ct = tf.placeholder(tf.float32, [1,None,None,None], name='images_ct')
    with tf.name_scope('MRI_input'):
        # MRI image patch
        images_mri = tf.placeholder(tf.float32, [1,None,None,None], name='images_mri')
    # Connect the CT and MRI images in the channel direction, the first channel is the CT image, the second channel is the MRI image
    with tf.name_scope('input'):
        input_image=tf.concat([images_ct,images_mri],axis=-1)
    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image)


    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        data_ct=prepare_data('Test_CT')
        data_mri=prepare_data('Test_MRI')
        for i in range(len(data_ct)):
            start=time.time()
            train_data_ct,train_data_mri=input_setup(i)
            result =sess.run(fusion_image,feed_dict={images_ct: train_data_ct,images_mri: train_data_mri})
            result=result*127.5+127.5
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if i<=9:
                image_path = os.path.join(image_path,'F9_0'+str(i)+".bmp")
            else:
                image_path = os.path.join(image_path,'F9_'+str(i)+".bmp")
            end=time.time()
            imsave(result, image_path)
            print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
    tf.reset_default_graph()
    num_epoch=num_epoch+1
