#!/bin/python3


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
import os
import time
import pathlib
import PIL


IMG_SIZE = 32


### Download and process the CIFAR dataset
(train_images, train_labels), (valid_images, valid_labels) = tf.keras.datasets.cifar10.load_data()
n_labels = 10
train_images, valid_images = train_images / 255.0, valid_images / 255.0
training_data = (train_images, train_labels)
valid_data = (valid_images, valid_labels)


### Quick and dirty parameter handling
net_type = ""
l=1
focus = False
try:
  print(sys.argv)
  net_type = sys.argv[1]
  print(net_type)
  l = int(sys.argv[2])
  if len(sys.argv)>3 and sys.argv[3] == "focus":
    focus = True
except:
  print("Provide net type, size and subnet parameters")
  print('Net type must be one of "std", "wide", "hierarchical" or "separable" ')
  print('Append "focus" to include the focus layer')
  exit(1)



# "Standard" conv layer with normalization and dropout
def conv2d_layer(inputs, size, stride, width):
  x = layers.Conv2D(size, (width, width),
      strides=(stride,stride), padding='same')(inputs)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)
  x = layers.BatchNormalization()(x)
  return x

# Split conv layer for the hierarchical version
def conv2d_split_layer(inputs, size, stride, width):
  out = []
  for sub_input in inputs:
    x = layers.Conv2D(size, (width, width),
        strides=(stride,stride), padding='same')(sub_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    out.append(x)
  return out

# Combine feature sets pairwise
def combine_2(inputs):
  axis = len(inputs[0].shape)-1
  out = []
  i1 = inputs[::2]
  i2 = inputs[1::2]
  for x,y in zip(i1,i2):
    x = layers.concatenate([x,y], axis=axis)
    out.append( x )
  return out

# Combine all remaining
def combine_all(inputs):
  axis = len(inputs[0].shape)-1
  return layers.concatenate(inputs, axis=axis)

# Replace Conv2D with SeparableConv2D
def separable_conv2d_layer(inputs, size, stride, width):
  x = layers.SeparableConv2D(size, (width, width), strides=(stride, stride), padding='same')(inputs)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)
  x = layers.BatchNormalization()(x)
  return x

# An attention layer that returns nv combinations of the 
# features at different spatial features after the
# convolutional part. Returns nv*d_out features
def focus_layer(inputs, l=4*4, d_in=64, d_out=64, nv = 4):
  x = layers.Reshape((l, d_in))(inputs)
  
  f = layers.Dense(nv, activation='relu')(inputs)
  v = layers.Dense(d_out, activation='relu')(inputs)

  f = layers.Reshape((l, nv))(f)
  v = layers.Reshape((l, d_out))(v)

  w = layers.Softmax(axis=1)(f)
  out = layers.Dot(axes=(1,1))([w,v])
  out = layers.Reshape((d_out*nv,))(out)
  x = layers.BatchNormalization()(x)

  return out



# Create the convolutional part
if net_type == "std":
  print("std")
  def conv_layers(inputs):
    x = conv2d_layer(inputs,   l, 2, 3)
    x = conv2d_layer(x,      2*l, 2, 3)
    x = conv2d_layer(x,      2*l, 2, 3)
    return x

if net_type == "wide":
  def conv_layers(inputs):
    x = conv2d_layer(inputs,   l, 3, 5)
    x = conv2d_layer(x,      2*l, 3, 5)
    return x

if net_type == "hierarchical":
  def conv_layers(inputs):
    x = conv2d_split_layer([inputs]*8,   l, 2, 3)
    x = combine_2(x)
    x = conv2d_split_layer(x,    4*l, 2, 3)
    x = combine_2(x)
    x = conv2d_split_layer(x,    8*l, 2, 3)
    x = combine_all(x)
    return x

if net_type == "separable":
  def conv_layers(inputs):
    x = separable_conv2d_layer(inputs,   l, 2, 3)
    x = separable_conv2d_layer(x,      2*l, 2, 3)
    x = separable_conv2d_layer(x,      2*l, 2, 3)
    return x



# Build the full model
def make_classifier(n_classes):
  inputs = layers.Input(shape = [IMG_SIZE, IMG_SIZE, 3])

  x = conv_layers(inputs)

  if(focus):
    x = focus_layer(x, l=4*4, d_in=2*l, d_out = 2*l, nv=4)

  x = layers.Flatten()(x)
  x = layers.Dense(256, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  x = layers.BatchNormalization()(x)
  out = layers.Dense(n_classes)(x)

  model = tf.keras.Model(inputs=inputs, outputs=out)

  model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
  return model



classifier = make_classifier(n_labels)



# Print summary and do test run
classifier.summary()
classifier.fit(train_images, train_labels, validation_data=valid_data, epochs=100)



