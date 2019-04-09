#!/usr/bin/env python
# coding: utf-8

# In[2]:

from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

import tensorflow as tf
import time
import random

from sklearn.datasets import make_moons, make_circles, make_classification


# In[3]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[22]:


randidx = [random.randrange(1, len(x_train), 1) for _ in range(4)]
fig = plt.figure()
for i in range(len(randidx)):
    idx=randidx[i]
    fig.add_subplot(1,len(randidx),i+1)
    img = x_train[idx].reshape( (28, 28) )
    plt.imshow(img, cmap='gray')
    plt.axis('off')


# Now we will declare and train our NN model

# In[6]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(6, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(6, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=32)

# print(model.evaluate(x_train, y_train))
print(model.evaluate(x_test, y_test))
# print(model.predict(x_test)[1])



# Now let's draw the border that we have learnt

# In[7]:


# x_test, y_test
pred = model.predict(x_test)

for i in [random.randrange(1, len(x_test), 1) for _ in range(10)]:
    img = x_test[i]
    plt.title("Predicted : " + str(np.argmax(pred[i])) )
    plt.imshow(img, cmap='gray')
    plt.show()

