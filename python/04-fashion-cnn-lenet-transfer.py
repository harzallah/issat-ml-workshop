#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import random
import matplotlib.pyplot as plt


# In[3]:


num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Now we will declare and train our NN model

# In[4]:


batch_size = 128
epochs = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[5]:


(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train_fashion = x_train_fashion.reshape(x_train_fashion.shape[0], 1, img_rows, img_cols)
    x_test_fashion = x_test_fashion.reshape(x_test_fashion.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_fashion = x_train_fashion.reshape(x_train_fashion.shape[0], img_rows, img_cols, 1)
    x_test_fashion = x_test_fashion.reshape(x_test_fashion.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train_fashion = x_train_fashion.astype('float32')
x_test_fashion = x_test_fashion.astype('float32')
x_train_fashion /= 255
x_test_fashion /= 255
print('x_train_fashion shape:', x_train_fashion.shape)
print(x_train_fashion.shape[0], 'train samples')
print(x_test_fashion.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train_fashion = keras.utils.to_categorical(y_train_fashion, num_classes)
y_test_fashion = keras.utils.to_categorical(y_test_fashion, num_classes)


# Sample of MNIST Fashion dataset

# In[9]:


randidx = [random.randrange(1, len(x_train_fashion), 1) for _ in range(4)]
fig = plt.figure()
for i in range(len(randidx)):
    idx=randidx[i]
    fig.add_subplot(1,len(randidx),i+1)
    img = x_train_fashion[idx].reshape( (28, 28) )
    plt.imshow(img, cmap='gray')
    plt.axis('off')


# In[6]:


batch_size_fashion = 128
epochs_fashion = 10

model_fashion = Sequential()
model_fashion.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model_fashion.add(Conv2D(64, (3, 3), activation='relu'))
model_fashion.add(MaxPooling2D(pool_size=(2, 2)))
model_fashion.add(Dropout(0.25))
model_fashion.add(Flatten())
model_fashion.add(Dense(128, activation='relu'))
model_fashion.add(Dropout(0.5))
model_fashion.add(Dense(num_classes, activation='softmax'))

model_fashion.set_weights(model.get_weights()) 



for layer in model_fashion.layers[:-3]:
    layer.trainable = False
    
model_fashion.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_fashion.fit(x_train_fashion, y_train_fashion,
          batch_size=batch_size_fashion,
          epochs=epochs_fashion,
          verbose=1,
          validation_data=(x_test_fashion, y_test_fashion)) 

score = model_fashion.evaluate(x_test_fashion, y_test_fashion, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ### Class Labels : 
# |Description|T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle |
# | -- | -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- |  -- |
# |Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
# 

# In[7]:


score = model_fashion.evaluate(x_test_fashion, y_test_fashion, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

randidx = [random.randrange(1, len(x_test), 1) for _ in range(10)]
pred = model_fashion.predict(x_test_fashion[randidx])

for i in range(len(randidx)):
    idx = randidx[i]
    img = x_test_fashion[idx].reshape( (28, 28) )
    plt.title("Predicted : " + str(np.argmax(pred[i])) )
    plt.imshow(img, cmap='gray')
    plt.show()

