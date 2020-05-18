# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:41:56 2020

@author: Eshika Mahajan
"""


#importing keras and loading the inbuilt MNIST dataset
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)



#displaying images to check the dataset
#subplot is used because we wish to plot all the figures on 1 plot 
#plt.imshow()creates image from 2darray. the image will have 1 sq for each element of the array
#the xolour of the image is determined by the cmap function used in imshow()
plt.subplot(221)  #221 means nrows=2 ncols=2 and fig=1
plt.imshow(x_train[0], cmap=plt.get_cmap('gray')) 

plt.subplot(222)  #nrows=2 ncols=2 and fig=2
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)  #nrows=2 ncols=2 and fig=3
plt.imshow(x_train[200], cmap=plt.get_cmap('gray'))
plt.subplot(224)  #nrows=2 ncols=2 and fig=4
plt.imshow(x_train[400], cmap=plt.get_cmap('gray'))
plt.show()



# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)



#Normalising the dataset-->Data scaling 
#converting the whole dataset into float datatype to initiate normalisation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

'''Output 
x_train shape: (60000, 28, 28, 1)
Number of images in x_train 60000
Number of images in x_test 10000 
'''



#intialising the classifier parameters
batch_size = 128   # it will take 128 images at one go
num_classes = 10   # total number of classes
epochs = 10        #no of times an ENTIRE dataset is passed forward and backward through the neural network.



# convert class vectors to binary class matrices
'''
This function takes a vector or 1 column matrix of class labels 
and converts it into a matrix with p columns, one for each category. (just like dummy variables after label encoding)
'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



#importing kera libraries to frame the neural network
from keras.models import Sequential   # we are preparin a sequential model and not a functional model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D  #these are the different layers we'll be adding to our network
from keras import backend as K



model = Sequential() #creating an object of class sequential
#adding a convulation layer with relu as activation finction 
#input shape is already given above
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
#pooling the above layer
model.add(MaxPooling2D(pool_size=(2, 2)))
#adding another convolution layer and then pooling it
model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flattening the 2D arrays for fully connected layers

model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has been succefully trained.")

#mnist.h5 contains the trained weights
model.save('mnist.h5')
print('The trained weights have been saved.')

model.evaluate(x_test, y_test)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#testing the image at index 4444
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('The number predicted is : ', pred.argmax())




image_index = 0
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('The number predicted is : ', pred.argmax())



image_index = 9920
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('The number predicted is : ', pred.argmax())




image_index = 400
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('The number predicted is : ', pred.argmax())





