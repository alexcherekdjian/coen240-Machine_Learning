import tensorflow as tf
import numpy as np
import math
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras import datasets, layers, models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape

# p values, image and psnr list
P_values = [10,50,200]
psnr = []
im = Image.new('RGB', (280, 112))

T = 2; # expansion factor
# image resolution (pixels)
m = 28; # rows
n = 28; # columns

# get data for fashion mnist
fashion_mnist = datasets.fashion_mnist

# split into training and testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# print first set of ten test images
test_imgs = np.concatenate((test_images[0:10]), axis = 1)
image = Image.fromarray((test_imgs*255.0).astype('uint8'))
im.paste(image, (0, 0))

# go through all p values
for i in range(0, len(P_values)):
    # Parameters
    P = P_values[i]; # P < m * n; 10, 50, 200

    # define the keras model
    model = Sequential()
    # A flattened input layer with m x n nodes
    model.add(Flatten(input_shape=(m, n)))
    # A compressed layer with P nodes (no activation function)
    model.add(Dense(P))
    # An expansion layer with m x n x T nodes, followed by ReLU activation
    model.add(Dense(m*n*T, activation='relu'))
    # An output layer with m x n nodes (no activation function)
    model.add(Dense(m*n))
    # A reshape layer that convert the 1-D vector output to the m x n 2-dimensional image
    model.add(Reshape((m, n)))

    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit the keras model on the dataset
    model.fit(train_images, train_images, epochs=10, batch_size=64)
    
    # evaluate the model and get mse
    mse = model.evaluate(test_images, test_images)
    
    # calculate psnr
    psnr.append(10*math.log10(1/mse))
    
    # append predictions to final image
    pred_image = model.predict(test_images)
    test_imgs = np.concatenate((pred_image[0:10]), axis = 1)
    image = Image.fromarray((test_imgs*255.0).astype('uint8'))
    im.paste(image, (0, 28 + i * 28))

# print p vs psnr values
print("P_values " + str(P_values))
print("PSNR " + str(psnr))

# save image
im.save('problem2.png')
print("png saved.")