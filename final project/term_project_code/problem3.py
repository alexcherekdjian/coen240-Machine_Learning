import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
%matplotlib inline

# class definitions for two models
class CNN_A:
    def __init__(self, stride1, stride2):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, kernel_size=(3, 3), padding = "SAME", strides = (stride1, stride2), activation='relu', input_shape=(240, 416, 3)))
        self.model.add(layers.Conv2D(32, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu'))
        self.model.add(layers.Conv2D(4, kernel_size=(1, 1), padding = "SAME", strides = (1,1), activation='relu'))

        self.model.add(layers.Conv2DTranspose(32, kernel_size=(1, 1), padding = "SAME", strides = (1,1), activation='relu'))
        self.model.add(layers.Conv2DTranspose(64, kernel_size=(5, 5), padding = "SAME", strides = (1,1), activation='relu'))
        self.model.add(layers.Conv2DTranspose(3, kernel_size=(3, 3), padding = "SAME", strides = (stride1, stride2), activation='relu'))

class CNN_B:
    def __init__(self, stride1, stride2):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, kernel_size = (3, 3), padding = "SAME", strides = (stride1, stride2), activation = 'relu', input_shape = (240, 416, 3)))
        self.model.add(layers.Conv2D(16, kernel_size = (5, 5), padding = "SAME", strides = (1, 1), activation = 'relu'))
        self.model.add(layers.Conv2D(8, kernel_size = (1, 1), padding = "SAME", strides = (1, 1), activation = 'relu'))
        
        self.model.add(layers.Conv2D(4, kernel_size = (3, 3), padding = "SAME", strides = (1, 1), activation = 'relu'))

        self.model.add(layers.Conv2DTranspose(8, kernel_size = (3, 3), padding = "SAME", strides = (1, 1), activation = 'relu'))
        self.model.add(layers.Conv2DTranspose(16, kernel_size = (1, 1), padding = "SAME", strides = (1, 1), activation = 'relu'))
        self.model.add(layers.Conv2DTranspose(64, kernel_size = (5, 5), padding = "SAME", strides = (1, 1), activation = 'relu'))
        self.model.add(layers.Conv2DTranspose(3, kernel_size = (3, 3), padding = "SAME", strides = (stride1, stride2), activation = 'relu'))


# pathname of files, change pathname manually for different picture sets
pathname = './BasketballDrill_832x480_50'

x_samples = []

# read in all files and resize images if needed
for im in os.listdir(pathname):
    img = Image.open(pathname + "/" + str(im)).convert('RGB')
    img = img.resize((416, 240)) 
    
    np_img = np.array(img)
    x_samples.append(np_img)

# create np array with all samples and create test train split
x_samples = np.array(x_samples)
x_train, x_test = train_test_split(x_samples, test_size=0.2)


x_train = x_train[:int(x_train.shape[0]/4)]
x_test = x_test[:int(x_train.shape[0]/4)]

# initializing compression ratio lists
compression_ratios = [1/2, 1/4, 1/8, 1/16, 1/32]

# initializing different A models and psnr list
model_A_psnr = []
model_1_2_A = CNN_A(1,2)
model_1_4_A = CNN_A(2,2)
model_1_8_A = CNN_A(2,4)
model_1_16_A = CNN_A(4,4)
model_1_32_A = CNN_A(4,8)

# initializing different B models and psnr list
model_B_psnr = []
model_1_2_B = CNN_B(1,2)
model_1_4_B = CNN_B(2,2)
model_1_8_B = CNN_B(2,4)
model_1_16_B = CNN_B(4,4)
model_1_32_B = CNN_B(4,8)

# using 5 epochs to train
EPOCHS = 20

# train and evaluate model A
model_1_2_A.model.compile(optimizer='adam', loss='mse')
model_1_2_A.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_2_A.model.evaluate(x_test, x_test)
model_A_psnr.append(10*math.log10((255**2)/mse))

model_1_4_A.model.compile(optimizer='adam', loss='mse')
model_1_4_A.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_4_A.model.evaluate(x_test, x_test)
model_A_psnr.append(10*math.log10((255**2)/mse))

model_1_8_A.model.compile(optimizer='adam', loss='mse')
model_1_8_A.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_8_A.model.evaluate(x_test, x_test)
model_A_psnr.append(10*math.log10((255**2)/mse))

model_1_16_A.model.compile(optimizer='adam', loss='mse')
model_1_16_A.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_16_A.model.evaluate(x_test, x_test)
model_A_psnr.append(10*math.log10((255**2)/mse))

model_1_32_A.model.compile(optimizer='adam', loss='mse')
model_1_32_A.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_32_A.model.evaluate(x_test, x_test)
model_A_psnr.append(10*math.log10((255**2)/mse))

# train and evaluate model B
model_1_2_B.model.compile(optimizer='adam', loss='mse')
model_1_2_B.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_2_B.model.evaluate(x_test, x_test)
model_B_psnr.append(10*math.log10((255**2)/mse))

model_1_4_B.model.compile(optimizer='adam', loss='mse')
model_1_4_B.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_4_B.model.evaluate(x_test, x_test)
model_B_psnr.append(10*math.log10((255**2)/mse))

model_1_8_B.model.compile(optimizer='adam', loss='mse')
model_1_8_B.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_8_B.model.evaluate(x_test, x_test)
model_B_psnr.append(10*math.log10((255**2)/mse))

model_1_16_B.model.compile(optimizer='adam', loss='mse')
model_1_16_B.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_16_B.model.evaluate(x_test, x_test)
model_B_psnr.append(10*math.log10((255**2)/mse))

model_1_32_B.model.compile(optimizer='adam', loss='mse')
model_1_32_B.model.fit(x_train, x_train, epochs=EPOCHS)
mse = model_1_32_B.model.evaluate(x_test, x_test)
model_B_psnr.append(10*math.log10((255**2)/mse))

# create images
im_1_2 = Image.new('RGB', (1248, 480))
im_1_4 = Image.new('RGB', (1248, 480))
im_1_8 = Image.new('RGB', (1248, 480))
im_1_16 = Image.new('RGB', (1248, 480))
im_1_32 = Image.new('RGB', (1248, 480))

# 2 frames of each 
for i in range(0, 2):
    # predict an image from each 1/2 compression rate model 
    pred_image_A = model_1_2_A.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_B = model_1_2_B.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_A = pred_image_A.reshape(240, 416, 3)
    pred_image_B = pred_image_B.reshape(240, 416, 3)

    # print predictions to image to display quality difference
    original = Image.fromarray(x_test[i].astype('uint8'))
    image_A = Image.fromarray(pred_image_A.astype('uint8'))
    image_B = Image.fromarray(pred_image_B.astype('uint8'))

    im_1_2.paste(original, (0, 0 + i*240))
    im_1_2.paste(image_A, (416, 0 + i*240))
    im_1_2.paste(image_B, (832, 0 + i*240))

for i in range(0, 2):
    # predict an image from each 1/4 compression rate model 
    pred_image_A = model_1_4_A.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_B = model_1_4_B.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_A = pred_image_A.reshape(240, 416, 3)
    pred_image_B = pred_image_B.reshape(240, 416, 3)

    # print predictions to image to display quality difference
    original = Image.fromarray(x_test[i].astype('uint8'))
    image_A = Image.fromarray(pred_image_A.astype('uint8'))
    image_B = Image.fromarray(pred_image_B.astype('uint8'))

    im_1_4.paste(original, (0, 0 + i*240))
    im_1_4.paste(image_A, (416, 0 + i*240))
    im_1_4.paste(image_B, (832, 0 + i*240))

for i in range(0, 2):
    # predict an image from each 1/8 compression rate model 
    pred_image_A = model_1_8_A.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_B = model_1_8_B.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_A = pred_image_A.reshape(240, 416, 3)
    pred_image_B = pred_image_B.reshape(240, 416, 3)

    # print predictions to image to display quality difference
    original = Image.fromarray(x_test[i].astype('uint8'))
    image_A = Image.fromarray(pred_image_A.astype('uint8'))
    image_B = Image.fromarray(pred_image_B.astype('uint8'))

    im_1_8.paste(original, (0, 0 + i*240))
    im_1_8.paste(image_A, (416, 0 + i*240))
    im_1_8.paste(image_B, (832, 0 + i*240))

for i in range(0, 2):
    # predict an image from each 1/16 compression rate model 
    pred_image_A = model_1_16_A.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_B = model_1_16_B.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_A = pred_image_A.reshape(240, 416, 3)
    pred_image_B = pred_image_B.reshape(240, 416, 3)

    # print predictions to image to display quality difference
    original = Image.fromarray(x_test[i].astype('uint8'))
    image_A = Image.fromarray(pred_image_A.astype('uint8'))
    image_B = Image.fromarray(pred_image_B.astype('uint8'))

    im_1_16.paste(original, (0, 0 + i*240))
    im_1_16.paste(image_A, (416, 0 + i*240))
    im_1_16.paste(image_B, (832, 0 + i*240))

for i in range(0, 2):
    # predict an image from each 1/32 compression rate model 
    pred_image_A = model_1_32_A.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_B = model_1_32_B.model.predict(x_test[i].reshape(1,240, 416, 3))
    pred_image_A = pred_image_A.reshape(240, 416, 3)
    pred_image_B = pred_image_B.reshape(240, 416, 3)

    # print predictions to image to display quality difference
    original = Image.fromarray(x_test[i].astype('uint8'))
    image_A = Image.fromarray(pred_image_A.astype('uint8'))
    image_B = Image.fromarray(pred_image_B.astype('uint8'))

    im_1_32.paste(original, (0, 0 + i*240))
    im_1_32.paste(image_A, (416, 0 + i*240))
    im_1_32.paste(image_B, (832, 0 + i*240))

# save images
im_1_2.save('1_2_reconstruct.png')
print("1/2 png saved.")
im_1_4.save('1_4_reconstruct.png')
print("1/4 png saved.")
im_1_8.save('1_8_reconstruct.png')
print("1/8 png saved.")
im_1_16.save('1_16_reconstruct.png')
print("1/16 png saved.")
im_1_32.save('1_32_reconstruct.png')
print("1/32 png saved.")

# plot PSNR vs Compression Ratios
plt.figure(num=None, figsize=(5, 5), dpi=150, facecolor='w', edgecolor='k')
plot_B, = plt.plot(compression_ratios, model_B_psnr)
plot_A, = plt.plot(compression_ratios, model_A_psnr)
plt.legend([plot_A, plot_B], ['Model A', 'Model B'])
plt.xlabel('Compression Ratios')
plt.ylabel('PSNR')
plt.show()
