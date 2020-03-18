import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# grab a reference to the dataset
fashion_mnist = datasets.fashion_mnist

# load the data set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# flatten the images
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))


# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# create model instance
model = models.Sequential()

# add layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(train_images, train_labels, epochs=5)

# calculate predictions
predictions_net = model.predict(test_images)

# finding values from prob percentages of each category
prediction = np.argmax(predictions_net, axis=1)

# create confusion matrix and accuracy score
predictions_cm = confusion_matrix(test_labels, prediction)
recognition_accuracy_rate_net = accuracy_score(test_labels, prediction)

# plot using seaborn to make it look nice
plt.figure(figsize=(9,9))
sns.heatmap(predictions_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(recognition_accuracy_rate_net)
plt.title(all_sample_title, size = 15)
plt.show()