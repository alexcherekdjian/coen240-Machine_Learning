import tensorflow as tf
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# uploading data from mnist dataset
mnist = tf.keras.datasets.mnist
(x_traino, y_train),(x_testo, y_test) = mnist.load_data()

# reshaping matrices and init the regression solver
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# fix random seed for reproducibility
numpy.random.seed(7)

# create the model
# model.add(Flatten(input_shape(28,28))
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu')) # input 784
#model.add(Dense(512, activation='relu')) # hidden 512, reLu 
model.add(Dense(10, activation='softmax')) # output 10 nodes, soft max

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=2)

# calculate predictions
predictions_net = model.predict(x_test)

# finding values from prob percentages of each category
prediction = np.argmax(predictions_net, axis=1)
    
# create confusion matrix and accuracy score
predictions_cm = confusion_matrix(y_test, prediction)
recognition_accuracy_rate_net = accuracy_score(y_test, prediction)

# plot using seaborn to make it look nice
plt.figure(figsize=(9,9))
sns.heatmap(predictions_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(recognition_accuracy_rate_net)
plt.title(all_sample_title, size = 15)
