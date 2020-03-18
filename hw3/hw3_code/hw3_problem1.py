import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# uploading data from mnist dataset
mnist = tf.keras.datasets.mnist
(x_traino, y_train),(x_testo, y_test) = mnist.load_data()

# reshaping matrices and init the regression solver
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0
logreg = LogisticRegression(solver='saga', multi_class='multinomial',max_iter = 100,verbose=2)

# train the algorithm
logreg.fit(x_train, y_train)

# predict all values from dataset
predictions = logreg.predict(x_test)

# create confusion matrix and accuracy score
predictions_cm = confusion_matrix(y_test, predictions)
recognition_accuracy_rate = accuracy_score(y_test, predictions)

# plot using seaborn to make it look nice
plt.figure(figsize=(9,9))
sns.heatmap(predictions_cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(recognition_accuracy_rate)
plt.title(all_sample_title, size = 15)