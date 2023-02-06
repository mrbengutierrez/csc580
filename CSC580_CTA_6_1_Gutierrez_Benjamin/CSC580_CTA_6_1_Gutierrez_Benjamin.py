"""
Student: Benjamin Gutierrez
Date: February 5, 2023
Course: Applying Machine Learning and Neural Networks - Capstone
Instructor: Lori Farr
Assignment: Module 6 Critical Thinking Assignment Option 1
"""

# Implementation of CIFAR10 with CNNs Using TensorFlow

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot
import ssl

# Load the CIFAR10 Dataset.
from tensorflow.keras.datasets import cifar10

if __name__ == "__main__":
    # get seed for reproducibility
    seed = 51
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # turn off ssl verification
    ssl._create_default_https_context = ssl._create_unverified_context

    # read the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalize test dataset
    devisor = 255.0
    x_train = x_train / devisor
    x_test = x_test / devisor

    # train on cifar10 using convolutional neural network

    # define the model
    replica = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='leaky_relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='leaky_relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # compile the model
    replica.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # train the model
    num_epochs = 15
    replica.fit(x_train, y_train, epochs=num_epochs)

    # evaluate the model
    evaluation = replica.evaluate(x_test, y_test)
    test_accuracy = evaluation[1]

    # print model accuracy
    print(f"Test Accuracy: {round(test_accuracy * 100, 2)}%")

    # make predictions
    y_pred = replica.predict(x_test)

    # plot the first 25 images from the test dataset and their predicted labels
    # if the prediction is correct, the color will be green, otherwise red
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    matplotlib.pyplot.figure(figsize=(10, 10))
    for i in range(25):
        matplotlib.pyplot.subplot(5, 5, i + 1)
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.grid(False)
        matplotlib.pyplot.imshow(x_test[i], cmap=matplotlib.pyplot.cm.binary)

        predicted_label = np.argmax(y_pred[i])
        true_label = y_test[i][0]
        color = 'green' if predicted_label == true_label else 'red'
        matplotlib.pyplot.xlabel(f"Predicted: {class_names[predicted_label]}\nActual: {class_names[true_label]}",
                                 color=color)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.suptitle("First 25 images from test dataset")
    matplotlib.pyplot.subplots_adjust(top=0.95)
    matplotlib.pyplot.show()
