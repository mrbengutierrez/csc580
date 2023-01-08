"""
Student: Benjamin Gutierrez
Date: January 7, 2023
Course: Applying Machine Learning and Neural Networks - Capstone
Instructor: Lori Farr
Assignment: Module 3 Critical Thinking Assignment Option 1
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tf.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

    # Make random numbers predictable
    seed_value = 101
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)

    # Generate some random data for training the linear regression model
    # There will be 50 data points ranging from 0 to 50
    num_points = 50
    x = np.linspace(0, 50, num_points)
    y = np.linspace(0, 50, num_points)

    # Adding noise to the random linear data
    x += np.random.uniform(-4, 4, num_points)
    y += np.random.uniform(-4, 4, num_points)
    n = len(x)  # Number of data points

    # 1) plot the training data
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Training Data")
    # save the plot
    plt.savefig("training_data.png")
    plt.show()


    # 2) Create a TensorFlow model by defining the placeholders X and Y
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # 3) Define the weights W and bias b
    Weight = tf.Variable(np.random.randn(), name="W")
    bias = tf.Variable(np.random.randn(), name="b")

    # 4) Define hyperparameters for the model
    learning_rate = 0.01
    training_epochs = 1000

    # 5) Define the hypothesis
    hypothesis = X * Weight + bias

    # Define the cost function
    cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (2 * n)

    # Define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 6) Implement the training process inside a TensorFlow session
    session = tf.InteractiveSession()

    # Initialize the variables W and b
    session.run(tf.global_variables_initializer())

    # Fit all training data
    for epoch in range(training_epochs):
        for (_x, _y) in zip(x, y):
            session.run(optimizer, feed_dict={X: _x, Y: _y})

        # Display logs per epoch step
        if (epoch + 1) % 50 == 0:
            # Calculate the cost
            c = session.run(cost, feed_dict={X: x, Y: y})
            print(f"Epoch {epoch + 1}: cost = {c} Weight = {session.run(Weight)}, bias = {session.run(bias)}")

    # 7) Print out the results for the training cost, weight, and bias
    training_cost = session.run(cost, feed_dict={X: x, Y: y})
    calculated_weight = session.run(Weight)
    calculated_bias = session.run(bias)
    print(f"Training cost = {training_cost} Weight = {calculated_weight} bias = {calculated_bias}")

    # 8) Plot the training data and the linear regression model
    plt.plot(x, y, 'ro', label='Original data')
    calculated_y = calculated_weight * x + calculated_bias
    plt.plot(x, calculated_y, label='Fitted line')
    plt.title('Linear Regression Result')
    plt.legend()
    # save the plot
    plt.savefig('linear_regression.png')
    plt.show()
