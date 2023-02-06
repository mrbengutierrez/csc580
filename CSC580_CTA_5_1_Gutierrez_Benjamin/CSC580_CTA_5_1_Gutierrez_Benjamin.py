"""
Student: Benjamin Gutierrez
Date: February 5, 2023
Course: Applying Machine Learning and Neural Networks - Capstone
Instructor: Lori Farr
Assignment: Module 5 Critical Thinking Assignment Option 1
"""

# Step 1: Load the Tox21 Dataset.

import threading
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
import time

best_n_epochs = None
best_learning_rate = None
best_batch_size = None
best_validation_accuracy = 0.0

def train(learning_rate, batch_size, n_epochs):
    # setup tensorflow
    tf.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

    # set seed for reproducibility
    seed = 456
    np.random.seed(seed)
    tf.set_random_seed(seed)

    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # Step 2: Remove extra datasets.

    # Remove extra tasks
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    # Step 3: Define placeholders that accept minibatches of different sizes.

    # Generate tensorflow graph
    d = 1024
    n_hidden = 50


    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, (None, d))

        y = tf.placeholder(tf.float32, (None,))

    # Step 4: Implement a hidden layer.

    with tf.name_scope("hidden-layer"):
        W = tf.Variable(tf.random_normal((d, n_hidden)))

        b = tf.Variable(tf.random_normal((n_hidden,)))

        x_hidden = tf.nn.relu(tf.matmul(x, W) + b)

    # Step 5: Complete the fully connected architecture.

    with tf.name_scope("output"):
        W = tf.Variable(tf.random_normal((n_hidden, 1)))
        b = tf.Variable(tf.random_normal((1,)))
        y_logit = tf.matmul(x_hidden, W) + b
        # the sigmoid gives the class probability of 1
        y_one_prob = tf.sigmoid(y_logit)
        # Rounding P(y=1) will give the correct prediction.
        y_pred = tf.round(y_one_prob)

    with tf.name_scope("loss"):
        # Compute the cross-entropy term for each datapoint
        y_expand = tf.expand_dims(y, 1)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
        # Sum all contributions
        l = tf.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", l)
        merged = tf.summary.merge_all()

    # Step 6: Add dropout to a hidden layer.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        x_hidden_drop = tf.nn.dropout(x_hidden, keep_prob)

    # Step 7: Define a hidden layer with dropout.
    with tf.name_scope("hidden-layer-dropout"):
        W = tf.Variable(tf.random_normal((d, n_hidden)))
        b = tf.Variable(tf.random_normal((n_hidden,)))
        x_hidden_drop = tf.nn.relu(tf.matmul(x, W) + b)
        x_hidden_drop = tf.nn.dropout(x_hidden_drop, keep_prob)

    # Step 8: Implement mini - batching training.

    train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21',tf.get_default_graph())
    N = train_X.shape[0]
    epoch_losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(n_epochs):
            pos = 0
            pos_losses = []
            while pos < N:
                batch_X = train_X[pos:pos + batch_size]
                batch_y = train_y[pos:pos + batch_size]
                feed_dict = {x: batch_X, y: batch_y}
                _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                pos_losses.append(loss)
                # print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                train_writer.add_summary(summary, step)
                step += 1
                pos += batch_size
            epoch_losses.append(np.mean(pos_losses))

        # Make Predictions

        valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X})

        # Step 9: Use TensorBoard to track model convergence
        # use tensorboard --logdir=/tmp/fcnet-tox21 to view the graph

        # print the model accuracy
        print(f"learning_rate: {learning_rate}, batch_size: {batch_size}, n_epochs: {n_epochs}")
        validation_accuracy = accuracy_score(valid_y, valid_y_pred)
        print("Validation Accuracy: %f" % validation_accuracy)
        global best_validation_accuracy
        global best_learning_rate
        global best_batch_size
        global best_n_epochs
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_learning_rate = learning_rate
            best_batch_size = batch_size
            best_n_epochs = n_epochs


if __name__ == "__main__":
    # setup timer
    start_time = time.time()
    learning_rate_list = [0.0001, 0.001, 0.01, 0.1, 0.5]
    batch_size_list = [10, 100, 1000]
    n_epochs_list = [10, 100, 1000]

    for learning_rate in learning_rate_list:
        for batch_size in batch_size_list:
            for n_epoch in n_epochs_list:
                thread = threading.Thread(target=train, args=(learning_rate,batch_size, n_epoch))
                thread.start()
                thread.join()

    print(f"best learning rate: {best_learning_rate}, best batch size: {best_batch_size}, best n_epochs: {best_n_epochs}, best validation accuracy: {best_validation_accuracy}")

    total_time = time.time() - start_time
    print(f"Total time: {round(total_time,2)} seconds")




