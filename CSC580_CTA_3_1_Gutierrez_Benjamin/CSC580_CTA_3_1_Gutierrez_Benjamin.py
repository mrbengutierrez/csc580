"""
Student: Benjamin Gutierrez
Date: January 7, 2023
Course: Applying Machine Learning and Neural Networks - Capstone
Instructor: Lori Farr
Assignment: Module 3 Critical Thinking Assignment Option 1
"""

# Load in the Tox21 Dataset
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Set seed value to generate reproducible results
    seed_value = 456
    np.random.seed(456)
    tf.set_random_seed(456)

    # Get training & testing data
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

