from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pathlib
import pandas as pd
import numpy as np

# Just disables AVX2 warning, offloads to GPU anyways
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

# Outputs all data set elements
for elem in dataset:
    print(elem.numpy())

# Outputs data set elements based on iteration, in this case, only twice
it = iter(dataset)
print(next(it).numpy())
print(next(it).numpy())

# Reduces data set into one number, in this case adds all the integers together
print(dataset.reduce(0, lambda state, value: state + value).numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
        tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="x", dtype=tf.float32)
b = tf.constant(13.0, name="x", dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))
