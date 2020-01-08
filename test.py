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
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

#
for a, (b, c) in dataset3:
    print('Shapes: {a.shape}, {b.shape}, {c.shape}' .format(a=a, b=b, c=c))

# Data set using Sparse Tensor
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

for z in dataset1:
    print(z.numpy())

# importing text data
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

textDataset = tf.data.TextLineDataset(file_paths)

# print first 5 lines of the Illiad
for line in textDataset.take(5):
    print(line.numpy())

files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
    if i % 3 == 0:
        print()
        print(line.numpy())

# Making variables
x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="x", dtype=tf.float32)
b = tf.constant(13.0, name="x", dtype=tf.float32)
y = tf.Variable(tf.add(tf.multiply(a, x), b))
