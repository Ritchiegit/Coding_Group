# TFrecord
# TF.train.Example
# TF.train.feature

# Byte
# Float
# Int64

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

print(_bytes_feature(b'goat'))
print(_float_feature(3.14))
print(_int64_feature(4))
print(_int64_feature(True))

# num to Feature
f1 = _float_feature(3.14)
# Feature to num
raw_value = f1.float_list.value[0]
print(raw_value)

# feature to str
pi_str = f1.SerializeToString()
print(pi_str)
# str to feature
f1_back = tf.train.Feature.FromString(pi_str)
print(f1_back)

# Example to str
def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        "feature0": _int64_feature(feature0),
        "feature1": _int64_feature(feature1),
        "feature2": _bytes_feature(feature2),
        "feature3": _float_feature(feature3),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example_serialized = example.SerializeToString()
    return example_serialized
eg = serialize_example(False, 4, b'first', 8976.434)
print(eg)
eg_example = tf.train.Example.FromString(eg)
print(eg_example)


# generate dataset
n = 3
feature0 = np.random.choice([True, False], n)
feature1 = np.random.randint(0, 5, n)
strings = np.array((b'first', b'second', b'third', b'forth', b'fifth'))
feature2 = strings[feature1]
feature3 = np.random.randn(n)
feature_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
for elem in feature_dataset:
    print(elem)


# A: generate all examples and save
# 1 map
# tf.function
def tf_serialize_example(f0, f1, f2, f3):
    res = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),
        tf.string
    )
    return tf.reshape(res, ())
# serialized_feature_dataset = feature_dataset.map(tf_serialize_example)
# print(1, serialized_feature_dataset)
# for elem in serialized_feature_dataset:
#     print(1, elem)
# map

# 2. generator
def generator():
    for feature in feature_dataset:
        serialized_feature = serialize_example(*feature)
        yield serialized_feature

serialized_feature_dataset = tf.data.Dataset.from_generator(generator, output_shapes=(), output_types=tf.string)
for elem in serialized_feature_dataset:
    print(1, elem)

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_feature_dataset)

# B save example one by one
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)