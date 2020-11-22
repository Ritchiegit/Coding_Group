import tensorflow as tf
tf.enable_eager_execution()
filename = "test.tfrecord"

filenames = [filename]

# tfrecord to str
raw_dataset = tf.data.TFRecordDataset(filenames)


# str to Example to num
print(raw_dataset)
for raw_record in raw_dataset:
    print(raw_record)
    example = tf.train.Example.FromString(raw_record.numpy())
    print(example.features.feature["feature0"].int64_list.value[0])
    print(example.features.feature["feature1"].int64_list.value[0])
    print(example.features.feature["feature2"].bytes_list.value[0])
    print(example.features.feature["feature3"].float_list.value[0])

# str to Feature to num
feature_description = {
    "feature0": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "feature1": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "feature2": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "feature3": tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}

def _parse_function(str1):
    return tf.io.parse_single_example(str1, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
for parsed_data in parsed_dataset:
    print(parsed_data)
    print(parsed_data['feature0'].numpy())
    print(parsed_data['feature1'].numpy())
    print(parsed_data['feature2'].numpy())
    print(parsed_data['feature3'].numpy())
