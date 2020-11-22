import tensorflow as tf
import glob
import os
class YT8MReader():
    def __init__(self,
                 num_classes=3862,
                 feature_size=[1024, 128],
                 feature_name=["mean_rgb", "mean_audio"]):
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.feature_name = feature_name

    def prepare_reader(self, filename_queue, batch_size=1024):
        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read_up_to(filename_queue, batch_size)
        parsed_examples = self.parsing_serialized_examples(serialized_examples)
        return parsed_examples

    def parsing_serialized_examples(self, serialized_examples):
        feature_description = {
            "id": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.VarLenFeature(tf.int64),
            "mean_audio": tf.FixedLenFeature(128, tf.float32),
            "mean_rgb": tf.FixedLenFeature(1024, tf.float32)
        }
        features = tf.parse_example(serialized_examples, features=feature_description)
        labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
        labels.set_shape([None, 3862])
        output_dict = {
            "id": features["id"],
            "mean_rgb": features["mean_audio"],
            "mean_audio": features["mean_audio"],
            "labels": features["labels"],
        }
        return output_dict

if __name__ == "__main__":
    FOLDER = ""
    tmp_file = "data"
    files = glob.glob(os.path.join(FOLDER, tmp_file, "*.tfrecord"))
    filename_queue = tf.train.string_input_producer(files, num_epochs=1)
    reader_y8tm = YT8MReader()
    out = reader_y8tm.prepare_reader(filename_queue, batch_size=5)
    print(out)
