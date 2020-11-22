import numpy as np
import glob
import tensorflow as tf
import os
# 3862
tf.enable_eager_execution()
def tf_itr(tmp_file="test", FOLDER="", label_num=3862, batch_size=128):
    tfiles = glob.glob(os.path.join(FOLDER, tmp_file, "*.tfrecord"))
    print("num of files is ", len(tfiles))

    # id labels mean_rgb mean_audio
    ids, audio, rgb, lbs = [], [], [], []
    for index_i, file_name in enumerate(tfiles):
        for example in tf.data.TFRecordDataset(file_name):
            tf_example = tf.train.Example.FromString(example.numpy())
            ids.append(tf_example.features.feature["id"].bytes_list.value[0].decode(encoding='UTF-8'))
            audio.append(tf_example.features.feature["mean_audio"].float_list.value[0])
            rgb.append(tf_example.features.feature["mean_rgb"].float_list.value[0])
            ys = tf_example.features.feature["labels"].int64_list.value
            label_in_onehot = np.zeros(label_num).astype(np.int8)
            for y in ys:
                label_in_onehot[y] = 1
            lbs.append(label_in_onehot)

            if len(ids) >= batch_size:
                yield np.array(ids), np.array(audio), np.array(rgb), np.array(lbs)
                ids, audio, rgb, lbs = [], [], [], []
        if index_i + 1 == len(tfiles):
            yield np.array(ids), np.array(audio), np.array(rgb), np.array(lbs)


if __name__ == "__main__":
    # g = tf_itr(tmp_file="", FOLDER='data')
    # ids, audio, rgb, lbs = next(g)
    # print(ids, audio, rgb, lbs)

    g = tf_itr(tmp_file="", FOLDER='data')
    for ids, audio, rgb, lbs in g:
        print(ids, audio, rgb, lbs)
        input()