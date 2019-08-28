"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import sys

# flags = tf.app.flags
# flags.DEFINE_string('csv_input', csvInput, 'Path to the CSV input')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('image_dir', imageDir, 'Path to images')
# FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'dumbbell_ziva_1':
        return 1
    elif row_label == 'dumbbell_ziva_2':
        return 2
    elif row_label == 'dumbbell_ziva_3':
        return 3
    elif row_label == 'dumbbell_ziva_4':
        return 4
    elif row_label == 'dumbbell_ziva_5':
        return 5
    elif row_label == 'dumbbell_ziva_6':
        return 6
    elif row_label == 'dumbbell_ziva_7':
        return 7
    elif row_label == 'dumbbell_ziva_8':
        return 8
    elif row_label == 'dumbbell_ziva_9':
        return 9
    elif row_label == 'dumbbell_ziva_10':
        return 10
    elif row_label == 'dumbbell_ziva_12':
        return 11
    elif row_label == 'dumbbell_ziva_14':
        return 12
    elif row_label == 'dumbbell_ziva_16':
        return 13
    elif row_label == 'dumbbell_ziva_18':
        return 14
    elif row_label == 'dumbbell_ziva_20':
        return 15
    elif row_label == 'dumbbell_ziva_22':
        return 16
    elif row_label == 'dumbbell_ziva_24':
        return 17
    elif row_label == 'dumbbell_ziva_26':
        return 18
    elif row_label == 'dumbbell_ziva_28':
        return 19
    elif row_label == 'dumbbell_ziva_30':
        return 20
    elif row_label == 'dumbbell_ziva_32':
        return 21
    elif row_label == 'dumbbell_ziva_34':
        return 22
    elif row_label == 'dumbbell_ziva_36':
        return 23
    elif row_label == 'dumbbell_ziva_38':
        return 24
    elif row_label == 'dumbbell_ziva_40':
        return 25
    elif row_label == 'kettlebell_ziva_4':
        return 26
    elif row_label == 'kettlebell_ziva_6':
        return 27
    elif row_label == 'kettlebell_ziva_8':
        return 28
    elif row_label == 'kettlebell_ziva_12':
        return 29
    elif row_label == 'kettlebell_ziva_14':
        return 30
    elif row_label == 'kettlebell_ziva_16':
        return 31
    elif row_label == 'kettlebell_ziva_20':
        return 32
    elif row_label == 'kettlebell_ziva_24':
        return 33
    elif row_label == 'kettlebell_ziva_28':
        return 34
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    jpg_path = os.path.join(path, '{}'.format(group.filename))
    if not os.path.exists(jpg_path):
        print("error",jpg_path)
        sys.exit(0)
    with tf.gfile.GFile(jpg_path, 'rb') as fid:
        # print(os.path.join(path, '{}'.format(group.filename)))
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    writer = tf.python_io.TFRecordWriter('C:\\models\\research\\object_detection\\wanling.record')
    
    imageDir = ['C:\\models\\research\\object_detection\\images\\train_ziva_1' , 
                'C:\\models\\research\\object_detection\\images\\train_ziva_2']

    for path in imageDir:
        # path = os.path.join(FLAGS.image_dir)
        # print(path)
        if (path == imageDir[0]):
            csvInput = './images/train_ziva_1_labels.csv'
        else:
            csvInput = './images/train_ziva_2_labels.csv'

        examples = pd.read_csv(csvInput)
        grouped = split(examples, 'filename')

        for group in grouped:
            # print(group, path)
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), "FLAGS.output_path")
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    # tf.app.run()
    main()

