import os
import cv2
import tensorflow as tf
from tensorflow.train import BytesList, Int64List
from tensorflow.train import Features, Feature, Example
from tensorflow.data.experimental import AUTOTUNE
from contextlib import ExitStack

class TFRecord:

    def __init__(self, n_classes, image_shape=[224, 224, 3]):
        """
        TFRecord class to write images to TF REcords and load TF Records to tf.data

        Attributes:
            n_classes (int) representing number of unique classes to classify.
            image_shape (list of integers) representing the target image shape.
        """
        self.image_shape = image_shape
        self.n_classes = n_classes
        
    @tf.autograph.experimental.do_not_convert
    def _load_img(self, img_path, label):
        """
        Function to read in image path and returns an image.

        Args:
            img_path (string) representing path to an image.
            label (int) encoded label for image

        Returns:
            tuple: (image, label)
        """
        shape=self.image_shape
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=shape[2])
        image = tf.image.resize(image, [shape[0], shape[1]])
        image = tf.cast(image, tf.uint8)
        return (image, label)
    
    def _create_example(self, image, label):
        """
        Function to create an Example Protobuf.

        Args:
            image: an array of shape [height, width, depth]
            label: corresponding label for an image

        Returns:
            example: Tensorflow Protobufs
        """
        # serialize tensor
        image_data = tf.io.serialize_tensor(image)
        # create feature dictionary
        feature_dict={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(int64_list=Int64List(value=[label.numpy()])),
                }
        # return example
        return Example(
            features=Features(
                feature=feature_dict
            ))
    
    def compute_nshards(self, image_paths):
        """
        Function to compute number of chunks required to write TFRecords.

        Args:
            image_paths (list of strings): paths to images
        
        Returns:
            int: number of chunks (shards).

        Note: each shard's size will be between 150 - 200 apprx.
        """
        print("[INFO] Computing n_shards required...")
        total_image_size = 0
        for path in image_paths:
            image = cv2.imread(path)
            dim = (self.image_shape[0], self.image_shape[1])
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            total_image_size += image.nbytes / (1024 * 1024.0)

        n_shards = total_image_size // 150
        return int(n_shards)
    
    # create a function to save dataset in TFRecords format
    def write_tfrecords(self, image_paths, labels, save_path, n_shards, prefix="file"):
        """
        Function to write images to TF Records.

        Args:
            image_paths (list of strings): path to images
            labels (list of integers): labels for images
            save_path (str): path to save generated TF Records.
            n_shards (int): number os shards to be created.
            prefix (str): filename prefix. # optional

        Returns:
            list: generated filenames.
        """
        print("[INFO] {} shards will be created".format(n_shards))
        # create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self._load_img, num_parallel_calls=AUTOTUNE)
        
        # create directories if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = save_path + '/' + prefix

        paths = ["{}.tfrecord-{}-of-{}".format(file_path, index+1, n_shards)
                 for index in range(n_shards)]
        print("[INFO] Writing TFRecords...")
        with ExitStack() as stack:
            writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                       for path in paths]
            for index, (image, label) in dataset.enumerate():
                shard = index % n_shards
                example = self._create_example(image, label)
                writers[shard].write(example.SerializeToString())
            print("[INFO] Done!\n")
            return paths
        
    @tf.autograph.experimental.do_not_convert
    def _preprocess(self, tfrecord):
        """
        Function to parse Examples (TensorFlow Protobufs).

        Returns:
            tuple: (image, label)
        """
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }

        # parse a single example
        example = tf.io.parse_single_example(tfrecord, feature_description)
        image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
        height = self.image_shape[0]
        width = self.image_shape[1]
        image = tf.reshape(image, [height, width, 3])

        # one hot encode label
        label_oh = tf.one_hot(example["label"], depth=self.n_classes)

        return (image, label_oh)

    def parse_tfrecord(self, tfrecord_path, batch_size=32, shuffle_buffer_size=None):
        """
        Function to parse TFRecords.

        Args:
            tfrecord_path (str): path to directory where all TFRecords are stored.
            batch_size (int): batch size for dataset
            shuffle_buffer_size (int or None): buffer size to shuffle training data.

        Returns:
            tf.data: tensorflow dataset.
        """
        file_paths = os.listdir(tfrecord_path)
        file_paths = [os.path.join(tfrecord_path, path) for path in file_paths 
                       if not path.startswith(".")]

        dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=AUTOTUNE)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size=shuffle_buffer_size)

        dataset = dataset.map(self._preprocess, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

        return dataset