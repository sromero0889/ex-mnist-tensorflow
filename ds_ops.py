from os import listdir
import tensorflow as tf
from datasets import load_dataset
from pathlib import Path
import re


def load_mnist_dataset(source) -> ((tf.Tensor, tf.Tensor), (tf.Tensor, tf.Tensor)):
    match source:
        case 0:
            print("Tensorflow Datasets")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

            x_train: tf.Tensor = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
            y_train: tf.Tensor = y_train.reshape(60000, 1)

            x_test: tf.Tensor = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
            y_test: tf.Tensor = y_test.reshape(10000, 1)
            return (x_train, y_train), (x_test, y_test)
        case 1:
            print("Hugging Face Datasets")
            dataset = load_dataset("mnist")
            return NotImplemented
        case _:
            return NotImplemented


def load_img(img_path: Path, target_size: (int, int), color_mode: str, channels: int) -> tf.Tensor:
    img = tf.convert_to_tensor(
        tf.keras.utils.load_img(
            img_path,
            color_mode=color_mode,
            target_size=target_size,
            interpolation='nearest',
            keep_aspect_ratio=True
        ),
        dtype=tf.float32
    )

    img = tf.reshape(img, (target_size[0], target_size[1], channels)) / 255
    return img


def get_label_from_name(filename: str) -> int:
    return int(re.findall(r'\d', filename)[-1])


def load_img_dir(img_folder: Path, target_size: (int, int), channels: int) -> (tf.Tensor, tf.Tensor):
    color_mode = 'grayscale' if channels == 1 else 'rgb'
    labels = []
    images = []
    for f in listdir(img_folder):
        labels.append(get_label_from_name(filename=f))
        images.append(load_img(img_path=img_folder.joinpath(f), target_size=target_size, color_mode=color_mode, channels=channels))
    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)
