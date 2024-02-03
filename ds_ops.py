import tensorflow as tf
from datasets import load_dataset


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
