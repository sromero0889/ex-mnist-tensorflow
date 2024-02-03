from collections import namedtuple
import tensorflow as tf


# Replicate structure to add a new model DEF
class CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.c1 = tf.keras.layers.Conv2D(32, 5, padding="same", activation=tf.nn.relu, name="c1")
        self.p1 = tf.keras.layers.MaxPooling2D(name="p1")
        self.c2 = tf.keras.layers.Conv2D(64, 5, padding="same", activation=tf.nn.relu, name="c2")
        self.p2 = tf.keras.layers.MaxPooling2D(name="p2")
        self.f1 = tf.keras.layers.Flatten(name="f1")
        self.fc1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu, name="fc1")
        self.fc2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, name="fc2")

    def call(self, inputs, training=False):
        # Todo add dropout when training True
        return self.fc2(self.fc1(self.f1(self.p2(self.c2(self.p1(self.c1(inputs)))))))


MODEL_DEF = namedtuple('MODEL_DEF', ['name', 'new', 'input', 'optimizer', 'loss', 'metrics'])

# Replicate structure to add a new model DEF
CNN_MODEL_DEF = MODEL_DEF(
    'cnn',
    CNN,
    (None, 28, 28, 1),
    tf.keras.optimizers.Adam(),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
