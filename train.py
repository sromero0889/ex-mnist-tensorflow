import tensorflow as tf
from ds_ops import load_mnist_dataset
from model_ops import load_model, save_model, ModelFrmt
from model_def import MODEL_DEF, CNN_MODEL_DEF
import argparse


def save_trained_model(model: tf.keras.Model, model_def: MODEL_DEF):
    for frmt in list(ModelFrmt):
        save_model(frmt=frmt, model=model, model_def=model_def)


def train_model(model_def: MODEL_DEF, from_pretrained: bool, epochs: int, batch_size: int, save_freq: int) -> None:
    # Load MNIST Dataset
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset(source=0)
    # Load Model (for training we want to also load training history if exists)
    model: tf.keras.Model = load_model(frmt=ModelFrmt.SavedModel, model_def=model_def, from_pretrained=from_pretrained)

    # TODO split trainings & save checkpoints f(save_freq, epochs)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    save_trained_model(model=model, model_def=model_def)


parser = argparse.ArgumentParser(description='Train TF Model')
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--save_freq", type=int, required=True)
parser.add_argument("--load", action='store_true')
args = parser.parse_args()

s_freq = 0 if args.save_freq <= 0 else args.epochs if args.save_freq >= args.epochs else args.save_freq
train_model(
    model_def=CNN_MODEL_DEF,  # We only have 1 model, add argument to use more
    from_pretrained=args.load,
    epochs=args.epochs,
    batch_size=args.batch_size,
    save_freq=s_freq
)
