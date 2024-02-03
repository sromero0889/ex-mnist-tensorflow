import tensorflow as tf
from ds_ops import load_mnist_dataset
from model_ops import load_model, ModelFrmt
from model_def import MODEL_DEF, CNN_MODEL_DEF
import argparse
from pathlib import Path



def predict(model_def: MODEL_DEF, load_from:ModelFrmt, img_folder: Path) -> None:
    # Load Data for inference
    # Todo load & resize images from img_folder
    (_, _), (x_test, y_test) = load_mnist_dataset(source=0)
    input_shape = (1, model_def.input[1], model_def.input[2], model_def.input[3])

    input_data = x_test[0].reshape(input_shape)

    # Load Model
    model: tf.keras.Model = load_model(frmt=load_from, model_def=model_def, from_pretrained=True)
    res = model.predict(input_data)
    predictions = tf.math.argmax(res, 1)

    print(f"Prediction: {predictions[0]}  - Result: {y_test[0]}")


parser = argparse.ArgumentParser(description='Inference with TF Model')
parser.add_argument("--load_from", type=ModelFrmt, choices=list(ModelFrmt), default=ModelFrmt.SavedModel)
parser.add_argument("--img_folder", type=str, default="samples")
args = parser.parse_args()

predict(
    model_def=CNN_MODEL_DEF,  # We only have 1 model, add argument to use more
    load_from=args.load_from,
    img_folder=args.img_folder
)
