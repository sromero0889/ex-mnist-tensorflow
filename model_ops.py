import tensorflow as tf
from model_def import MODEL_DEF
from pathlib import Path
from enum import Enum


class ModelFrmt(Enum):
    SavedModel = 'keras',
    WeightsH5 = 'h5',
    SafeTensors = 'safetensors'


def save_model(frmt: ModelFrmt, model: tf.keras.Model, model_def: MODEL_DEF) -> None:
    model_dir: Path = Path("models").joinpath(model_def.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    match frmt:
        case ModelFrmt.SavedModel:
            print("SavedModel - Tensorflow")
            model.save(model_dir.joinpath(f"{model_def.name}.keras"))
        case ModelFrmt.WeightsH5:
            print("H5 weights")
            model.save_weights(model_dir.joinpath(f"{model_def.name}.weights.h5"))
        case ModelFrmt.SafeTensors:
            print("SafeTensors")
        case _:
            raise NotImplemented


def load_model(frmt: ModelFrmt, model_def: MODEL_DEF, from_pretrained: bool) -> tf.keras.Model:
    try:
        if from_pretrained:
            model_dir: Path = Path("models").joinpath(model_def.name)
            match frmt:
                case ModelFrmt.SavedModel:
                    print("SavedModel - Tensorflow")
                    return tf.keras.models.load_model(model_dir.joinpath(f"{model_def.name}.keras"))
                case ModelFrmt.WeightsH5:
                    print("H5 weights")
                    model = model_def.new()
                    model.build(model_def.input)
                    model.load_weights(model_dir.joinpath(f"{model_def.name}.weights.h5"))
                    return model
                case ModelFrmt.SafeTensors:
                    print("SafeTensors")
                case _:
                    raise NotImplemented
    except OSError as e:
        print("File Not found, create new instance: ", e)

    model = model_def.new()
    model.compile(optimizer=model_def.optimizer, loss=model_def.loss, metrics=model_def.metrics)
    model.build(model_def.input)
    return model
