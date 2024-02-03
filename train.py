import tensorflow as tf
from ds_ops import load_mnist_dataset
from model_ops import load_model, save_model
from model_def import MODEL_DEF, CNN_MODEL_DEF



#https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist#1.-How-many-convolution-subsambling-pairs?


def save_trained_model(model: tf.keras.Model, model_def: MODEL_DEF):
    fmts = [0,1,2]
    for frmt in fmts:
        save_model(frmt=frmt, model=model, model_def=model_def)



def train_model(model_def: MODEL_DEF, from_pretrained:bool, epochs: int, batch_size: int, save_freq:int) -> None:
    # Load MNIST Dataset
    (x_train, y_train), (x_test, y_test) = load_mnist_dataset(source=0)
    #
    model: tf.keras.Model = load_model(frmt=0, model_def=model_def, from_pretrained=from_pretrained)

    if save_freq < epochs:
        print("TODO")

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    save_trained_model(model=model, model_def=CNN_MODEL_DEF)








#print(model.get_weights())

#model2 = CNN()
#model2.build((None, 28, 28, 1))
#model2.set_weights(model.get_weights())