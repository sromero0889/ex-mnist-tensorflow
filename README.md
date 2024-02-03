# Example MNIST Tensorflow
This example shows how to:
  - Define a CNN by subclassing the tf.keras.Model class
  - Train the model (from scratch or loading an existing model)
  - Visualize training history
  - Load & use a dataset from Tensorflow & Hugging Face
  - Load & Save models using SavedModel & SafeTensors formats
  - Inference

## Index
  - [Setup Virtual environment with venv](#virtual-environment--dependencies) 
  - [Training](#training) 
  - [Inference](#inference) 
  - [Visualization](#visualization) 
  - [Resources](#resources) 

## Virtual Environment & Dependencies

Create Virtual environment & install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Activate Virtual environment

```bash
source venv/bin/activate
```

Deactivate Virtual environment

```bash
deactivate
```

## Training

--save_freq=0 if you don't want to save the model

Training from Scratch
```bash
python train.py --epochs=10 --save_freq=10 --batch_size=64
```

Training loading previous version
```bash
python train.py --epochs=10 --save_freq=5 --load --batch_size=64
```

## Inference

Loads all images inside the selected folder and makes predictions.

*images inside sample's folder have different sizes*

```bash
python inference.py --img_folder="samples"
```

## Visualization


## Resources

* Datasets
  - [MNIST database](http://yann.lecun.com/exdb/mnist/)
  - [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets/index)
  - [Hugging Face MNIST Dataset](https://huggingface.co/datasets/mnist)
  - [Tensorflow MNIST Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)

* Save & Load Models
  - [SafeTensors](https://github.com/huggingface/safetensors)
  - [SavedModel Tensorflow](https://www.tensorflow.org/guide/keras/serialization_and_saving#how_to_save_and_load_a_model)

* Model Definition
  - [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)