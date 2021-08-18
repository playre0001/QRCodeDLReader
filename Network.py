import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import learning_phase

import os

def CreateNetwork(max_string_length,image_size,loss,optimizer,metrics):
    """
    Create Deep Learning Network and compile

    Args:
        max_string_length: Max string lengh for generateing QRcode
        loss: loss function
        optimizer: optimizer
        metrics: metrics of learning

    Return: 
        tf.keras.Model
    """

    resnet=tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=image_size,weights=None)

    x=layers.GlobalAveragePooling2D()(resnet.output)
    x=layers.Dense(4096,kernel_initializer='he_normal',bias_initializer="random_normal")(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Dense(4096,kernel_initializer='he_normal',bias_initializer="random_normal")(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Dense(2048,kernel_initializer='he_normal',bias_initializer="random_normal")(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Dense(1024,kernel_initializer='he_normal',bias_initializer="random_normal")(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Dense(max_string_length,kernel_initializer='he_normal',bias_initializer="random_normal")(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)
    x=layers.Activation("sigmoid",name="sigmoid")(x)

    model=tf.keras.Model(inputs=resnet.input,outputs=x)

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return model

def LoadNetwork(model_path):
    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path)