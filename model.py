"""Можно использовать любые модели из application"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, mobilenet_v3, efficientnet


def get_model_builder(model_name):
    print('loading model {}'.format(model_name))
    build_model_function = globals()[model_name]
    return build_model_function


def get_model(model_name, image_size, num_classes, freezing):
    kwargs = {'weights': 'imagenet',
              'include_top': False,
              'input_shape': image_size + (3,),
              'pooling': 'avg'}

    if 'fficient' in model_name:
        model_function = get_model_builder('efficientnet')
    else:
        model_function = get_model_builder(model_name)

    base_model = getattr(model_function, model_name)(**kwargs)

    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # ])
    if freezing:
        print('Основная модель будет заморожена')
        for layer in base_model.layers:
            layer.trainable = False
    else:
        print('Основная модель будет разморожена')
        for layer in base_model.layers:
            layer.trainable = True

    inputs = tf.keras.Input(shape=image_size + (3,))
    # x = data_augmentation(inputs)
    x = base_model(inputs)
    # x = tf.keras.layers.Dense(512,
    #                          activation='relu',
    #                          kernel_initializer='he_normal',
    #                          kernel_regularizer=l2(0.01))(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return Model(inputs, outputs), model_function
