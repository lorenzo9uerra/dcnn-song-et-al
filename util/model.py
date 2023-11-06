import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Dense,
    Activation,
    Concatenate,
    GlobalAveragePooling2D,
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score
from keras.optimizers import Adam
from util.data import load_data_road

def conv2d(x, numfilt, filtsz, strides=1, pad="same", act=True, name=None):
    x = Conv2D(
        numfilt,
        filtsz,
        strides,
        padding=pad,
        data_format="channels_last",
        use_bias=False,
        name=name + "conv2d",
    )(x)
    x = BatchNormalization(axis=3, scale=False, name=name + "conv2d" + "bn")(x)
    if act:
        x = Activation("relu", name=name + "conv2d" + "act")(x)
    return x

def incresA(x, name=None):
    pad = "same"
    branch0 = conv2d(x, 32, 1, 1, pad, True, name=name + "b0")
    branch1 = conv2d(x, 32, 1, 1, pad, True, name=name + "b1_1")
    branch1 = conv2d(branch1, 32, 3, 1, pad, True, name=name + "b1_2")
    branch2 = conv2d(x, 32, 1, 1, pad, True, name=name + "b2_1")
    branch2 = conv2d(branch2, 32, 3, 1, pad, True, name=name + "b2_2")
    branch2 = conv2d(branch2, 32, 3, 1, pad, True, name=name + "b2_3")
    branches = [branch0, branch1, branch2]
    mixed = Concatenate(axis=3, name=name + "_concat")(branches)
    final_lay = conv2d(mixed, 128, 1, 1, pad, False, name=name + "filt_exp_1x1")
    return final_lay

def incresB(x, name=None):
    pad = "same"
    branch0 = conv2d(x, 64, 1, 1, pad, True, name=name + "b0")
    branch1 = conv2d(x, 64, 1, 1, pad, True, name=name + "b1_1")
    branch1 = conv2d(branch1, 64, [1, 3], 1, pad, True, name=name + "b1_2")
    branch1 = conv2d(branch1, 64, [3, 1], 1, pad, True, name=name + "b1_3")
    branches = [branch0, branch1]
    mixed = Concatenate(axis=3, name=name + "_mixed")(branches)
    final_lay = conv2d(mixed, 448, 1, 1, pad, False, name=name + "filt_exp_1x1")
    return final_lay

def DCNN(seed_value = 42):
    # Set a seed value:
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)    

    img_input = Input(shape=(29, 29, 1))

    # Stem
    x = conv2d(img_input, 32, 3, 1, "same", True, name="conv1")
    x = conv2d(x, 32, 3, 1, "valid", True, name="conv2")

    x = MaxPooling2D(3, strides=2, padding="valid", name="maxpool1")(x)
    x = conv2d(x, 64, 1, 1, "same", True, name="conv3")

    x = conv2d(x, 128, 3, 1, "same", True, name="conv4")
    x = conv2d(x, 128, 3, 1, "same", True, name="conv5")

    # Inception-ResNet-A module
    x = incresA(x, name="incresA")

    # Reduction A module
    x_red_11 = MaxPooling2D(3, strides=2, padding="valid", name="red_maxpool_1")(x)

    x_red_12 = conv2d(x, 192, 3, 2, "valid", True, name="x_red1_c1")

    x_red_13 = conv2d(x, 96, 1, 1, "same", True, name="x_red1_c2_1")
    x_red_13 = conv2d(x_red_13, 96, 3, 1, "same", True, name="x_red1_c2_2")
    x_red_13 = conv2d(x_red_13, 128, 3, 2, "valid", True, name="x_red1_c2_3")

    x = Concatenate(axis=3, name="red_concat_1")([x_red_11, x_red_12, x_red_13])

    # Inception-ResNet-B module
    x = incresB(x, name="incresB")

    # Reduction B module
    x_red_21 = MaxPooling2D(3, 2, padding="valid", name="red_maxpool_2")(x)

    x_red_22 = conv2d(x, 128, 1, 1, "same", True, name="x_red2_c11")
    x_red_22 = conv2d(x_red_22, 192, 3, 2, "valid", True, name="x_red2_c12")

    x_red_23 = conv2d(x, 128, 1, 1, "same", True, name="x_red2_c21")
    x_red_23 = conv2d(x_red_23, 128, 3, 2, "valid", True, name="x_red2_c22")

    x_red_24 = conv2d(x, 128, 1, 1, "same", True, name="x_red2_c31")
    x_red_24 = conv2d(x_red_24, 128, 3, 1, "same", True, name="x_red2_c32")
    x_red_24 = conv2d(x_red_24, 128, 3, 2, "valid", True, name="x_red2_c33")

    x = Concatenate(axis=3, name="red_concat_2")([x_red_21, x_red_22, x_red_23, x_red_24])

    # TOP
    x = GlobalAveragePooling2D(data_format="channels_last")(x)
    x = Dropout(0.6, seed=seed_value)(x)
    x = Dense(1, activation="sigmoid")(x)
    
    return Model(img_input, x, name="inception_resnet_v2")