import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as tf_k
from DAC import InvertedResidualBlock

## Toy Dataset : FashionMNIST because I like cloths.

(x_train, y_train), (x_test, y_test) = tf_k.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test  = x_test[..., None]

# Mini DAC built for the inverted Res Block
x = tf_k.Input(shape=(28, 28, 1), name="x")
z = tf_k.Sequential([
    tf_k.layers.Conv2D(16, kernel_size=3, strides=2, padding="same", use_bias=False, name="stemConv2D"),
    tf_k.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="stemBatchNorm"),
    tf_k.layers.ReLU(6.0, name="stemReLU")
], name="stem")(x)

# x is the image from FashionMNIST
# z is a hidden tensor, and it passes through
# A conv which takes in 16 channels
# Batch nomed with x_hat = (gamma elmenet wise multiply (x - mean)/ std) + betaaaa
# gamma is the learnt scale and bettaaaa is a learnt shift from that distirbution i think you know this already
# relu 6 is the min(max(0, x), 6)
z = InvertedResidualBlock(inputChannels=16, outputChannels=16, expansionFactor=4,
                          stride=1, transposed=False, name="block0")(z)   # stays as it is

out = tf_k.Sequential([
    tf_k.layers.GlobalAveragePooling2D(name="pool"),
    tf_k.layers.Dense(10, name="logits"),
], name="head")(z)

model = tf_k.Model(x, out, name="miniDAC")
model.summary()








if __name__ == "__main__":
    print(x_train.shape, x_train.min(), x_train.max())
    print(x_test.shape,  x_test.min(),  x_test.max())
