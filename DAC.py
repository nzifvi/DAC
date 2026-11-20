import os
os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1"
)

import numpy
import tensorflow
import tensorflow_model_optimization
import tf_keras

_BN_MOMENTUM = 0.9
_BN_EPSILON  = 1e-5

class InvertedResidualBlock:
    def __init__(self, inputChannels:int, outputChannels:int, expansionFactor:int, stride:int, transposed:bool, name:str):
        hiddenChannels = inputChannels * expansionFactor
        self._useResidual = stride == 1 and inputChannels == outputChannels

        layers = []
        if expansionFactor != 1:
            layers.extend([
                tf_keras.layers.Conv2D(
                    hiddenChannels,
                    kernel_size = 1,
                    use_bias    = False,
                    name        = f"{name}Expansion"
                ),
                tf_keras.layers.BatchNormalization(
                    momentum = _BN_MOMENTUM,
                    epsilon  = _BN_EPSILON,
                    name     = f"{name}BNExpansion"
                ),
                tf_keras.layers.ReLU(
                    6.0,
                    name = f"{name}ReLUExpansion"
                )
            ])

        if stride == 2 and transposed:
            layers.extend([
                tf_keras.layers.UpSampling2D(
                    2,
                    interpolation = "nearest",
                    name          = f"{name}UpSampling"
                ),
                tf_keras.layers.DepthwiseConv2D(
                    3,
                    padding  = "same",
                    use_bias = False,
                    name     = f"{name}DepthwiseConv2d"
                )
            ])
        else:
            layers.extend([
                tf_keras.layers.DepthwiseConv2D(
                    3,
                    stride   = stride,
                    padding  = "same",
                    use_bias = False,
                    name     = f"{name}DepthwiseConv2d"
                )
            ])
        layers.extend([
            tf_keras.layers.BatchNormalization(
                momentum = _BN_MOMENTUM,
                epsilon  = _BN_EPSILON,
                name     = f"{name}DepthwiseBatchNorm"
            ),
            tf_keras.layers.ReLU(
                6.0,
                name = f"{name}DepthwiseReLU"
            )
        ])

        layers.extend([
            tf_keras.layers.Conv2D(
                outputChannels,
                kernel_size = 1,
                use_bias    = False,
                name        = f"{name}Projection"
            ),
            tf_keras.layers.BatchNormalization(
                name = f"{name}ProjectionBatchNorm"
            )
        ])

        self._f = layers
        if self._useResidual:
            self._add = tf_keras.layers.Add(name = f"{name}Add")
        else:
            self._add = None

    def __call__(self, z):
        out = z
        for layer in self._f:
            out = layer(out)
        if self._useResidual:
            return self._add([z, out])
        else:
            return out

class DACMobileNetV2Encoder(tf_keras.Model):
    ...

class DACMobileNetV2Decoder(tf_keras.Model):
    ...