import os
os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1"
)

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
                    name     = f"{name}ExpansionBatchNorm"
                ),
                tf_keras.layers.ReLU(
                    6.0,
                    name = f"{name}ExpansionReLU"
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
                    strides   = stride,
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

class DAC(tf_keras.Model):
    def __init__(self, opticalMarkerAmount:int, inputShape:tuple, widthMultiplier:float, encoderSettings:list, decoderSettings, stemChannels:int):
        def width(channels):
            return makeDivisible(
                value   = channels * widthMultiplier,
                divisor = 8
            )

        x = tf_keras.Input(
            shape = inputShape,
            name  = "x"
        )

        z = tf_keras.layers.Conv2D(
            width(stemChannels),
            kernel_size = 3,
            strides     = 2,
            padding     = "same",
            use_bias    = False,
            name        = "stemConv2D"
        )(x)
        z = tf_keras.layers.BatchNormalization(
            momentum = _BN_MOMENTUM,
            epsilon  = _BN_EPSILON,
            name     = "stemBatchNorm"
        )(z)
        z = tf_keras.layers.ReLU(
            6.0,
            name = "stemReLU"
        )(z)
        channels = width(stemChannels)

        skips = {}
        index = 0
        for expansionFactor, outputChannels, repeats, stride in encoderSettings:
            for i in range(repeats):
                block = InvertedResidualBlock(
                    inputChannels   = channels,
                    outputChannels  = width(outputChannels),
                    expansionFactor = expansionFactor,
                    stride          = stride if i == 0 else 1,
                    name            = f"DACEncoderBlock{index}",
                    transposed      = False
                )
                z = block(z)
                channels = width(outputChannels)
                index += 1
            skips[z.shape[1]] = z

        index = 0
        for expansionFactor, outputChannels, repeats, stride in decoderSettings:
            for i in range(repeats):
                upsample = i == 0 and stride == 2
                block = InvertedResidualBlock(
                    inputChannels   = channels,
                    outputChannels  = width(outputChannels),
                    expansionFactor = expansionFactor,
                    stride          = 2 if upsample else 1,
                    transposed      = True,
                    name            = f"DACDecoderBlock{index}",
                )
                z = block(z)
                channels = width(outputChannels)
                if upsample and z.shape[1] in skips:
                    skip = skips[z.shape[1]]
                    if skip.shape[-1] != channels:
                        skip = tf_keras.layers.Conv2D(
                            channels,
                            kernel_size = 1,
                            name        = f"DACDecoderSkipProjection{index}"
                        )(skip)
                    z = tf_keras.layers.Add(
                        name = f"DACDecoderSkip{index}"
                    )([z, skip])
                index += 1

        heatmaps = tf_keras.layers.Conv2D(
            opticalMarkerAmount,
            kernel_size = 1,
            name = "opticalMarkerHeatmaps"
        )(z)
        depths   = tf_keras.layers.Conv2D(
            opticalMarkerAmount,
            kernel_size = 1,
            name = "opticalMarkerDepths"
        )(z)

        super().__init__(
            inputs  = x,
            outputs = [heatmaps, depths],
            name    = "DAC"
        )

def makeDivisible(value:float, divisor:int) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded

