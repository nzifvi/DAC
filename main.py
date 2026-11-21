from DAC import DAC

_ENCODER_CONFIGS = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1)
]
_DECODER_CONFIGS = [
    (6, 64, 3, 1),
    (6, 32, 3, 2),
    (6, 24, 2, 2),
    (6, 16, 1, 2)
]


dac = DAC(
    opticalMarkerAmount = 12,
    inputShape          = (256, 256, 1),
    widthMultiplier     = 0.5,
    encoderSettings     = _ENCODER_CONFIGS,
    decoderSettings     = _DECODER_CONFIGS,
    stemChannels        = 32
)

