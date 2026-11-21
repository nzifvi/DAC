from DAC import DAC, DACMobileNetV2Decoder, DACMobileNetV2Encoder

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

encoder = DACMobileNetV2Encoder(
    latentChannels    = 16,
    bottleneckConfigs = _ENCODER_CONFIGS,
    stemChannels      = 32,
    inputShape        = (256, 256)
),
decoder = DACMobileNetV2Decoder(
    latentChannels    = 16,
    bottleneckConfigs = _DECODER_CONFIGS,
    latentSize        = 4
)
dac = DAC(
    encoder = encoder,
    decoder = decoder,
)

