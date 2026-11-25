import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as tf_k
from DAC import InvertedResidualBlock
from Auxiliaries.log import logRun
from Auxiliaries.plots import loadRuns, plotRuns
from ai_edge_litert.interpreter import Interpreter
from pathlib import Path
tf_k.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

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
# A conv which takes in 1 channels to 16 channels
# Batch nomed with x_hat = (gamma elmenet wise multiply (x - mean)/ std) + betaaaa
# gamma is the learnt scale and bettaaaa is a learnt shift from that distirbution i think you know this already
# relu 6 is the min(max(0, x), 6)
z = InvertedResidualBlock(inputChannels=16, outputChannels=16, expansionFactor=4,
                          stride=1, transposed=False, name="block0")(z)   # The imported D*C model

out = tf_k.Sequential([
    tf_k.layers.GlobalAveragePooling2D(name="pool"),
    tf_k.layers.Dense(10, name="logits"),
], name="head")(z)

model = tf_k.Model(x, out, name="miniDAC")
# model.summary()

if __name__ == "__main__":
    # 1. train + float baseline
    model.compile(
        optimizer=tf_k.optimizers.Adam(1e-3),
        loss=tf_k.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    tb = tf_k.callbacks.TensorBoard(log_dir=str(RESULTS / "logs"))
    history = model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
    floatLoss, floatAcc = model.evaluate(x_test, y_test, verbose=0)
    print(f"float32 test accuracy: {floatAcc:.4f}")

    # 2. calibration indices
    rng = np.random.default_rng(0)
    calibIdx = rng.choice(len(x_train), size=200, replace=False)

    # 3. calibration data + convert to full int8
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([x_train[i:i+1]] for i in calibIdx)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tfliteInt8 = converter.convert()
    (RESULTS / "miniDAC_int8.tflite").write_bytes(tfliteInt8)

    tfliteFloat = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    print(f"float32: {len(tfliteFloat)/1024:.1f} KB | int8: {len(tfliteInt8)/1024:.1f} KB")

    # 4. run the int8 model and measure accuracy
    interpreter = Interpreter(model_content=tfliteInt8)
    interpreter.allocate_tensors()
    inputDetails  = interpreter.get_input_details()[0]
    outputDetails = interpreter.get_output_details()[0]
    inScale, inZeroPoint = inputDetails["quantization"]      # the converter's Δ and z for the input

    correct = 0
    for i in range(len(x_test)):
        q = np.clip(np.round(x_test[i:i+1] / inScale + inZeroPoint), -128, 127).astype(np.int8)
        interpreter.set_tensor(inputDetails["index"], q)
        interpreter.invoke()
        logits = interpreter.get_tensor(outputDetails["index"])
        correct += int(np.argmax(logits) == y_test[i])
    int8Acc = correct / len(x_test)

    drop = floatAcc - int8Acc
    print(f"int8 test accuracy: {int8Acc:.4f} | drop: {drop*100:.2f} pts | within 1%: {drop <= 0.01}")

    # 5. log the run
    logRun({
        "config":  {"epochs": 3, "batch_size": 128, "calib_size": 200, "seed": 0},
        "float32": {"test_accuracy": float(floatAcc), "test_loss": float(floatLoss),
                    "size_kb": len(tfliteFloat) / 1024},
        "int8":    {"test_accuracy": int8Acc, "size_kb": len(tfliteInt8) / 1024},
        "drop": float(drop),
        "within_budget": bool(drop <= 0.01),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    })
    print(f"saved {plotRuns(loadRuns()[-1:])}")
