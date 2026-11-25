"""Training, loss evaluation and loss plots for the DAC on the TFRecord written by Synthesis.py.

    prepare       raw items -> batches of (image, targets, weights) keyed by the DAC's output names
    trainDAC      compile and fit; Keras records every loss per epoch (History, history.csv, TensorBoard)
    evaluateLoss  the same weighted losses on held-out data, computed by Keras (model.evaluate)
    plotHistory   train / validation loss curves per output, plus the learning rate, as a PNG

Live curves while training:  tensorboard --logdir runs/dac/logs   then open http://localhost:6006
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
import tf_keras

from Synthesis import for_model, pixel_squared_error

# Pixels near a marker get (1 + FOREGROUND_WEIGHT * target) times the weight of background pixels.
# Almost every heatmap pixel is 0, so without this the cheapest solution is to predict all zeros.
FOREGROUND_WEIGHT = 10.0

# Plot colours: two categorical series (train, validation), neutral inks for everything else
_TRAIN_COLOUR = "#257de0"
_VALIDATION_COLOUR = "#f47230"
_INK = "#191919"
_MUTED = "#666666"
_GRID = "#e7e7e7"


def makeDACMapper(model, foregroundWeight: float = FOREGROUND_WEIGHT):
    """for_model keys its targets 'heatmaps' and 'depths'. Keras matches targets to outputs by name,
    so re-key them with the model's own output names."""
    heatName, depthName = model.output_names

    def toDAC(item):
        image, targets, weights = for_model(item)
        heatWeights = weights["heatmaps"] * (1.0 + foregroundWeight * targets["heatmaps"])
        return (image,
                {heatName: targets["heatmaps"], depthName: targets["depths"]},
                {heatName: heatWeights, depthName: weights["depths"]})

    return toDAC


def prepare(rawDataset, model, batchSize: int = 16, shuffle: bool = False, shuffleBuffer: int = 256):
    """Each element is ~3 MB once mapped, so the shuffle buffer costs ~shuffleBuffer x 3 MB of RAM."""
    dataset = rawDataset.map(makeDACMapper(model), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(shuffleBuffer, reshuffle_each_iteration=True)
    return dataset.batch(batchSize).prefetch(tf.data.AUTOTUNE)


def trainDAC(model, trainData, valData, epochs: int = 100, learningRate: float = 1e-3,
             depthLossWeight: float = 1.0, runDir: str = "runs/dac"):
    """Returns Keras's History: history.history holds every loss for every epoch, e.g.
    'loss', 'opticalMarkerHeatmaps_loss', 'opticalMarkerDepths_loss' and the same with a 'val_' prefix."""
    os.makedirs(runDir, exist_ok=True)
    heatName, depthName = model.output_names
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learningRate),
        loss={heatName: pixel_squared_error, depthName: pixel_squared_error},
        # Depth is in raw sensor units and heatmaps are in [0, 1]: balance them with this weight
        loss_weights={heatName: 1.0, depthName: depthLossWeight},
    )
    callbacks = [
        tf_keras.callbacks.ModelCheckpoint(os.path.join(runDir, "best.h5"), monitor="val_loss",
                                           save_best_only=True, save_weights_only=True),
        tf_keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        tf_keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf_keras.callbacks.CSVLogger(os.path.join(runDir, "history.csv")),      # the losses, for replotting later
        tf_keras.callbacks.TensorBoard(log_dir=os.path.join(runDir, "logs")),    # live plots in TensorBoard
    ]
    history = model.fit(trainData, validation_data=valData, epochs=epochs, callbacks=callbacks)
    model.save_weights(os.path.join(runDir, "dac.h5"))
    return history


def evaluateLoss(model, rawDataset, batchSize: int = 16) -> dict:
    """The same weighted losses as in training, computed by Keras on held-out data (e.g. the test session).
    Returns e.g. {'loss': ..., 'opticalMarkerHeatmaps_loss': ..., 'opticalMarkerDepths_loss': ...}."""
    results = model.evaluate(prepare(rawDataset, model, batchSize=batchSize), return_dict=True, verbose=0)
    print("test: " + " | ".join(f"{name} {value:.5g}" for name, value in results.items()))
    return results


def plotHistory(source, outPath: str, testLoss: dict | None = None) -> str:
    """One panel per loss (total, then each output), train vs validation, plus the learning rate.
    source: the History returned by trainDAC, its .history dict, or the path to history.csv.
    testLoss: optional dict from evaluateLoss, drawn as a dotted reference line in each panel."""
    if isinstance(source, (str, os.PathLike)):
        import pandas
        history = pandas.read_csv(source).to_dict(orient="list")
    else:
        history = getattr(source, "history", source)

    lossKeys = ["loss"] + [k for k in history if k.endswith("_loss") and not k.startswith("val_")]
    lrKey = next((k for k in ("lr", "learning_rate") if k in history), None)
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1 if "val_loss" in history else None

    panels = len(lossKeys) + (1 if lrKey else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.3 * panels, 3.7), constrained_layout=True, squeeze=False)
    axes = axes[0]

    def style(ax, title):
        ax.set_title(title, color=_INK, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=_MUTED, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c8c8c8")
        ax.tick_params(colors=_MUTED, labelsize=8)

    for ax, key in zip(axes, lossKeys):
        train = np.asarray(history[key], dtype=float)
        ax.plot(epochs, train, color=_TRAIN_COLOUR, linewidth=2, label="train")
        values = [train]
        if f"val_{key}" in history:
            validation = np.asarray(history[f"val_{key}"], dtype=float)
            ax.plot(epochs, validation, color=_VALIDATION_COLOUR, linewidth=2, label="validation")
            values.append(validation)
            if best is not None:
                ax.plot([best], [validation[best - 1]], "o", markersize=7, color=_VALIDATION_COLOUR,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        if testLoss and key in testLoss:
            ax.axhline(testLoss[key], color=_MUTED, linewidth=1.2, linestyle=":")
            ax.annotate("test", (epochs[-1], testLoss[key]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=_MUTED, annotation_clip=False)
            values.append(np.array([testLoss[key]]))
        if np.concatenate(values).min() > 0:
            ax.set_yscale("log")  # losses fall over orders of magnitude
        name = "Total" if key == "loss" else key[:-len("_loss")].replace("opticalMarker", "")
        style(ax, f"{name} loss")

    if best is not None:
        axes[0].annotate(f"best epoch ({best})", (best, history["val_loss"][best - 1]), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=8, color=_MUTED)

    if lrKey:
        ax = axes[-1]
        ax.plot(epochs, history[lrKey], color=_MUTED, linewidth=2)
        ax.set_yscale("log")
        style(ax, "Learning rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=len(labels), frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(outPath) or ".", exist_ok=True)
    fig.savefig(outPath, dpi=200, facecolor="white")
    plt.close(fig)
    return outPath

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info and warning messages

from DAC import DAC
from Synthesis import load_dataset
import tensorflow

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

# Whole sessions are held out: neighbouring frames are near-duplicates, so splitting frames at random
# would put near-copies of the test frames in the training set
_VALIDATION_SESSION = "session_002"   # picks the best epoch
_TEST_SESSION       = "session_003"   # touched once, at the end


dac = DAC(
    opticalMarkerAmount = 12,
    inputShape          = (256, 256, 1),
    widthMultiplier     = 0.5,
    encoderSettings     = _ENCODER_CONFIGS,
    decoderSettings     = _DECODER_CONFIGS,
    stemChannels        = 32
)

from Auxiliaries import Visualisations

Visualisations.drawArchitecture(
    model  = dac,
    outDir = "/home/ben/PycharmProjects/DAC/"
)

data = load_dataset(
    path = "marker_frames.tfrecord"
)
trainRaw = data.filter(
    lambda item : tensorflow.logical_and(
        tensorflow.not_equal(item["session"], _VALIDATION_SESSION),
        tensorflow.not_equal(item["session"], _TEST_SESSION)
    )
)
validationRaw = data.filter(
    lambda item : tensorflow.equal(item["session"], _VALIDATION_SESSION)
)
testRaw = data.filter(
    lambda item : tensorflow.equal(item["session"], _TEST_SESSION)
)

history = trainDAC(
    model           = dac,
    trainData       = prepare(trainRaw, dac, batchSize = 16, shuffle = True),
    valData         = prepare(validationRaw, dac, batchSize = 16),
    epochs          = 100,
    learningRate    = 1e-3,
    depthLossWeight = 1.0,
    runDir          = "runs/dac"
)

testLoss = evaluateLoss(
    model      = dac,
    rawDataset = testRaw
)

plotHistory(
    source   = history,
    outPath  = "runs/dac/loss.png",
    testLoss = testLoss
)
