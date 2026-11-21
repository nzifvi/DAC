import os
os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1"
)
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import tf_keras

_BLOCK_PATTERN = re.compile(
    r"^DAC(Encoder|Decoder)Block(\d+)"
)
_SKIP_PATTERN = re.compile(
    r"^DACDecoderSkip(\d+)$"
)
_HEAD_PREFIX = "opticalMarker"

_BLOCK_COLOUR = ("#CBE3F7", "#E3F0FB", "#A3CBEC", "#1D5F99")
_COLOURS = {kind : _BLOCK_COLOUR for kind in ("input", "stem", "encoder", "decoder", "head")}
_TEXT = "#2C2C2A"
_MUTED = "#6B6962"

def _macs(layer) -> int:
    if isinstance(layer, (tf_keras.layers.Conv2D, tf_keras.layers.DepthwiseConv2D)):
        _, h, w, cOut = layer.output_shape
        k = layer.kernel_size[0] * layer.kernel_size[1]
        cIn = layer.input_shape[-1]
        perPixel = cIn if isinstance(layer, tf_keras.layers.DepthwiseConv2D) else cIn * cOut
        return h * w * k * perPixel
    return 0


def _params(layers) -> int:
    return sum(int(sum(math.prod(w.shape) for w in layer.trainable_weights)) for layer in layers)


def extractStages(model) -> tuple:
    layers = model.layers
    _, h, w, c = model.inputs[0].shape
    stages = [{"kind": "input", "name": "Input", "h": h, "w": w, "c": c, "info": "greyscale\nimage", "macs": 0,
               "params": 0}]

    stem = [layer for layer in layers if layer.name.startswith("stem")]
    _, h, w, c = stem[-1].output_shape
    stride = next(l.strides[0] for l in stem if isinstance(l, tf_keras.layers.Conv2D))
    stages.append({"kind": "stem", "name": "Stem", "h": h, "w": w, "c": c,
                   "info": f"conv 3×3\n{'↓' + str(stride) if stride > 1 else ''}",
                   "macs": sum(_macs(l) for l in stem), "params": _params(stem)})

    blocks = {}
    for layer in layers:
        match = _BLOCK_PATTERN.match(layer.name)
        if match:
            blocks.setdefault((match.group(1), int(match.group(2))), []).append(layer)

    def describe(blockLayers):
        inputChannels = blockLayers[0].input_shape[-1]
        expansion = next((l for l in blockLayers if l.name.endswith("Expansion")), None)
        t = expansion.filters // inputChannels if expansion else 1
        down = any(isinstance(l, tf_keras.layers.DepthwiseConv2D) and l.strides[0] == 2 for l in blockLayers)
        up = any(isinstance(l, tf_keras.layers.UpSampling2D) for l in blockLayers)
        return t, down, up, blockLayers[-1].output_shape

    for kind in ("Encoder", "Decoder"):
        indices = sorted(i for k, i in blocks if k == kind)
        current = None
        for i in indices:
            blockLayers = blocks[(kind, i)]
            t, down, up, (_, h, w, c) = describe(blockLayers)
            if current is None or down or up or c != current["c"]:
                current = {"kind": kind.lower(), "h": h, "w": w, "c": c, "t": t, "n": 0,
                           "resample": "↓2" if down else "↑2" if up else "", "blocks": [], "macs": 0, "params": 0}
                stages.append(current)
            current["n"] += 1
            current["blocks"].append(i)
            current["macs"] += sum(_macs(l) for l in blockLayers)
            current["params"] += _params(blockLayers)

    number = {"encoder": 0, "decoder": 0}
    for stage in stages:
        if stage["kind"] in number:
            number[stage["kind"]] += 1
            stage["name"] = f"{'Enc' if stage['kind'] == 'encoder' else 'Dec'} {number[stage['kind']]}"
            plural = "s" if stage["n"] > 1 else ""
            stage["info"] = f"{stage['n']} IR block{plural}\n{stage['resample']}"

    skips = []
    for layer in layers:
        match = _SKIP_PATTERN.match(layer.name)
        if match:
            blockIndex, h = int(match.group(1)), layer.output_shape[1]
            target = next(s for s in stages if s["kind"] == "decoder" and blockIndex in s["blocks"])
            source = [s for s in stages if s["kind"] == "encoder" and s["h"] == h][-1]
            skips.append((stages.index(source), stages.index(target)))
            target["params"] += _params([l for l in layers if l.name == f"DACDecoderSkipProjection{blockIndex}"])

    heads = []
    for layer in layers:
        if layer.name.startswith(_HEAD_PREFIX):
            _, h, w, c = layer.output_shape
            heads.append({"kind": "head", "name": layer.name[len(_HEAD_PREFIX):], "h": h, "w": w, "c": c,
                          "info": "conv 1×1", "macs": _macs(layer), "params": _params([layer])})
    return stages, skips, heads


def drawArchitecture(model, outDir: str, fileStem: str = "dac_architecture", formats: tuple = ("png", "svg")) -> list:
    stages, skips, heads = extractStages(model)

    size = lambda h: 0.42 * math.log2(h)
    thickness = lambda c: 0.16 + 0.065 * math.sqrt(c)
    ox, oy = 0.42, 0.30
    gap = 0.62

    fig, ax = plt.subplots(figsize=(17, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")

    def prism(x, h, w, c, kind):
        front, top, side, edge = _COLOURS[kind]
        height, width, depth = size(h), thickness(c), 0.5 * size(w)
        y0 = -height / 2
        dx, dy = ox * depth, oy * depth
        style = dict(edgecolor=edge, linewidth=0.8, joinstyle="round")
        ax.add_patch(Polygon([(x, y0 + height), (x + dx, y0 + height + dy), (x + width + dx, y0 + height + dy),
                              (x + width, y0 + height)], facecolor=top, **style))
        ax.add_patch(Polygon([(x + width, y0), (x + width + dx, y0 + dy), (x + width + dx, y0 + height + dy),
                              (x + width, y0 + height)], facecolor=side, **style))
        ax.add_patch(Rectangle((x, y0), width, height, facecolor=front, **style))
        return {"left": x, "right": x + width + dx, "top": y0 + height + dy, "centre": x + (width + dx) / 2,
                "frontRight": x + width}

    def arrow(x0, y0, x1, y1, color=_MUTED, **kwargs):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color,
                                     linewidth=0.9, shrinkA=0, shrinkB=0, **kwargs))

    labelY = -size(max(s["h"] for s in stages)) / 2 - 0.35
    boxes, x = [], 0.0
    for i, stage in enumerate(stages):
        box = prism(x, stage["h"], stage["w"], stage["c"], stage["kind"])
        boxes.append(box)
        if i > 0:
            arrow(boxes[i - 1]["right"] + 0.04, 0, box["left"] - 0.04, 0)
        ax.text(box["centre"], labelY, stage["name"], ha="center", va="top", fontsize=9, fontweight="bold",
                color=_TEXT)
        ax.text(box["centre"], labelY - 0.32, f"{stage['h']}×{stage['w']}×{stage['c']}", ha="center", va="top",
                fontsize=8, color=_TEXT)
        ax.text(box["centre"], labelY - 0.6, stage["info"], ha="center", va="top", fontsize=7, color=_MUTED,
                linespacing=1.35)
        x = box["right"] + gap

    last = boxes[-1]
    headX = x
    spacing = size(heads[0]["h"]) + 0.9
    headOutputs = []
    lowestHead = 0.0
    for k, head in enumerate(heads):
        yOffset = (k - (len(heads) - 1) / 2) * -spacing
        height, width, depth = size(head["h"]), thickness(head["c"]), 0.5 * size(head["w"])
        y0 = yOffset - height / 2
        front, top, side, edge = _COLOURS["head"]
        dx, dy = ox * depth, oy * depth
        style = dict(edgecolor=edge, linewidth=0.8)
        ax.add_patch(Polygon([(headX, y0 + height), (headX + dx, y0 + height + dy),
                              (headX + width + dx, y0 + height + dy), (headX + width, y0 + height)],
                             facecolor=top, **style))
        ax.add_patch(Polygon([(headX + width, y0), (headX + width + dx, y0 + dy),
                              (headX + width + dx, y0 + height + dy), (headX + width, y0 + height)],
                             facecolor=side, **style))
        ax.add_patch(Rectangle((headX, y0), width, height, facecolor=front, **style))
        arrow(last["right"] + 0.04, 0, headX - 0.04, yOffset)
        centreX = headX + (width + dx) / 2
        label = f"{head['h']}×{head['w']}×{head['c']} · {head['info']}"
        if yOffset >= 0:
            ax.text(centreX, y0 + height + dy + 0.42, head["name"], ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=_TEXT)
            ax.text(centreX, y0 + height + dy + 0.1, label, ha="center", va="bottom", fontsize=8, color=_TEXT)
        else:
            ax.text(centreX, y0 - 0.1, head["name"], ha="center", va="top", fontsize=9, fontweight="bold",
                    color=_TEXT)
            ax.text(centreX, y0 - 0.42, label, ha="center", va="top", fontsize=8, color=_TEXT)
        headOutputs.append((headX + width + dx, yOffset))
        lowestHead = min(lowestHead, y0)
    rightEdge = headX + width + dx + 1.1

    top = max(b["top"] for b in boxes)
    for level, (source, target) in enumerate(sorted(skips, key=lambda st: st[1] - st[0])):
        y = top + 0.35 + 0.38 * level
        sx, tx = boxes[source]["centre"], boxes[target]["centre"]
        ax.plot([sx, sx, tx], [boxes[source]["top"] + 0.05, y, y], color=_COLOURS["decoder"][3], linewidth=1.0,
                linestyle=(0, (4, 2.5)))
        arrow(tx, y, tx, boxes[target]["top"] + 0.22, color=_COLOURS["decoder"][3], linestyle=(0, (4, 2.5)))
        ax.text(tx, boxes[target]["top"] + 0.12, "⊕", ha="center", va="center", fontsize=11,
                color=_COLOURS["decoder"][3])
    skipTop = top + 0.35 + 0.38 * max(len(skips) - 1, 0)

    os.makedirs(outDir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(outDir, f"{fileStem}.{fmt}")
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
