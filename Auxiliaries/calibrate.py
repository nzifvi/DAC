import os
import argparse
import json
from pathlib import Path
import cv2
import numpy
import time

def _loadParams(name:str) -> dict:
    configDir = Path(__file__).resolve().parent.parent / "configs"
    with open(configDir / f"{name}.json", "r") as f:
        return json.load(f)


def _saveParams(name:str, params:dict):
    configDir = Path(__file__).resolve().parent.parent / "configs"

    width = max(len(json.dumps(key)) for key in params)
    lines = [f" {json.dumps(key):<{width}} : {json.dumps(value)}" for key, value in params.items()]
    temp  = configDir.with_suffix(".json.temp")
    with open(temp, "w") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")
    os.replace(temp, configDir)

class RealSenseCamera:
    def __init__(self, realSenseParams, name = "realSense"):
        params = _loadParams(name + "Params")

        self._name = name

        self._imageShape          = tuple(int(dim) for dim in params["imageShape"])
        self._recordedColourShape = tuple(int(dim) for dim in params["recordedColourShape"])
        self._recordedDepthShape  = tuple(int(dim) for dim in params["recordedDepthShape"])
        self._recordedFPS         = int(params["recordedFPS"])
        self._f                   = tuple(float(dim) for dim in params["f"])
        self._c                   = tuple(float(dim) for dim in params["c"])
        self._depthUnitMm         = float(params["depthUnitMm"])
        self._distortion          = [float(k) for k in params["distortion"]]
        self._distortionModel     = params["distortionModel"] if params["distortionModel"] != "NAN" else None

        if not self._checkCalibration():
            print("! RealSense intrinsics uncalibrated. Calibration beginning\n"
                  "    - Waiting up to 1m for the RealSense camera to be plugged in.")
            self._waitForDevice(timeoutS=60)
            self._readIntrinsicsFromDevice()
            self._save()

    def _checkCalibration(self) -> bool:
        return not numpy.isnan([*self._f, *self._c, self._depthUnitMm]).any()

    def _waitForDevice(self, timeoutS: float = 60.0) -> None:
        import pyrealsense2 as rs

        deadline = time.monotonic() + timeoutS
        while len(rs.context().query_devices()) == 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"No RealSense connected within {timeoutS:g} s")
            time.sleep(0.5)

    def _readIntrinsicsFromDevice(self):
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config   = rs.config()

        config.enable_stream(
            rs.stream.color,
            *self._recordedColourShape,
            rs.format.bgr8,
            self._recordedFPS
        )
        config.enable_stream(
            rs.stream.depth,
            *self._recordedDepthShape,
            rs.format.z16,
            self._recordedFPS
        )
        profile = pipeline.start(config)

        try:
            intr       = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            depthScale = profile.get_device().first_depth_sensor().get_depth_scale()
        finally:
            pipeline.stop()

        w, h = self._imageShape
        sx, sy = w / intr.width, h / intr.height

        self._f               = (intr.fx * sx, intr.fy * sy)
        self._c               = ((intr.ppx + 0.5) * sx - 0.5, (intr.ppy + 0.5) * sy - 0.5)
        self._depthUnitMm     = depthScale * 1000.0          # the SDK gives metres per raw depth unit
        self._distortion      = [float(k) for k in intr.coeffs]
        self._distortionModel = str(intr.model)

    def _save(self) -> None:
        """Write the intrinsics into configs/<name>Params.json. The file is read first and only these keys
        change, so the recording settings and any keys of your own stay as they are."""
        params = _loadParams(self._name + "Params")
        params.update({
            "f": list(self._f),
            "c": list(self._c),
            "depthUnitMm": self._depthUnitMm,
            "distortion": self._distortion,
            "distortionModel": self._distortionModel,
        })
        _saveParams(self._name + "Params", params)

class OV5640Camera:
    def __init__(self, K, D, maxTheta, rmsPx, imagesUsed = none, name = "ov5640"):
        ...

class CanonicalCamera:
    def __init__(self, canonicalParams, imageParams):