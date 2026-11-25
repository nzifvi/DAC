import os
import argparse
import json
from pathlib import Path
import cv2
import numpy
import time

def _loadParams(name:str) -> dict:
    with open(os.path.join("../configs/", name + ".json"), "r") as f:
        params = json.load(f)
        return params

def _saveParams(name:str, params:dict)

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

        if self._checkCalibration():
            print("! RealSense Intrinsics uncalibrated. Calibration beginning\n    - Main thread will sleep for 1m. Ensure RealSense camera is plugged in before 1m period ends.")
            time.sleep(60)
            intrinsics = self._readIntrinsicsFromDevice()
            self._save(
                path = "../configs/" + name + "Params.json",
            )

    def _checkCalibration(self):
        return not (numpy.isnan(self._f).any() or numpy.isnan(self._c).any())

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

    def _save(self, path):


class OV5640Camera:
    def __init__(self, K, D, maxTheta, rmsPx, imagesUsed = none, name = "ov5640"):
        ...

class CanonicalCamera:
    def __init__(self, canonicalParams, imageParams):