import os
import argparse
import json
from pathlib import Path
import cv2
import numpy

def _loadParams(name:str) -> dict:
    with open(os.path.join("../configs/", name + ".json"), "r") as f:
        params = json.load(f)
        return params

class RealSenseCamera:
    def __init__(self, realSenseParams, name = "realSense"):
        params = _loadParams(name + "Params")

        self._name = name

        self._imageShape          = tuple(int(dim) for dim in params["imageShape"])
        self._recordedColourShape = tuple(int(dim) for dim in params["recordedColourShape"])
        self._recordedDepthShape  = tuple(int(dim) for dim in params["recordedDepthShape"])
        self._recordedFPS         = int(params["recordedFPS"])

    def loadIntrinsics(self, path:str):
        ...

    def readIntrinsicsFromDevice(self):
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



class OV5640Camera:
    def __init__(self, K, D, maxTheta, rmsPx, imagesUsed = none, name = "ov5640"):
        ...

class CanonicalCamera:
    def __init__(self, canonicalParams, imageParams):