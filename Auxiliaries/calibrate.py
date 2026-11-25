import os
import argparse
import json
from pathlib import Path
import cv2
import numpy

def loadCanonicalParameters(data) -> dict:
