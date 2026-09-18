"""
Keypoint / descriptor extractors used to build DescPP training data.

Every extractor takes a BGR uint8 image and returns:
    keypoints   : np.float32 [N, K], (u, v, ...), K must match model.kp_dims
    descriptors : [N, D], packed uint8 for binary descriptors, float32 otherwise

To add a new feature:
    1. Implement an extractor class below and decorate it with @register("name").
    2. Add a matching entry under `features:` in config/train_config.yaml.
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTORS = {}


def register(name):
    def deco(cls):
        EXTRACTORS[name] = cls
        return cls
    return deco


def build_extractor(name, **kwargs):
    if name not in EXTRACTORS:
        raise KeyError(f"Unknown feature '{name}'. Available: {list(EXTRACTORS)}")
    return EXTRACTORS[name](**kwargs)


def _add_to_path(path):
    """Make a third-party module that uses absolute imports importable."""
    path = str(path)
    if path not in sys.path:
        sys.path.append(path)


def _empty(k, d, dtype=np.float32):
    return np.zeros((0, k), np.float32), np.zeros((0, d), dtype)


# Third-party dependencies are imported lazily so that only the
# packages required by the selected feature need to be installed.

@register("orb")
class ORBExtractor:
    """keypoints: (u, v, size / 31, angle [rad]); descriptors: packed uint8 [N, 32]."""

    def __init__(self, n_features=3000, scale_factor=1.2, n_levels=8):
        _add_to_path(Path(__file__).parent / "orbslam2_features/lib")
        from orbslam2_features import ORBextractor
        self.ext = ORBextractor(n_features, scale_factor, n_levels)

    def __call__(self, bgr):
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        kps_tuples, desc = self.ext.detectAndCompute(gray)
        if len(kps_tuples) == 0:
            return _empty(4, 32, np.uint8)
        kps = [cv.KeyPoint(*kp) for kp in kps_tuples]
        kps = np.array([[k.pt[0], k.pt[1], k.size / 31, np.deg2rad(k.angle)] for k in kps],
                       dtype=np.float32)
        return kps, np.asarray(desc, dtype=np.uint8)


@register("superpoint")
class SuperPointExtractor:
    """keypoints: (u, v, confidence); descriptors: float32 [N, 256]."""

    def __init__(self, weights_path="lib/SuperPointPretrainedNetwork/superpoint_v1.pth",
                 nms_dist=4, conf_thresh=0.015, nn_thresh=0.7):
        from .SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
        self.ext = SuperPointFrontend(weights_path=weights_path, nms_dist=nms_dist,
                                      conf_thresh=conf_thresh, nn_thresh=nn_thresh,
                                      cuda=torch.cuda.is_available())

    def __call__(self, bgr):
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.
        pts, desc, _ = self.ext.run(gray)  # pts: [3, N], desc: [256, N]
        if pts is None or pts.shape[1] == 0:
            return _empty(3, 256)
        return pts.T.astype(np.float32), desc.T.astype(np.float32)


@register("alike")
class ALIKEExtractor:
    """keypoints: (u, v, score); descriptors: float32 [N, 128]."""

    def __init__(self, model="alike-l", top_k=-1, scores_th=0.2):
        _add_to_path(Path(__file__).parent / "ALIKE")
        import alike
        self.ext = alike.ALike(**alike.configs[model], device=_DEVICE, top_k=top_k, scores_th=scores_th)

    def __call__(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        pred = self.ext(rgb, sub_pixel=True)
        kps = pred["keypoints"]
        if len(kps) == 0:
            return _empty(3, 128)
        kps = np.hstack([kps, pred["scores"][:, None]]).astype(np.float32)
        return kps, pred["descriptors"].astype(np.float32)


@register("sift")
class SIFTExtractor:
    """keypoints: (u, v, scale, orientation); descriptors: L2-normalized float32 [N, 128]."""

    def __init__(self, nfeatures=-1):
        from .dog import Dog
        self.ext = Dog(nfeatures=nfeatures)

    def __call__(self, bgr):
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.
        kps, _scores, desc = self.ext.detectAndCompute(gray)
        if len(kps) == 0:
            return _empty(4, 128)
        return np.asarray(kps, np.float32), np.asarray(desc, np.float32)