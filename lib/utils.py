import os
import yaml
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def unpack_orb_descriptor(desc):
    # desc: [N, 32]，dtype=torch.uint8
    N = desc.shape[0]
    desc = desc.unsqueeze(-1)                             # [N, 32, 1]
    bits = torch.bitwise_right_shift(desc, 7 - torch.arange(8, device=desc.device))  # [N, 32, 8]
    bits = torch.bitwise_and(bits, 1).view(N, 256).float()  # [N, 256]
    bits = bits * 2.0 - 1.0
    return bits

def normalize_keypoints(keypoints, image_shape):
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps

# Morton LUT for 8-bit
_MORTON256 = np.zeros(256, dtype=np.uint32)
for i in range(256):
    x = i
    x = (x | (x << 4)) & 0x0F0F
    x = (x | (x << 2)) & 0x3333
    x = (x | (x << 1)) & 0x5555
    _MORTON256[i] = x

def morton16(x, y):
    """x, y: uint16 arrays"""
    xl =  x        & 0xFF
    xh = (x >> 8) & 0xFF
    yl =  y        & 0xFF
    yh = (y >> 8) & 0xFF

    lo = _MORTON256[xl] | (_MORTON256[yl] << 1)
    hi = _MORTON256[xh] | (_MORTON256[yh] << 1)

    return (hi << 16) | lo

def z_ordering_encode(kp_xy, desc, scale=1):
    kp_xy = np.asarray(kp_xy, dtype=np.float32)
    desc  = np.asarray(desc)

    u = (kp_xy[:, 0] * scale).astype(np.uint16)
    v = (kp_xy[:, 1] * scale).astype(np.uint16)

    morton = morton16(u, v)

    order = np.argsort(morton)

    return kp_xy[order], desc[order], order

def z_ordering_decode(desc_sorted, order):
    """
    desc_sorted: (N, D) descriptors after model
    order      : indices returned by encode()

    Return:
        desc_recovered in original order
    """
    desc_recovered = np.zeros_like(desc_sorted)
    desc_recovered[order] = desc_sorted
    return desc_recovered