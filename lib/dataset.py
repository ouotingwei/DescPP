"""
MegaDepth image-pair dataset for DescPP training.

Adapted from FeatureBooster (https://github.com/SJTU-ViSYS/FeatureBooster).
"""
import os

import cv2 as cv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import normalize_keypoints, z_ordering_encode

# Keypoint coordinates are scaled by ORDER_SCALE and cast to uint16 before
# Z-order sorting, so the image side length must be smaller than 65536 / ORDER_SCALE.
ORDER_SCALE = 64


def add_random_motion_blur(img, kernel_size=15):
    """Apply a random directional motion blur. `kernel_size=None` disables it."""
    if not kernel_size:
        return img
    ksize = np.random.randint(0, (kernel_size + 1) // 2) * 2 + 1
    center = (ksize - 1) // 2
    kernel = np.zeros((ksize, ksize))
    direction = np.random.randint(4)
    if direction == 0:
        kernel[center, :] = 1
    elif direction == 1:
        kernel[:, center] = 1
    elif direction == 2:
        np.fill_diagonal(kernel, 1)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1)

    gx, gy = np.meshgrid(np.arange(ksize), np.arange(ksize))
    var = (ksize ** 2) / 16
    kernel *= np.exp(-((gx - center) ** 2 + (gy - center) ** 2) / (2 * var))
    kernel /= np.sum(kernel)
    return cv.filter2D(img.astype(np.uint8), -1, kernel)


def read_scene_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class MegaDepth(Dataset):
    """
    Args:
        scene_list_path:   Text file listing one scene id per line.
        scene_info_path:   Directory of preprocessed scene .npz files.
        base_path:         MegaDepth root directory.
        extractor:         Callable returning (keypoints, descriptors), see lib/features.py.
        binary_desc:       Whether descriptors are packed binary (e.g. ORB).
        motion_blur:       Motion blur kernel size, or None to disable.
        pairs_per_scene:   Maximum number of image pairs sampled per scene.
        kps_per_image:     Maximum number of keypoints per image.
        crop_image_size:   Crop size around a co-visible point, or -1 for no cropping.
    """

    def __init__(
        self,
        scene_list_path,
        scene_info_path,
        base_path,
        extractor,
        binary_desc=False,
        motion_blur=None,
        min_overlap_ratio=0.1,
        max_overlap_ratio=1.0,
        pairs_per_scene=300,
        kps_per_image=2048,
        crop_image_size=512,
        min_keypoints=2,
    ):
        if crop_image_size == -1 or crop_image_size * ORDER_SCALE >= 65536:
            raise ValueError(f"Z-ordering requires 0 < crop_image_size < {65536 // ORDER_SCALE}.")

        self.scenes = read_scene_list(scene_list_path)
        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.extractor = extractor
        self.binary_desc = binary_desc
        self.motion_blur = motion_blur

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.pairs_per_scene = pairs_per_scene
        self.num_kps = kps_per_image
        self.crop_image_size = crop_image_size
        self.min_keypoints = min_keypoints

        self.dataset = []

    # ------------------------------------------------------------------ build
    def build_dataset(self, desc="Building dataset"):
        """Sample image pairs from every scene and extract features."""
        self.dataset = []
        n_skipped = 0
        for scene in tqdm(self.scenes, desc=desc):
            info_path = os.path.join(self.scene_info_path, f"{scene}.npz")
            if not os.path.exists(info_path):
                continue

            scene_info = np.load(info_path, allow_pickle=True)
            overlap = scene_info["overlap_matrix"]
            valid = (overlap >= self.min_overlap_ratio) & (overlap <= self.max_overlap_ratio)
            pairs = np.vstack(np.where(valid))
            pair_order = np.random.permutation(pairs.shape[1])

            image_paths = scene_info["image_paths"]
            depth_paths = scene_info["depth_paths"]
            p3d_to_2d = scene_info["points3D_id_to_2D"]
            intrinsics = scene_info["intrinsics"]
            poses = scene_info["poses"]

            count = 0
            for pair_idx in pair_order:
                if count == self.pairs_per_scene:
                    break
                idx0, idx1 = pairs[0, pair_idx], pairs[1, pair_idx]

                matches = np.array(list(p3d_to_2d[idx0].keys() & p3d_to_2d[idx1]))
                if len(matches) == 0:
                    continue
                match = np.random.choice(matches)
                pt0, pt1 = p3d_to_2d[idx0][match], p3d_to_2d[idx1][match]

                pair = {
                    "image_path0": image_paths[idx0], "depth_path0": depth_paths[idx0],
                    "intrinsics0": intrinsics[idx0], "poses0": poses[idx0],
                    "image_path1": image_paths[idx1], "depth_path1": depth_paths[idx1],
                    "intrinsics1": intrinsics[idx1], "poses1": poses[idx1],
                    "central_match": np.array([pt0[1], pt0[0], pt1[1], pt1[0]]),
                }
                try:
                    pair.update(self.extract_pair(pair))
                except Exception:
                    n_skipped += 1
                    continue

                self.dataset.append(pair)
                count += 1

        np.random.shuffle(self.dataset)
        print(f"{len(self.dataset)} pairs built, {n_skipped} pairs skipped.")

    # -------------------------------------------------------------- features
    def extract_feature(self, bgr):
        bgr = add_random_motion_blur(bgr, self.motion_blur)
        keypoints, descriptors = self.extractor(bgr)
        if keypoints.shape[0] < self.min_keypoints:
            raise ValueError("Not enough keypoints.")

        if keypoints.shape[0] > self.num_kps:
            keypoints, descriptors = keypoints[:self.num_kps], descriptors[:self.num_kps]

        keypoints, descriptors, _ = z_ordering_encode(keypoints, descriptors, scale=ORDER_SCALE)

        normalized_kps = normalize_keypoints(keypoints, bgr.shape)
        return keypoints, normalized_kps, descriptors

    def extract_pair(self, meta):
        image0 = cv.imread(os.path.join(self.base_path, meta["image_path0"]))
        image1 = cv.imread(os.path.join(self.base_path, meta["image_path1"]))
        if image0 is None or image1 is None:
            raise FileNotFoundError("Image not found.")

        if self.crop_image_size != -1:
            image0, bbox0, image1, bbox1 = self.crop(image0, image1, meta["central_match"])
        else:
            bbox0, bbox1 = np.array([0, 0]), np.array([0, 0])

        kp0, nkp0, desc0 = self.extract_feature(image0)
        kp1, nkp1, desc1 = self.extract_feature(image1)

        # Keypoints are stored as (i, j) = (row, col) for depth-based warping.
        return {
            "kp0": np.ascontiguousarray(kp0[:, :2][:, ::-1]), "normalized_kp0": nkp0,
            "desc0": desc0, "bbox0": bbox0,
            "kp1": np.ascontiguousarray(kp1[:, :2][:, ::-1]), "normalized_kp1": nkp1,
            "desc1": desc1, "bbox1": bbox1,
        }

    def crop(self, image0, image1, central_match):
        s = self.crop_image_size

        def _corner(c, limit):
            if limit < s:
                raise ValueError("Image is smaller than the crop size.")
            return int(min(max(int(c) - s // 2, 0), limit - s))

        i0, j0 = _corner(central_match[0], image0.shape[0]), _corner(central_match[1], image0.shape[1])
        i1, j1 = _corner(central_match[2], image1.shape[0]), _corner(central_match[3], image1.shape[1])
        return (image0[i0:i0 + s, j0:j0 + s], np.array([i0, j0]),
                image1[i1:i1 + s, j1:j1 + s], np.array([i1, j1]))

    # --------------------------------------------------------------- loading
    def _load_depth(self, path, bbox):
        with h5py.File(os.path.join(self.base_path, path), "r") as f:
            depth = np.array(f["/depth"])
        assert np.min(depth) >= 0
        if self.crop_image_size != -1:
            s = self.crop_image_size
            depth = depth[bbox[0]:bbox[0] + s, bbox[1]:bbox[1] + s]
        return depth

    def _desc_tensor(self, desc):
        if self.binary_desc:
            # Packed uint8 [N, D/8] -> {0, 1} float [N, D]
            desc = np.unpackbits(desc.astype(np.uint8), axis=1, bitorder="little")
        return torch.from_numpy(desc.astype(np.float32))

    def __getstate__(self):
        # The extractor is not needed by DataLoader workers and may hold a CUDA model.
        state = self.__dict__.copy()
        state["extractor"] = None
        return state

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))
        out = {}
        for k in ("0", "1"):
            out.update({
                f"kp{k}": f32(d[f"kp{k}"]),
                f"normalized_kp{k}": f32(d[f"normalized_kp{k}"]),
                f"desc{k}": self._desc_tensor(d[f"desc{k}"]),
                f"depth{k}": f32(self._load_depth(d[f"depth_path{k}"], d[f"bbox{k}"])),
                f"intrinsics{k}": f32(d[f"intrinsics{k}"]),
                f"pose{k}": f32(d[f"poses{k}"]),
                f"bbox{k}": f32(d[f"bbox{k}"]),
            })
        return out


def custom_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}