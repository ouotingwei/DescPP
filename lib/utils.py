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

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

def save_config_log(cfg, save_dir, filename="setup_log.txt"):
    log_path = os.path.join(save_dir, filename)
    with open(log_path, "w") as f:
        f.write("Training Configuration:\n")
        f.write("="*30 + "\n")
        for k, v in vars(cfg).items():
            f.write(f"{k}: {v}\n")
        f.write("="*30 + "\n")
    print(f"[Info] Saved training setup to {log_path}")

def warp_by_homography(pos, homography, H_img: int, W_img: int):
    if pos.shape[0] != 2:
        pos = pos.transpose(0, 1)

    pos_h = torch.cat([pos, torch.ones((1, pos.shape[1]), device=pos.device)], dim=0)  # [3, N]
    warp_pos_h = homography @ pos_h  # [3, N]
    warp_pos = warp_pos_h[:2, :] / (warp_pos_h[2:, :] + 1e-8)  # [2, N]

    u, v = warp_pos[0], warp_pos[1]
    margin = 16
    valid = (u >= margin) & (u < W_img - margin) & \
            (v >= margin) & (v < H_img - margin) & \
            torch.isfinite(u) & torch.isfinite(v)

    ids = torch.nonzero(valid, as_tuple=False).squeeze(1)

    return pos[:, ids], warp_pos[:, ids], ids


def check_matching(kps0, kps1, h10, h01, min_matching=1024):
    pos_r = 3
    enough = True
    
    kps0 = np.array([[kp[0], kp[1]] for kp in kps0], dtype=np.float32)
    kps1 = np.array([[kp[0], kp[1]] for kp in kps1], dtype=np.float32)

    kps0_pos = torch.from_numpy(kps0[:, :2].T).cuda()
    kps1_pos = torch.from_numpy(kps1[:, :2].T).cuda()
    h10 = torch.from_numpy(h10.astype(np.float32)).view(3, 3).cuda()
    h01 = torch.from_numpy(h01.astype(np.float32)).view(3, 3).cuda()

    H_img = int(kps1_pos[1].max().item() + 1)
    W_img = int(kps1_pos[0].max().item() + 1)

    _, kps0_warp, _ = warp_by_homography(kps0_pos, h10, H_img, W_img)
    dist0 = torch.max(torch.abs(kps0_warp.unsqueeze(2) - kps1_pos.unsqueeze(1)), dim=0)[0]
    ids0 = (dist0 <= pos_r).sum(dim=1) >= 1
    if ids0.sum() < min_matching:
        enough = False

    _, kps1_warp, _ = warp_by_homography(kps1_pos, h01, H_img, W_img)
    dist1 = torch.max(torch.abs(kps1_warp.unsqueeze(2) - kps0_pos.unsqueeze(1)), dim=0)[0]
    ids1 = (dist1 <= pos_r).sum(dim=1) >= 1
    if ids1.sum() < min_matching:
        enough = False

    # pos_rate_0 = ids0.sum().item() / len(kps0)
    # pos_rate_1 = ids1.sum().item() / len(kps1)
    # print(f"pos_rate_0: {pos_rate_0:.4f}, pos_rate_1: {pos_rate_1:.4f}")

    return enough

def sample_descriptors_from_map(
    desc_map: torch.Tensor,              # [B, C, H', W']
    kps_batch: List[torch.Tensor],       # List of [N_i, 2]
    img_h: int,
    img_w: int,
    scale_factor: float = 1/8
) -> torch.Tensor:
    B, C, Hf, Wf = desc_map.shape
    H = img_h
    W = img_w
    all_descs = []

    for b in range(B):
        kps = kps_batch[b]
        if len(kps) == 0:
            all_descs.append(torch.zeros((0, C), device=desc_map.device))
            continue

        if not isinstance(kps, torch.Tensor):
            kps = torch.tensor(kps, dtype=torch.float32, device=desc_map.device)
        else:
            kps = kps.to(dtype=torch.float32, device=desc_map.device)

        kps_norm = kps.clone()
        kps_norm[:, 0] = (kps[:, 0] / (W - 1)) * 2 - 1
        kps_norm[:, 1] = (kps[:, 1] / (H - 1)) * 2 - 1
        kps_norm *= scale_factor

        grid = kps_norm.view(1, 1, -1, 2)  # [1, 1, N, 2]
        sampled = F.grid_sample(
            desc_map[b:b+1], grid, mode='bicubic', align_corners=True
        )  # [1, C, 1, N]

        desc = sampled.squeeze(2).squeeze(0).permute(1, 0)  # [N, C]
        all_descs.append(desc)

    # 保證維度對齊（需要 padding 或保證每張圖有相同 N）
    return torch.stack(all_descs, dim=0)  # [B, N, C]

def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise Exception

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise Exception

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(
        pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2
):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention
    u1 = pos1[1, :] + bbox1[1] + .5
    v1 = pos1[0, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)
    # XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2_hom = torch.linalg.multi_dot((pose2, torch.inverse(pose1), XYZ1_hom))
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise Exception

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids

def custom_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

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