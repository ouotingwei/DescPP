import os
import sys
import yaml
import argparse
import cv2 as cv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

orb_path = Path(__file__).parent / "lib/orbslam2_features/lib"
if str(orb_path) not in sys.path:
    sys.path.append(str(orb_path))
import orbslam2_features

alike_root = Path(__file__).parent / "lib" / "ALIKE"
if str(alike_root) not in sys.path:
    sys.path.append(str(alike_root))

from lib.ALIKE import alike 
from lib.ALIKE.alike import ALike
from lib.SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from lib.utils import *
from lib.dog import Dog
from model.DescPP import DescPP 

parser = argparse.ArgumentParser(description="Descriptor Enhancement with Desc++")
parser.add_argument('--feature', type=str, required=True, 
                    help='[ORB, ORB-Desc++, SIFT, SIFT-Desc++, SuperPoint, SuperPoint-Desc++, ALIKE, ALIKE-Desc++]')
parser.add_argument('--model_path', type=str, required=False, help='Path to weights file (.pt)')
parser.add_argument('--config', type=str, default='DescPP_config.yml', help='Path to config file')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

with open(args.config, "r") as f:
    config_data = yaml.safe_load(f)

dataset_path = 'hpatches-benchmarking/hpatches-sequences-release'
sp_weights_path = 'lib/SuperPointPretrainedNetwork/superpoint_v1.pth'
feature = args.feature.strip()
prefix = feature.split('-')[0].lower() 
use_descpp = 'Desc++' in feature

extractor = None
enhancer = None

if prefix == 'orb':
    extractor = orbslam2_features.ORBextractor(3000, 1.2, 8)
    if use_descpp:
        enhancer = DescPP(config_data['orb']).to(device)

elif prefix == 'sift':
    extractor = Dog()
    if use_descpp:
        enhancer = DescPP(config_data['sift']).to(device)

elif prefix == 'superpoint':
    extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, 
                                   conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())
    if use_descpp:
        enhancer = DescPP(config_data['superpoint']).to(device)

elif prefix == 'alike':
    extractor = ALike(**alike.configs['alike-l'], device=device, top_k=-1, scores_th=0.2)
    if use_descpp:
        enhancer = DescPP(config_data['alike']).to(device)
else:
    sys.exit(f"Error: Unsupported feature type '{feature}'")

# load weight
if use_descpp:
    if args.model_path is None:
        raise ValueError(f"Feature '{feature}' requires --model_path")
    print(f"Loading Desc++ weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    enhancer.load_state_dict(state_dict)
    enhancer.eval()

seq_names = sorted(os.listdir(dataset_path))

for seq_name in tqdm(seq_names):
    seq_dir = os.path.join(dataset_path, seq_name)
    if not os.path.isdir(seq_dir): continue

    for im_idx in range(1, 7):
        img_path = os.path.join(seq_dir, f"{im_idx}.ppm")
        
        # featrue extraction
        if prefix == 'orb':
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            kps_tuples, descriptors = extractor.detectAndCompute(img)
            keypoints_cv = [cv.KeyPoint(*kp) for kp in kps_tuples]
            # [u, v, size, angle]
            keypoints_np = np.array([[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] 
                                     for kp in keypoints_cv], dtype=np.float32)
        
        elif prefix == 'sift':
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img_float = (img.astype('float32') / 255.)
            keypoints_np, scores, descriptors = extractor.detectAndCompute(img_float)
            
        elif prefix == 'superpoint':
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img_float = (img.astype('float32') / 255.)
            keypoints, descriptors, _ = extractor.run(img_float)
            keypoints_np, descriptors = keypoints.T, descriptors.T
            
        elif prefix == 'alike':
            img = cv.imread(img_path, cv.COLOR_BGR2RGB)
            pred = extractor(img, sub_pixel=True)
            keypoints = pred['keypoints']
            descriptors = pred['descriptors']
            scores = pred['scores']
            keypoints_np = np.hstack((keypoints, np.expand_dims(scores, 1)))

        # Desc++ enhance
        if use_descpp and enhancer is not None:
            # Z-ordering 
            keypoints_np, descriptors, _ = z_ordering_encode(keypoints_np, descriptors)
            
            # Normalizaiton
            kps_norm = normalize_keypoints(keypoints_np, img.shape).astype(np.float32)
            
            with torch.no_grad():
                kps_t = torch.from_numpy(kps_norm).to(device)
                
                if prefix == 'orb':
                    desc_bits = np.unpackbits(descriptors, axis=1, bitorder='little')
                    desc_bits = desc_bits.astype(np.float32) * 2.0 - 1.0
                    desc_t = torch.from_numpy(desc_bits).to(device)
                    
                    out = enhancer(desc_t, kps_t)
                    
                    out_bits = (out >= 0).to(torch.uint8).cpu().numpy()
                    descriptors = np.packbits(out_bits, axis=1, bitorder='little')
                else:
                    desc_t = torch.from_numpy(descriptors.astype(np.float32)).to(device)
                    out = enhancer(desc_t, kps_t)
                    descriptors = out.cpu().numpy()

        out_dir = os.path.join(seq_dir, f"{im_idx}.ppm.{feature}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'keypoints.npy'), keypoints_np)
        np.save(os.path.join(out_dir, 'descriptors.npy'), descriptors)

print("Feature extraction complete.")