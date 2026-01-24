import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

methods = ['ORB', 'ORB-Desc++', 'SIFT', 'SIFT-Desc++', 'SuperPoint', 'SuperPoint-Desc++', 'ALIKE', 'ALIKE-Desc++']
names = ['ORB', 'ORB-Desc++', 'SIFT', 'SIFT-Desc++', 'SuperPoint', 'SuperPoint-Desc++', 'ALIKE', 'ALIKE-Desc++']
colors = ['red', 'red', 'purple', 'purple', 'green', 'green', 'blue', 'blue']
linestyles = ['-', '--', '-', '--', '-', '--', '-', '--']

dataset_path = 'hpatches-sequences-release'

n_i = 52
n_v = 56

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def mnn_matcher(descA, descB):
    sim = descA @ descB.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=descA.device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().cpu().numpy()

def benchmark_features(read_feats):
    seq_names = sorted(os.listdir(dataset_path))
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}
    i_matches = {thr: 0 for thr in rng}
    v_matches = {thr: 0 for thr in rng}
    seq_type = []
    n_feats = []
    n_matches = []

    for seq_name in tqdm(seq_names):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device),
                torch.from_numpy(descriptors_b).to(device)
            )

            H = np.loadtxt(os.path.join(dataset_path, seq_name, f"H_1_{im_idx}"))
            pos_a = keypoints_a[matches[:, 0], :2]
            pos_b = keypoints_b[matches[:, 1], :2]

            pos_a_h = np.concatenate([pos_a, np.ones((pos_a.shape[0], 1))], axis=1)
            pos_b_proj_h = (H @ pos_a_h.T).T
            pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

            dist = np.linalg.norm(pos_b - pos_b_proj, axis=1)
            if dist.shape[0] == 0:
                dist = np.array([float('inf')])

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                    i_matches[thr] += np.sum(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
                    v_matches[thr] += np.sum(dist <= thr)


    return i_err, v_err, i_matches, v_matches, [seq_type, np.array(n_feats), np.array(n_matches)]

def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5), 
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )

def getBit(des):
    res = []
    for d in des:
        for i in range(8):
            res.append(((d >> i) & 1) * 2 - 1)
    return res

def generate_read_function(method, extension='ppm', type='float'):
    def read_function(seq_name, im_idx):
        folder_path = os.path.join(dataset_path, seq_name, f"{im_idx}.{extension}.{method}")

        keypoints_path = os.path.join(folder_path, 'keypoints.npy')
        descriptors_path = os.path.join(folder_path, 'descriptors.npy')

        # Check existence
        if not os.path.exists(keypoints_path):
            raise FileNotFoundError(f"Missing keypoints file: {keypoints_path}")

        if not os.path.exists(descriptors_path):
            raise FileNotFoundError(f"Missing descriptors file: {descriptors_path}")

        # Load files
        keypoints = np.load(keypoints_path)
        descriptors = np.load(descriptors_path)

        # Convert format
        if type != 'float':
            descriptors = np.unpackbits(descriptors.astype(np.uint8), axis=1, bitorder='little')
            descriptors = descriptors * 2.0 - 1.0

        return keypoints, descriptors

    return read_function


errors = {}
for method in methods:
    print(f"Processing {method}")
    if method.upper().startswith("ORB"):
        read_function = generate_read_function(method, type='bool')
    else:
        read_function = generate_read_function(method, type='float')
    errors[method] = benchmark_features(read_function)

for method in methods:
    i_err, v_err, i_matches, v_matches, _ = errors[method]
    print(method)
    for thr in [3, 5]:
        print('# MMA@{:d}: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
            thr,
            (i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5), 
            i_err[thr] / (n_i * 5), 
            v_err[thr] / (n_v * 5))
        )
    
        print('# inliers@{:d}: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
            thr,
            (i_matches[thr] + v_matches[thr]) / ((n_i + n_v) * 5), 
            i_matches[thr] / (n_i * 5), 
            v_matches[thr] / (n_v * 5))
        )

# MMA plot
plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

fig = plt.figure(figsize=(12, 6))  

# ---------- Overall ----------
plt.subplot(1, 3, 1)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _, _, _ = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim([0.13, 0.9])
plt.grid(True,  alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=20)

# ---------- Illumination ----------
plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _, _, _ = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Illumination')
plt.xlabel('threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0.13, 0.9])
plt.gca().axes.set_yticklabels([])
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=20)

# ---------- Viewpoint ----------
plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _, _, _ = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng],
             color=color, ls=ls, linewidth=2, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0.13, 0.9])
plt.gca().axes.set_yticklabels([])
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=20)

handles, labels = [], []
for method, name, color, ls in zip(methods, names, colors, linestyles):
    h = plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=3)
    handles.append(h)
    labels.append(name)

fig.legend(handles, labels,
           loc='lower center',
           ncol=4,            
           fontsize=15,
           frameon=False,
           bbox_to_anchor=(0.5, 0.01))

plt.subplots_adjust(bottom=0.25, wspace=0.1) 

plt.savefig('mma.pdf', dpi=300)
plt.close()