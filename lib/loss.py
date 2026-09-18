"""
Unified DescPP loss (FastAP + boost loss) for binary (ORB) and float (SP / ALIKE / SIFT) descriptors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import warp


def binarize_pm1(x):
    """Straight-through binarization to {-1, 1}. Forward: sign, backward: identity."""
    return (x + ((x >= 0).type_as(x) - x).detach()) * 2.0 - 1.0


class FastAPLoss(nn.Module):
    def __init__(self, num_bins=10, max_distance=4.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_distance = max_distance

    def forward(self, dist, pos_labels, neg_labels, size_average=True):
        """dist / pos_labels / neg_labels: [N, M]. Returns 1 - AP."""
        device = dist.device
        delta = self.max_distance / self.num_bins
        Z = torch.linspace(0., self.max_distance, steps=self.num_bins + 1, device=device).view(-1, 1, 1)
        N_pos = pos_labels.sum(dim=1)
        pulse = F.relu(1. - torch.abs(dist - Z) / delta)
        h_pos = (pulse * pos_labels).sum(dim=2).t()
        h_neg = (pulse * neg_labels).sum(dim=2).t()
        H_pos = torch.cumsum(h_pos, dim=1)
        H = torch.cumsum(h_pos + h_neg, dim=1)

        h_product = h_pos * H_pos
        safe = (h_product > 0) & (H > 0)
        ap = torch.zeros_like(h_pos)
        ap[safe] = h_product[safe] / H[safe]
        ap = torch.clamp(ap.sum(dim=1) / N_pos, max=1.0)
        return 1 - ap.mean() if size_average else 1 - ap


class LossProcessor:
    """
    descriptor_type:
        "binary"
        "float" 
    """
    def __init__(self, descriptor_type="float", num_bins=10, pos_radius=3, neg_radius=16,
                 min_pos=0, check_finite=False):
        assert descriptor_type in ("binary", "float")
        self.binary = descriptor_type == "binary"
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius
        self.min_pos = min_pos
        self.num_bins = num_bins
        self.check_finite = check_finite
        self._fastap_cache = {}

    # ------------------------------------------------------------ helpers
    def _fastap(self, dist, pos, neg, max_distance):
        key = float(max_distance)
        if key not in self._fastap_cache:
            self._fastap_cache[key] = FastAPLoss(self.num_bins, max_distance)
        return self._fastap_cache[key](dist, pos, neg, size_average=False).view(-1)

    def encode(self, model, desc, kp_norm):
        if self.binary:
            return binarize_pm1(model(desc * 2 - 1, kp_norm))
        return model(desc, kp_norm)

    def distance(self, a, b, already_pm1=False):
        """return (distance matrix, max_distance)"""
        if self.binary:
            if not already_pm1:
                a, b = a * 2. - 1., b * 2. - 1.
            D = a.shape[1]
            return (D - a @ b.t()) * 0.5, D
        return 2 - 2 * a @ b.t(), 4.0

    def _build_labels(self, src_warp_pos, dst_pos):
        pos_dist = torch.max(torch.abs(src_warp_pos.unsqueeze(2).float() - dst_pos.unsqueeze(1)), dim=0)[0]
        ids_has_gt = (pos_dist <= self.pos_radius).sum(dim=1) >= 1
        if ids_has_gt.sum() <= self.min_pos:
            return None
        pos_dist = pos_dist[ids_has_gt]
        return ids_has_gt, (pos_dist <= self.pos_radius).float(), (pos_dist >= self.neg_radius).float()

    # ---------------------------------------------------------- one direction
    def _one_dir(self, src, dst, desc_src, desc_dst, boost_src, boost_dst):
        try:
            _, src_warp_pos, ids_valid = warp(
                src["kp"][:, :2].t(),
                src["depth"], src["K"], src["T"], src["bbox"],
                dst["depth"], dst["K"], dst["T"], dst["bbox"],
            )
        except Exception:
            return None

        labels = self._build_labels(src_warp_pos, dst["kp"][:, :2].t())
        if labels is None:
            return None
        ids_has_gt, pos, neg = labels

        desc_src_sel = desc_src[ids_valid][ids_has_gt]
        boost_src_sel = boost_src[ids_valid][ids_has_gt]

        dist_origin, Do = self.distance(desc_src_sel, desc_dst)
        dist_boost, Db = self.distance(boost_src_sel, boost_dst, already_pm1=True)

        if self.check_finite and not torch.isfinite(dist_boost).all():
            print(f"[warn] non-finite boosted distance "
                  f"(max|boost_src|={boost_src_sel.abs().max():.3e}, max|boost_dst|={boost_dst.abs().max():.3e})")
            return None

        origin_ap = self._fastap(dist_origin, pos, neg, Do)
        boost_ap = self._fastap(dist_boost, pos, neg, Db)
        return origin_ap, boost_ap

    # ------------------------------------------------------------ one pair
    def forward_pair(self, model, s):
        """
        s: dict of tensors for one image pair
        Returns (origin_ap_loss, boost_ap_loss) 
        """
        boost0 = self.encode(model, s["desc0"], s["normalized_kp0"])
        boost1 = self.encode(model, s["desc1"], s["normalized_kp1"])

        if self.check_finite and not (torch.isfinite(boost0).all() and torch.isfinite(boost1).all()):
            print("[warn] model output contains NaN/Inf, skip pair")
            return None, None

        v0 = dict(kp=s["kp0"], depth=s["depth0"], K=s["intrinsics0"], T=s["pose0"], bbox=s["bbox0"])
        v1 = dict(kp=s["kp1"], depth=s["depth1"], K=s["intrinsics1"], T=s["pose1"], bbox=s["bbox1"])

        outs = [o for o in (
            self._one_dir(v0, v1, s["desc0"], s["desc1"], boost0, boost1),
            self._one_dir(v1, v0, s["desc1"], s["desc0"], boost1, boost0),
        ) if o is not None]

        if not outs:
            return None, None
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def boost_objective(origin_ap_loss, boost_ap_loss, boost_weight=1.0, eps=1e-6):
    """
    match_loss : mean(1 - AP_boost)
    boost_loss : mean(relu(AP_origin / AP_boost - 1))
    """
    match_loss = boost_ap_loss.mean()
    boost_loss = F.relu((1. - origin_ap_loss) / (1. - boost_ap_loss).clamp_min(eps) - 1.).mean()
    return match_loss + boost_weight * boost_loss, match_loss, boost_loss