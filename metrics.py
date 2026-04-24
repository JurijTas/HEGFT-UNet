from torch import nn
import torch
import torch.nn.functional as F



# -------------------------
# Losses: DiceLoss + BoundaryLoss
# -------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # logits: Bx1xHxW, targets: Bx1xHxW (0/1)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        # return mean dice loss
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary loss that multiplies prediction probability map by absolute SDF magnitude
    and takes the mean. Target SDF should be precomputed signed distance maps.
    Reference: Kervadec et al. (Boundary loss).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target_sdf):
        
        probs = torch.sigmoid(logits)
        loss = (probs * torch.abs(target_sdf)).mean()
        return loss


    
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return  self.bce(logits, targets)



def distance_transform_tensor(mask):
    """
    Computes distance transform on a binary mask (B x 1 x H x W) using PyTorch only.
    This approximates Euclidean distance using 2D convolutions (fast).
    """
    

    # Create a big kernel with 1s to propagate distance outward
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)

    # Initialize distances
    dist = torch.zeros_like(mask)

    # Copy mask: we want distance to foreground, so background=1; foreground=0
    inverted = 1 - mask  # foreground→0, background→1

    # We run iterative passes expanding outward from mask edges
    wave = inverted.clone()

    max_iters = mask.shape[-1] + mask.shape[-2]  # safe upper bound
    current_dist = 1.0

    for _ in range(max_iters):
        # Convolve "wave" to expand it
        new_wave = F.conv2d(wave, kernel, padding=1)

        # Keep only pixels newly reached
        new_wave = (new_wave > 0).float() * (dist == 0).float() * inverted

        if new_wave.sum() == 0:
            break

        dist += new_wave * current_dist

        wave = new_wave
        current_dist += 1.0

    return dist


class HausdorffDistanceLoss(nn.Module):
    """
    Karimi et al. (2019) differentiable formulation:
        L = mean( (p - t)^2 * (D_t^2 + D_p^2) )
    where D_t and D_p are distance transforms of target and prediction.

    Notes:
    - logits passed in are raw; sigmoid applied inside.
    - very sensitive to scale → use small weights (0.1–0.2).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Binarize predictions
        preds = (probs > 0.5).float()

        # Compute DT of target and prediction masks
        dt_targets = distance_transform_tensor(targets)
        dt_preds = distance_transform_tensor(preds)

        # HD Loss (differentiable approximation)
        loss = (probs - targets)**2 * (dt_targets**2 + dt_preds**2)
        return loss.mean()

######################################################
################   Metrics For Testing  ##############
#####################################################
def compute_dice(preds, targets, eps=1e-8):
    return (2 * (preds * targets).sum() + eps) / (preds.sum() + targets.sum() + eps)


def compute_iou(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    union = (preds + targets - preds * targets).sum()
    return (intersection + eps) / (union + eps)


def compute_accuracy(preds, targets):
    correct = (preds == targets).sum()
    total = torch.numel(preds)
    return correct, total

def compute_confusion(preds, targets):
    preds = preds.int()
    targets = targets.int()

    TP = ((preds == 1) & (targets == 1)).sum().float()
    TN = ((preds == 0) & (targets == 0)).sum().float()
    FP = ((preds == 1) & (targets == 0)).sum().float()
    FN = ((preds == 0) & (targets == 1)).sum().float()

    return TP, FP, FN, TN


def compute_precision(preds, targets, eps=1e-8):
    TP, FP, _, _ = compute_confusion(preds, targets)
    return TP / (TP + FP + eps)


def compute_recall(preds, targets, eps=1e-8):
    TP, _, FN, _ = compute_confusion(preds, targets)
    return TP / (TP + FN + eps)


def compute_sensitivity(preds, targets, eps=1e-8):
    return compute_recall(preds, targets, eps)


def compute_specificity(preds, targets, eps=1e-8):
    _, FP, _, TN = compute_confusion(preds, targets)
    return TN / (TN + FP + eps)



def compute_hausdorff(preds, targets):
    """
    Compute Hausdorff distance per image, ignoring empty masks.
    
    Inputs:
        preds   - binary mask tensor (N,1,H,W) or (N,H,W)
        targets - binary mask tensor (N,1,H,W) or (N,H,W)
    
    Returns:
        hd_vals - tensor of shape (num_valid_images,)
                  only includes images where both pred and target have foreground.
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
        targets = targets.squeeze(1)

    batch_distances = []

    for p, t in zip(preds, targets):
        p_idx = torch.nonzero(p, as_tuple=False).float()
        t_idx = torch.nonzero(t, as_tuple=False).float()

        # Skip invalid masks
        if p_idx.numel() == 0 or t_idx.numel() == 0:
            continue  # ignore this image

        pdist = torch.cdist(p_idx, t_idx)
        h1 = pdist.min(dim=1)[0].max()
        h2 = pdist.min(dim=0)[0].max()
        batch_distances.append(torch.max(h1, h2))

    if len(batch_distances) == 0:
        return torch.tensor([])  # no valid images
    return torch.stack(batch_distances)


def compute_hd95(preds, targets):
    """
    Compute 95th percentile Hausdorff (HD95) per image, ignoring empty masks.
    
    Inputs:
        preds   - binary mask tensor (N,1,H,W) or (N,H,W)
        targets - binary mask tensor (N,1,H,W) or (N,H,W)
    
    Returns:
        hd95_vals - tensor of shape (num_valid_images,)
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
        targets = targets.squeeze(1)

    batch_hd95 = []

    for p, t in zip(preds, targets):
        p_idx = torch.nonzero(p, as_tuple=False).float()
        t_idx = torch.nonzero(t, as_tuple=False).float()

        if p_idx.numel() == 0 or t_idx.numel() == 0:
            continue  # skip invalid masks

        pdist = torch.cdist(p_idx, t_idx)
        p_to_t = pdist.min(dim=1)[0]
        t_to_p = pdist.min(dim=0)[0]
        all_distances = torch.cat([p_to_t, t_to_p], dim=0)

        k = int(0.95 * (all_distances.numel() - 1))
        hd95_val = torch.kthvalue(torch.sort(all_distances).values, k+1).values
        batch_hd95.append(hd95_val)

    if len(batch_hd95) == 0:
        return torch.tensor([])
    return torch.stack(batch_hd95)





