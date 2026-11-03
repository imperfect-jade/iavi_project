"""Small PointNet-style candidate gain regressor."""

import torch
from torch import nn


class GainPolicy(nn.Module):
    def __init__(self, action_count: int = 24, center_mm=(0.0, 0.0, 0.0),
                 scale_mm=100.0, use_points: bool = True):
        super().__init__()
        if scale_mm <= 0:
            raise ValueError("scale_mm must be positive")
        self.action_count = action_count
        self.use_points = use_points
        self.register_buffer("center_mm", torch.tensor(center_mm, dtype=torch.float32))
        self.register_buffer("scale_mm", torch.tensor(float(scale_mm), dtype=torch.float32))
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
        ) if use_points else None
        self.head = nn.Sequential(
            nn.Linear(256 + 3 * action_count + 1, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_count), nn.Sigmoid(),
        )

    def forward(self, points, point_mask, visited_mask, available_mask,
                current_action_id, used_scans):
        if self.use_points:
            if not point_mask.any(dim=1).all():
                raise ValueError("each state needs at least one valid point")
            x = ((points - self.center_mm) / self.scale_mm).transpose(1, 2)
            x = self.point_encoder(x)
            x = x.masked_fill(~point_mask[:, None, :], -torch.inf)
            pooled = x.max(dim=2).values
        else:
            pooled = torch.zeros((visited_mask.shape[0], 256),
                                 device=visited_mask.device, dtype=torch.float32)
        current = torch.nn.functional.one_hot(current_action_id.long(), self.action_count)
        state = torch.cat((pooled, visited_mask.float(), available_mask.float(),
                           current.float(), (used_scans / self.action_count)[:, None]), dim=1)
        return self.head(state)


class GainPolicyV2(nn.Module):
    """Gain regressor with an ungated point-to-action path.

    Raw scores are intentional: the policy ranks actions, and removing the
    output sigmoid avoids suppressing gradients when gains are near zero.
    """

    def __init__(self, action_count: int = 24, center_mm=(0.0, 0.0, 0.0),
                 scale_mm=100.0, use_points: bool = True):
        super().__init__()
        if scale_mm <= 0:
            raise ValueError("scale_mm must be positive")
        self.action_count = action_count
        self.use_points = use_points
        self.register_buffer("center_mm", torch.tensor(center_mm, dtype=torch.float32))
        self.register_buffer("scale_mm", torch.tensor(float(scale_mm), dtype=torch.float32))
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, 1), nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, 1), nn.LeakyReLU(0.1),
        ) if use_points else None
        self.point_head = nn.Linear(256, action_count, bias=False) if use_points else None
        self.history_head = nn.Sequential(
            nn.Linear(3 * action_count + 1, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, action_count),
        )
        if use_points:
            nn.init.normal_(self.point_head.weight, mean=0.0, std=0.01)

    def forward(self, points, point_mask, visited_mask, available_mask,
                current_action_id, used_scans):
        current = torch.nn.functional.one_hot(current_action_id.long(), self.action_count)
        history = torch.cat((visited_mask.float(), available_mask.float(), current.float(),
                             (used_scans / self.action_count)[:, None]), dim=1)
        scores = self.history_head(history)
        if self.use_points:
            if not point_mask.any(dim=1).all():
                raise ValueError("each state needs at least one valid point")
            x = ((points - self.center_mm) / self.scale_mm).transpose(1, 2)
            x = self.point_encoder(x)
            x = x.masked_fill(~point_mask[:, None, :], -torch.inf)
            pooled = x.max(dim=2).values
            scores = scores + self.point_head(pooled)
        return scores


def make_gain_policy(config: dict) -> nn.Module:
    version = config.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown model version: {version}")
    policy = GainPolicy if version == "v1" else GainPolicyV2
    return policy(config["action_count"], config["center_mm"], config["scale_mm"],
                  config.get("use_points", True))


def masked_smooth_l1(prediction, target, mask, sample_weight=None):
    per_action = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    per_state = (per_action * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    if sample_weight is None:
        return per_state.mean()
    return (per_state * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def select_action(prediction, visited_mask, available_mask):
    eligible = available_mask.bool() & ~visited_mask.bool()
    if not eligible.any(dim=1).all():
        raise ValueError("no eligible action")
    return prediction.masked_fill(~eligible, -torch.inf).argmax(dim=1)
