"""Small PointNet-style candidate gain regressor."""

import torch
from torch import nn


class GainPolicy(nn.Module):
    def __init__(self, action_count: int = 36, center_mm=(0.0, 0.0, 0.0), scale_mm=100.0):
        super().__init__()
        if scale_mm <= 0:
            raise ValueError("scale_mm must be positive")
        self.action_count = action_count
        self.register_buffer("center_mm", torch.tensor(center_mm, dtype=torch.float32))
        self.register_buffer("scale_mm", torch.tensor(float(scale_mm), dtype=torch.float32))
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 3 * action_count + 1, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_count), nn.Sigmoid(),
        )

    def forward(self, points, point_mask, visited_mask, available_mask,
                current_action_id, used_scans):
        if not point_mask.any(dim=1).all():
            raise ValueError("each state needs at least one valid point")
        x = ((points - self.center_mm) / self.scale_mm).transpose(1, 2)
        x = self.point_encoder(x)
        x = x.masked_fill(~point_mask[:, None, :], -torch.inf)
        pooled = x.max(dim=2).values
        current = torch.nn.functional.one_hot(current_action_id.long(), self.action_count)
        state = torch.cat((pooled, visited_mask.float(), available_mask.float(),
                           current.float(), (used_scans / self.action_count)[:, None]), dim=1)
        return self.head(state)


def masked_smooth_l1(prediction, target, mask):
    per_action = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    per_state = (per_action * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_state.mean()


def select_action(prediction, visited_mask, available_mask):
    eligible = available_mask.bool() & ~visited_mask.bool()
    if not eligible.any(dim=1).all():
        raise ValueError("no eligible action")
    return prediction.masked_fill(~eligible, -torch.inf).argmax(dim=1)
