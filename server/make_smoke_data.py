"""Generate artificial samples for pipeline checks only; never report as experiments."""

import argparse
from pathlib import Path
import numpy as np


def make_smoke_data(output: str | Path, point_count: int = 2048, action_count: int = 24):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for split, objects, states in (("train", 4, 32), ("val", 2, 16)):
        rows = []
        for object_index in range(objects):
            center = rng.uniform(-30, 30, size=3)
            object_id = f"synthetic_{split}_{object_index:02d}"
            preference = rng.uniform(0.1, 0.7, size=action_count)
            for _ in range(states):
                points = rng.normal(center, 20, size=(point_count, 3)).astype(np.float32)
                visited = np.zeros(action_count, bool)
                visited[:2] = True
                extra = rng.choice(np.arange(2, action_count), size=int(rng.integers(0, max(1, min(6, action_count - 2)))), replace=False)
                visited[extra] = True
                target = np.clip(preference + 0.002 * center[0], 0, 1).astype(np.float32)
                target[visited] = 0
                rows.append((points, np.ones(point_count, bool), visited,
                             np.ones(action_count, bool), target,
                             int(np.flatnonzero(visited)[-1]), float(visited.sum()), object_id))
        columns = list(zip(*rows))
        keys = ("points", "point_mask", "visited_mask", "available_mask",
                "gain_target", "current_action_id", "used_scans", "object_id")
        np.savez_compressed(output / f"{split}.npz", **{key: np.asarray(value) for key, value in zip(keys, columns)})
    print(f"Synthetic train/val files written to {output}; do not use for results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "smoke_data")
    args = parser.parse_args()
    make_smoke_data(args.output)
