import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_sim import coverage_score, rollout


def test_oracle_rollout_is_monotone_and_chooses_new_coverage():
    table = np.zeros((24, 4), bool)
    table[0, 0] = True
    table[6, 1] = True
    table[2, 2:] = True
    clouds = [np.array([[i, 0, 0]], np.float32) for i in range(24)]
    result = rollout("oracle", table, clouds, None, [0, 6], 3, 1,
                     np.random.default_rng(1), "cpu")
    assert result["actions"] == [0, 6, 2]
    assert result["coverage"] == [.5, 1.]
    assert coverage_score(table, [0, 6]) == .5
