import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rebase_camera_poses import pose_in_lower_reference


class RebaseTests(unittest.TestCase):
    def test_same_lower_pose_is_identity_and_upper_maps_to_lower(self):
        lower = np.eye(4)
        lower[:3, 3] = [10, 20, 30]
        upper = np.eye(4)
        upper[:3, 3] = [10, 80, 30]
        np.testing.assert_allclose(pose_in_lower_reference(lower, lower), np.eye(4))
        np.testing.assert_allclose(pose_in_lower_reference(lower, upper)[:3, 3], [0, -60, 0])


if __name__ == "__main__":
    unittest.main()
