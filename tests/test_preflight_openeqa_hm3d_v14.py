from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from analysis.preflight_openeqa_hm3d_v14 import audit_episode


class PreflightOpenEQAHM3DV14Tests(unittest.TestCase):
    def make_episode(self, root, frames=20):
        np.savetxt(root / "intrinsic_color.txt", np.eye(4))
        for index in range(frames):
            prefix = f"{index:03d}"
            Image.fromarray(
                np.full((3, 4, 3), index, dtype=np.uint8)
            ).save(root / f"{prefix}-rgb.png")
            Image.fromarray(
                np.full((3, 4), index * 100, dtype=np.uint16)
            ).save(root / f"{prefix}-depth.png")
            np.savetxt(root / f"{prefix}.txt", np.eye(4))

    def test_complete_episode_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_episode(root)
            result = audit_episode(root)
            self.assertEqual(result["available_frames"], 20)
            self.assertEqual(result["selected_frames"], 20)
            self.assertEqual(result["resolution"], [4, 3])
            self.assertEqual(
                result["selected_depth_uint16_range"],
                [0, 1900],
            )

    def test_invalid_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_episode(root)
            np.savetxt(root / "000.txt", np.zeros((4, 4)))
            with self.assertRaises(ValueError):
                audit_episode(root)


if __name__ == "__main__":
    unittest.main()
