import hashlib
import json
import unittest
from dataclasses import replace

import numpy as np
import torch

from maip.data import DEFAULT_DATA, available_cases, load_case, normalize_sensitivity
from maip.model import MultiBranchNetwork
from maip.reconstruction import Config, reconstruct


class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_distributed_data_checksums(self):
        manifest = json.loads((DEFAULT_DATA / "manifest.json").read_text(encoding="utf-8"))
        for filename, record in manifest["files"].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((DEFAULT_DATA / filename).read_bytes()).hexdigest(), record["sha256"])

    def test_all_cases_and_coordinate_maps(self):
        self.assertEqual(len(available_cases()), 8)
        for case in available_cases():
            with self.subTest(case=case):
                data = load_case(case)
                self.assertEqual(data.voltages.shape, (104, 4))
                n = data.sensitivity.shape[1]
                np.testing.assert_array_equal(data.eit2rect.T, data.rect2eit)
                self.assertEqual(np.count_nonzero(data.rect2eit), n)
                np.testing.assert_array_equal(data.rect2eit.sum(1), np.ones(n))
                values = np.arange(n, dtype=np.float32)
                row, col = data.positions[:, 2:4].T - 1
                np.testing.assert_array_equal(data.to_images(values)[row, col], values)
                self.assertEqual(int(data.display_mask.sum()), n)

    def test_grid_operator_agrees_with_mesh_operator(self):
        generator = np.random.default_rng(7)
        for case in ("case2_sim", "DBLung_sim"):
            with self.subTest(case=case):
                data = load_case(case)
                matrix = normalize_sensitivity(data.sensitivity)
                grid = generator.normal(size=(4096, 4)).astype(np.float32)
                mapped = torch.from_numpy(matrix).matmul(torch.from_numpy(data.rect2eit))
                on_grid = mapped.matmul(torch.from_numpy(grid)).numpy()
                on_mesh = matrix @ (data.rect2eit @ grid)
                np.testing.assert_allclose(on_grid, on_mesh, atol=2e-6, rtol=2e-5)

    def test_joint_loss_reaches_all_branches_and_attention(self):
        torch.manual_seed(2)
        model = MultiBranchNetwork()
        output = model(torch.rand(1, 4, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 64, 64))
        target = torch.rand_like(output)
        (output - target).abs().mean().backward()
        for branch in model.branches:
            gradient = branch.input_layer[0].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        for gradient in (model.attention.attention_weights.grad, model.attention.channel_weights.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_invalid_case_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown case"):
            load_case("missing-case")
        data = load_case()
        for voltages in (np.zeros((104, 3), np.float32), np.full((104, 4), np.nan, np.float32)):
            with self.assertRaisesRegex(ValueError, "Expected finite voltages"):
                reconstruct(replace(data, voltages=voltages), Config(epochs=1), device="cpu", progress=False)


if __name__ == "__main__":
    unittest.main()
