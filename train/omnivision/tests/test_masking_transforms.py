import unittest
from typing import List

import torch
from omnivore.data.transforms.mask_image_modeling import MaskForPerformance

from omnivore.utils.testing import (
    compose_omnivore_config,
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestMakingTransform(unittest.TestCase):
    def test_mask_for_performance_size224_p16(self):
        self._assert_drop_count(
            patch_size=16,
            image_size=224,
            expected_mask_size=[14, 14],
            expected_drop=5,
        )

    def test_mask_for_performance_size256_p16(self):
        self._assert_drop_count(
            patch_size=16,
            image_size=256,
            expected_mask_size=[16, 16],
            expected_drop=1,
        )

    def test_mask_for_performance_size224_p14(self):
        self._assert_drop_count(
            patch_size=14,
            image_size=224,
            expected_mask_size=[16, 16],
            expected_drop=1,
        )

    def _assert_drop_count(
        self,
        patch_size: int,
        image_size: int,
        expected_mask_size: List[int],
        expected_drop: int,
    ):
        transform = MaskForPerformance(patch_size=patch_size, class_token=True)
        image = torch.randn(size=(3, image_size, image_size))
        out = transform(image)
        self.assertTrue(torch.allclose(out["data"], image))
        self.assertEqual(torch.Size(expected_mask_size), out["mask"].shape)
        self.assertEqual(expected_drop, out["mask"].count_nonzero().item())

    @gpu_test(gpu_count=2)
    def test_masking_transform_end_to_end(self):
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/vit_train_synthetic_opt",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)
