# test cases modified from classy_vision/test/meters_accuracy_meter_test.py

import multiprocessing
import tempfile
import unittest

import torch
import torch.distributed
from omnivision.meters.accuracy_meter import AccuracyMeter, MultilabelModes
from tests.utils import run_distributed_test
from torch.distributed import ReduceOp


def test_meter_distributed(rank, world_size):
    tc = unittest.TestCase()
    for top_k in [1, 2]:
        meter = AccuracyMeter(top_k)
        meter.set_sync_device(torch.device("cpu"))

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = {
            0: [
                torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Rank 0
                torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Rank 0
            ],
            1: [
                torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Rank 1
                torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Rank 1
            ],
        }[rank]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = {
            0: [
                torch.tensor([0, 1, 2]),  # Rank 0
                torch.tensor([0, 1, 2]),  # Rank 0
            ],
            1: [
                torch.tensor([0, 1, 2]),  # Rank 1
                torch.tensor([0, 1, 2]),  # Rank 1
            ],
        }[rank]

        # In first two updates there are 3 correct top-2, 5 correct in top 2
        # The same occurs in the second two updates and is added to first
        expected_values = [
            {1: 300 / 6.0, 2: 500 / 6.0}[top_k],  # After one update
            {1: 600 / 12.0, 2: 1000 / 12.0}[top_k],  # After two updates
        ]
        for i in range(2):
            meter.update(model_outputs[i], targets[i])
            tc.assertAlmostEqual(
                meter.compute_synced()[""], expected_values[i], places=4
            )


class TestOmnivisionMeter(unittest.TestCase):
    def test_meter_synced(self):
        run_distributed_test(test_meter_distributed, 2)

    def _apply_updates_and_test_meter(
        self, meter, model_output, target, expected_value, **kwargs
    ):
        """
        Runs a valid meter test. Does not reset meter before / after running
        """
        if not isinstance(model_output, list):
            model_output = [model_output]

        if not isinstance(target, list):
            target = [target]

        for i in range(len(model_output)):
            meter.update(model_output[i], target[i], **kwargs)

        meter_value = meter.compute()
        self.assertAlmostEqual(meter_value[""], expected_value, places=4)

    def meter_update_and_reset_test(
        self, meter, model_outputs, targets, expected_value, **kwargs
    ):
        """
        This test verifies that a single update on the meter is successful,
        resets the meter, then applies the update again.
        """
        # If a single output is provided, wrap in list
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]
            targets = [targets]

        self._apply_updates_and_test_meter(
            meter, model_outputs, targets, expected_value, **kwargs
        )

        meter.reset()

        # Verify reset works by reusing single update test
        self._apply_updates_and_test_meter(
            meter, model_outputs, targets, expected_value, **kwargs
        )

    def meter_invalid_meter_input_test(self, meter, model_output, target):
        # Invalid model
        with self.assertRaises(Exception):
            meter.update(model_output.shape, target.shape)

    def meter_invalid_update_test(self, meter, model_output, target, **kwargs):
        """
        Runs a valid meter test. Does not reset meter before / after running
        """
        if not isinstance(model_output, list):
            model_output = [model_output]

        if not isinstance(target, list):
            target = [target]

        with self.assertRaises(AssertionError):
            for i in range(len(model_output)):
                meter.update(model_output[i], target[i], **kwargs)

    def test_single_meter_update_and_reset(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        for multilabel_mode in MultilabelModes:
            for top_k in [1, 2]:
                meter = AccuracyMeter(top_k, multilabel_mode)

                # Batchsize = 3, num classes = 3, score is a value in {1, 2,
                # 3}...3 is the highest score
                model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])

                # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
                target = torch.tensor([0, 1, 2])

                # Only the first sample has top class correct, first and third
                # sample have correct class in top 2
                expected_value = {1: 100 / 3.0, 2: 200 / 3.0}[top_k]

                self.meter_update_and_reset_test(
                    meter, model_output, target, expected_value
                )

    def test_double_meter_update_and_reset(self):
        for multilabel_mode in MultilabelModes:
            for top_k in [1, 2]:
                meter = AccuracyMeter(top_k, multilabel_mode)

                # Batchsize = 3, num classes = 3, score is a value in {1, 2,
                # 3}...3 is the highest score...two batches in this test
                model_outputs = [
                    torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),
                    torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),
                ]

                # Class 0 is the correct class for sample 1, class 2 for
                # sample 2, etc, in both batches
                targets = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]

                # First batch has top-1 accuracy of 1/3.0, top-2 accuracy of 2/3.0
                # Second batch has top-1 accuracy of 2/3.0, top-2 accuracy of 3/3.0
                expected_value = {1: 300 / 6.0, 2: 500 / 6.0}[top_k]

                self.meter_update_and_reset_test(
                    meter, model_outputs, targets, expected_value
                )

    def test_single_meter_update_and_reset_onehot(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with onehot target.
        """
        for multilabel_mode in MultilabelModes:
            for top_k in [1, 2]:
                meter = AccuracyMeter(top_k, multilabel_mode)

                # Batchsize = 3, num classes = 3, score is a value in {1, 2,
                # 3}...3 is the highest score
                model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])

                # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
                target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                # Only the first sample has top class correct, first and third
                # sample have correct class in top 2
                expected_value = {1: 100 / 3.0, 2: 200 / 3.0}[top_k]

                self.meter_update_and_reset_test(
                    meter, model_output, target, expected_value
                )

    def test_single_meter_update_and_reset_multilabel(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with multilabel target.
        """
        for top_k in [1, 2]:
            # Batchsize = 7, num classes = 3, score is a value in {1, 2,
            # 3}...3 is the highest score
            model_output = torch.tensor(
                [
                    [3, 2, 1],
                    [3, 1, 2],
                    [1, 3, 2],
                    [1, 2, 3],
                    [2, 1, 3],
                    [2, 3, 1],
                    [1, 3, 2],
                ]
            )

            target = torch.tensor(
                [
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 0, 1],
                ]
            )

            meter = AccuracyMeter(top_k, MultilabelModes.CLASSY)
            # 1st, 4th, 5th, 6th sample has top class correct, 2nd and 7th have at least
            # one correct class in top 2.
            expected_value = {1: 400 / 7.0, 2: 600 / 7.0}[top_k]
            self.meter_update_and_reset_test(
                meter, model_output, target, expected_value
            )

            meter = AccuracyMeter(top_k, MultilabelModes.RECALL)
            # How many positive classes did we recall in top-k out of all we had
            # to recall
            expected_value = {1: 400 / 12.0, 2: 800 / 12.0}[top_k]
            self.meter_update_and_reset_test(
                meter, model_output, target, expected_value
            )

    def test_meter_invalid_model_output(self):
        for top_k in [1, 2]:
            meter = AccuracyMeter(top_k)
            # This model output has 3 dimensions instead of expected 2
            model_output = torch.tensor(
                [[[3, 2, 1], [1, 2, 3]], [[-1, -3, -4], [-10, -90, -100]]]
            )
            target = torch.tensor([0, 1, 2])

            self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_invalid_target(self):
        for top_k in [1, 2]:
            meter = AccuracyMeter(top_k)
            model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])
            # Target has 3 dimensions instead of expected 1 or 2
            target = torch.tensor([[[0, 1, 2], [0, 1, 2]]])

            self.meter_invalid_meter_input_test(meter, model_output, target)

    def test_meter_invalid_topk(self):
        for top_k in [1, 5]:
            meter = AccuracyMeter(top_k)
            model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])
            target = torch.tensor([0, 1, 2])

            self.meter_invalid_meter_input_test(meter, model_output, target)
