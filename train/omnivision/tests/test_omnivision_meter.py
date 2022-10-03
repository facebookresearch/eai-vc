import unittest

import torch
import torch.distributed
from omnivision.meters.omnivision_meter import OmnivisionMeter
from tests.utils import run_distributed_test
from torch.distributed import ReduceOp


class DummyMeter(OmnivisionMeter):
    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.tensor(0.0), ReduceOp.SUM)
        self.register_buffer("b", torch.tensor(0.0), ReduceOp.SUM)
        self.register_buffer("c", [], None)

    def update(self, a, b, c) -> None:
        self.a += a
        self.b += b
        self.c.append(c)

    def compute(self) -> torch.Tensor:
        return self.a, self.b, self.c


def test_meter_synced(rank, world_size):
    tc = unittest.TestCase()

    meter = DummyMeter()
    meter.set_sync_device(torch.device("cpu"))
    a = torch.tensor(1.0)
    b = torch.tensor(2.0)
    c = torch.ones((1, 10)) * rank
    meter.update(a, b, c)
    res = meter.compute()
    torch.testing.assert_close(res, (a, b, [c]))
    res = meter.compute_synced()
    torch.testing.assert_close(
        res,
        (
            a * world_size,
            b * world_size,
            [torch.vstack([torch.ones((1, 10)) * i for i in range(world_size)])],
        ),
    )
    res = meter.compute()
    torch.testing.assert_close(res, (a, b, [c]))


class TestOmnivisionMeter(unittest.TestCase):
    def test_meter_synced(self):
        run_distributed_test(test_meter_synced, 2)
