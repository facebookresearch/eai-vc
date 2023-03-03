#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--nocluster",
        action="store_true",
        default=False,
        help="Run outside of FAIR cluster.",
    )


@pytest.fixture
def nocluster(request):
    return request.config.getoption("--nocluster")
