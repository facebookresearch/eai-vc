import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--simple",
        action="store_true",
        default=False,
        help="Only run a basic resnet50 with random init",
    )


@pytest.fixture
def simple(request):
    return request.config.getoption("--simple")
