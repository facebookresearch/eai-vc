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
