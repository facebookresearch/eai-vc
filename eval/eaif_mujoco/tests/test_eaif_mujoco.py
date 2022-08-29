import pytest

from eaif_mujoco.gym_wrapper import env_constructor
from eaif_models import eaif_model_zoo

# Full Env list for testing
history_window = 3
seed = 123


@pytest.fixture(params=eaif_model_zoo)
def embedding_name(request, simple):
    model_name = request.param

    # Skip everything except randomly-initialized ResNet50 if
    # option "--simple" is applied
    if simple and model_name != "rn50_rand":
        pytest.skip()
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


@pytest.fixture(params=["dmc_walker_stand-v1", "relocate-v0", "kitchen_sdoor_open-v3"])
def env_name(request):
    return request.param


def test_env_embedding(env_name, embedding_name, device):
    e = env_constructor(
        env_name=env_name,
        embedding_name=embedding_name,
        history_window=history_window,
        seed=seed,
        device=device,
    )
    o = e.reset()
    assert o.shape[0] == e.env.embedding_dim * history_window
    o, r, d, ifo = e.step(e.action_space.sample())
    assert o.shape[0] == e.env.embedding_dim * history_window
