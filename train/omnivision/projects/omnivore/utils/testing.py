import functools
import os
import tempfile
from contextlib import contextmanager
from typing import Callable, Mapping, Sequence
from unittest import TestCase

import torch
import torch.distributed as dist


def assert_all_close_recursive(a, b, tc: TestCase, msg: str = "", atol: int = 1e-8):
    if isinstance(a, Mapping):
        assert isinstance(b, Mapping)
        assert tc.assertSetEqual(a.keys(), b.keys())
        for k in a:
            assert_all_close_recursive(a[k], b[k], tc, msg=msg, atol=atol)
    elif isinstance(a, Sequence):
        assert isinstance(b, Sequence)
        assert len(a) == len(b)
        for i in range(len(a)):
            assert_all_close_recursive(a[i], b[i], tc, msg=msg, atol=atol)
    else:
        assert torch.allclose(a, b, atol=atol), msg


@contextmanager
def with_temp_files(count: int):
    """
    Context manager to create temporary files and remove them
    after at the end of the context
    """
    if count == 1:
        fd, file_name = tempfile.mkstemp()
        yield file_name
        os.close(fd)
    else:
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])


@contextmanager
def in_temporary_directory(enabled: bool = True):
    """
    Context manager to create a temporary directory and remove
    it at the end of the context
    """
    if enabled:
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                yield temp_dir
            finally:
                os.chdir(old_cwd)
    else:
        yield os.getcwd()


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """

    def gpu_test_decorator(test_function: Callable):
        @functools.wraps(test_function)
        def wrapped_test(*args, **kwargs):
            if torch.cuda.device_count() >= gpu_count:
                return test_function(*args, **kwargs)

        return wrapped_test

    return gpu_test_decorator


def init_distributed_on_file(world_size: int, gpu_id: int, sync_file: str):
    """
    Init the process group need to do distributed training, by syncing
    the different workers on a file.
    """
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="file://" + sync_file,
        world_size=world_size,
        rank=gpu_id,
    )


def compose_omnivore_config(overrides):
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="omnivore.config"):
        return compose("defaults", overrides=overrides)


def run_integration_test(cfg):
    from omnivore.train_app_submitit import single_node_runner

    single_node_runner(cfg, main_port=5556)


def _identity(obj):
    return obj


def _do_nothing(obj):
    """
    Decorator that ignores the input and returns a function which does nothing instead
    """

    def _pass(*args, **kwargs):
        pass

    return _pass


def skip_test_if(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return _do_nothing(reason)
    return _identity
