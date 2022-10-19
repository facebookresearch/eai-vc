import os
import unittest

from omnivore.utils.testing import (
    compose_omnivore_config,
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestFSDPTrainer(unittest.TestCase):
    @gpu_test(gpu_count=2)
    def test_fsdp_vit_integration(self):
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/vit_train_synthetic",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)

        with in_temporary_directory() as pretrain_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/vit_train_synthetic_fsdp",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={pretrain_dir}",
                ]
            )
            run_integration_test(cfg)
            cp_path = os.path.join(pretrain_dir, "checkpoints", "checkpoint.pt")

            with in_temporary_directory() as eval_dir:
                cfg = compose_omnivore_config(
                    [
                        "+experiments=tests/vit_eval_synthetic_fsdp",
                        "launcher.gpus_per_node=2",
                        f"launcher.experiment_log_dir={eval_dir}",
                        f"variables.path_to_checkpoint={cp_path}",
                    ]
                )
                run_integration_test(cfg)

    @gpu_test(gpu_count=2)
    def test_fsdp_mae_integration(self):
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/mae_train_synthetic",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)

        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/mae_train_synthetic_fsdp",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)

    @gpu_test(gpu_count=2)
    def test_fsdp_slip_integration(self):
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/slip_train_synthetic_partial",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)

        # TODO - does not work because of the way SLIP works
        #  Fix is simple but analyse the impact first
        """
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/slip_train_synthetic_fsdp",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)
        """

    @gpu_test(gpu_count=2)
    def test_fsdp_clip_integration(self):
        with in_temporary_directory() as exp_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/clip_train_synthetic",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={exp_dir}",
                ]
            )
            run_integration_test(cfg)

        with in_temporary_directory() as pretrain_dir:
            cfg = compose_omnivore_config(
                [
                    "+experiments=tests/clip_train_synthetic_fsdp",
                    "launcher.gpus_per_node=2",
                    f"launcher.experiment_log_dir={pretrain_dir}",
                ]
            )
            run_integration_test(cfg)
            cp_path = os.path.join(pretrain_dir, "checkpoints", "checkpoint.pt")
            # TODO - evaluate cp_path
            print(cp_path)
