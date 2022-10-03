import contextlib
import unittest
from functools import partial
from typing import Any, Dict

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from omnivore.data.api import VisionSample
from omnivore.data.transforms.mask_image_modeling import MaskImageModeling, RandMasking
from omnivore.data.transforms.transform_wrappers import MaskingTransform
from omnivore.models.fsdp_model_utils import FSDPSettings, is_valid_fsdp_model
from omnivore.models.vision_transformer import Attention, Decoder, VisionTransformer
from omnivore.utils.testing import (
    assert_all_close_recursive,
    gpu_test,
    init_distributed_on_file,
    with_temp_files,
)
from torch.nn.parallel import DistributedDataParallel


class TestFullyShardedVisionTransformer(unittest.TestCase):
    @gpu_test(gpu_count=2)
    def test_vit_supervised(self):
        self._test_fsdp_vit_with_options(
            self._build_vit_supervised,
            self._get_input_target_loss_supervised,
            {
                "mixed_precision": False,
                "use_checkpoint": False,
                "reshard_after_forward": False,
            },
        )

    @gpu_test(gpu_count=2)
    def test_vit_supervised_optimized(self):
        self._test_fsdp_vit_with_options(
            self._build_vit_supervised,
            self._get_input_target_loss_supervised,
            {
                "mixed_precision": True,
                "use_checkpoint": True,
                "reshard_after_forward": False,
            },
        )

    @gpu_test(gpu_count=2)
    def test_vit_supervised_optimized_with_sharding(self):
        self._test_fsdp_vit_with_options(
            self._build_vit_supervised,
            self._get_input_target_loss_supervised,
            {
                "mixed_precision": True,
                "use_checkpoint": True,
                "reshard_after_forward": True,
            },
        )

    @gpu_test(gpu_count=2)
    def test_vit_mae(self):
        self._test_fsdp_vit_with_options(
            self._build_vit_mae,
            self._get_input_target_loss_mae,
            {
                "mixed_precision": False,
                "use_checkpoint": False,
                "reshard_after_forward": True,
            },
        )

    @gpu_test(gpu_count=2)
    def test_vit_mae_optimized(self):
        self._test_fsdp_vit_with_options(
            self._build_vit_mae,
            self._get_input_target_loss_mae,
            {
                "mixed_precision": True,
                "use_checkpoint": True,
                "reshard_after_forward": True,
            },
        )

    def _test_fsdp_vit_with_options(
        self, vit_builder, input_target_loss_builder, options, world_size: int = 2
    ):
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                self._worker_test_fsdp_vit,
                (
                    vit_builder,
                    input_target_loss_builder,
                    sync_file,
                    world_size,
                    options,
                ),
                nprocs=world_size,
            )

    @staticmethod
    def _build_vit_supervised(options, gpu_id):
        mixed_precision = options.get("mixed_precision", True)
        embed_dim = options.get("embed_dim", 192)
        num_heads = options.get("num_heads", 6)
        vit_depth = options.get("vit_depth", 6)

        attn_target = partial(
            Attention,
            num_heads=num_heads,
            proj_drop=0.0,
            qk_scale=None,
            qkv_bias=True,
            attn_drop=0,
        )

        torch.random.manual_seed(0)
        ref_vit = VisionTransformer(
            attn_target=attn_target,
            embed_dim=embed_dim,
            depth=vit_depth,
            drop_path_rate=0.0,
        ).cuda(gpu_id)

        torch.random.manual_seed(0)
        sharded_vit = VisionTransformer(
            fsdp_settings=FSDPSettings(
                compute_dtype="float32",
                mixed_precision=mixed_precision,
                fp32_reduce_scatter=mixed_precision,
                full_precision_layers=["torch.nn.LayerNorm"],
            ),
            attn_target=attn_target,
            embed_dim=embed_dim,
            depth=vit_depth,
            drop_path_rate=0.0,
        ).cuda(gpu_id)
        return ref_vit, sharded_vit

    @staticmethod
    def _get_input_target_loss_supervised(options, gpu_id):
        embed_dim = options.get("embed_dim", 192)
        batch_size = 2
        x = torch.randn(size=(batch_size, 3, 224, 224)).cuda(gpu_id)
        targets = torch.randn(size=(batch_size, embed_dim)).cuda(gpu_id)
        criterion = nn.MSELoss()
        return {"x": x}, targets, criterion

    @staticmethod
    def _build_vit_mae(options, gpu_id):
        mixed_precision = options.get("mixed_precision", True)
        embed_dim = options.get("embed_dim", 192)
        dec_embed_dim = options.get("dec_embed_dim", 96)
        dec_depth = options.get("dec_depth", 2)
        num_heads = options.get("num_heads", 6)
        vit_depth = options.get("vit_depth", 2)

        attn_target = partial(
            Attention,
            num_heads=num_heads,
            proj_drop=0.0,
            qk_scale=None,
            qkv_bias=True,
            attn_drop=0,
        )

        decoder_fn = partial(
            Decoder,
            embed_dim=embed_dim,
            decoder_depth=dec_depth,
            decoder_embed_dim=dec_embed_dim,
            learnable_pos_embed=False,
            attn_target=attn_target,
        )

        torch.random.manual_seed(0)
        ref_vit = VisionTransformer(
            attn_target=attn_target,
            embed_dim=embed_dim,
            depth=vit_depth,
            drop_path_rate=0.0,
            learnable_pos_embed=False,
            masked_image_modeling=True,
            patch_dropping=True,
            decoder=decoder_fn,
        ).cuda(gpu_id)

        torch.random.manual_seed(0)
        sharded_vit = VisionTransformer(
            fsdp_settings=FSDPSettings(
                compute_dtype="float32",
                mixed_precision=mixed_precision,
                fp32_reduce_scatter=mixed_precision,
                full_precision_layers=["torch.nn.LayerNorm"],
            ),
            attn_target=attn_target,
            embed_dim=embed_dim,
            depth=vit_depth,
            drop_path_rate=0.0,
            learnable_pos_embed=False,
            masked_image_modeling=True,
            patch_dropping=True,
            decoder=decoder_fn,
        ).cuda(gpu_id)
        return ref_vit, sharded_vit

    @staticmethod
    def _get_input_target_loss_mae(options, gpu_id):
        def mae_loss(model_out, target):
            return nn.MSELoss()(model_out[1], target)

        dec_embed_dim = options.get("dec_embed_dim", 96)
        x = torch.randn(size=(3, 224, 224)).cuda(gpu_id)
        mask = MaskingTransform(
            masking_object=MaskImageModeling(
                pred_ratio=0.0,  # TODO 0.75,
                pred_ratio_var=0.0,
                pred_shape=RandMasking(),
                patch_size=[1, 16, 16],
            )
        )(VisionSample(vision=x)).mask
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        targets = torch.randn(size=(1, 196, dec_embed_dim)).cuda(gpu_id)
        criterion = mae_loss
        return {"x": x, "mask": mask}, targets, criterion

    @staticmethod
    def _worker_test_fsdp_vit(
        gpu_id: int,
        vit_build_fn,
        input_output_loss_fn,
        sync_file: str,
        world_size: int,
        options: Dict[str, Any],
    ):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        mixed_precision = options.get("mixed_precision", True)
        use_checkpoint = options.get("use_checkpoint", True)

        ref_vit, sharded_vit = vit_build_fn(options, gpu_id)
        ip, targets, criterion = input_output_loss_fn(options, gpu_id)

        # Check that the parameters are compatible between
        # ViT with FSDP enabled and ViT without FSDP enabled
        sharded_vit.load_state_dict(ref_vit.state_dict())
        assert is_valid_fsdp_model(sharded_vit)

        # Wrap non FSDP model with DDP
        ref_vit = DistributedDataParallel(ref_vit, device_ids=[gpu_id])

        # Create the optimizers
        ref_optim = optim.AdamW(ref_vit.parameters())
        fsdp_optim = optim.AdamW(sharded_vit.parameters())

        # Run a forward pass with mixed precision enabled
        context = (
            torch.cuda.amp.autocast() if mixed_precision else contextlib.suppress()
        )
        with context:
            ref_out = ref_vit(**ip, use_checkpoint=use_checkpoint)
            loss_ref = criterion(ref_out, targets)
            fsdp_out = sharded_vit(**ip, use_checkpoint=use_checkpoint)
            loss_fsdp = criterion(fsdp_out, targets)

        tc = unittest.TestCase()

        # Compare that the outputs are the same
        assert_all_close_recursive(
            ref_out,
            fsdp_out,
            tc,
            msg="FSDP enabled should not change the results",
            atol=1e-5,
        )
        assert torch.allclose(
            loss_ref, loss_fsdp, atol=1e-5
        ), "FSDP enabled should not change the results"

        # Run the backward and check the next output
        # which verifies that both DDP and FSDP lead to
        # the same gradients being computed
        ref_optim.zero_grad(set_to_none=True)
        fsdp_optim.zero_grad(set_to_none=True)
        loss_ref.backward()
        loss_fsdp.backward()
        ref_optim.step()
        fsdp_optim.step()

        with context:
            ref_out = ref_vit(**ip, use_checkpoint=use_checkpoint)
            fsdp_out = sharded_vit(**ip, use_checkpoint=use_checkpoint)
            loss_ref = criterion(ref_out, targets)
            loss_fsdp = criterion(fsdp_out, targets)

        assert torch.allclose(
            loss_ref, loss_fsdp, atol=1e-4
        ), "FSDP enabled should not change the results"
