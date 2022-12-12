import time
from functools import partial

import numpy as np
import torch
from omnivore.data.api import VisionSample
from omnivore.data.transforms.mask_image_modeling import (
    RandMasking,
    MaskImageModeling,
    TmaeMaskImageModeling,
)
from omnivore.data.transforms.transform_wrappers import MaskingTransform
from omnivore.models.vision_transformer import Attention, Decoder, VisionTransformer

if __name__ == "__main__":
    patch_embed_type = "tmae"  # "tmae" or "generic"
    approach = "TMAE" if patch_embed_type == "tmae" else "OmniMAE"
    print("approach:", approach)

    BZ, T = 32, 4

    img_size = [3, T, 224, 224]
    patch_size = [1, 16, 16]

    enc_embed_dim = 768
    enc_depth = 12
    enc_num_heads = 12

    dec_embed_dim = 384
    dec_depth = 4
    dec_num_heads = 16

    pred_ratio = [0.85, 0.95]
    pred_ratio_var = [0.0, 0.0]
    avg_pred_ratio = sum(pred_ratio) / len(pred_ratio)
    dec_pred_ratio = 0.25
    dec_pred_ratio_var = 0.0

    dec_attn_target = partial(
        Attention,
        num_heads=dec_num_heads,
        proj_drop=0.0,
        qk_scale=None,
        qkv_bias=True,
        attn_drop=0,
    )

    decoder_fn = partial(
        Decoder,
        embed_dim=enc_embed_dim,
        decoder_depth=dec_depth,
        decoder_embed_dim=dec_embed_dim,
        learnable_pos_embed=False,
        attn_target=dec_attn_target,
    )

    vit_attn_target = partial(
        Attention,
        num_heads=enc_num_heads,
        proj_drop=0.0,
        qk_scale=None,
        qkv_bias=True,
        attn_drop=0,
    )

    vit = VisionTransformer(
        img_size=img_size,  # type: ignore
        patch_size=patch_size,  # type: ignore
        in_chans=3,
        embed_dim=enc_embed_dim,
        depth=enc_depth,
        mlp_ratio=4,
        attn_target=vit_attn_target,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        layer_scale_type=None,
        layer_scale_init_value=1e-4,
        patch_embed_type=patch_embed_type,
        patch_embed_params_list=[
            torch.nn.Conv3d(
                in_channels=3,
                out_channels=enc_embed_dim,
                kernel_size=patch_size,  # type: ignore
                stride=patch_size,  # type: ignore
            )
        ],
        layer_norm_eps=1e-6,
        masked_image_modeling=True,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=decoder_fn,
        mask_token_embed_dim=None,
        patch_drop_max_patches=-1,
        fsdp_settings=None,
    ).cuda()

    x = torch.randn(size=img_size).cuda()
    x = x.unsqueeze(0).repeat(BZ, 1, 1, 1, 1)
    print("x.shape: {}".format(x.shape))

    if patch_embed_type == "tmae":
        sample = MaskingTransform(
            masking_object=TmaeMaskImageModeling(
                pred_ratio=pred_ratio,
                pred_ratio_var=pred_ratio_var,
                pred_shape=RandMasking(),
                patch_size=patch_size,
                decoder_pred_ratio=dec_pred_ratio,
                decoder_pred_ratio_var=dec_pred_ratio_var,
            )
        )(VisionSample(vision=x))
    else:
        sample = MaskingTransform(
            masking_object=MaskImageModeling(
                pred_ratio=pred_ratio,
                pred_ratio_var=pred_ratio_var,
                pred_shape=RandMasking(),
                patch_size=patch_size,
            )
        )(VisionSample(vision=x))

    mask = sample.mask.unsqueeze(0).repeat(BZ, 1, 1, 1)
    print("mask.shape: {}".format(mask.shape))

    decoder_mask = None
    if hasattr(sample, "decoder_mask"):
        decoder_mask = sample.decoder_mask.unsqueeze(0).repeat(BZ, 1, 1, 1)
        assert decoder_mask.shape == mask.shape

    # initial execution for timing
    _, feats = vit(x=x, mask=mask, decoder_mask=decoder_mask)
    num_tokens_per_sample = int(np.prod(np.array(img_size[1:]) / np.array(patch_size)))
    assert feats.shape == (BZ, num_tokens_per_sample, dec_embed_dim)
    del feats  # clear memory

    print("max memory: {:0.1f} GB".format(torch.cuda.max_memory_allocated() / 1e9))

    # get timing over N batches
    N = 10
    tic = time.time()
    for _ in range(N):
        _, feats = vit(x=x, mask=mask, decoder_mask=decoder_mask)
        del feats  # clear memory
    print(
        "processing time: {:.1f} ms [{:0.1f} GB]".format(
            1000 * (time.time() - tic) / N, torch.cuda.max_memory_allocated() / 1e9
        )
    )
