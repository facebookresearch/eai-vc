from functools import partial

import numpy as np
import torch
from omnivore.data.api import VisionSample
from omnivore.data.transforms.mask_image_modeling import MaskImageModeling, RandMasking
from omnivore.data.transforms.transform_wrappers import MaskingTransform
from omnivore.models.vision_transformer import Attention, Decoder, VisionTransformer

if __name__ == "__main__":
    patch_embed_type = "tmae"
    # patch_embed_type = "generic"
    img_size = [3, 16, 224, 224]
    patch_size = [2, 16, 16]

    vit_embed_dim = 768
    vit_depth = 2
    vit_num_heads = 12

    dec_embed_dim = 384
    dec_depth = 2
    dec_num_heads = 16

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
        embed_dim=dec_embed_dim,
        decoder_depth=dec_depth,
        decoder_embed_dim=dec_embed_dim,
        learnable_pos_embed=False,
        attn_target=dec_attn_target,
    )

    vit_attn_target = partial(
        Attention,
        num_heads=vit_num_heads,
        proj_drop=0.0,
        qk_scale=None,
        qkv_bias=True,
        attn_drop=0,
    )
    torch.random.manual_seed(0)
    vit = VisionTransformer(
        img_size=img_size,  # type: ignore
        patch_size=patch_size,  # type: ignore
        in_chans=3,
        embed_dim=vit_embed_dim,
        depth=vit_depth,
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
                out_channels=vit_embed_dim,
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
    mask = MaskingTransform(
        masking_object=MaskImageModeling(
            pred_ratio=0.75,  # type: ignore
            pred_ratio_var=0.0,  # type: ignore
            pred_shape=RandMasking(),
            patch_size=patch_size,  # type: ignore
        )
    )(VisionSample(vision=x)).mask

    x = x.unsqueeze(0)
    mask = mask.unsqueeze(0)

    _, feats = vit(x=x, mask=mask)

    print("x.shape {}".format(x.shape))
    print("mask.shape {}".format(mask.shape))
    print("feats.shape {}".format(feats.shape))

    num_tokens = int(np.prod(np.array(img_size[1:]) / np.array(patch_size)))
    assert feats.shape == (1, num_tokens, dec_embed_dim)
