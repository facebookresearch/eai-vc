import logging
from typing import List, Dict, Tuple, Any
import torch
from torch.nn import Module
from torch import nn
from torch.utils.checkpoint import checkpoint
from classy_vision.models import build_model, RegNet as ClassyRegNet

_bn_cls = (nn.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)


class Flatten(nn.Module):
    """
    Flatten module attached in the model. It basically flattens the input tensor.
    """

    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        """
        flatten the input feat
        """
        return torch.flatten(feat, start_dim=self.dim)

    def flops(self, x):
        """
        number of floating point operations performed. 0 for this module.
        """
        return 0


def _as_tensor(x: Tuple[int, int]) -> torch.Tensor:
    """
    An equivalent of `torch.as_tensor`, but works under tracing if input
    is a list of tensor. `torch.as_tensor` will record a constant in tracing,
    but this function will use `torch.stack` instead.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x)
    if isinstance(x, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in x):
        return torch.stack(x)
    return torch.as_tensor(x)


class MultiDimensionalTensor(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
    ):
        self.tensor = tensor
        self.mask = mask
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        """
        Effective batch size. For multi-crop augmentations,
        (as in SwAV https://arxiv.org/abs/2006.09882) this returns N * num_crops.
        Otherwise returns N.
        """
        return len(self.tensor)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.
        Args:
            idx: int or slice
        Returns:
            Tensor: an image of shape (H, W) or (C, H, W)
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @property
    def device(self):
        return self.tensor.device

    def to(self, device, non_blocking: bool):
        """
        Move the tensor and mask to the specified device.
        """
        # type: (Device) -> MultiDimensionalTensor # noqa
        cast_tensor = self.tensor.to(device, non_blocking=non_blocking)
        cast_mask = self.mask.to(device, non_blocking=non_blocking)
        return MultiDimensionalTensor(cast_tensor, cast_mask, self.image_sizes)

    @classmethod
    def from_tensors(cls, tensor_list: List[torch.Tensor]) -> "MultiDimensionalTensor":
        assert len(tensor_list) > 0
        assert isinstance(tensor_list, list)
        for t in tensor_list:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensor_list[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensor_list]
        image_sizes_tensor = [_as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values
        num_tensors = len(tensor_list)
        batch_shape_per_crop = list(tensor_list[0].shape[:-2]) + list(max_size)

        b, c, h, w = batch_shape_per_crop
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        nested_output_tensor = torch.zeros(
            (b * num_tensors, c, h, w), dtype=dtype, device=device
        )
        mask = torch.ones((b * num_tensors, h, w), dtype=torch.bool, device=device)

        for crop_num in range(num_tensors):
            img = tensor_list[crop_num]
            nested_output_tensor[
                (crop_num * b) : (crop_num + 1) * b, :, : img.shape[2], : img.shape[3]
            ].copy_(img)
            mask[
                (crop_num * b) : (crop_num + 1) * b, : img.shape[2], : img.shape[3]
            ] = False
        return MultiDimensionalTensor(nested_output_tensor, mask, image_sizes)


def transform_model_input_data_type(model_input, input_type: str):
    """
    Default model input follow RGB format. Based the model input specified,
    change the type. Supported types: RGB, BGR, LAB
    """
    model_output = model_input
    # In case the model takes BGR input type, we convert the RGB to BGR
    if input_type == "bgr":
        model_output = model_input[:, [2, 1, 0], :, :]
    # In case of LAB image, we take only "L" channel as input. Split the data
    # along the channel dimension into [L, AB] and keep only L channel.
    if input_type == "lab":
        model_output = torch.split(model_input, [1, 2], dim=1)[0]
    return model_output


def get_tunk_forward_interpolated_outputs(
    input_type: str,  # bgr or rgb or lab
    interpolate_out_feat_key_name: str,
    remove_padding_before_feat_key_name: str,
    feat: MultiDimensionalTensor,
    feature_blocks: nn.ModuleDict,
    feature_mapping: Dict[str, str] = None,
    use_checkpointing: bool = False,
    checkpointing_splits: int = 2,
) -> List[torch.Tensor]:
    """
    Args:
        input_type (AttrDict): whether the model input should be RGB or BGR or LAB
        interpolate_out_feat_key_name (str): what feature dimensions should be
            used to interpolate the mask
        remove_padding_before_feat_key_name (str): name of the feature block for which
            the input should have padding removed using the interpolated mask
        feat (MultiDimensionalTensor): model input
        feature_blocks (nn.ModuleDict): ModuleDict containing feature blocks in the model
        feature_mapping (Dict[str, str]): an optional correspondence table in between
            the requested feature names and the model's.
    Returns:
        out_feats: a list with the asked output features placed in the same order as in
            `out_feat_keys`.
    """
    if feature_mapping is not None:
        interpolate_out_feat_key_name = feature_mapping[interpolate_out_feat_key_name]

    model_input = transform_model_input_data_type(feat.tensor, input_type)
    out = get_trunk_forward_outputs(
        feat=model_input,
        out_feat_keys=[interpolate_out_feat_key_name],
        feature_blocks=feature_blocks,
        use_checkpointing=use_checkpointing,
        checkpointing_splits=checkpointing_splits,
    )
    # mask is of shape N x H x W and has 1.0 value for places with padding
    # we interpolate the mask spatially to N x out.shape[-2] x out.shape[-1].
    interp_mask = F.interpolate(feat.mask[None].float(), size=out[0].shape[-2:]).to(
        torch.bool
    )[0]

    # we want to iterate over the rest of the feature blocks now
    _, max_out_feat = parse_out_keys_arg(
        [interpolate_out_feat_key_name], list(feature_blocks.keys())
    )
    for i, (feature_name, feature_block) in enumerate(feature_blocks.items()):
        # We have already done the forward till the max_out_feat.
        # we forward through rest of the blocks now.
        if i >= (max_out_feat + 1):
            if remove_padding_before_feat_key_name and (
                feature_name == remove_padding_before_feat_key_name
            ):
                # negate the mask so that the padded locations have 0.0 and the non-padded
                # locations have 1.0. This is used to extract the h, w of the original tensors.
                interp_mask = (~interp_mask).chunk(len(feat.image_sizes))
                tensors = out[0].chunk(len(feat.image_sizes))
                res = []
                for i, tensor in enumerate(tensors):
                    w = torch.sum(interp_mask[i][0], dim=0)[0]
                    h = torch.sum(interp_mask[i][0], dim=1)[0]
                    res.append(feature_block(tensor[:, :, :w, :h]))
                out[0] = torch.cat(res)
            else:
                out[0] = feature_block(out[0])
    return out


def parse_out_keys_arg(
    out_feat_keys: List[str], all_feat_names: List[str]
) -> Tuple[List[str], int]:
    """
    Checks if all out_feature_keys are mapped to a layer in the model.
    Returns the last layer to forward pass through for efficiency.
    Allow duplicate features also to be evaluated.
    Adapted from (https://github.com/gidariss/FeatureLearningRotNet).
    """

    # By default return the features of the last layer / module.
    if out_feat_keys is None or (len(out_feat_keys) == 0):
        out_feat_keys = [all_feat_names[-1]]

    if len(out_feat_keys) == 0:
        raise ValueError("Empty list of output feature keys.")
    for _, key in enumerate(out_feat_keys):
        if key not in all_feat_names:
            raise ValueError(
                f"Feature with name {key} does not exist. "
                f"Existing features: {all_feat_names}."
            )

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max(all_feat_names.index(key) for key in out_feat_keys)

    return out_feat_keys, max_out_feat


def layer_splittable_before(m: Module) -> bool:
    """
    Return if this module can be split in front of it for checkpointing.
    We don't split the relu module.
    """
    return str(m) != "ReLU(inplace=True)"


def checkpoint_trunk(
    feature_blocks: Dict[str, Module],
    unique_out_feat_keys: List[str],
    checkpointing_splits: int,
) -> Dict[str, Module]:
    """
    Checkpoint a list of blocks and return back the split version.
    """
    # If checkpointing, split the model appropriately. The number of features requested
    # can be a limiting factor and override the number of activation chunks requested
    feature_blocks_bucketed = []

    # The features define the splits, first pass
    bucket = []

    for feature_name, feature_block in feature_blocks.items():
        # expand the res2,res3, res4, res5 kind of stages into sub-blocks so that we can
        # checkpoint them.
        if feature_name.startswith("res"):
            for b in feature_block:
                bucket.append(b)
        else:
            bucket.append(feature_block)

        if feature_name in unique_out_feat_keys:
            # Boundary, add to current bucket and move to next
            feature_blocks_bucketed.append([feature_name, bucket])
            bucket = []

    # If there are not enough splits, split again
    split_times = 0
    while len(feature_blocks_bucketed) < checkpointing_splits:
        # Find the biggest block
        lengths = [len(v) for _, v in feature_blocks_bucketed]
        assert max(lengths) > 0, "Something wrong, we shouldn't have an empty list"
        if max(lengths) == 1:
            # Everything is already split.
            break
        if max(lengths) == 2:
            # Find a splittable 2-element element.
            found = False
            for i, (_, v) in enumerate(feature_blocks_bucketed):
                if len(v) == 2 and layer_splittable_before(v[1]):
                    found = True
                    i_max = i
                    break
            if not found:
                # Didn't find good 2-element block, we are done.
                break
        else:
            # TODO: here we assume all >2-element blocks are splittable,
            #       i.e. there is not layer-relu-relu, case. But in general
            #       this is not the case. We can extend in the future.
            i_max = lengths.index(max(lengths))

        # Split the biggest block in two, keep the rest unchanged
        # Avoid inplace-relu.
        n_split_layers = len(feature_blocks_bucketed[i_max][1]) // 2
        biggest_block = feature_blocks_bucketed[i_max]
        if not layer_splittable_before(biggest_block[1][n_split_layers]):
            assert len(biggest_block[1]) > 2
            if n_split_layers == len(biggest_block[1]) - 1:
                n_split_layers -= 1
            else:
                n_split_layers += 1
        assert n_split_layers > 0 and n_split_layers < len(
            biggest_block[1]
        ), "Should never split into an empty list and the rest"

        feature_blocks_bucketed = (
            feature_blocks_bucketed[:i_max]
            + [[f"activation_split_{split_times}", biggest_block[1][:n_split_layers]]]
            + [[biggest_block[0], biggest_block[1][n_split_layers:]]]
            + feature_blocks_bucketed[(i_max + 1) :]
        )
        split_times += 1

    # Replace the model with the bucketed version, checkpoint friendly
    feature_blocks = {
        block[0]: nn.Sequential(*block[1]) for block in feature_blocks_bucketed
    }
    # Make sure we didn't loss anything
    assert len(feature_blocks) == len(feature_blocks_bucketed)
    return feature_blocks


def get_trunk_forward_outputs(
    feat: torch.Tensor,
    out_feat_keys: List[str],
    feature_blocks: nn.ModuleDict,
    feature_mapping: Dict[str, str] = None,
    use_checkpointing: bool = False,
    checkpointing_splits: int = 2,
) -> List[torch.Tensor]:
    """
    Args:
        feat: model input.
        out_feat_keys: a list/tuple with the feature names of the features that
            the function should return. By default the last feature of the network
            is returned.
        feature_blocks: ModuleDict containing feature blocks in the model
        feature_mapping: an optional correspondence table in between the requested
            feature names and the model's.
    Returns:
        out_feats: a list with the asked output features placed in the same order as in
        `out_feat_keys`.
    """

    # Sanitize inputs
    if feature_mapping is not None:
        out_feat_keys = [feature_mapping[f] for f in out_feat_keys]

    out_feat_keys, max_out_feat = parse_out_keys_arg(
        out_feat_keys, list(feature_blocks.keys())
    )

    # Forward pass over the trunk
    unique_out_feats = {}
    unique_out_feat_keys = list(set(out_feat_keys))

    # FIXME: Ideally this should only be done once at construction time
    if use_checkpointing:
        feature_blocks = checkpoint_trunk(
            feature_blocks, unique_out_feat_keys, checkpointing_splits
        )

        # If feat is the first input to the network, it doesn't have requires_grad,
        # which will make checkpoint's backward function not being called. So we need
        # to set it to true here.
        feat.requires_grad = True

    # Go through the blocks, and save the features as we go
    # NOTE: we are not doing several forward passes but instead just checking
    # whether the feature is requested to be returned.
    for i, (feature_name, feature_block) in enumerate(feature_blocks.items()):
        # The last chunk has to be non-volatile
        if use_checkpointing and i < len(feature_blocks) - 1:
            # Un-freeze the running stats in any BN layer
            for m in filter(lambda x: isinstance(x, _bn_cls), feature_block.modules()):
                m.track_running_stats = m.training

            feat = checkpoint(feature_block, feat)

            # Freeze the running stats in any BN layer
            # the checkpointing process will have to do another FW pass
            for m in filter(lambda x: isinstance(x, _bn_cls), feature_block.modules()):
                m.track_running_stats = False
        else:
            feat = feature_block(feat)

        # This feature is requested, store. If the same feature is requested several
        # times, we return the feature several times.
        if feature_name in unique_out_feat_keys:
            unique_out_feats[feature_name] = feat

        # Early exit if all the features have been collected
        if i == max_out_feat and not use_checkpointing:
            break

    # now return the features as requested by the user. If there are no duplicate keys,
    # return as is.
    if len(unique_out_feat_keys) == len(out_feat_keys):
        return list(unique_out_feats.values())

    output_feats = []
    for key_name in out_feat_keys:
        output_feats.append(unique_out_feats[key_name])
    return output_feats


class RegNet(torch.nn.Module):
    def __init__(
        self,
        trunk_config,
        INPUT_TYPE="bgr",
        use_activation_checkpointing=False,
        NUM_ACTIVATION_CHECKPOINTING_SPLITS=8,
    ):
        super().__init__()
        self.INPUT_TYPE = INPUT_TYPE
        self.use_activation_checkpointing = use_activation_checkpointing
        self.activation_checkpointing_splits = NUM_ACTIVATION_CHECKPOINTING_SPLITS

        if "name" in trunk_config:
            name = trunk_config["name"]
            if name == "anynet":
                model = build_model(trunk_config)
            else:
                logging.info(f"Building model: RegNet: {name}")
                model = build_model({"name": name})
        else:
            logging.info("Building model: RegNet from yaml config")
            model = ClassyRegNet.from_config(trunk_config)

        # Now map the models to the structure we want to expose for SSL tasks
        # The upstream RegNet model is made of :
        # - `stem`
        # - n x blocks in trunk_output, named `block1, block2, ..`

        # We're only interested in the stem and successive blocks
        # everything else is not picked up on purpose
        feature_blocks: List[Tuple[str, nn.Module]] = []

        # - get the stem
        feature_blocks.append(("conv1", model.stem))

        # - get all the feature blocks
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        # - finally, add avgpool and flatten.
        feature_blocks.append(("avgpool", nn.AdaptiveAvgPool2d((1, 1))))
        feature_blocks.append(("flatten", Flatten(1)))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                use_checkpointing=self.use_activation_checkpointing,
                checkpointing_splits=self.activation_checkpointing_splits,
            )
        else:
            model_input = transform_model_input_data_type(x, self.INPUT_TYPE)
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                use_checkpointing=self.use_activation_checkpointing,
                checkpointing_splits=self.activation_checkpointing_splits,
            )
        return out[0]


def load_vissl_checkpoint(checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    return checkpoint_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
