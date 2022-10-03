# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
from omnivore.data.api import Sample, VisionTextSample


class MIMOHeadWrapper(nn.Module):
    """Attaches multiple input multiple output heads to the trunk using forward hooks.

    Args:
        trunk: Any model to which you want to attach the heads to.
        heads: A list of dicts with the following keys:
            fork_module: The module which the head will be applied to. It can be an
                empty string, in which case the head is attached to the trunk's output.
            head: The head which is to be attached.
            input_key: The head will only run on inputs with this key. If set to
                `None` the head will be applied to all inputs.
            output_key: The head will produce this output key. If set to `None`, the
                output key will be the same as the input key.

            An example heads value can look like -
            ```
            [
                {
                    "fork_module": "layer_1.layer_a.layer_alpha",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_1",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_2",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_2",
                    "output_key": "out_3",
                },
                {
                    "fork_module": "",
                    "head": nn.Conv2d(in_feat, out_feat),
                    "input_key": None,
                    "output_key": None,
                },
            ]
            ```
        trunk_fields: A list of dicts with the following keys:
            input_key: The input key this rule applies to. If `None`, applies to all
                inputs.
            args: These specific keys will be fetched from the sample and passed as
                *args to the trunk for the specified `input_key`.
            kwargs: These specific keys will be fetched from the sample and passed as
                **kwargs to the trunk for the specified `input_key`.

            Example -
            ```
            [
                {
                    "input_key": "dataset_1",
                    "args": ["vision"]
                },
                {
                    "input_key": "dataset_2",
                    "args": ["vision"],
                    "kwargs": {"mask": "mask"}
                },
                {
                    "input_key": "dataset_3",
                    "args": ["text"]
                },
            ]
            ```

        Note that two heads cannot produce the same output key in the same forward pass.

    Returns:
        A dict with keys corresponding to the output keys which match with the input key.
    """

    @dataclass
    class HeadArgs:
        fork_module: str
        head: nn.Module
        input_key: Optional[str]
        output_key: Optional[str]

    @dataclass
    class TrunkFieldArgs:
        input_key: Optional[str]
        args: List[str] = field(default_factory=list)
        kwargs: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        trunk: nn.Module,
        heads: List[Dict],
        trunk_fields: List[Dict],
        handle_list_inputs=False,
    ) -> None:
        """WARNING: handle_list_inputs is a hack which needs to be refactored away."""
        super().__init__()

        self.trunk = trunk
        self.handle_list_inputs = handle_list_inputs

        # cast to HeadArgs for input validation
        heads = [self.HeadArgs(**head_dict) for head_dict in heads]
        # cast to TrunkFieldArgs for input validation
        trunk_fields = [
            self.TrunkFieldArgs(**trunk_fields_dict)
            for trunk_fields_dict in trunk_fields
        ]

        self.head_name_to_fork_module = {}
        self.heads = nn.ModuleList()
        self.head_input_keys = []
        self.head_output_keys = []
        self.head_fork_modules = []

        for head_args in heads:
            self.heads.append(head_args.head)
            self.head_input_keys.append(head_args.input_key)
            self.head_output_keys.append(head_args.output_key)
            self.head_fork_modules.append(head_args.fork_module)

        self.trunk_field_args = {}
        self.trunk_field_kwargs = {}
        for trunk_fields_elem in trunk_fields:
            input_key = trunk_fields_elem.input_key
            if input_key in self.trunk_field_args:
                raise KeyError(
                    f"Multiple trunk_fields specified for the same input_key: {input_key}"
                )
            self.trunk_field_args[input_key] = trunk_fields_elem.args
            self.trunk_field_kwargs[input_key] = trunk_fields_elem.kwargs

        # outputs is used as a temporary storage of the head outputs
        self.outputs = {}

        # input_key is used to specify which key is currently being processed
        self.input_key = None

        # handles to the hooks which can be used for removing the hooks if needed
        self.hook_handles = []
        self._register_hooks()

    def __deepcopy__(self, memo: Dict):
        # when we register a `hook_fn` inside `_register_hooks`, we bind it to `self`.
        # when creating a deepcopy, this binding refers to the original object
        # instead of the new object. this means that the hooks on the new object are
        # registered, but refer to the `self` of the original object.
        # to solve this, we remove the handles and re-register.

        # this portion is to call the original deepcopy by deleting this function
        # and adding it back after the deepcopy has been created.
        # we delete the function so that when we call copy.deepcopy() it uses python's
        # default deepcopy behavior, and then we override it in our custom __deepcopy__
        deepcopy_fn = self.__deepcopy__
        self.__deepcopy__ = None
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_fn
        cp.__deepcopy__ = deepcopy_fn

        # now that we have a default deepcopied object, we remove the hooks with the
        # incorrect binding and register them again.
        for handle in cp.hook_handles:
            handle.remove()
        cp.hook_handles = []
        cp._register_hooks()
        return cp

    def _register_hooks(self):
        for i, head in enumerate(self.heads):
            fork_module_name = self.head_fork_modules[i]

            def hook_fn(
                module,
                module_in,
                module_out,
                # the following variables are passed as kwargs in the closure to avoid
                # late binding in python
                head_method=head,
                in_key=self.head_input_keys[i],
                out_key=self.head_output_keys[i],
            ):
                if in_key is not None and self.input_key != in_key:
                    return
                if out_key is None:
                    out_key = self.input_key
                if out_key in self.outputs:
                    # reset state before raising
                    self.outputs = {}
                    self.input_key = None
                    raise ValueError(
                        f"Two heads produced the same output key `{out_key}` during forward"
                    )
                self.outputs[out_key] = head_method(module_out)

            fork_module = self.trunk.get_submodule(fork_module_name)
            self.hook_handles.append(fork_module.register_forward_hook(hook_fn))

    def _get_trunk_fields(self):
        fields_args = self.trunk_field_args.get(self.input_key)
        fields_kwargs = self.trunk_field_kwargs.get(self.input_key)
        if fields_args is None:
            assert fields_kwargs is None
            fields_args = self.trunk_field_args.get(None)
            fields_kwargs = self.trunk_field_kwargs.get(None)
            if fields_args is None:
                assert fields_kwargs is None
                raise ValueError(
                    f"No trunk fields specified for input key: {self.input_key}"
                )
        return fields_args, fields_kwargs

    def forward_sub_batch(self, sub_batch, *args, **kwargs):
        assert isinstance(sub_batch, Sample), f"Received {type(sub_batch)}"
        fields_args, fields_kwargs = self._get_trunk_fields()
        sample_args = [getattr(sub_batch, arg) for arg in fields_args]
        sample_kwargs = {
            key: getattr(sub_batch, field) for key, field in fields_kwargs.items()
        }
        self.trunk(*sample_args, *args, **sample_kwargs, **kwargs)

    def forward(self, batch, *args, **kwargs) -> Dict:
        assert isinstance(batch, Mapping)
        assert len(self.outputs) == 0, f"self.outputs has keys: {self.outputs.keys()}"
        for key, sub_batch in batch.items():
            self.input_key = key
            if self.handle_list_inputs and isinstance(sub_batch.vision, Sequence):
                # FIXME: this only handles list inputs for the field "vision"
                assert len(batch) == 1
                out_vals = []
                for e in sub_batch.vision:
                    e_batch = copy.copy(sub_batch)
                    e_batch.vision = e
                    self.forward_sub_batch(e_batch, *args, **kwargs)
                    assert len(self.outputs) == 1
                    out_key, out_val = self.outputs.popitem()
                    out_vals.append(out_val)
                return {out_key: torch.cat(out_vals)}
            else:
                self.forward_sub_batch(sub_batch, *args, **kwargs)
        outputs = self.outputs
        self.input_key = None
        self.outputs = {}
        return outputs


class MultiModalZeroShotEvalWrapper(nn.Module):
    """
    Takes a multimodal input and computes features for each modality using
    corresponding models.
    """

    def __init__(
        self,
        vision_trunk,
        text_trunk,
        label_strings,
        logit_scale_output_key="logit_scale",
        temp_init_value=0.07,
        learnable_logit_scale=True,
    ) -> None:
        super().__init__()
        self.vision_trunk = vision_trunk
        assert isinstance(vision_trunk, MIMOHeadWrapper)
        self.text_trunk = text_trunk
        assert isinstance(text_trunk, MIMOHeadWrapper)
        self.label_strings = label_strings
        # To be used for classifying actions by matching to
        # Will be set before each validation run
        self.target_text_features = None
        logit_scale_data = torch.ones([], dtype=torch.float32) * np.log(
            1 / temp_init_value
        )
        if learnable_logit_scale:
            self.logit_scale = torch.nn.Parameter(logit_scale_data, requires_grad=True)
        else:
            self.register_buffer("logit_scale", logit_scale_data)
        self.logit_scale_output_key = logit_scale_output_key

    def _get_feat_from_dict(self, feat_dict):
        """
        This function is needed because the 2 trunks in here are MIMOWrappers,
        which return a dict of features. So this function reads out the features.
        Assumes only 1 output feature
        """
        assert len(feat_dict) == 1
        return feat_dict[list(feat_dict.keys())[0]]

    def parse_kwargs_per_trunk(self, kwargs):
        vision_trunk_kwargs = kwargs["vision_trunk"] if "vision_trunk" in kwargs else {}
        text_trunk_kwargs = kwargs["text_trunk"] if "text_trunk" in kwargs else {}
        return vision_trunk_kwargs, text_trunk_kwargs

    def forward_train(self, batch, *args, **kwargs):
        assert isinstance(batch, Mapping)
        outputs = {}
        for key, sub_batch in batch.items():
            kwargs_vision, kwargs_text = self.parse_kwargs_per_trunk(kwargs)
            out_vision = self.vision_trunk({key: sub_batch}, *args, **kwargs_vision)
            out_text = self.text_trunk({key: sub_batch}, *args, **kwargs_text)
            assert len(set(out_vision.keys()) & set(out_text.keys())) == 0
            outputs[key] = {**out_vision, **out_text}
        outputs[key].update({self.logit_scale_output_key: self.logit_scale.exp()})
        return outputs

    def on_validation_epoch_start(self):
        assert isinstance(self.label_strings, Mapping)
        self.target_text_features = {}
        for key, label_strings in self.label_strings.items():
            num_classes = len(label_strings)
            print(
                f"Validation start: Computing target string features for "
                f"{num_classes} classes in {key}..."
            )
            with torch.no_grad():
                all_label_embeddings = []
                for cls_idx in range(num_classes):
                    all_label_embeddings.append(
                        self._get_feat_from_dict(
                            self.text_trunk(
                                {
                                    key: VisionTextSample(
                                        text=label_strings[cls_idx].to(
                                            next(self.text_trunk.parameters()).device
                                        )
                                    )
                                }
                            )
                        )
                    )
                self.target_text_features[key] = torch.stack(all_label_embeddings)
                # normalize all text features
                self.target_text_features[key] = torch.nn.functional.normalize(
                    self.target_text_features[key], dim=-1, p=2
                )

                # mean across templates (dim=1)
                assert self.target_text_features[key].ndim == 3
                self.target_text_features[key] = self.target_text_features[key].mean(
                    dim=1
                )

                # renormalize
                self.target_text_features[key] = torch.nn.functional.normalize(
                    self.target_text_features[key], dim=-1, p=2
                )

            print("...computing target strings done.")

    def on_validation_epoch_end(self):
        # Setting back to None so we don't mistakenly use the same features
        # again in the next epoch the evaluation is done. Must be recomputed
        # in on_validation_epoch_start.
        self.target_text_features = None

    def forward_val(self, batch, *args, **kwargs):
        assert isinstance(batch, Mapping)
        outputs = {}
        kwargs_vision, _ = self.parse_kwargs_per_trunk(kwargs)
        for key, sub_batch in batch.items():
            image_feature = self._get_feat_from_dict(
                self.vision_trunk({key: sub_batch}, *args, **kwargs_vision)
            )
            image_feature = torch.nn.functional.normalize(image_feature, p=2, dim=-1)
            img_txt_matches = image_feature @ self.target_text_features[key].t()
            outputs[key] = img_txt_matches
        return outputs

    def forward(self, batch, *args, **kwargs):
        if self.target_text_features is not None:
            # The text features are set by the on_validation_epoch_start function
            # Hence the validation epoch has started, so that forward should be called.
            # When that epoch finishes, the text features are set back to None
            return self.forward_val(batch, *args, **kwargs)
        return self.forward_train(batch, *args, **kwargs)
