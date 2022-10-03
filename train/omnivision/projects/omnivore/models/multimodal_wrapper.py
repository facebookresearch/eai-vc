#!/usr/bin/env python3

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np

import torch
import torch.nn as nn
from omnivore.data.api import BatchSample, BatchTextSample, FIELDS_WITH_LIST, Sample
from omnivore.meters import FINAL_LOGITS_NAME
from omnivore.models.helpers import singleton_dict_to_item
from omnivore.utils import Phase


@dataclass
class PreprocessorArgs:
    name: str
    preprocessor: nn.Module


@dataclass
class PostprocessorArgs:
    name: str
    postprocessor: nn.Module


@dataclass
class HeadArgs:
    fork_module: str
    head: nn.Module
    preprocessed_input_key: Optional[str]
    output_key: Optional[str]
    name: str = None
    head_to_clone: str = None


@dataclass
class SampleFieldMapping:
    input_fields: List[str]
    preprocessor_name: str
    output_key: str
    output_key_for_dict: bool


@dataclass
class SampleToModalityArgs:
    sample_type: Callable
    sample_field_to_modality: List[SampleFieldMapping]

    def parse_sample_field_to_modality(self):
        sample_field_to_modality = []
        for k in self.sample_field_to_modality:
            sample_field_to_modality.append(SampleFieldMapping(**k))
        self.sample_field_to_modality = sample_field_to_modality


@dataclass
class HeadToPostprocessorArgs:
    input_key: str
    postprocessor_name: str


@dataclass
class TrunkArgs:
    name: str
    trunk: nn.Module
    trunk_to_clone: str = None


@dataclass
class TokensToTrunkArgs:
    input_keys: str
    trunk_name: str


class ListInputReductionOps(Enum):
    MEAN = "mean"
    NO_OP = "no_op"
    CAT = "cat"


@dataclass
class DatasetListInputReductionElem:
    field_name: str
    reduction_op: ListInputReductionOps


class MultimodalWrapper(nn.Module):
    """
    Wrapper class that mainly handles modality specific preprocessing

    Control flow

    Data --> Preprocessor (modality aware) --> trunk (kwargs aware) --> head (modality aware) --> Postprocessor (head output aware)
    The returned output is grouped according to the keys in the Batch (for example `dataset keys`)

    **Preprocessors**
    Modality preprocessors are specified per input_modality_key, i.e., the key inside the Sample
    The core trunk model should be implemented separately
    Example:
    ```
    modality_preprocessors:
    - name: str
      preprocessor:
        _target_: ...
    - name: str
      preprocessor:
        _target_: ...
    sample_to_modality_preprocessor:
    - sample_type: <class of the sample>
      sample_field_to_modality:
      - input_fields: ["vision", "depth"]
        preprocessor_name: <str "name" field of the preprocessor>
        output_key: <str>
      - input_fields: ["vision", "depth"]
        preprocessor_name: <str "name" field of the preprocessor>
        output_key: <str>
    ```
    This specification returns two preprocessors one that operates on vision + depth (single forward call)
    and one that operates on text.
    Each preprocessor will return Dict of kwargs that the trunk can directly ingest
    In a forward pass, after all modality preprocessors run, we will have a List of size <= 2 that contains Tensors.

    **Trunk**
    The trunk takes the preprocessor's Dict output as kwargs.

    **Head**
    The heads operate on the `output_key` of the preprocessor
    ```
    heads:
        - head:
            _target_: ...
            fork_module: "" # layer of the trunk to use
            preprocessed_input_key: <output_key> produced by preprocessor
            output_key: str output produced by the heads
    ````

    **Postprocessor**
    ```
    postprocessors:
        - name: str
          postprocessor:
            _target_: ...
      head_to_postprocessor:
        - input_key: <output_key> produced by the head
          postprocessor_name: <str "name" of the postprocessor>
    ```
    """

    def __init__(
        self,
        modality_preprocessors: List[Dict],
        sample_to_modality_preprocessor: List[Dict],
        trunks: List[Dict],
        tokens_to_trunks: List[Dict],
        heads: List[Dict],
        postprocessors: List[Dict],
        head_to_postprocessor: List[Dict],
        list_input_reduction: str = ListInputReductionOps.CAT,
        dataset_specific_list_input_reduction: Dict = None,
    ) -> None:
        super().__init__()
        # cast to PreprocessorArgs for input validation
        modality_preprocessors = [
            PreprocessorArgs(**preprocessor) for preprocessor in modality_preprocessors
        ]
        self.modality_preprocessors = nn.ModuleDict()
        for el in modality_preprocessors:
            self.modality_preprocessors[el.name] = el.preprocessor

        # cast to SampleToModalityArgs for input validation
        sample_to_modality_preprocessor = [
            SampleToModalityArgs(**sample_mapping)
            for sample_mapping in sample_to_modality_preprocessor
        ]

        # perform some basic checks to spot errors in the config
        for sample_mapping in sample_to_modality_preprocessor:
            assert issubclass(sample_mapping.sample_type, Sample)
            sample_mapping.parse_sample_field_to_modality()
            output_keys = []
            for inputs in sample_mapping.sample_field_to_modality:
                assert inputs.preprocessor_name in self.modality_preprocessors
                output_keys.append(inputs.output_key)
            assert len(set(output_keys)) == len(output_keys)

        # save the mapping
        self.sample_to_modality_preprocessor = sample_to_modality_preprocessor
        self.list_input_reduction = ListInputReductionOps(list_input_reduction)
        self.dataset_specific_list_input_reduction = (
            dataset_specific_list_input_reduction
        )

        if dataset_specific_list_input_reduction is not None:
            assert isinstance(
                dataset_specific_list_input_reduction, Mapping
            ), f"dataset_specific_list_input_reduction should be mapping. Found {type(dataset_specific_list_input_reduction)}"
            self.dataset_specific_list_input_reduction = {}
            for (
                dataset_key,
                field_items,
            ) in dataset_specific_list_input_reduction.items():
                self.dataset_specific_list_input_reduction[dataset_key] = {}
                for field_item in field_items:
                    field_item = DatasetListInputReductionElem(**field_item)
                    self.dataset_specific_list_input_reduction[dataset_key][
                        field_item.field_name
                    ] = ListInputReductionOps(field_item.reduction_op)

        # Adding all the trunks
        # Validation
        self.trunks = nn.ModuleDict()
        trunks = [TrunkArgs(**trunk) for trunk in trunks]

        # Copy trunks if specified
        for trunk in trunks:
            if trunk.trunk_to_clone is not None:
                # Copy trunk by reference except for optionally some unshared parameters
                # (e.g. omnivore.models.simple_transformer.copy_trunk_with_unshared_layernorm)
                assert not isinstance(trunk.trunk, nn.Module)
                assert trunk.trunk_to_clone in [
                    t.name for t in trunks
                ], f"{trunk.trunk_to_clone} is not one of the defined trunks names."
                trunk_to_clone = next(
                    filter(lambda a: a.name == trunk.trunk_to_clone, trunks)
                ).trunk
                assert isinstance(
                    trunk_to_clone, nn.Module
                ), "Trunk to be cloned has to be already instantiated."
                assert isinstance(
                    trunk.trunk, Callable
                ), "Clone trunk has to be instantiated using a callable (e.g. omnivore.models.simple_transformer.copy_trunk_with_unshared_layernorm)."
                trunk.trunk = trunk.trunk(trunk_to_clone)

            assert isinstance(trunk.trunk, nn.Module)
            self.trunks[trunk.name] = trunk.trunk

        # Input validation
        tokens_to_trunks = [
            TokensToTrunkArgs(**tokens_to_trunk) for tokens_to_trunk in tokens_to_trunks
        ]
        self.tokens_to_trunks = {}
        for mapping in tokens_to_trunks:
            for input_key in mapping.input_keys:
                assert (
                    input_key not in self.tokens_to_trunks
                ), f"Already exists {input_key} -> {self.tokens_to_trunks[input_key]}"
                assert mapping.trunk_name in self.trunks
                self.tokens_to_trunks[input_key] = mapping.trunk_name

        # cast to HeadArgs for input validation
        heads = [HeadArgs(**head_dict) for head_dict in heads]

        # Add hooks based on what layer features the head operates on
        # Also clone heads (share parameters etc.) if specified
        self.head_name_to_fork_module = {}
        self.heads = nn.ModuleList()
        self.head_input_keys = []
        self.head_output_keys = []
        self.head_fork_modules = []

        for head_args in heads:
            # Copying head parameters
            if head_args.head_to_clone is not None:
                assert not isinstance(
                    head_args.head, nn.Module
                ), "head to clone should be a Callable"
                assert head_args.head_to_clone in [
                    h.name for h in heads
                ], f"{head_args.head_to_clone} is not one of the defined heads names."
                head_to_clone = next(
                    filter(lambda a: a.name == head_args.head_to_clone, heads)
                ).head
                assert isinstance(
                    head_to_clone, nn.Module
                ), "Head to be cloned has to be already instantiated."
                assert isinstance(
                    trunk.trunk, Callable
                ), "Clone trunk has to be instantiated using a callable (e.g. omnivore.models.simple_transformer.copy_trunk_with_unshared_layernorm)."
                head_args.head = head_args.head(head_to_clone)

            self.heads.append(head_args.head)
            self.head_input_keys.append(head_args.preprocessed_input_key)
            self.head_output_keys.append(head_args.output_key)
            self.head_fork_modules.append(head_args.fork_module)

        # outputs is used as a temporary storage of the head outputs
        self.outputs = {}

        # handles to the hooks which can be used for removing the hooks if needed
        self.hook_handles = []
        self._register_hooks()

        # setup post processors
        self.postprocessors = None
        if postprocessors is not None:
            # Input arg validation
            postprocessors = [
                PostprocessorArgs(**postprocessor) for postprocessor in postprocessors
            ]
            self.postprocessors = nn.ModuleDict()
            for el in postprocessors:
                self.postprocessors[el.name] = el.postprocessor
            head_to_postprocessor = [
                HeadToPostprocessorArgs(**mapping) for mapping in head_to_postprocessor
            ]
            self.head_to_postprocessor = {}
            for mapping in head_to_postprocessor:
                assert mapping.input_key in self.head_output_keys
                assert mapping.postprocessor_name in self.postprocessors
                self.head_to_postprocessor[
                    mapping.input_key
                ] = mapping.postprocessor_name

    def _register_hooks(self):
        for i, (head, head_input_key) in enumerate(
            zip(self.heads, self.head_input_keys)
        ):
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
                if in_key is not None and self.head_routing_key != in_key:
                    return
                if out_key is None:
                    out_key = self.head_routing_key
                if out_key in self.outputs:
                    # reset state before raising
                    self.outputs = {}
                    self.head_routing_key = None
                    raise ValueError(
                        f"Two heads produced the same output key `{out_key}` during forward"
                    )
                head_output = head_method(module_out, **self.tokenized_input_for_head)
                head_output = self.run_postprocessor(out_key, head_output)
                self.outputs[out_key] = head_output

            fork_module = self.trunks[
                self.tokens_to_trunks[head_input_key]
            ].get_submodule(fork_module_name)
            self.hook_handles.append(fork_module.register_forward_hook(hook_fn))

    def forward_sub_batch(self, sub_batch, *args, **kwargs):
        assert isinstance(sub_batch, Sample), f"Received {type(sub_batch)}"

        sub_batch_match = [
            isinstance(sub_batch, x.sample_type)
            for x in self.sample_to_modality_preprocessor
        ]
        # ensure that we have exactly one match
        assert (
            sub_batch_match.count(True) == 1
        ), f"{type(sub_batch)} matches zero or more than one sample_mapping!"
        matched_index = sub_batch_match.index(True)

        sample_mapping = self.sample_to_modality_preprocessor[matched_index]

        for inputs in sample_mapping.sample_field_to_modality:
            preprocessor = self.modality_preprocessors[inputs.preprocessor_name]
            preprocessor_input = {x: getattr(sub_batch, x) for x in inputs.input_fields}
            tokenized_input = preprocessor(**preprocessor_input)
            tokenized_input_for_trunk = tokenized_input["trunk"]
            tokenized_input_for_head = tokenized_input["head"]
            # set self.head_routing_key so the right heads are called
            self.head_routing_key = inputs.output_key
            self.tokenized_input_for_head = tokenized_input_for_head
            # optionally use output key to construct a dict
            # this is useful if the trunk is modality aware
            # and wants to share/separate parameters based on input modality
            if inputs.output_key_for_dict:
                tokenized_input_for_trunk = {
                    inputs.output_key: tokenized_input_for_trunk
                }
            # FIXME: The kwargs are not being passed into the preprocessor,
            # so that won't be checkpointed. Probably not a problem since that is
            # likely a small part of the model, but ideally should allow for
            # checkpointing the full end-to-end model eventually
            self.trunks[self.tokens_to_trunks[self.head_routing_key]](
                *args,
                **tokenized_input_for_trunk,
                **kwargs,
            )
            # reset
            self.head_routing_key = None
            self.tokenized_input_for_head = None

    def run_postprocessor(self, head_output_key: str, head_output):
        # This function is called in the head hooks
        if (
            self.postprocessors is not None
            and head_output_key in self.head_to_postprocessor
        ):
            postprocessor_name = self.head_to_postprocessor[head_output_key]
            postprocessor = self.postprocessors[postprocessor_name]
            head_output = postprocessor(head_output)
        return head_output

    def forward(self, batch: Mapping[str, BatchSample], *args, **kwargs) -> Dict:
        assert isinstance(batch, Mapping), f"Received {type(batch)}"
        assert len(self.outputs) == 0
        self.head_routing_key = None
        outputs = {}
        for dataset_key, sub_batch in batch.items():
            # Handle in case the data is a list. This can happen,
            # for example, in multi-crop testing
            all_data = []
            all_keys = []
            for field in FIELDS_WITH_LIST:
                if not hasattr(sub_batch, field):
                    continue
                all_keys.append(field)
                this_data = getattr(sub_batch, field)
                if isinstance(getattr(sub_batch, field), List):
                    all_data.append(this_data)
                else:
                    # Make singleton list to match
                    all_data.append([this_data])
            assert all(
                [len(el) == len(all_data[0]) for el in all_data]
            ), "All data fields should be same len"
            assert len(batch) == 1
            if len(all_data) == 0:
                # No list field found. Just do a forward for the
                # Sample as is
                self.forward_sub_batch(sub_batch, *args, **kwargs)
            else:
                out_vals = {}
                for items in zip(*all_data):
                    e_batch = copy.copy(sub_batch)
                    for item, key in zip(items, all_keys):
                        setattr(e_batch, key, item)
                    self.forward_sub_batch(e_batch, *args, **kwargs)
                    for out_key, out_val in self.outputs.items():
                        if out_key not in out_vals:
                            out_vals[out_key] = []
                        out_vals[out_key].append(out_val)
                    # Reset the outputs to capture the next forward sub batch
                    self.outputs = {}
                for field_key in out_vals.keys():
                    if (
                        self.dataset_specific_list_input_reduction is not None
                        and dataset_key in self.dataset_specific_list_input_reduction
                    ):
                        field_to_ops = self.dataset_specific_list_input_reduction[
                            dataset_key
                        ]
                        reduction_op_name = field_to_ops[field_key]
                    else:
                        reduction_op_name = self.list_input_reduction

                    if reduction_op_name == ListInputReductionOps.CAT:
                        self.outputs[field_key] = torch.cat(out_vals[field_key])
                    elif reduction_op_name == ListInputReductionOps.MEAN:
                        self.outputs[field_key] = torch.mean(
                            torch.stack(out_vals[field_key]), dim=0
                        )
                    elif reduction_op_name == ListInputReductionOps.NO_OP:
                        self.outputs[field_key] = out_vals[field_key]
                    else:
                        raise NotImplementedError(reduction_op_name)

            assert dataset_key not in outputs
            outputs.update({dataset_key: self.outputs})
            self.outputs = {}
        return outputs


class MultiModalZeroShotWithTextTargetsWrapper(nn.Module):
    def __init__(
        self,
        zero_shot_with_text_targets: List[Mapping],
        multimodal_model: nn.Module,
    ):
        super().__init__()
        self.zero_shot_with_text_targets = zero_shot_with_text_targets
        self.multimodal_model = multimodal_model
        self.target_text_features = None
        self.phase = None

    def on_train_epoch_start(self):
        self.phase = Phase.TRAIN

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.phase = Phase.VAL
        assert isinstance(self.zero_shot_with_text_targets, Mapping)
        self.target_text_features = {}
        model_device = next(self.multimodal_model.parameters()).device

        for dataset_key, dataset_spec in self.zero_shot_with_text_targets.items():
            label_strings = dataset_spec["label_strings"]
            num_classes = len(label_strings)
            logging.info(
                f"Validation start: Computing target string features for "
                f"{num_classes} classes for dataset {dataset_key}..."
            )
            all_label_embeddings = []
            num_label_names_per_class = []
            # loop over classes to avoid OOMs
            for cls_idx in range(num_classes):
                num_label_names_per_class.append(len(label_strings[cls_idx]))
                for templates_per_label_name in label_strings[cls_idx]:
                    batch = {
                        dataset_key: BatchTextSample(
                            text=templates_per_label_name.to(model_device)
                        )
                    }
                    output = self.multimodal_model(batch)[dataset_key]
                    all_label_embeddings.append(singleton_dict_to_item(output))
            target_text_features = torch.stack(all_label_embeddings)
            # normalize all text features
            target_text_features = torch.nn.functional.normalize(
                target_text_features, dim=-1, p=2
            )

            # mean across templates (dim=1)
            assert target_text_features.ndim == 3
            target_text_features = target_text_features.mean(dim=1)

            # renormalize
            target_text_features = torch.nn.functional.normalize(
                target_text_features, dim=-1, p=2
            )

            # transpose since we can easily compute an inner_product with image features
            target_text_features = target_text_features.t()
            num_label_names_per_class = np.array(num_label_names_per_class)
            assert target_text_features.shape[1] == num_label_names_per_class.sum()
            num_label_names_per_class_cumsum = np.cumsum(
                num_label_names_per_class
            ).tolist()
            num_label_names_per_class_cumsum.insert(0, 0)
            num_label_names_per_class_cumsum_idx = (
                torch.Tensor(num_label_names_per_class_cumsum).long().to(model_device)
            )
            self.target_text_features[dataset_key] = (
                target_text_features,
                num_label_names_per_class_cumsum_idx,
                num_classes,
            )
        logging.info("...computing target strings done.")

    def on_validation_epoch_end(self):
        # Setting back to None so we don't mistakenly use the same features
        # again in the next epoch the evaluation is done. Must be recomputed
        # in on_validation_epoch_start.
        self.target_text_features = None

    def forward_val(self, batch, *args, **kwargs):
        model_output = self.multimodal_model(batch, *args, **kwargs)
        # Add the final logits to the model output
        for dataset_key, dataset_outputs in model_output.items():
            non_txt_feature = singleton_dict_to_item(dataset_outputs)
            non_txt_feature = torch.nn.functional.normalize(
                non_txt_feature, p=2, dim=-1
            )
            (
                target_text_feature,
                num_label_names_per_class_cumsum_idx,
                num_classes,
            ) = self.target_text_features[dataset_key]
            matches = non_txt_feature @ target_text_feature
            # aggregate predictions per class by computing a max over the logits
            # for label_names in that class
            aggregated_logits = []
            for k in range(num_classes):
                logits_per_label_names = matches[
                    :,
                    num_label_names_per_class_cumsum_idx[
                        k
                    ] : num_label_names_per_class_cumsum_idx[k + 1],
                ]
                aggregated_logits.append(logits_per_label_names.max(dim=1).values)
            aggregated_logits = torch.stack(aggregated_logits).t()
            assert aggregated_logits.shape[1] == num_classes

            # make the logits contiguous so that they can be gathered
            # downstream meters like mAP etc. may gather `scores`
            # The column indexing op for `matches` makes the
            # logits_per_label_names and aggregated_logits non-contiguous
            aggregated_logits = aggregated_logits.contiguous()

            model_output[dataset_key][FINAL_LOGITS_NAME] = aggregated_logits
        return model_output

    def forward(self, batch: Mapping[str, BatchSample], *args, **kwargs) -> Dict:
        assert isinstance(batch, Mapping), f"Received {type(batch)}"
        if self.phase == Phase.VAL:
            # The text features are set by the on_validation_epoch_start function
            # Hence the validation epoch has started, so that forward should be called.
            # When that epoch finishes, the text features are set back to None
            return self.forward_val(batch, *args, **kwargs)
        return self.multimodal_model(batch, *args, **kwargs)
