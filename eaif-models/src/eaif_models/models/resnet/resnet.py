import types

import torch
import torchvision  # noqa
import clip


def forward_without_avgpool_flatten(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


# Remove fully-connected layer from each torchvision.models.resnet model
# Also remove AvgPool and flatten if asked to do so
def resnet50(use_avgpool_and_flatten=True, *args, **kwargs):
    model = torchvision.models.resnet50(*args, **kwargs)
    model.fc = torch.nn.modules.linear.Identity()
    if not use_avgpool_and_flatten:
        funcType = types.MethodType
        model.forward = funcType(forward_without_avgpool_flatten, model)
    return model


def resnet50_vip(use_avgpool_and_flatten=True, *args, **kwargs):
    model = torchvision.models.resnet50(*args, **kwargs)
    model.fc = torch.nn.modules.linear.Linear(2048, 1024)
    if not use_avgpool_and_flatten:
        funcType = types.MethodType
        model.forward = funcType(forward_without_avgpool_flatten, model)
    return model


def load_moco_checkpoint(checkpoint_path, moco_version="v2"):
    assert moco_version in ["v2", "v3"], "MoCo version has to be either v2 or v3"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    old_state_dict = checkpoint["state_dict"]
    state_dict = {}
    for k in list(old_state_dict.keys()):
        if moco_version == "v2":
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = old_state_dict[k]
        else:
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not (
                k.startswith("module.base_encoder.head")
                or k.startswith("module.base_encoder.fc")
            ):
                # remove prefix
                updated_key = k[len("module.base_encoder.") :]
                state_dict[updated_key] = old_state_dict[k]
        # delete renamed or unused k
        del old_state_dict[k]
    return state_dict


def load_r3m_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["r3m"]

    result = {}

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key, value in state_dict.items():
        if key.startswith("module.convnet."):
            no_prefix_key = key.replace("module.convnet.", "")
            if no_prefix_key.startswith("fc."):
                continue
            result[no_prefix_key] = value

    return result


def load_vip_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["vip"]

    result = {}

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key, value in state_dict.items():
        if key.startswith("module.convnet."):
            no_prefix_key = key.replace("module.convnet.", "")
            result[no_prefix_key] = value

    return result


# Original forward: https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L223
def forward_without_avgpool_flatten_clip(self, x: torch.Tensor) -> torch.Tensor:
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


# Explanation of difference between torchvision resnet with clip:
# https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L95
def resnet50_clip(use_avgpool_and_flatten=True, *args, **kwargs):
    model = clip.model.ModifiedResNet(*args, **kwargs)
    if not use_avgpool_and_flatten:
        funcType = types.MethodType
        model.forward = funcType(forward_without_avgpool_flatten_clip, model)
    return model


def load_clip_resnet50_checkpoint(checkpoint_path):
    checkpoint = torch.jit.load(checkpoint_path, map_location="cpu").state_dict()
    state_dict = {
        k.replace("visual.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("visual.")
    }
    return state_dict
