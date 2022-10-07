import types

import torch
import torchvision  # noqa


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

def load_moco_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
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
