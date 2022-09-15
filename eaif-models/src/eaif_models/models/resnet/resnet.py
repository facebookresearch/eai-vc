import torch
import torchvision  # noqa


# Remove fully-connected layer from each torchvision.models.resnet model
for i in [18, 34, 50]:
    exec(
        f"""
def resnet{i}(*args, **kwargs):
    model = torchvision.models.resnet{i}(*args, **kwargs)
    model.fc = torch.nn.modules.linear.Identity()
    return model
    """
    )


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
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(state_dict.keys()):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]

    return state_dict
