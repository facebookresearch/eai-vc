import omegaconf


def get_model_tag(metadata: omegaconf.DictConfig):
    return f"{metadata.algo}_{metadata.model}_{metadata.data}"