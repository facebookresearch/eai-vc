import omegaconf


def get_model_tag(metadata: omegaconf.DictConfig):
    if isinstance(metadata.data, omegaconf.ListConfig):
        data = "_".join(sorted(metadata.data))
    else:
        data = metadata.data

    comment = ""
    if "comment" in metadata:
        comment = f"_{metadata.comment}"

    return f"{metadata.algo}_{metadata.model}_{data}{comment}"
