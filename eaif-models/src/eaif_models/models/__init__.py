import logging
import torch
import PIL

log = logging.getLogger(__name__)

zero_img = PIL.Image.new("RGB", (100, 100))


def load_model(
    model,
    transform,
    metadata=None,
    checkpoint_dict=None,
):
    if checkpoint_dict is not None:
        msg = model.load_state_dict(checkpoint_dict)
        log.warning(msg)

    with torch.no_grad():
        transformed_img = transform(zero_img).unsqueeze(0)
        embedding_dim = model.eval()(transformed_img).shape[1]
        model.train()

    return model, embedding_dim, transform, metadata
