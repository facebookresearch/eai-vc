import torch
import torch.nn as nn
import torch.nn.functional as F
from omnivision.utils.distributed import get_rank
from omnivore.losses import CORE_LOSS_KEY
from omnivore.utils.distributed import all_gather_batch

IGNORE_INDEX = -100


class CLIPLoss(nn.Module):
    def __init__(
        self,
        all_gather_fn: callable = all_gather_batch,
        normalize: bool = True,
        loss1_weight: float = 0.5,
        loss2_weight: float = 0.5,
        label_smoothing: float = 0.0,
        mask_with_data_valid: bool = False,
    ):
        super().__init__()
        self.all_gather_fn = all_gather_fn
        self.labels = None
        self.last_local_batch_size = None
        self.normalize = normalize
        self.loss1_weight = loss1_weight
        self.loss2_weight = loss2_weight
        self.label_smoothing = label_smoothing
        self.mask_with_data_valid = mask_with_data_valid

    def forward(self, outputs):
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        logit_scale = outputs["logit_scale"]
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        if self.normalize:
            # normalized features
            image_embed = F.normalize(image_embed, dim=-1, p=2)
            text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = self.all_gather_fn([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        # labels for cross-entropy
        labels = self.labels

        if self.mask_with_data_valid:
            data_valid_mask = outputs["data_valid"]
            # clone the labels since we are caching them
            labels = labels.clone()
            # make the mask bool so we may index
            data_valid_mask = data_valid_mask.bool()
            labels[~data_valid_mask] = IGNORE_INDEX

        loss1 = F.cross_entropy(
            logits_per_image,
            labels,
            label_smoothing=self.label_smoothing,
            ignore_index=IGNORE_INDEX,
        )
        loss2 = F.cross_entropy(
            logits_per_text,
            labels,
            label_smoothing=self.label_smoothing,
            ignore_index=IGNORE_INDEX,
        )

        loss = loss1 * self.loss1_weight + loss2 * self.loss2_weight

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {
            CORE_LOSS_KEY: loss,
            "contrastive_acc": acc,
            "loss1": loss1.detach(),
            "loss2": loss2.detach(),
        }
