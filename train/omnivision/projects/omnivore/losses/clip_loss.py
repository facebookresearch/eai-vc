import torch
import torch.nn as nn
import torch.nn.functional as F
from omnivision.utils.distributed import get_rank
from omnivore.losses import CORE_LOSS_KEY
from omnivore.utils.distributed import all_gather_batch


class CLIPLoss(nn.Module):
    def __init__(
        self,
        all_gather_fn: callable = all_gather_batch,
        normalize: bool = True,
        loss1_weight: float = 0.5,
        loss2_weight: float = 0.5,
    ):
        super().__init__()
        self.all_gather_fn = all_gather_fn
        self.labels = None
        self.last_local_batch_size = None
        self.normalize = normalize
        self.loss1_weight = loss1_weight
        self.loss2_weight = loss2_weight

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

        loss1 = F.cross_entropy(logits_per_image, self.labels)
        loss2 = F.cross_entropy(logits_per_text, self.labels)

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
