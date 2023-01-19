import torch
import torch.nn as nn
import torchvision


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, load_path=None):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        if load_path not in [None, ""]:
            self.base_encoder = load_pretrained_model(base_encoder, load_path, mlp_dim)
            self.momentum_encoder = load_pretrained_model(
                base_encoder, load_path, mlp_dim
            )
        else:
            self.base_encoder = base_encoder(num_classes=mlp_dim)
            self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(
            self.base_encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(
            self.base_encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # print('--------------------------- Before ------------------------')
        # print(k.shape)
        k = concat_all_gather(k)
        # print('--------------------------- After ------------------------')
        # print(type(k))
        # print(k.shape)
        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (
            torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()
        ).cuda()
        loss = nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        acc1, acc5 = acc1[0], acc5[0]  # list(tensor) to tensor
        return loss, acc1, acc5

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        q1_loss, q1_acc1, q1_acc5 = self.contrastive_loss(q1, k2)
        q2_loss, q2_acc1, q2_acc5 = self.contrastive_loss(q2, k1)

        total_loss = q1_loss + q2_loss
        acc1, acc5 = (q1_acc1 + q2_acc1) / 2, (q1_acc5 + q2_acc5) / 2

        return total_loss, acc1, acc5


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del (
            self.base_encoder.head,
            self.momentum_encoder.head,
        )  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# model loading function
def load_pretrained_model(
    base_encoder: callable,
    load_path: str,
    dim: int,
) -> nn.Module:
    """Load a pretrained model for downstream adaptation with MoCo"""
    # num_classes is the output fc dimension
    model = base_encoder(num_classes=dim)
    arch = "resnet" if isinstance(model, torchvision.models.resnet.ResNet) else "vit"
    old_state_dict = torch.load(load_path, map_location="cpu")
    if "state_dict" in old_state_dict.keys():
        old_state_dict = old_state_dict["state_dict"]
        mae = False
    elif "r3m" in old_state_dict.keys():
        old_state_dict = old_state_dict["r3m"]
        mae = False
    else:
        old_state_dict = old_state_dict["model"]
        mae = True
    # load keys correctly by removing prefix
    if mae:
        print("Trying to load model as | Arch = %s | Pretraining alg = MAE" % arch)
        model = load_mae_encoder(model, load_path)
    else:
        if any(["base_encoder" in k for k in old_state_dict.keys()]):
            print(
                "Trying to load model as | Arch = %s | Pretraining alg = MoCo-v3" % arch
            )
            state_dict = load_moco_checkpoint(load_path, moco_version="v3")
        elif any(["module.convnet" in k for k in old_state_dict.keys()]):
            print("Trying to load model as | Arch = %s | Pretraining alg = R3M" % arch)
            from eaif_models.models.resnet.resnet import load_r3m_checkpoint

            state_dict = load_r3m_checkpoint(checkpoint_path=load_path)
        else:
            print(
                "Trying to load model as | Arch = %s | Pretraining alg = MoCo-v2" % arch
            )
            state_dict = load_moco_checkpoint(load_path, moco_version="v2")
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or set(  # resnet
            msg.missing_keys
        ) == {
            "head.bias",
            "head.weight",
        }  # vit
        # print(msg)
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


def load_mae_encoder(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    # filter out keys with name decoder or mask_token
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if "decoder" not in k and "mask_token" not in k
    }

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"head.bias", "head.weight"}
    # print(msg)
    return model


if __name__ == "__main__":
    # check if prior moco-v3 models can be loaded
    from functools import partial
    import torchvision
    import moco_vit

    model = load_pretrained_model(
        base_encoder=partial(
            torchvision.models.__dict__["resnet50"], zero_init_residual=True
        ),
        load_path="/checkpoint/aravraj/moco_v3/moco_v3_rn50_try1/checkpoints/moco_v3_rn50_try1/checkpoint_0024.pth",
        dim=256,
    )

    model = load_pretrained_model(
        base_encoder=partial(
            torchvision.models.__dict__["resnet50"], zero_init_residual=True
        ),
        load_path="/checkpoint/maksymets/eaif/models/moco_ego4d/moco_ego4d_5m.pth",
        dim=256,
    )

    model = load_pretrained_model(
        base_encoder=partial(
            torchvision.models.__dict__["resnet50"], zero_init_residual=True
        ),
        load_path="/checkpoint/maksymets/eaif/models/r3m/r3m_50/model.pt",
        dim=256,
    )

    model = load_pretrained_model(
        base_encoder=partial(moco_vit.__dict__["vit_base"]),
        load_path="/checkpoint/aravraj/moco_v3/moco-v3_vitb_adroit_IDP_try1/checkpoints/moco-v3_vitb_adroit_IDP_try1/checkpoint_0299.pth",
        dim=256,
    )

    model = load_pretrained_model(
        base_encoder=partial(moco_vit.__dict__["vit_base"]),
        load_path="/checkpoint/maksymets/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth",
        dim=256,
    )
