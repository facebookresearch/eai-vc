import numpy as np
import torch
from slip.models import LayerNorm, Transformer
from torch import nn


class SLIPTextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.transformer_width = transformer_width
        self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, tokenized):
        # adding reshape so it works when 1 text is passed here (the tokenizer
        # eats up the batch dimension)
        # tokenized = (
        #     self.tokenizer(text)
        #     .to(self.positional_embedding.device)
        #     .reshape((len(text), -1))
        # )
        x = self.token_embedding(tokenized)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)]

        return x

    def forward(self, text):
        return self.encode_text(text)
