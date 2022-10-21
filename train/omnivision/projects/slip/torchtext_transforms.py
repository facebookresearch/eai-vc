# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# File modified from - https://github.com/pytorch/text/blob/main/torchtext/transforms.py
# to be mult-cluster compatible.

import json
from copy import deepcopy
from functools import lru_cache
from typing import Any, List, Mapping, Sequence, Union

import torch
import torchtext  # noqa: F401
import torchtext.transforms as T
from iopath.common.file_io import g_pathmgr
from torch.nn import Module
from torchtext._torchtext import GPT2BPEEncoder as GPT2BPEEncoderPyBind
from torchtext.models.roberta.model import RobertaEncoderConf


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_valid_path(path_list):
    path_found = False
    for path in path_list:
        if g_pathmgr.exists(path):
            valid_path = path
            path_found = True
            break

    if not path_found:
        raise Exception(f"None of the paths exist in {path_list}")

    return valid_path


def build_bert_tokenizer(
    encoder_json_path_list: List[str],
    vocab_bpe_path_list: List[str],
    vocal_model_path_list: List[str],
    context_length: int,
):
    encoder_json_path = get_valid_path(encoder_json_path_list)
    vocab_bpe_path = get_valid_path(vocab_bpe_path_list)
    vocal_model_path = get_valid_path(vocal_model_path_list)

    with g_pathmgr.open(vocal_model_path, "rb") as f:
        vocab_model_state_dict = torch.load(f, map_location="cpu")

    return T.Sequential(
        GPT2BPETokenizer(
            encoder_json_path=encoder_json_path,
            vocab_bpe_path=vocab_bpe_path,
        ),
        T.VocabTransform(vocab_model_state_dict),
        T.Truncate(context_length),
        T.AddToken(token=0, begin=True),
        T.AddToken(token=2, begin=False),
        T.ToTensor(padding_value=RobertaEncoderConf.padding_idx),
        T.PadTransform(
            max_length=context_length + 2, pad_value=RobertaEncoderConf.padding_idx
        ),
    )


class GPT2BPETokenizer(Module):
    """
    Transform for GPT-2 BPE Tokenizer.
    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py
    :param encoder_json_path: Path to GPT-2 BPE encoder json file.
    :type encoder_json_path: str
    :param vocab_bpe_path: Path to bpe vocab file.
    :type vocab_bpe_path: str
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
    __jit_unused_properties__ = ["is_jitable"]
    _seperator: torch.jit.Final[str]

    def __init__(
        self, encoder_json_path: str, vocab_bpe_path: str, return_tokens: bool = False
    ) -> None:
        super().__init__()
        self._seperator = "\u0001"
        # load bpe encoder and bpe decoder
        with g_pathmgr.open(encoder_json_path, "r", encoding="utf-8") as f:
            bpe_encoder = json.load(f)
        # load bpe vocab
        with g_pathmgr.open(vocab_bpe_path, "r", encoding="utf-8") as f:
            bpe_vocab = f.read()

        bpe_merge_ranks = {
            self._seperator.join(merge_pair.split()): i
            for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
        }
        # Caching is enabled in Eager mode
        self.bpe = GPT2BPEEncoderPyBind(
            bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True
        )

        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs
        Args:
            text: An input text string.
        Returns:
            A list of bpe token ids represents each bpe tokens
        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens
        Args:
            text: An input text string.
        Returns:
            A list of bpe token ids represents each bpe tokens
        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
        """
        return self.bpe.tokenize(text)

    def add_special_tokens(
        self, special_tokens_dict: Mapping[str, Union[str, Sequence[str]]]
    ) -> int:
        """Add a dictionary of special tokens (eos, pad, cls…) to the encoder
        :param special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
        [bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
        Tokens are only added if they are not already in the vocabulary.
        :type special_tokens_dict: Dict[str, Union[str, List[str]]]
        :return: Number of tokens added to the vocabulary.
        :rtype: int
        """
        for key in special_tokens_dict.keys():
            assert (
                key in self.SPECIAL_TOKENS_ATTRIBUTES
            ), f"Key '{key}' is not in the special token list: {self.SPECIAL_TOKENS_ATTRIBUTES}"

        return self.bpe.add_special_tokens(
            {
                k: v
                for k, v in special_tokens_dict.items()
                if k != "additional_special_tokens"
            },
            special_tokens_dict.get("additional_special_tokens", []),
        )

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.GPT2BPEEncoder(
                self.bpe.bpe_encoder_,
                self.bpe.bpe_merge_ranks_,
                self.bpe.seperator_,
                self.bpe.byte_encoder_,
                False,
            )
            return tokenizer_copy
        return self

    def decode(self, tokens: List[str]) -> str:
        """Return a decoded string given a list of string token ids.
        :param input: A list of strings, each string corresponds to token ids.
        :type input: List[str]
        :return: decoded text
        :rtype: str
        """
        return self.bpe.decode([int(token) for token in tokens])
