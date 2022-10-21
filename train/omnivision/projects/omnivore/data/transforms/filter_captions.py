# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import json
import re
from typing import List, Optional, Set, Tuple, Union

from omnivore.data.api import VisionTextHashtagSample, VisionTextSample
from omnivore.utils.data import FileLoader

try:
    import fasttext as ft
except ImportError:
    try:
        import fastText as ft
    except ImportError:
        raise ImportError("Missing fasttext dependency")

import numpy as np
from iopath.common.file_io import g_pathmgr


def clean_double_spaces(text: str) -> str:
    return " ".join(s for s in text.split(" ") if s)


def remove_special_tokens(text: str, tokens: List[str]) -> str:
    for token in tokens:
        text = text.replace(token, " ")
    return clean_double_spaces(text)


def get_first_valid_path(paths: List[str]) -> str:
    for path in paths:
        if g_pathmgr.exists(path):
            return path
    raise Exception(f"No file path found among: {paths}")


IG_CAPTIONS_SPECIAL_TOKENS = [
    "<NEWLINE>",
    "<URL>",
    "<EMAIL>",
    "<USERNAME>",
]


class RemoveSpecialTokens:
    """
    Remove the special tokens present in the captions
    """

    def __init__(self, tokens: Optional[List[str]] = None):
        self.tokens = tokens
        if self.tokens is None:
            self.tokens = list(IG_CAPTIONS_SPECIAL_TOKENS)

    def __call__(self, text: str) -> str:
        return remove_special_tokens(text, self.tokens)


class RemovePoundSigns:
    """
    Remove the pound signs from the text
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, text: str) -> str:
        if self.enabled:
            text = text.replace("#", " ")
            return clean_double_spaces(text)
        else:
            return text


class RemoveHashtags:
    """
    Remove all form of hashtag from the text provided as input
    """

    def __init__(self):
        super().__init__()
        self.hashtag_pat = re.compile(r"#\w+ ?")

    def __call__(self, text: str) -> str:
        text = self.hashtag_pat.sub("", text)
        return clean_double_spaces(text)


class HashTagsToCaptions:
    """
    Transformation used to map a list of hashtags as strings
    to the associated captions

    Input format is the content of the 'hashtags' column, i.e.
    strings separated with comas, without spaces

    For instance: "motherboard,electroniccomputer,graphics"

    Args:
        separator: separator between the hashtags
        prompt (str): fixed prompt to use in front of hashtags
        template_paths (list): templates for prompts to pick from
    """

    def __init__(
        self,
        separator: str = ", ",
        prompt: Optional[str] = None,
        template_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.prompt = prompt
        self.separator = separator
        if template_paths is not None:
            self.templates = FileLoader.load(template_paths)[0]
        else:
            self.templates = None

    def __call__(self, hashtags: str) -> str:
        hashtags = hashtags.split(",")
        hashtags = self.separator.join(hashtags)
        if self.templates is not None:
            prompt = np.random.choice(self.templates)
            return prompt.format(hashtags)
        elif self.prompt is not None and self.prompt != "":
            return " ".join([self.prompt, hashtags])
        else:
            return hashtags


class FilterHashtagsIn:
    """
    Transform which allows to remove the hashtags of the captions based on
    a provided set of hashtags we want to keep
    """

    def __init__(self, hashtags: Set[str]):
        super().__init__()
        self.hashtags = {
            hashtag.replace("#", "").strip().lower() for hashtag in hashtags
        }
        self.hashtag_pat = re.compile(r"#\w+")

    def __call__(self, caption: str) -> str:
        out_caption = caption
        for hashtag in self.hashtag_pat.findall(caption):
            cleaned_hashtag = hashtag.replace("#", "").strip().lower()
            if cleaned_hashtag not in self.hashtags:
                out_caption = out_caption.replace(hashtag, "")
        return clean_double_spaces(out_caption)


class FilterHashtags:
    """
    Transform which allows to remove the hashtags of the captions based on
    a white list of hashtags provided in an input file
    """

    def __init__(self, hashtag_list_paths: List[str]):
        super().__init__()

        # Find the correct path to load from
        selected_path = get_first_valid_path(hashtag_list_paths)

        # Read the hashtags from the selected path
        self.valid_hashtags = set()
        with g_pathmgr.open(selected_path, "r") as f:
            for line in f:
                self.valid_hashtags.add(line.strip())
        self.filter = FilterHashtagsIn(self.valid_hashtags)

    def __call__(self, caption: str) -> str:
        return self.filter(caption)


class FilterCaptionLanguage:
    """
    Filter the captions based on the language it contains:
    - captions matching the chosen language are kept
    - otherwise, only the hashtags inside the caption are kept
    """

    def __init__(
        self,
        model_paths: List[str],
        target_label: str = "__label__en",
        threshold: float = 0.8,
        replace_by_hashtags: bool = True,
    ):
        super().__init__()
        self.target_label = target_label
        self.threshold = threshold
        self.replace_by_hashtags = replace_by_hashtags
        self.hashtag_pat = re.compile(r"#\w+ ?")

        # Load the fastText model from the selected path
        selected_path = get_first_valid_path(model_paths)
        self.selected_path = g_pathmgr.get_local_path(selected_path)
        self.lid_model = None

    def __call__(self, caption: str) -> str:
        self._lazy_load()
        if self.is_target_language(caption)[0]:
            return caption
        elif self.replace_by_hashtags:
            return self.hashtags_only(caption)
        else:
            return ""

    def _lazy_load(self):
        # FastText object cannot be pickled so we delay it's creation
        # after the pickling (to spawn DataLoader workers)
        if self.lid_model is None:
            self.lid_model = ft.load_model(self.selected_path)

    def is_target_language(self, text: str) -> Tuple[bool, str, float]:
        text = self.remove_hashtags(text)
        text = remove_special_tokens(text, IG_CAPTIONS_SPECIAL_TOKENS)
        labels, probs = self.lid_model.predict(text)
        idx = np.argmax(probs)
        label, prob = labels[idx], probs[idx]
        is_target = label == self.target_label and prob >= self.threshold
        return is_target, label, prob

    def remove_hashtags(self, text: str) -> str:
        return self.hashtag_pat.sub("", text)

    def hashtags_only(self, text: str) -> str:
        return "".join(hashtag for hashtag in self.hashtag_pat.findall(text))


class DropSamplesWithEmptyCaption:
    def __call__(
        self, sample: Union[VisionTextSample, VisionTextHashtagSample]
    ) -> Union[VisionTextSample, VisionTextHashtagSample]:
        if sample.text == "":
            return None
        else:
            return sample


class ReplaceEmptyCaptionWithHashtags:
    """
    Replace the caption inside the VisionTextSample with the hashtags if the caption is empty

    Args:
        new_value: replacement string for the label
        separator: separator between hashtags in the text
        template_paths: templates for the text around the hashtags

    """

    def __init__(
        self,
        separator: str = " ",
        template_paths: Optional[List[str]] = None,
    ):
        # TODO - accept a list of transforms instead?
        self.transform = HashTagsToCaptions(
            separator=separator, template_paths=template_paths
        )

    def __call__(
        self, sample: Union[VisionTextSample, VisionTextHashtagSample]
    ) -> Union[VisionTextSample, VisionTextHashtagSample]:
        if isinstance(sample, VisionTextHashtagSample):
            if sample.text == "":
                sample.text = self.transform(sample.hashtags)
            return sample
        elif isinstance(sample, VisionTextSample):
            if sample.text == "":
                sample.text = self.transform(sample.label)
            return sample


class AnalyseCaption:
    """
    Transform that can be plugged into a transformation pipeline in order to
    replace the captions with statistics on the captions
    """

    def __init__(
        self,
        model_paths: List[str],
        hashtag_list_paths: List[str],
        target_label: str = "__label__en",
        dump_text: bool = False,
    ):
        super().__init__()
        self.target_label = target_label
        self.dump_text = dump_text
        self.hashtag_pat = re.compile(r"#\w+")

        # Load the fastText model from the first valid path
        lid_model_path = get_first_valid_path(model_paths)
        self.lid_model_path = g_pathmgr.get_local_path(lid_model_path)
        self.lid_model = None

        # Load the hashtags from the first valid path
        self.hashtags_path = get_first_valid_path(hashtag_list_paths)
        self.valid_hashtags = None

    def __call__(self, caption: str) -> str:
        self._lazy_load()
        caption = remove_special_tokens(caption, IG_CAPTIONS_SPECIAL_TOKENS)
        is_target, label, prob = self.infer_target_language(caption)
        hashtags = self.select_hashtags(caption)
        stats = {
            "is_target": 1 if is_target else 0,
            "label": label,
            "prob": prob,
            "num_hashtags": len(hashtags),
            "num_wordnet": len(hashtags & self.valid_hashtags),
        }
        if self.dump_text:
            stats["text"] = caption
        return json.dumps(stats)

    def _lazy_load(self):
        # FastText object cannot be pickled so we delay it's creation
        # after the pickling (to spawn DataLoader workers)
        if self.lid_model is None:
            self.lid_model = ft.load_model(self.lid_model_path)

        if self.valid_hashtags is None:
            with g_pathmgr.open(self.hashtags_path, "r") as f:
                self.valid_hashtags = {line.strip() for line in f}

    def infer_target_language(self, text: str) -> Tuple[bool, str, float]:
        text = self.remove_hashtags(text)
        text = self.clean_double_spaces(text)
        labels, probs = self.lid_model.predict(text)
        idx = np.argmax(probs)
        label, prob = labels[idx], probs[idx]
        is_target = label == self.target_label
        return is_target, label, prob

    def clean_double_spaces(self, text: str) -> str:
        return " ".join(s for s in text.split(" ") if s)

    def remove_hashtags(self, text: str) -> str:
        return self.hashtag_pat.sub("", text)

    def select_hashtags(self, text: str) -> Set[str]:
        return {
            hashtag.replace("#", "").strip()
            for hashtag in self.hashtag_pat.findall(text)
        }
