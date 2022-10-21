import json
import os
import unittest

import numpy as np

from omnivore.data.api import VisionTextSample
from omnivore.data.tokenizers.simple_tokenizer import (
    HashtagSimpleTokenizer,
    SimpleTokenizer,
)
from omnivore.data.transforms.filter_captions import (
    AnalyseCaption,
    FilterCaptionLanguage,
    FilterHashtagsIn,
    HashTagsToCaptions,
    RemoveHashtags,
    RemovePoundSigns,
    RemoveSpecialTokens,
    ReplaceEmptyCaptionWithHashtags,
)
from omnivore.utils.testing import in_temporary_directory, skip_test_if


BPE_PATH = "/checkpoint/imisra/datasets/SLIP/bpe_simple_vocab_16e6.txt.gz"
LID_PATH = "/checkpoint/qduval/omniscale/lid.176.bin"

skip_if_no_bpe_path = skip_test_if(
    not os.path.exists(BPE_PATH), "Missing resource file to run the test"
)

skip_if_no_lid_path = skip_test_if(
    not os.path.exists(LID_PATH), "Missing resource file to run the test"
)


class TestCaptionParsing(unittest.TestCase):
    @skip_if_no_bpe_path
    def test_simple_tokenizer_on_hashtags(self):
        tokenizer = SimpleTokenizer(bpe_path_list=[BPE_PATH])
        encoded1 = tokenizer.encode(
            "Look at this #amazing #picture of my #crazydog! #fun #realhappylifeforever #lol"
        )
        decoded1 = tokenizer.decode(encoded1)
        encoded2 = tokenizer.encode(
            "Look at this amazing picture of my crazy dog! fun real happy life forever lol"
        )
        decoded2 = tokenizer.decode(encoded2)

        self.assertEqual(
            decoded1.strip(),
            "look at this # amazing # picture of my # crazydog ! # fun # realhappylifeforever # lol",
        )
        self.assertEqual(
            decoded2.strip(),
            "look at this amazing picture of my crazy dog ! fun real happy life forever lol",
        )

        encoded1_no_hashs = list(filter(lambda x: x != 258, encoded1))
        print(encoded1_no_hashs)
        print(encoded2)
        self.assertNotEqual(
            encoded1_no_hashs, encoded2, "Spaces after words are encoded"
        )

    @skip_if_no_bpe_path
    def test_hashtag_simple_tokenizer(self):
        ref_tokenizer = SimpleTokenizer(context_length=77, bpe_path_list=[BPE_PATH])
        tokenizer_1 = HashtagSimpleTokenizer(
            context_length=77,
            bpe_path_list=[BPE_PATH],
            keep_pound_sign=True,
            conservative_mode=False,
        )
        tokenizer_2 = HashtagSimpleTokenizer(
            context_length=77,
            bpe_path_list=[BPE_PATH],
            keep_pound_sign=True,
            conservative_mode=True,
        )
        ref_encoded = ref_tokenizer.encode("#so amazing #real fun moment #pontifun")
        new_encoded = tokenizer_1.encode("#soamazing #realfun moment #pontifun")
        self.assertEqual(ref_encoded, new_encoded, "New should split if possible")
        new_encoded = tokenizer_2.encode("#soamazing #realfun moment #pontifun")
        self.assertEqual(ref_encoded, new_encoded, "New should split if possible")

    def test_remove_special_tokens(self):
        transform = RemoveSpecialTokens()
        text = (
            "Incredible!<NEWLINE><NEWLINE>These #cats are way more clever than #dogs!"
        )
        self.assertEqual(
            transform(text), "Incredible! These #cats are way more clever than #dogs!"
        )
        text = "Check the latest of <USERNAME> post at <URL>. Ping me at <EMAIL>"
        self.assertEqual(transform(text), "Check the latest of post at . Ping me at")

    def test_remove_pounds(self):
        transform = RemovePoundSigns()
        text = "Say hello to my #amazing #dog! #fun"
        out = transform(text)
        self.assertEqual(out, "Say hello to my amazing dog! fun")

    def test_remove_hashtags(self):
        transform = RemoveHashtags()
        text = "Say hello to my #amazing #dog! #fun"
        out = transform(text)
        self.assertEqual(out, "Say hello to my !")

    def test_filter_hashtags(self):
        transform = FilterHashtagsIn({"cat", "dog", "amazing"})
        text = "Say hello to my #amazing #dog! #fun"
        out = transform(text)
        self.assertEqual(out, "Say hello to my #amazing #dog!")

    def test_hashtag_as_captions(self):
        transform = HashTagsToCaptions(prompt="A photo containing:")
        self.assertEqual(
            "A photo containing: zebra, elephant, cat", transform("zebra,elephant,cat")
        )
        transform = HashTagsToCaptions(prompt="")
        self.assertEqual("zebra, elephant, cat", transform("zebra,elephant,cat"))

        with in_temporary_directory():
            templates = np.array(["a photo of {}.", "an image of {}."])
            np.save("prompts.npy", templates)
            transform = HashTagsToCaptions(template_paths=["prompts.npy"])
            out = transform("zebra,elephant,cat")
            self.assertIn(
                out,
                {
                    "a photo of zebra, elephant, cat.",
                    "an image of zebra, elephant, cat.",
                },
            )

        with in_temporary_directory():
            templates = np.array(["{}"])
            np.save("prompts.npy", templates)
            transform = HashTagsToCaptions(
                template_paths=["prompts.npy"], separator=" "
            )
            self.assertEqual(transform("zebra,elephant,cat"), "zebra elephant cat")

    @skip_if_no_lid_path
    def test_lid_model_filtering(self):
        transform = FilterCaptionLanguage([LID_PATH])
        self.assertEqual(
            "this is english for sure", transform("this is english for sure")
        )
        self.assertEqual(
            "this is #english for sure #fun",
            transform("this is #english for sure #fun"),
        )
        self.assertEqual(
            "#fun #caption",
            transform("c'est du français pour sûr #fun #caption"),
        )
        multilingual_samples = [
            "moité français half english #fun #caption",
            "half english moité français #fun #caption",
            "half english moite francais #fun #caption",
        ]
        for sample in multilingual_samples:
            self.assertEqual(
                "#fun #caption",
                transform(sample),
            )
            print(transform.is_target_language(sample))

    @skip_if_no_lid_path
    def test_analyze_captions(self):
        with in_temporary_directory():
            with open("hashtags.txt", "w") as f:
                for hashtag in ["fun", "dog"]:
                    f.write(hashtag + "\n")

            transform = AnalyseCaption(
                model_paths=[LID_PATH], hashtag_list_paths=["hashtags.txt"]
            )
            out = transform("This could have been a #fun #caption")
            out = json.loads(out)
            self.assertEqual(2, out["num_hashtags"])
            self.assertEqual("__label__en", out["label"])
            self.assertGreater(out["prob"], 0.95)

    def test_replace_empty_captions_with_uru_hashtags(self):
        transform = ReplaceEmptyCaptionWithHashtags(separator=" ")
        sample = VisionTextSample(vision=None, text="", label="cat,dog")
        sample = transform(sample)
        self.assertEqual("cat dog", sample.text)
