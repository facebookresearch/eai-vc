from typing import Mapping

# A key used to specify the final logits, if multiple features are returned,
# that is used to evaluate accuracy/mAP etc.
FINAL_LOGITS_NAME = "logits"


class DictApplyMeterWrapper:
    """
    Applies a meter on a specific key
    Performing "Containment and delegation", as in https://stackoverflow.com/a/1383646
    """

    def __init__(self, base_meter, key=FINAL_LOGITS_NAME):
        self.base_meter = base_meter
        self.key = key

    def __getattr__(self, attr):
        return getattr(self.base_meter, attr)

    def update(self, preds: Mapping, *args, **kwargs):
        assert isinstance(preds, Mapping)
        return self.base_meter.update(preds[self.key], *args, **kwargs)
