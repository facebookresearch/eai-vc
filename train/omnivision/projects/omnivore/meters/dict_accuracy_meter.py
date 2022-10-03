import logging

from omnivision.meters.accuracy_meter import AccuracyMeter


class DictAccuracyMeter(AccuracyMeter):
    def __init__(self, top_k: int, dict_key: str) -> None:
        logging.warn(
            "DEPRECATED: Please use the DictApplyMeterWrapper with AccuracyMeter"
        )
        super().__init__(top_k=top_k)
        self.dict_key = dict_key

    def update(self, preds, target):
        assert isinstance(preds, dict)
        super().update(preds[self.dict_key], target)
