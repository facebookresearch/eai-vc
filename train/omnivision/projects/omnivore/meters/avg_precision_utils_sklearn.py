# This is the SKLEARN version of mAP
# This is not accurate, and is only being added for comparison to
# prior work in audio classification (eg AST) uses this form of
# AP computation.
# Based on https://github.com/YuanGongND/ast/blob/master/src/utilities/stats.py


import numpy as np
from scipy.special import expit
from sklearn import metrics


def get_precision_recall(target, output, apply_sigmoid=False):
    if apply_sigmoid:
        output = expit(output)
    avg_precision = metrics.average_precision_score(
        target.astype(np.bool), output, average=None
    )
    return None, None, None, [avg_precision]
