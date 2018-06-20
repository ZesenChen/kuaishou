from sklearn.metrics import f1_score
import numpy as np
from copy import copy

def try_threshold(y_true, y_pred, min_t, max_t):
    best_t = 0
    best_score = 0
    threshold  = min_t
    while threshold <= max_t:
        tmp = copy(y_pred)
        tmp[tmp>threshold] = 1
        tmp[tmp<=threshold] = 0
        f1 = f1_score(y_true,y_pred)
        if f1 > best_score:
            best_score = f1
            best_t = threshold
        threshold += 0.002
    return best_score,best_t
