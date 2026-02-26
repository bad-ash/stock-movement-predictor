"""Cross-validation window utilities for time-series evaluation."""

import numpy as np

def rolling_windows(n, min_train=1500, test_h=125, max_folds=None, drop_last_partial=True):
    """
    Expanding-window CV with contiguous test blocks.

    n_folds is derived from n, min_train, test_h:
      floor((n - min_train) / test_h)  [or include partial last block]
    """
    if n <= min_train:
        return  # yields nothing

    if drop_last_partial:
        n_folds = (n - min_train) // test_h
    else:
        n_folds = int(np.ceil((n - min_train) / test_h))

    if max_folds is not None:
        n_folds = min(n_folds, max_folds)

    start = min_train
    for _ in range(n_folds):
        end = start + test_h
        if end > n:
            if drop_last_partial:
                break
            end = n
        yield np.arange(0, start), np.arange(start, end)
        start += test_h
