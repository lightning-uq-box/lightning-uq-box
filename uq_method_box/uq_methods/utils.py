"""Utilities for UQ-Method Implementations."""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd


def save_predictions_to_csv(outputs: List[Dict[str, np.ndarray]], path: str) -> None:
    """Save model predictions to csv file.

    Args:
        outputs: a NxO array where N is the number of predictions and
            O the number of variables to be stored
        path: path where csv should be saved
    """
    # concatenate the predictions into a single dictionary
    save_pred_dict = defaultdict(list)

    for out in outputs:
        for k, v in out.items():
            save_pred_dict[k].extend(v.tolist())

    import pdb

    pdb.set_trace()
    # save the outputs, i.e. write them to file
    df = pd.DataFrame.from_dict(save_pred_dict)

    df.to_csv(path, index=False)
