from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .dummy_hparams import DummyParams


def apply_nothing_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DummyParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Do nothing!
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    return model, weights_copy
