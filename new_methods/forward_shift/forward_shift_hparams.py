from dataclasses import dataclass
from typing import List
from util.hparams import HyperParams


@dataclass
class ForwardShiftHyperParams(HyperParams):
    algo_version: int

    # Method
    layers: List[int]
    num_steps: int
    lr: float

    # Module templates
    # mlp_module_ff1: str
    mlp_module_ff2: str

    stop_on_success: float # actually a bool. not really used
    annotate: float # actually a bool

    # Defaults
    batch_size: int = 1
    alpha: float = 0.0
    # hyperparameters for multi-step optimization. in the final version, we used only a single step
    minimum_loss_for_step: float = 1e-2
    tmp_param: float = 1.0
    n_acts_to_mask: int = 0
    
