from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class DummyParams(HyperParams):
    # Method
    we_do_need_any: str = "this hparams file is just a place-holder (a dummy) file to make the common structure"
    