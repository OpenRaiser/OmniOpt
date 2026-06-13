"""
Stage 2 Optimizer implementations for FineWeb-Edu benchmark.
"""

from .adam_mini import Adam_mini
from .lamb import Lamb
from .shampoo import Shampoo
from .galore_adamw import AdamW as GaLore_AdamW
from .galore_adafactor import Adafactor as GaLoreAdafactor
from .svd_projector import GaLoreProjector
from .random_projector import GradientProjector
from .adan import Adan
from .apollo import AdamW as APOLLO_AdamW
from .came import CAME
from .conda import Conda, CondaProjector
from .lion import Lion
from .mars import MARS
from .muon import Muon
from .nadam import NAdamLegacy as NAdam
from .radam import RAdamLegacy as RAdam
from .sophia import SophiaG
from .soap import SOAP
from .rmnp import RMNP
from .adabelief import AdaBelief
from .adamp import AdamP
from .adamw import AdamWLegacy
from .adopt import Adopt
from .kron import Kron
from .laprop import LaProp
from .lars import LARS
from .nvnovograd import NvNovoGrad
from .prodigy import Prodigy

# Optional (requires bitsandbytes)
try:
    from .galore_adamw8bit import AdamW8bit as GaLoreAdamW8bit
except (ImportError, ModuleNotFoundError):
    GaLoreAdamW8bit = None

try:
    from .q_apollo import AdamW as QAPOLLOAdamW
except (ImportError, ModuleNotFoundError):
    QAPOLLOAdamW = None

__all__ = [
    'Adam_mini', 'Lamb', 'Shampoo', 'GaLore_AdamW', 'GaLoreAdafactor',
    'GaLoreProjector', 'GradientProjector', 'CondaProjector',
    'Adan', 'APOLLO_AdamW', 'CAME', 'Conda', 'Lion', 'MARS', 'Muon',
    'NAdam', 'RAdam', 'SophiaG', 'SOAP', 'RMNP',
    'AdaBelief', 'AdamP', 'AdamWLegacy', 'Adopt', 'Kron',
    'LaProp', 'LARS', 'NvNovoGrad', 'Prodigy',
]
