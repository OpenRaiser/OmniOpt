__version__ = "0.1.0"

# GaLore optimizer
from .galore_adafactor import Adafactor as GaLoreAdafactor
from .galore_adamw import AdamW as GaLoreAdamW
# GaLoreAdamW8bit requires bitsandbytes, make it optional
try:
    from .galore_adamw8bit import AdamW8bit as GaLoreAdamW8bit
except (ImportError, ModuleNotFoundError) as e:
    GaLoreAdamW8bit = None

# apollo optimizer
from .apollo import AdamW as APOLLOAdamW
# QAPOLLOAdamW requires bitsandbytes, make it optional
try:
    from .q_apollo import AdamW as QAPOLLOAdamW
except (ImportError, ModuleNotFoundError) as e:
    QAPOLLOAdamW = None

# Standard optimizers
from .adabelief import AdaBelief
from .adam_mini import Adam_mini
from .adamp import AdamP
from .adamw import AdamWLegacy
from .adan import Adan
from .adopt import Adopt
from .came import CAME
from .conda import Conda as CondaAdamW
from .kron import Kron
from .lamb import Lamb
from .lars import LARS
from .lion import Lion
from .laprop import LaProp
from .nadam import NAdamLegacy
from .mars import MARS
from .muon import Muon
from .rmnp import RMNP
from .nvnovograd import NvNovoGrad
from .prodigy import Prodigy
from .radam import RAdamLegacy
from .shampoo import Shampoo
from .soap import SOAP
from .sophia import SophiaG
from .sgg_adamw import SGGAdamW
from .sgg_lamb import SGGLAMB
from .sgg_shampoo import SGGShampoo
