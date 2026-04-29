from .fgsm  import fgsm_attack
from .ifgsm import IFGSMAttack, ifgsm_attack

__all__ = [
    "fgsm_attack",
    "IFGSMAttack",
    "ifgsm_attack",
]
