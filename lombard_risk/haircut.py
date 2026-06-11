from dataclasses import dataclass

from .config import VOL_BUFFER


@dataclass
class HaircutComponents:
    base: float
    vol_addon: float
    liq_addon: float
    conc_addon: float

    @property
    def total(self) -> float:
        return self.base + self.vol_addon + self.liq_addon + self.conc_addon


def vol_addon_from_var(var_abs: float) -> float:
    return var_abs + VOL_BUFFER
