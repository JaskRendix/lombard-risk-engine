import numpy as np

from lombard_risk.haircut import HaircutComponents, vol_addon_from_var


def test_haircut_components_total():
    hc = HaircutComponents(base=0.20, vol_addon=0.03, liq_addon=0.02, conc_addon=0.05)
    assert np.isclose(hc.total, 0.30)


def test_vol_addon_from_var():
    assert np.isclose(vol_addon_from_var(0.10), 0.12)
