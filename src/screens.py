"""Financial-ratio Shariah screens (denominator: total assets) and task construction.

Ratios: r_debt  = interest-bearing debt / total assets
        r_cash  = (cash + interest-bearing securities) / total assets
        r_rec   = (accounts receivable + cash) / total assets
A firm passes a rulebook when every listed ratio is strictly below its threshold.
"""
import numpy as np, pandas as pd

RATIOS = ['r_debt', 'r_cash', 'r_rec']
OFF = 9.0  # threshold value that switches a ratio screen off

RULEBOOKS = {
    'MSCI':   {'r_debt': 1 / 3, 'r_cash': 1 / 3, 'r_rec': 1 / 3},
    'MSCI-M': {'r_debt': 1 / 3, 'r_cash': 1 / 3, 'r_rec': 0.49},
    'FTSE':   {'r_debt': 1 / 3, 'r_cash': 1 / 3, 'r_rec': 0.50},
    'SC':     {'r_debt': 0.33, 'r_cash': 0.33, 'r_rec': OFF},
}
# screens used only for zero-shot evaluation (never seen in training of any model)
UNSEEN = {
    'MSCI-entry': {'r_debt': 0.30, 'r_cash': 0.30, 'r_rec': 0.30},     # MSCI threshold for new inclusions
    'MSCI-rec70': {'r_debt': 1 / 3, 'r_cash': 1 / 3, 'r_rec': 0.70},   # hypothetical stress case: receivables threshold raised to 70%
    'Debt-25':    {'r_debt': 0.25, 'r_cash': OFF, 'r_rec': OFF},        # stricter single-ratio screen
    'Cash-40':    {'r_debt': OFF, 'r_cash': 0.40, 'r_rec': OFF},
}


def tau_vec(rb):
    return np.array([rb.get(k, OFF) for k in RATIOS])


def passes(R, tau):
    """R: (n,3) ratios, tau: (3,) or (n,3) thresholds -> boolean pass."""
    return np.all(R < tau, axis=1)


PROHIBITED_NOTE = 'SIC 6000-6799, 2080-2085, 2100-2199, 2011-2013, 3480-3489, 5180-5182, 5921, 7800-7849, 7990-7999'
