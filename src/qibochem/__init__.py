import importlib.metadata as im

from qibochem import ansatz, driver, measurement, selected_ci
from qibochem.selected_ci import QSCI, QSCIConfig, QSCIResult, qsci_ground_state

__version__ = im.version(__package__)
