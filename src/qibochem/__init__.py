import importlib.metadata as im

__version__ = im.version(__package__)

from qibo import (
    driver,
    ansatz,
    measurement,
)

