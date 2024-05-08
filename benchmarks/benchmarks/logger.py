import datetime
import json
import logging
import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable Tensorflow warnings


class CustomHandler(logging.StreamHandler):
    """Custom handler for stdout logging."""

    def format(self, record):
        """Format the record with specific format."""
        fmt = f"[BENCHMARKS|%(levelname)s|%(asctime)s]: %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(CustomHandler())


class JsonLogger(list):

    def __init__(self, filename=None):
        self.filename = filename
        if filename is not None:
            if os.path.isfile(filename):
                with open(filename) as file:
                    super().__init__(json.load(file))
                log.info("Extending existing logs from {}.".format(filename))
            else:
                log.info("Creating new logs in {}.".format(filename))
                super().__init__()
        else:
            log.warning("Filename was not provided and logs will not be saved.")
            super().__init__()
        self.append(dict())
        now = datetime.datetime.now()
        self.log(datetime=now.strftime("%Y-%m-%d %H:%M:%S"))

    def log(self, **kwargs):
        self[-1].update(kwargs)
        for k, v in kwargs.items():
            log.info(f"{k}: {v}")

    def average(self, key):
        self[-1][f"{key}_mean"] = np.mean(self[-1][key])
        if len(self[-1][key]) == 1:
            self[-1][f"{key}_std"] = 0.0
        else:
            self[-1][f"{key}_std"] = np.std(self[-1][key], ddof=1)
        log.info("{}_mean: {}".format(key, self[-1][f"{key}_mean"]))
        log.info("{}_std: {}".format(key, self[-1][f"{key}_std"]))

    def __str__(self):
        return "\n" + "\n".join(f"{k}: {v}" for k, v in self[-1].items())

    def dump(self):
        if self.filename is not None:
            with open(self.filename, "w") as file:
                json.dump(self, file)
