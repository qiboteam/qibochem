from benchmarks.libraries import abstract
from benchmarks.logger import log


class Qibo(abstract.AbstractBackend):

    def __init__(self, max_qubits="0", backend="qibojit", platform=None, accelerators=""):
        import qibo

        qibo.set_backend(backend=backend, platform=platform)
        from qibo import models

        self.name = "qibo"
        self.qibo = qibo
        self.models = models
        self.__version__ = qibo.__version__
        self.max_qubits = int(max_qubits)
        self.accelerators = self._parse_accelerators(accelerators)

    def from_qasm(self, qasm):
        circuit = self.models.Circuit.from_qasm(qasm, accelerators=self.accelerators)
        if self.max_qubits > 1:
            if self.max_qubits > 2:
                log.warn(
                    "Fusion with {} qubits is not yet supported by Qibo. " "Using max_qubits=2.".format(self.max_qubits)
                )
            circuit = circuit.fuse()
        return circuit

    def __call__(self, circuit):
        # transfer final state to numpy array because that's what happens
        # for all backends
        return circuit().state(numpy=True)

    def transpose_state(self, x):
        return x

    def get_precision(self):
        return self.qibo.get_precision()

    def set_precision(self, precision):
        self.qibo.set_precision(precision)

    def get_device(self):
        return self.qibo.get_device()

    @staticmethod
    def _parse_accelerators(accelerators):
        """Transforms string that specifies accelerators to dictionary.

        The string that is parsed has the following format:
            n1device1+n2device2+n3device3,...
        and is transformed to the dictionary:
            {'device1': n1, 'device2': n2, 'device3': n3, ...}

        Example:
            2/GPU:0+2/GPU:1 --> {'/GPU:0': 2, '/GPU:1': 2}
        """
        if not accelerators or accelerators is None:
            return None

        def read_digit(x):
            i = 0
            while x[i].isdigit():
                i += 1
            return x[i:], int(x[:i])

        accelerator_dict = {}
        for entry in accelerators.split("+"):
            device, n = read_digit(entry)
            if device in accelerator_dict:
                accelerator_dict[device] += n
            else:
                accelerator_dict[device] = n
        return accelerator_dict
