import numpy as np
from qibo import gates

from benchmarks.circuits import qasm


class OneQubitGate(qasm.OneQubitGate):

    def __init__(self, nqubits, nlayers="1", gate="H", **params):
        super().__init__(nqubits, nlayers=nlayers, gate=gate)
        self.gate = getattr(gates, gate)
        self.angles = {k: float(v) for k, v in params.items()}
        self.parameters = {"nqubits": nqubits, "nlayers": nlayers, "gate": gate, "params": params}

    def to_qasm(self):
        raise NotImplementedError

    def base_command(self, i):
        return self.gate(i, **self.angles)


class TwoQubitGate(OneQubitGate):

    def __init__(self, nqubits, nlayers="1", gate="CNOT", **params):
        super().__init__(nqubits, nlayers, gate, **params)

    def to_qasm(self):
        raise NotImplementedError

    def base_command(self, i):
        return self.gate(i, i + 1, **self.angles)

    def __iter__(self):
        return qasm.TwoQubitGate.__iter__(self)


class QFT(qasm.QFT):

    def to_qasm(self):
        raise NotImplementedError

    def __iter__(self):
        for i1 in range(self.nqubits):
            yield gates.H(i1)
            for i2 in range(i1 + 1, self.nqubits):
                theta = np.pi / 2 ** (i2 - i1)
                yield gates.CU1(i2, i1, theta)

        if self.swaps:
            for i in range(self.nqubits // 2):
                yield gates.SWAP(i, self.nqubits - i - 1)


class VariationalCircuit(qasm.VariationalCircuit):

    def __init__(self, nqubits, nlayers=1, varlayer="False", seed="123"):
        super().__init__(nqubits, nlayers, seed)
        self.varlayer = varlayer == "True"
        self.parameters["varlayer"] = varlayer

    def to_qasm(self):
        raise NotImplementedError

    def varlayer_circuit(self, theta):
        theta = theta.reshape((2 * self.nlayers, self.nqubits))
        pairs = list((i, i + 1) for i in range(0, self.nqubits - 1, 2))
        for l in range(self.nlayers):
            yield gates.VariationalLayer(range(self.nqubits), pairs, gates.RY, gates.CZ, theta[2 * l], theta[2 * l + 1])
            for i in range(1, self.nqubits - 2, 2):
                yield gates.CZ(i, i + 1)
            yield gates.CZ(0, self.nqubits - 1)

    def standard_circuit(self, theta):
        theta = iter(theta)
        for l in range(self.nlayers):
            for i in range(self.nqubits):
                yield gates.RY(i, next(theta))
            for i in range(0, self.nqubits - 1, 2):
                yield gates.CZ(i, i + 1)
            for i in range(self.nqubits):
                yield gates.RY(i, next(theta))
            for i in range(1, self.nqubits - 2, 2):
                yield gates.CZ(i, i + 1)
            yield gates.CZ(0, self.nqubits - 1)

    def __iter__(self):
        np.random.seed(self.seed)
        theta = 2 * np.pi * np.random.random(2 * self.nlayers * self.nqubits)
        if self.varlayer:
            return self.varlayer_circuit(theta)
        else:
            return self.standard_circuit(theta)


class BernsteinVazirani(qasm.BernsteinVazirani):

    def to_qasm(self):
        raise NotImplementedError

    def __iter__(self):
        yield gates.X(self.nqubits - 1)
        for i in range(self.nqubits):
            yield gates.H(i)
        for i in range(self.nqubits - 1):
            yield gates.CNOT(i, self.nqubits - 1)
        for i in range(self.nqubits - 1):
            yield gates.H(i)
        for i in range(self.nqubits - 1):
            yield gates.M(i)


class HiddenShift(qasm.HiddenShift):

    def to_qasm(self):
        raise NotImplementedError

    def oracle(self):
        for i in range(self.nqubits // 2):
            yield gates.CZ(2 * i, 2 * i + 1)

    def __iter__(self):
        for i in range(self.nqubits):
            yield gates.H(i)
        for i, ish in enumerate(self.shift):
            if ish:
                yield gates.X(i)
        yield from self.oracle()
        for i, ish in enumerate(self.shift):
            if ish:
                yield gates.X(i)
        for i in range(self.nqubits):
            yield gates.H(i)
        yield from self.oracle()
        for i in range(self.nqubits):
            yield gates.H(i)
        yield gates.M(*range(self.nqubits))


class QAOA(qasm.QAOA):

    def to_qasm(self):
        raise NotImplementedError

    @staticmethod
    def RX(q, theta):
        return gates.RX(q, theta=theta)

    @staticmethod
    def RZZ(q0, q1, theta):
        phase = np.exp(0.5j * theta)
        phasec = np.conj(phase)
        matrix = np.diag([phasec, phase, phase, phasec])
        return gates.Unitary(matrix, q0, q1)

    def __iter__(self):
        np.random.seed(self.seed)
        betas = np.random.uniform(-np.pi, np.pi, size=self.nparams)
        gammas = np.random.uniform(-np.pi, np.pi, size=self.nparams)
        # Prepare uniform superposition
        for i in range(self.nqubits):
            yield gates.H(i)
        # Apply QAOA unitary
        yield from self.maxcut_unitary(betas, gammas)
        # Measure
        yield gates.M(*range(self.nqubits))


class SupremacyCircuit(qasm.SupremacyCircuit):

    def __init__(self, nqubits, depth="2", seed="123"):
        super().__init__(nqubits, depth, seed)
        from qibo import models

        parent = qasm.SupremacyCircuit(nqubits, depth, seed)
        self.qibo_circuit = models.Circuit.from_qasm(parent.to_qasm())

    def to_qasm(self):
        raise NotImplementedError

    def __iter__(self):
        yield from self.qibo_circuit.queue


class BasisChange(qasm.BasisChange):

    def __init__(self, nqubits, simulation_time="1", seed="123"):
        super().__init__(nqubits, simulation_time, seed)
        from qibo import models

        parent = qasm.BasisChange(nqubits, simulation_time, seed)
        self.qibo_circuit = models.Circuit.from_qasm(parent.to_qasm())

    def to_qasm(self):
        raise NotImplementedError

    def __iter__(self):
        yield from self.qibo_circuit.queue


class QuantumVolume(qasm.QuantumVolume):

    def __init__(self, nqubits, depth="1", seed="123"):
        super().__init__(nqubits, depth, seed)
        from qibo import models

        parent = qasm.QuantumVolume(nqubits, depth, seed)
        self.qibo_circuit = models.Circuit.from_qasm(parent.to_qasm())

    def to_qasm(self):
        raise NotImplementedError

    def __iter__(self):
        yield from self.qibo_circuit.queue
