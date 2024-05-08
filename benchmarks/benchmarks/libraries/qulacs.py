import numpy as np

from benchmarks.libraries import abstract


class Qulacs(abstract.ParserBackend):

    def __init__(self):
        import qulacs

        self.name = "qulacs"
        self.qulacs = qulacs
        self.__version__ = None
        self.QuantumState = self.qulacs.QuantumState

    def RX(self, target, theta):
        return self.qulacs.gate.RX(target, -theta)

    def RY(self, target, theta):
        return self.qulacs.gate.RY(target, -theta)

    def RZ(self, target, theta):
        return self.qulacs.gate.RZ(target, -theta)

    def CU1(self, control, target, theta):
        # See `https://github.com/qulacs/qulacs/issues/278` for CU1 on Qulacs
        matrix = np.diag([1, np.exp(1j * theta)])
        gate = self.qulacs.gate.DenseMatrix([target], matrix)
        gate.add_control_qubit(control, 1)
        return gate

    def CU3(self, control, target, theta, phi, lam):
        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus) * cost, -np.conj(pminus) * sint], [pminus * sint, pplus * cost]])
        gate = self.qulacs.gate.DenseMatrix([target], matrix)
        gate.add_control_qubit(control, 1)
        return gate

    def RZZ(self, target1, target2, theta):
        phase = np.exp(0.5j * theta)
        phasec = np.conj(phase)
        matrix = np.diag([phasec, phase, phase, phasec])
        gate = self.qulacs.gate.DenseMatrix([target1, target2], matrix)
        return gate

    def __getattr__(self, x):
        return getattr(self.qulacs.gate, x)

    def __getitem__(self, x):
        return getattr(self.qulacs.gate, x)

    def from_qasm(self, qasm):
        nqubits, gatelist = self.parse(qasm)
        circuit = self.qulacs.QuantumCircuit(nqubits)
        for gatename, qubits, params in gatelist:
            gate = getattr(self, gatename)
            args = list(qubits)
            if params is not None:
                args.extend(params)
            circuit.add_gate(gate(*args))
        return circuit

    def __call__(self, circuit):
        nqubits = circuit.get_qubit_count()
        state = self.QuantumState(nqubits)
        circuit.update_quantum_state(state)
        return state.get_vector()

    def set_precision(self, precision):
        if precision != "double":
            raise NotImplementedError(f"Cannot set {precision} precision for {self.name} backend.")

    def get_precision(self):
        return "double"

    def get_device(self):
        return None


class QulacsGpu(Qulacs):

    def __init__(self):
        super().__init__()
        self.name = "qulacs-gpu"
        self.QuantumState = self.qulacs.QuantumStateGpu
