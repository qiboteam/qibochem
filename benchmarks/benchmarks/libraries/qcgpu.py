import numpy as np

from benchmarks.libraries import abstract


class QCGPU(abstract.ParserBackend):

    def __init__(self):
        import os

        os.environ["PYOPENCL_CTX"] = "0"
        import qcgpu

        self.name = "qcgpu"
        self.qcgpu = qcgpu
        self.__version__ = None

    def RX(self, target, theta):
        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        matrix = np.array([[cost, -1j * sint], [-1j * sint, cost]])
        gate = self.qcgpu.Gate(matrix)
        return ("apply_gate", (gate, target))

    def RY(self, target, theta):
        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        matrix = np.array([[cost, -sint], [sint, cost]])
        gate = self.qcgpu.Gate(matrix)
        return ("apply_gate", (gate, target))

    def RZ(self, target, theta):
        phase = np.exp(0.5j * theta)
        matrix = np.diag([np.conj(phase), phase])
        gate = self.qcgpu.Gate(matrix)
        return ("apply_gate", (gate, target))

    def U1(self, target, theta):
        phase = np.exp(1j * theta)
        matrix = np.diag([1, phase])
        gate = self.qcgpu.Gate(matrix)
        return ("apply_gate", (gate, target))

    def CU1(self, control, target, theta):
        phase = np.exp(1j * theta)
        matrix = np.diag([1, phase])
        gate = self.qcgpu.Gate(matrix)
        return ("apply_controlled_gate", (gate, control, target))

    def RZZ(self, target1, target2, theta):
        raise NotImplementedError

    class QCGPUCircuit(list):

        def __init__(self, nqubits):
            self.nqubits = nqubits

    def from_qasm(self, qasm):
        nqubits, gatelist = self.parse(qasm)
        circuit = self.QCGPUCircuit(nqubits)
        for gate, qubits, params in gatelist:
            args = list(qubits)
            if params is not None:
                args.extend(params)
            if gate == "SWAP":
                target1, target2 = qubits
                circuit.append(("cx", (target1, target2)))
                circuit.append(("cx", (target2, target1)))
                circuit.append(("cx", (target1, target2)))
            elif gate in {"RX", "RY", "RZ", "U1", "CU1"}:
                circuit.append(getattr(self, gate)(*args))
            else:
                circuit.append((gate.lower(), args))
        return circuit

    def __call__(self, circuit):
        state = self.qcgpu.State(circuit.nqubits)
        for gate, args in circuit:
            getattr(state, gate)(*args)
        return state.amplitudes()

    def set_precision(self, precision):
        if precision != "single":
            raise NotImplementedError(f"Cannot set {precision} precision for {self.name} backend.")

    def get_precision(self):
        return "single"

    def get_device(self):
        return None
