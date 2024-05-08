import os

import numpy as np

from benchmarks.libraries import abstract


class HybridQ(abstract.ParserBackend):

    def __init__(self, max_qubits="0", simplify="False"):
        from hybridq.gate import Gate, MatrixGate

        self.name = "hybridq"
        self.__version__ = "0.7.7.post2"
        self.Gate = Gate
        self.MatrixGate = MatrixGate
        self.max_qubits = int(max_qubits)
        if simplify in ("true", "True"):
            self.simplify = True
        else:
            self.simplify = False
        self.complex_type = "complex128"
        self.max_qubits = int(max_qubits)

    def H(self, q):
        return self.Gate("H", qubits=(q,))

    def X(self, q):
        return self.Gate("X", qubits=(q,))

    def Y(self, q):
        return self.Gate("Y", qubits=(q,))

    def Z(self, q):
        return self.Gate("Z", qubits=(q,))

    def RX(self, q, theta):
        return self.Gate("RX", params=[theta], qubits=(q,))

    def RY(self, q, theta):
        return self.Gate("RY", params=[theta], qubits=(q,))

    def RZ(self, q, theta):
        return self.Gate("RZ", params=[theta], qubits=(q,))

    def U1(self, q, theta):
        phase = np.exp(1j * theta)
        matrix = np.diag([1, phase])
        return self.MatrixGate(U=matrix, qubits=(q,))

    def U2(self, q, phi, lam):
        plus = np.exp(0.5j * (phi + lam))
        minus = np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(plus), -np.conj(minus)], [minus, plus]]) / np.sqrt(2)
        return self.MatrixGate(U=matrix, qubits=(q,))

    def U3(self, q, theta, phi, lam):
        return self.Gate("U3", params=[theta, phi, lam], qubits=(q,))

    def CNOT(self, q1, q2):
        return self.Gate("CNOT", qubits=(q1, q2))

    def SWAP(self, q1, q2):
        return self.Gate("SWAP", qubits=(q1, q2))

    def CZ(self, q1, q2):
        return self.Gate("CZ", qubits=(q1, q2))

    def CU1(self, q1, q2, theta):
        return self.Gate("CPHASE", params=[theta], qubits=(q1, q2))

    def CU3(self, q1, q2, theta, phi, lam):
        from hybridq.gate import Control

        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus) * cost, -np.conj(pminus) * sint], [pminus * sint, pplus * cost]])
        gate = self.MatrixGate(U=matrix, qubits=(q2,))
        return Control((q1,), gate=gate)

    def RZZ(self, q1, q2, theta):
        phase = np.exp(0.5j * theta)
        phasec = np.conj(phase)
        matrix = np.diag([phasec, phase, phase, phasec])
        return self.MatrixGate(U=matrix, qubits=(q1, q2))

    def from_qasm(self, qasm):
        from hybridq.circuit import Circuit

        nqubits, gatelist = self.parse(qasm)
        circuit = Circuit()
        for gatename, qubits, params in gatelist:
            args = list(qubits)
            if params:
                args.extend(params)
            gate = getattr(self, gatename)(*args)
            circuit.append(gate)
        return circuit

    def __call__(self, circuit):
        from hybridq.circuit.simulation import simulate

        initial_state = len(circuit.all_qubits()) * "0"
        final_state = simulate(
            circuit,
            optimize="evolution",
            initial_state=initial_state,
            complex_type=self.complex_type,
            simplify=self.simplify,
            compress=self.max_qubits,
            max_largest_intermediate=2**40,
        )
        return final_state.ravel()

    def transpose_state(self, x):
        return x

    def set_precision(self, precision):
        if precision == "single":
            self.complex_type = "complex64"
        else:
            self.complex_type = "complex128"

    def get_precision(self):
        if self.complex_type == "complex64":
            return "single"
        else:
            return "double"

    def get_device(self):
        return None


class HybridQGPU(HybridQ):

    def __init__(self, max_qubits="0", simplify="False"):
        super().__init__(max_qubits=max_qubits, simplify=simplify)
        self.name = "hybridq-gpu"

    def __call__(self, circuit):
        from hybridq.circuit.simulation import simulate

        initial_state = len(circuit.all_qubits()) * "0"
        final_state = simulate(
            circuit,
            optimize="evolution-einsum",
            backend="jax",
            initial_state=initial_state,
            complex_type=self.complex_type,
            simplify=self.simplify,
            compress=self.max_qubits,
            max_largest_intermediate=2**40,
        )
        return final_state.ravel()
