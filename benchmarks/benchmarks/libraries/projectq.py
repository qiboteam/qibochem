import numpy as np

from benchmarks.libraries import abstract


class ProjectQ(abstract.ParserBackend):

    def __init__(self, max_qubits="0", local_optimizer="0"):
        """Initialize data members.

        Args:
            max_qubits (str): if "0", gate fusion is disabled, otherwise it's enabled.
                              Note that it's not possible to set the maximum fused gate size.
            local_optimizer (str): if "0", local optimization of circuits is disabled,
                              otherwise it's enabled.
        """
        import projectq

        self.name = "projectq"
        self.projectq = projectq
        self.__version__ = None
        self.gate_fusion = int(max_qubits) > 0
        self.local_optimizer = bool(int(local_optimizer))

    def RX(self, theta):
        return self.projectq.ops.Rx(theta)

    def RY(self, theta):
        return self.projectq.ops.Ry(theta)

    def RZ(self, theta):
        return self.projectq.ops.Rz(theta)

    def U1(self, theta):
        return self.projectq.ops.R(theta)

    def U2(self, phi, lam):
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus), -np.conj(pminus)], [pminus, pplus]])
        matrix /= np.sqrt(2)
        return self.projectq.ops.MatrixGate(matrix)

    def U3(self, theta, phi, lam):
        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus) * cost, -np.conj(pminus) * sint], [pminus * sint, pplus * cost]])
        return self.projectq.ops.MatrixGate(matrix)

    def SWAP(self):
        return self.projectq.ops.Swap

    def CRX(self, theta):
        return self.projectq.ops.C(self.RX(theta))

    def CRY(self, theta):
        return self.projectq.ops.C(self.RY(theta))

    def CRZ(self, theta):
        return self.projectq.ops.CRz(theta)

    def CU1(self, theta):
        U1 = self.projectq.ops.R(theta)
        return self.projectq.ops.C(U1, n_qubits=1)

    def CU3(self, theta):
        raise NotImplementedError

    def RZZ(self, theta):
        return self.projectq.ops.Rzz(theta)

    def __getattr__(self, x):
        return getattr(self.projectq.ops, x)

    def __item__(self, x):
        return getattr(self.projectq.ops, x)

    def from_qasm(self, qasm):
        nqubits, gatelist = self.parse(qasm)
        backend = self.projectq.backends.Simulator(gate_fusion=self.gate_fusion)
        if self.local_optimizer:
            self.eng = self.projectq.MainEngine(backend=backend, engine_list=[self.projectq.cengines.LocalOptimizer()])
        else:
            self.eng = self.projectq.MainEngine(backend=backend)
        qureg = self.eng.allocate_qureg(nqubits)
        for gatename, qubits, params in gatelist:
            gate = getattr(self, gatename)
            if params is not None:
                parameters = list(params)
                if len(qubits) > 1:
                    gate(*parameters) | tuple(qureg[i] for i in qubits)
                else:
                    gate(*parameters) | qureg[qubits[0]]
            elif len(qubits) > 1:
                if gatename == "SWAP":
                    gate() | tuple(qureg[i] for i in qubits)
                else:
                    gate | tuple(qureg[i] for i in qubits)
            else:
                gate | qureg[qubits[0]]

        return qureg

    def __call__(self, qureg):
        self.eng.flush()
        self.qubit_id, wave = self.eng.backend.cheat()
        # measure everything to avoid error when running
        self.projectq.ops.All(self.projectq.ops.Measure) | qureg
        return np.array(wave)

    def transpose_state(self, x):
        shape = tuple(x.shape)
        nqubits = int(np.log2(shape[0]))
        x = np.reshape(x, nqubits * (2,))
        x = np.transpose(x, range(nqubits - 1, -1, -1))
        x = np.transpose(x, tuple(self.qubit_id[key] for key in self.qubit_id))
        x = np.reshape(x, shape)
        return x

    def set_precision(self, precision):
        if precision != "double":
            raise NotImplementedError(f"Cannot set {precision} precision for {self.name} backend.")

    def get_precision(self):
        return "double"

    def get_device(self):
        return None
