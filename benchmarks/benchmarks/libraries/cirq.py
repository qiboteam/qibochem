import numpy as np

from benchmarks.libraries import abstract


class Cirq(abstract.ParserBackend):

    def __init__(self):
        import cirq

        self.name = "cirq"
        self.__version__ = cirq.__version__
        self.cirq = cirq
        self.precision = "double"
        self.simulator = cirq.Simulator(dtype=np.complex128)

    def RX(self, theta):
        return self.cirq.rx(theta)

    def RY(self, theta):
        return self.cirq.ry(theta)

    def RZ(self, theta):
        return self.cirq.rz(theta)

    def CU1(self, theta):
        return self.cirq.CZPowGate(exponent=theta / np.pi)

    def CU3(self, theta, phi, lam):
        gate = self.cirq.circuits.qasm_output.QasmUGate(theta / np.pi, phi / np.pi, lam / np.pi)
        return gate.controlled(num_controls=1)

    def RZZ(self, theta):
        import numpy as np

        return self.cirq.ZZPowGate(exponent=theta / np.pi, global_shift=-0.5)

    def __getattr__(self, x):
        return getattr(self.cirq, x)

    def __getitem__(self, x):
        return getattr(self.cirq, x)

    def from_qasm(self, qasm):
        from cirq.contrib.qasm_import import circuit_from_qasm, exception

        try:
            return circuit_from_qasm(qasm)
        except exception.QasmException:
            nqubits, gatelist = self.parse(qasm)
            qubits = [self.cirq.GridQubit(i, 0) for i in range(nqubits)]
            circuit = self.cirq.Circuit()
            for gatename, qid, params in gatelist:
                if params is not None:
                    gate = getattr(self, gatename)(*params)
                else:
                    gate = getattr(self, gatename)
                circuit.append(gate(*(qubits[i] for i in qid)))
            return circuit

    def __call__(self, circuit):
        result = self.simulator.simulate(circuit)
        return result.final_state_vector

    def transpose_state(self, x):
        return x

    def get_precision(self):
        return self.precision

    def set_precision(self, precision):
        import numpy as np

        self.precision = precision
        if precision == "single":
            self.simulator = self.cirq.Simulator(dtype=np.complex64)
        else:
            self.simulator = self.cirq.Simulator(dtype=np.complex128)

    def get_device(self):
        return None


class TensorflowQuantum(Cirq):

    def __init__(self):
        import cirq
        import tensorflow_quantum as tfq

        self.name = "tfq"
        self.cirq = cirq
        self.precision = "single"
        self.__version__ = tfq.__version__
        self.state_layer = tfq.layers.State()

    def set_precision(self, precision):
        if precision == "double":
            raise NotImplementedError(f"Cannot set precision '{precision}' for {self.name} backend.")

    def from_qasm(self, qasm):
        circuit = super().from_qasm(qasm)
        # change `NamedQubit`s to `GridQubit`s as TFQ understands only `GridQubit`
        qubit_map = {}
        for q in circuit.all_qubits():
            if isinstance(q, self.cirq.NamedQubit):
                i = int(str(q).split("_")[-1])
                qubit_map[q] = self.cirq.GridQubit(i, 0)
        if qubit_map:
            return circuit.transform_qubits(qubit_map)
        return circuit

    def __call__(self, circuit):
        # transfer final state to numpy array because that's what happens
        # for all backends
        return self.state_layer(circuit)[0].numpy()


class QSim(Cirq):

    def __init__(self, max_qubits="0", nthreads=None):
        import cirq
        import qsimcirq

        self.name = "qsim"
        self.cirq = cirq
        self.qsimcirq = qsimcirq
        self.precision = "single"
        self.__version__ = qsimcirq.__version__

        if nthreads is None:
            from multiprocessing import cpu_count

            self.nthreads = cpu_count()
        else:
            self.nthreads = int(nthreads)
        self.max_qubits = int(max_qubits)

        self.simulator = self.get_simulator()

    def get_simulator(self):
        return self.qsimcirq.QSimSimulator({"t": self.nthreads, "f": self.max_qubits})

    def set_precision(self, precision):
        if precision == "double":
            raise NotImplementedError(f"Cannot set precision '{precision}' for {self.name} backend.")


class QSimGpu(QSim):

    def __init__(self, max_qubits="0"):
        super().__init__(max_qubits)
        self.name = "qsim-gpu"

    def get_simulator(self):
        qsim_options = self.qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0, max_fused_gate_size=self.max_qubits)
        return self.qsimcirq.QSimSimulator(qsim_options)


class QSimCuQuantum(QSim):

    def __init__(self, max_qubits="0"):
        super().__init__(max_qubits)
        self.name = "qsim-cuquantum"

    def get_simulator(self):
        qsim_options = self.qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1, max_fused_gate_size=self.max_qubits)
        return self.qsimcirq.QSimSimulator(qsim_options)
