from benchmarks.libraries import abstract


class Qiskit(abstract.AbstractBackend):

    def __init__(
        self, max_qubits="0", fusion_threshold="1", max_parallel_threads="0", statevector_parallel_threshold="14"
    ):
        import qiskit
        from qiskit.providers.aer import StatevectorSimulator

        self.name = "qiskit"
        self.__version__ = qiskit.__version__
        self.max_qubits = int(max_qubits)
        self.sim_options = dict(
            max_parallel_threads=int(max_parallel_threads),
            statevector_parallel_threshold=int(statevector_parallel_threshold),
            fusion_enable=self.max_qubits > 0,
            fusion_max_qubit=self.max_qubits,
            fusion_threshold=int(fusion_threshold),
            precision="double",
        )
        self.simulator = StatevectorSimulator(**self.sim_options)

    def from_qasm(self, qasm):
        from qiskit import QuantumCircuit

        # TODO: Consider using `circ = transpile(circ, simulator)`
        if "cu3" in qasm:
            import re

            theta, phi, lam = re.findall(r"cu3\((.*)\)", qasm)[0].split(",")
            gamma = -(float(phi) + float(lam)) / 2
            qasm = re.sub(rf"cu3\((.*)\)", f"cu({theta},{phi},{lam},{gamma})", qasm)
        return QuantumCircuit.from_qasm_str(qasm)

    def __call__(self, circuit):
        result = self.simulator.run(circuit).result()
        return result.get_statevector(circuit)

    def get_precision(self):
        return self.sim_options.get("precision")

    def set_precision(self, precision):
        from qiskit.providers.aer import StatevectorSimulator

        self.sim_options["precision"] = precision
        self.simulator = StatevectorSimulator(**self.sim_options)

    def get_device(self):
        return None


class QiskitGpu(Qiskit):

    def __init__(self, max_qubits="0", fusion_threshold="1"):
        from qiskit.providers.aer import StatevectorSimulator

        super().__init__(max_qubits)
        self.name = "qiskit-gpu"
        self.sim_options = dict(
            device="GPU",
            fusion_enable=self.max_qubits > 0,
            fusion_max_qubit=self.max_qubits,
            fusion_threshold=int(fusion_threshold),
            precision="double",
        )
        self.simulator = StatevectorSimulator(**self.sim_options)
