from abc import abstractmethod

import numpy as np


class AbstractCircuit:

    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.parameters = {}

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def to_qasm(self, theta=None):
        """Creates the circuit in OpenQASM format.

        Args:
            theta (np.ndarray): If not ``None`` ``RX`` gates with the given
                angles are added before the actual circuit gates so that the
                initial state is non-trivial. Useful for testing.

        Returns:
            A string with the circuit in OpenQASM format.
        """
        code = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.nqubits}];", f"creg m[{self.nqubits}];"]
        if theta is not None:
            code.extend(f"rx({t}) q[{i}];" for i, t in enumerate(theta))
        code.extend(iter(self))
        return "\n".join(code)

    def __str__(self):
        return ", ".join(f"{k}={v}" for k, v in self.parameters.items())


class OneQubitGate(AbstractCircuit):
    """Applies a specific one qubit gate to all qubits."""

    def __init__(self, nqubits, nlayers="1", gate="h", angles=""):
        super().__init__(nqubits)
        self.gate = gate
        self.nlayers = int(nlayers)
        self.angles = angles
        self.parameters = {"nqubits": nqubits, "nlayers": nlayers, "gate": gate, "params": angles}

    def base_command(self, i):
        if self.angles:
            return "{}({}) q[{}];".format(self.gate, self.angles, i)
        else:
            return "{} q[{}];".format(self.gate, i)

    def __iter__(self):
        for _ in range(self.nlayers):
            for i in range(self.nqubits):
                yield self.base_command(i)


class TwoQubitGate(OneQubitGate):
    """Applies a specific two qubit gate to all pairs of adjacent qubits."""

    def __init__(self, nqubits, nlayers="1", gate="cx", angles=""):
        super().__init__(nqubits, nlayers, gate, angles)

    def base_command(self, i):
        if self.angles:
            return "{}({}) q[{}],q[{}];".format(self.gate, self.angles, i, i + 1)
        else:
            return "{} q[{}],q[{}];".format(self.gate, i, i + 1)

    def __iter__(self):
        for _ in range(self.nlayers):
            for i in range(0, self.nqubits - 1, 2):
                yield self.base_command(i)
            for i in range(1, self.nqubits - 1, 2):
                yield self.base_command(i)


class QFT(AbstractCircuit):
    """Applies the Quantum Fourier Transform."""

    def __init__(self, nqubits, swaps="True"):
        super().__init__(nqubits)
        self.swaps = swaps == "True"
        self.parameters = {"nqubits": nqubits, "swaps": swaps}

    def __iter__(self):
        for i1 in range(self.nqubits):
            yield f"h q[{i1}];"
            for i2 in range(i1 + 1, self.nqubits):
                theta = np.pi / 2 ** (i2 - i1)
                yield f"cu1({theta}) q[{i2}],q[{i1}];"

        if self.swaps:
            for i in range(self.nqubits // 2):
                yield f"swap q[{i}],q[{self.nqubits - i - 1}];"


class VariationalCircuit(AbstractCircuit):
    """Example variational circuit consisting of alternating layers of RY and CZ gates."""

    def __init__(self, nqubits, nlayers="1", seed="123"):
        super().__init__(nqubits)
        self.nlayers = int(nlayers)
        self.seed = int(seed)
        self.parameters = {"nqubits": nqubits, "nlayers": nlayers, "seed": seed}

    def __iter__(self):
        nparams = 2 * self.nlayers * self.nqubits
        np.random.seed(self.seed)
        theta = iter(2 * np.pi * np.random.random(nparams))
        for l in range(self.nlayers):
            for i in range(self.nqubits):
                yield f"ry({next(theta)}) q[{i}];"
            for i in range(0, self.nqubits - 1, 2):
                yield f"cz q[{i}],q[{i + 1}];"
            for i in range(self.nqubits):
                yield f"ry({next(theta)}) q[{i}];"
            for i in range(1, self.nqubits - 2, 2):
                yield f"cz q[{i}],q[{i + 1}];"
            yield f"cz q[{0}],q[{self.nqubits - 1}];"


class BernsteinVazirani(AbstractCircuit):
    """Applies the Bernstein-Vazirani algorithm from Qiskit/openqasm.

    See `https://github.com/Qiskit/openqasm/tree/0af8b8489f32d46692b3a3a1421e98c611cd86cc/benchmarks/bv`
    for the OpenQASM code.
    Note that `Barrier` gates are excluded for simulation.
    """

    def __init__(self, nqubits):
        super().__init__(nqubits)
        self.parameters = {"nqubits": nqubits}

    def __iter__(self):
        yield f"x q[{self.nqubits - 1}];"
        for i in range(self.nqubits):
            yield f"h q[{i}];"
        for i in range(self.nqubits - 1):
            yield f"cx q[{i}],q[{self.nqubits - 1}];"
        for i in range(self.nqubits - 1):
            yield f"h q[{i}];"
        # for i in range(self.nqubits - 1):
        #    yield f"measure m[{i}];"


class HiddenShift(AbstractCircuit):
    """Applies the Hidden Shift algorithm.

    See `https://github.com/quantumlib/Cirq/blob/master/examples/hidden_shift_algorithm.py`
    for the Cirq code.
    If the shift (hidden bitstring) is not given then it is randomly generated
    using `np.random.randint`.
    """

    def __init__(self, nqubits, shift=""):
        super().__init__(nqubits)
        if len(shift):
            if len(shift) != nqubits:
                raise ValueError(
                    "Shift bitstring of length {} was given for " "circuit of {} qubits." "".format(len(shift), nqubits)
                )
            self.shift = [int(x) for x in shift]
        else:
            self.shift = np.random.randint(0, 2, size=(self.nqubits,))
        self.parameters = {"nqubits": nqubits, "shift": shift}

    def oracle(self):
        for i in range(self.nqubits // 2):
            yield f"cz q[{2 * i}],q[{2 * i + 1}];"

    def __iter__(self):
        for i in range(self.nqubits):
            yield f"h q[{i}];"
        for i, ish in enumerate(self.shift):
            if ish:
                yield f"x q[{i}];"
        yield from self.oracle()
        for i, ish in enumerate(self.shift):
            if ish:
                yield f"x q[{i}];"
        for i in range(self.nqubits):
            yield f"h q[{i}];"
        yield from self.oracle()
        for i in range(self.nqubits):
            yield f"h q[{i}];"
        # for i in range(self.nqubits):
        #    yield f"measure m[{i}];"


class QAOA(AbstractCircuit):
    """Example QAOA circuit for a MaxCut problem instance.

    See `https://github.com/quantumlib/Cirq/blob/master/examples/qaoa.py` for
    the Cirq code.
    If a JSON file containing the node link structure is given then the graph
    is loaded using `networkx.readwrite.json_graph.node_link_graph`, otherwise
    the graph is generated randomly using `networkx.random_regular_graph`.
    Note that different graphs may lead to different performance as the graph
    structure affects circuit depth.
    """

    def __init__(self, nqubits, nparams="2", graph="", seed="123"):
        super().__init__(nqubits)
        import networkx

        self.nparams = int(nparams)
        self.seed = int(123)
        if len(graph):
            import json

            with open(graph) as file:
                data = json.load(file)
            self.graph = networkx.readwrite.json_graph.node_link_graph(data)
        else:
            self.graph = networkx.random_regular_graph(3, self.nqubits)
        self.parameters = {"nqubits": nqubits, "nparams": nparams, "graph": graph, "seed": seed}

    @staticmethod
    def RX(q, theta):
        return f"rx({theta}) q[{q}];"

    @staticmethod
    def RZZ(q0, q1, theta):
        return f"rzz({theta}) q[{q0}],q[{q1}];"

    def maxcut_unitary(self, betas, gammas):
        for beta, gamma in zip(betas, gammas):
            for i, j in self.graph.edges:
                yield self.RZZ(i, j, -0.5 * gamma)
            for i in range(self.nqubits):
                yield self.RX(i, 2 * beta)

    def dump(self, dir):
        """Saves graph data as JSON in given directory."""
        import json

        import networkx

        data = networkx.readwrite.json_graph.node_link_data(self.graph)
        with open(dir, "w") as file:
            json.dump(data, file)

    def __iter__(self):
        np.random.seed(self.seed)
        betas = np.random.uniform(-np.pi, np.pi, size=self.nparams)
        gammas = np.random.uniform(-np.pi, np.pi, size=self.nparams)
        # Prepare uniform superposition
        for i in range(self.nqubits):
            yield f"h q[{i}];"
        # Apply QAOA unitary
        yield from self.maxcut_unitary(betas, gammas)
        # Measure
        # yield gates.M(*range(self.nqubits))


class SupremacyCircuit(AbstractCircuit):
    """Random circuit by Boixo et al 2018 for demonstrating quantum supremacy.

    See `https://github.com/quantumlib/Cirq/blob/v0.11.0/cirq-core/cirq/experiments/google_v2_supremacy_circuit.py`
    for the Cirq code.
    This circuit is constructed using `cirq` by exporting to OpenQASM and
    importing back to Qibo.
    """

    def __init__(self, nqubits, depth="2", seed="123"):
        super().__init__(nqubits)
        self.depth = int(depth)
        self.seed = int(seed)
        self.parameters = {"nqubits": nqubits, "depth": depth, "seed": seed}
        self.cirq_circuit = self.create_cirq_circuit()

    def create_cirq_circuit(self):
        import cirq
        from cirq.experiments import google_v2_supremacy_circuit as spc

        qubits = [cirq.GridQubit(i, 0) for i in range(self.nqubits)]
        return spc.generate_boixo_2018_supremacy_circuits_v2(qubits, self.depth, self.seed)

    def __iter__(self):
        qasm = self.cirq_circuit.to_qasm()
        for line in qasm.split("\n"):
            first_word = line.split(" ")[0]
            if first_word not in {"//", "OPENQASM", "include", "qreg"}:
                if first_word == "sx":
                    yield line.replace("sx", "rx(pi*0.5)")  # see issue #13
                else:
                    yield line


class BasisChange(AbstractCircuit):
    """Basis change fermionic circuit.

    See `https://quantumai.google/openfermion/tutorials/circuits_1_basis_change`
    for OpenFermion/Cirq code.
    This circuit is constructed using `openfermion` and `cirq` by exporting
    to OpenQASM and importing back to Qibo.
    """

    def __init__(self, nqubits, simulation_time="1", seed="123"):
        super().__init__(nqubits)
        self.simulation_time = float(simulation_time)
        self.seed = int(seed)
        self.parameters = {"nqubits": nqubits, "simulation_time": simulation_time, "seed": seed}
        self.openfermion_circuit = self.create_openfermion_circuit()

    def create_openfermion_circuit(self):
        import cirq
        import openfermion

        # Generate the random one-body operator.
        T = openfermion.random_hermitian_matrix(self.nqubits, seed=self.seed)
        # Diagonalize T and obtain basis transformation matrix (aka "u").
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        basis_transformation_matrix = eigenvectors.transpose()
        # Initialize the qubit register.
        qubits = cirq.LineQubit.range(self.nqubits)
        # Start circuit with the inverse basis rotation, print out this step.
        inverse_basis_rotation = cirq.inverse(openfermion.bogoliubov_transform(qubits, basis_transformation_matrix))
        circuit = cirq.Circuit(inverse_basis_rotation)
        # Add diagonal phase rotations to circuit.
        for k, eigenvalue in enumerate(eigenvalues):
            phase = -eigenvalue * self.simulation_time
            circuit.append(cirq.rz(rads=phase).on(qubits[k]))
        # Finally, restore basis.
        basis_rotation = openfermion.bogoliubov_transform(qubits, basis_transformation_matrix)
        circuit.append(basis_rotation)
        return circuit

    def __iter__(self):
        qasm = self.openfermion_circuit.to_qasm()
        for line in qasm.split("\n"):
            first_word = line.split(" ")[0]
            if first_word not in {"//", "OPENQASM", "include", "qreg"}:
                yield line


class QuantumVolume(AbstractCircuit):
    """Quantum Volume circuit from Qiskit.

    See `https://qiskit.org/documentation/stubs/qiskit.circuit.library.QuantumVolume.html`
    for the Qiskit model.
    This circuit is constructed using `qiskit` by exporting to OpenQASM and
    importing back to Qibo.
    """

    def __init__(self, nqubits, depth="1", seed="123"):
        super().__init__(nqubits)
        self.depth = int(depth)
        self.seed = int(seed)
        self.parameters = {"nqubits": nqubits, "depth": depth, "seed": seed}
        self.qiskit_circuit = self.create_qiskit_circuit()
        self.expression_symbols = {"*", "/"}
        self.expression_symbols.update(str(x) for x in range(10))

    def create_qiskit_circuit(self):
        from qiskit.circuit.library import QuantumVolume

        circuit = QuantumVolume(self.nqubits, self.depth, seed=self.seed)
        return circuit.decompose().decompose()

    def __iter__(self):
        raise NotImplementedError(
            "Iteration is not available for " "`QuantumVolume` because it is prepared " "using Qiskit."
        )

    def evaluate_pi(self, qasm):
        left = qasm.find("pi")
        if left < 0:
            return qasm

        import sympy

        right = left + 2
        left = left - 1
        while qasm[left] in self.expression_symbols:
            left -= 1
        while qasm[right] in self.expression_symbols:
            right += 1
        expr = qasm[left + 1 : right]
        evaluated = sympy.sympify(expr).evalf()
        return self.evaluate_pi(qasm.replace(expr, str(evaluated)))

    def __iter__(self):
        qasm = self.qiskit_circuit.qasm()
        for line in qasm.split("\n"):
            first_word = line.split(" ")[0]
            if first_word not in {"//", "OPENQASM", "include", "qreg"}:
                yield line.replace("1/(15*pi)", str(1.0 / (15.0 * np.pi)))

    def to_qasm(self, theta=None):
        qasm = super().to_qasm(theta)
        return self.evaluate_pi(qasm)
