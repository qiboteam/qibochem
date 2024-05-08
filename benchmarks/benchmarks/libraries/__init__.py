def parse(options):
    """Parse options from string.

    Args:
        options (str): String with options.
                       It should have the form 'arg1=value1,arg2=value2,...'.

    Returns:
        dict: {'arg1': value1, 'arg2': value2, ...}

    """
    kwargs = {}
    if options is not None:
        for parameter in options.split(","):
            if "=" in parameter:
                k, v = parameter.split("=")
                kwargs[k] = v
            else:
                raise ValueError(f"Cannot parse parameter {parameter}.")
    return kwargs


def get(backend_name, options=None):
    options = parse(options)
    if backend_name == "qibo":
        from benchmarks.libraries.qibo import Qibo

        return Qibo(**options)

    elif backend_name == "qiskit":
        from benchmarks.libraries.qiskit import Qiskit

        return Qiskit(**options)
    elif backend_name == "qiskit-gpu":
        from benchmarks.libraries.qiskit import QiskitGpu

        return QiskitGpu(**options)

    elif backend_name == "cirq":
        from benchmarks.libraries.cirq import Cirq

        return Cirq()
    elif backend_name == "qsim":
        from benchmarks.libraries.cirq import QSim

        return QSim(**options)
    elif backend_name == "qsim-gpu":
        from benchmarks.libraries.cirq import QSimGpu

        return QSimGpu(**options)
    elif backend_name == "qsim-cuquantum":
        from benchmarks.libraries.cirq import QSimCuQuantum

        return QSimCuQuantum(**options)
    elif backend_name == "tfq":
        from benchmarks.libraries.cirq import TensorflowQuantum

        return TensorflowQuantum()

    elif backend_name == "qulacs":
        from benchmarks.libraries.qulacs import Qulacs

        return Qulacs()
    elif backend_name == "qulacs-gpu":
        from benchmarks.libraries.qulacs import QulacsGpu

        return QulacsGpu()

    elif backend_name == "qcgpu":
        from benchmarks.libraries.qcgpu import QCGPU

        return QCGPU()

    elif backend_name == "projectq":
        from benchmarks.libraries.projectq import ProjectQ

        return ProjectQ(**options)

    elif backend_name == "hybridq":
        from benchmarks.libraries.hybridq import HybridQ

        return HybridQ(**options)
    elif backend_name == "hybridq-gpu":
        from benchmarks.libraries.hybridq import HybridQGPU

        return HybridQGPU(**options)

    raise KeyError(f"Unknown simulation library {backend_name}.")
