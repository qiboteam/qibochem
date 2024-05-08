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


def get(circuit_name, nqubits, options=None, qibo=False):
    if qibo:
        from benchmarks.circuits import qibo as module
    else:
        from benchmarks.circuits import qasm as module

    if circuit_name in ("qft", "QFT"):
        circuit = module.QFT
    elif circuit_name == "one-qubit-gate":
        circuit = module.OneQubitGate
    elif circuit_name == "two-qubit-gate":
        circuit = module.TwoQubitGate
    elif circuit_name in ("variational", "variational-circuit"):
        circuit = module.VariationalCircuit
    elif circuit_name in ("bernstein-vazirani", "bv"):
        circuit = module.BernsteinVazirani
    elif circuit_name in ("hidden-shift", "hs"):
        circuit = module.HiddenShift
    elif circuit_name == "qaoa":
        circuit = module.QAOA
    elif circuit_name == "supremacy":
        circuit = module.SupremacyCircuit
    elif circuit_name in ("basis-change", "bc"):
        circuit = module.BasisChange
    elif circuit_name in ("quantum-volume", "qv"):
        circuit = module.QuantumVolume
    else:
        raise NotImplementedError(f"Cannot find circuit {circuit_name}.")

    kwargs = parse(options)
    return circuit(nqubits, **kwargs)
