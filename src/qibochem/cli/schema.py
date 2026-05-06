"""Configuration schema for the qibochem CLI.

Top-level structure of a qibochem YAML input:

    version: 1
    molecule: { ... }
    hamiltonian: { ... }
    method: { kind: hf|vqe|qsci|qse, ... }
    output: { ... }

Each block is parsed into a frozen dataclass with ``__post_init__``
validation, matching the convention already used by ``QSCIConfig`` in
``qibochem.selected_ci``. Frozen instances make accidental mid-pipeline
mutation a hard error, and the post-init methods here only validate —
they never normalise fields — so freezing is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_VERSION = 1
VALID_MAPPINGS = ("jw", "bk")
VALID_METHODS = ("hf", "vqe", "qsci", "qse")
VALID_ANSATZ_KINDS = ("ucc", "givens", "he", "qeb", "symm", "basis_rotation")
VALID_EXCITATIONS = ("singles", "doubles")
VALID_INIT_PARAMS = ("zeros", "random", "mp2")
VALID_LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(ValueError):
    """Raised when a CLI input file fails schema validation."""


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return d[key]


def _unknown_keys(d: dict, allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ConfigError(f"{where}: unknown keys {sorted(extra)}; allowed: {sorted(allowed)}")


def _coerce_int(value: Any, where: str, *, allow_none: bool = False) -> Any:
    """Coerce ``value`` to int, raising :class:`ConfigError` on bad input.

    Accepts bool-free numeric types only (``True``/``False`` are *not* valid
    integers here even though Python's ``int(True) == 1`` would silently work).
    Without this guard, downstream comparisons like ``self.n_shots <= 0``
    raise raw ``TypeError`` and surface as exit code 1 instead of 2.
    """
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{where}: must be an integer; got {type(value).__name__}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: must be an integer; got {value!r}") from None


@dataclass(frozen=True)
class MoleculeConfig:
    """Molecule block.

    Exactly one of ``xyz_file`` or ``geometry`` must be set.
    ``geometry`` follows OpenFermion format: list of ``[symbol, [x, y, z]]``.
    """

    xyz_file: str | None = None
    geometry: list | None = None
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"
    active: list[int] | None = None

    def __post_init__(self) -> None:
        if (self.xyz_file is None) == (self.geometry is None):
            raise ConfigError("molecule: exactly one of 'xyz_file' or 'geometry' must be set")
        if self.multiplicity < 1:
            raise ConfigError("molecule.multiplicity must be >= 1")
        if self.geometry is not None:
            if not isinstance(self.geometry, list) or not self.geometry:
                raise ConfigError("molecule.geometry must be a non-empty list")
            for i, atom in enumerate(self.geometry):
                if not (isinstance(atom, (list, tuple)) and len(atom) == 2):
                    raise ConfigError(f"molecule.geometry[{i}]: expected [symbol, [x,y,z]]")
                sym, xyz = atom
                if not isinstance(sym, str):
                    raise ConfigError(f"molecule.geometry[{i}]: atom symbol must be a string")
                if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                    raise ConfigError(f"molecule.geometry[{i}]: coordinates must be a 3-list")
        if self.active is not None:
            if not isinstance(self.active, list) or not all(isinstance(i, int) for i in self.active):
                raise ConfigError("molecule.active must be a list of ints")

    @classmethod
    def parse(cls, d: dict) -> MoleculeConfig:
        _unknown_keys(d, {"xyz_file", "geometry", "charge", "multiplicity", "basis", "active"}, "molecule")
        return cls(
            xyz_file=d.get("xyz_file"),
            geometry=d.get("geometry"),
            charge=_coerce_int(d.get("charge", 0), "molecule.charge"),
            multiplicity=_coerce_int(d.get("multiplicity", 1), "molecule.multiplicity"),
            basis=str(d.get("basis", "sto-3g")),
            active=d.get("active"),
        )


@dataclass(frozen=True)
class HamiltonianConfig:
    mapping: str = "jw"

    def __post_init__(self) -> None:
        if self.mapping not in VALID_MAPPINGS:
            raise ConfigError(f"hamiltonian.mapping must be one of {VALID_MAPPINGS}; got {self.mapping!r}")

    @classmethod
    def parse(cls, d: dict) -> HamiltonianConfig:
        _unknown_keys(d, {"mapping"}, "hamiltonian")
        return cls(mapping=str(d.get("mapping", "jw")))


@dataclass(frozen=True)
class AnsatzConfig:
    kind: str
    excitations: list[str] = field(default_factory=lambda: ["singles", "doubles"])
    layers: int = 1

    def __post_init__(self) -> None:
        if self.kind not in VALID_ANSATZ_KINDS:
            raise ConfigError(f"method.ansatz.kind must be one of {VALID_ANSATZ_KINDS}; got {self.kind!r}")
        bad = [e for e in self.excitations if e not in VALID_EXCITATIONS]
        if bad:
            raise ConfigError(
                f"method.ansatz.excitations contains invalid entries {bad}; allowed: {list(VALID_EXCITATIONS)}"
            )
        if self.layers < 1:
            raise ConfigError("method.ansatz.layers must be >= 1")

    @classmethod
    def parse(cls, d: dict) -> AnsatzConfig:
        _unknown_keys(d, {"kind", "excitations", "layers"}, "method.ansatz")
        return cls(
            kind=str(_require(d, "kind", "method.ansatz")),
            excitations=list(d.get("excitations", ["singles", "doubles"])),
            layers=_coerce_int(d.get("layers", 1), "method.ansatz.layers"),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "BFGS"
    options: dict = field(default_factory=dict)
    initial_parameters: Any = "zeros"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.options, dict):
            raise ConfigError("method.optimizer.options must be a mapping")
        if isinstance(self.initial_parameters, str):
            if self.initial_parameters not in VALID_INIT_PARAMS:
                raise ConfigError(
                    f"method.optimizer.initial_parameters must be one of "
                    f"{VALID_INIT_PARAMS} or an explicit list; got {self.initial_parameters!r}"
                )
        elif isinstance(self.initial_parameters, list):
            if not all(isinstance(x, (int, float)) for x in self.initial_parameters):
                raise ConfigError("method.optimizer.initial_parameters list must contain only numbers")
        else:
            raise ConfigError("method.optimizer.initial_parameters must be a string strategy or a list of numbers")

    @classmethod
    def parse(cls, d: dict) -> OptimizerConfig:
        _unknown_keys(d, {"name", "options", "initial_parameters", "seed"}, "method.optimizer")
        # Pass `options` through unchanged; __post_init__ validates the type so
        # users get a clean ConfigError rather than a generic dict-coercion error.
        return cls(
            name=str(d.get("name", "BFGS")),
            options=d.get("options", {}),
            initial_parameters=d.get("initial_parameters", "zeros"),
            seed=_coerce_int(d.get("seed"), "method.optimizer.seed", allow_none=True),
        )


@dataclass(frozen=True)
class HFMethodConfig:
    kind: str = "hf"

    def __post_init__(self) -> None:
        if self.kind != "hf":  # pragma: no cover - parse() always sets kind="hf"
            raise ConfigError(f"HFMethodConfig.kind must be 'hf'; got {self.kind!r}")

    @classmethod
    def parse(cls, d: dict) -> HFMethodConfig:
        _unknown_keys(d, {"kind"}, "method (hf)")
        return cls(kind="hf")


@dataclass(frozen=True)
class VQEMethodConfig:
    kind: str = "vqe"
    ansatz: AnsatzConfig = field(default_factory=lambda: AnsatzConfig(kind="ucc"))
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    n_shots: int | None = None

    def __post_init__(self) -> None:
        if self.kind != "vqe":  # pragma: no cover - parse() always sets kind="vqe"
            raise ConfigError(f"VQEMethodConfig.kind must be 'vqe'; got {self.kind!r}")
        if self.n_shots is not None and self.n_shots <= 0:
            raise ConfigError("method.n_shots must be a positive integer")

    @classmethod
    def parse(cls, d: dict) -> VQEMethodConfig:
        _unknown_keys(d, {"kind", "ansatz", "optimizer", "n_shots"}, "method (vqe)")
        return cls(
            kind="vqe",
            ansatz=AnsatzConfig.parse(_require(d, "ansatz", "method (vqe)")),
            optimizer=OptimizerConfig.parse(d.get("optimizer", {})),
            n_shots=_coerce_int(d.get("n_shots"), "method.n_shots", allow_none=True),
        )


def _parse_method(d: dict) -> HFMethodConfig | VQEMethodConfig:
    """Discriminator parse on ``method.kind``.

    QSCI/QSE blocks parse-fail with a clear "not yet wired in this build"
    message until the ``selected_ci`` module lands on main.
    """
    if not isinstance(d, dict):
        raise ConfigError("method: expected a mapping")
    kind = _require(d, "kind", "method")
    if kind == "hf":
        return HFMethodConfig.parse(d)
    if kind == "vqe":
        return VQEMethodConfig.parse(d)
    if kind in ("qsci", "qse"):
        raise ConfigError(
            f"method.kind={kind!r} is recognised but not available in this build; "
            "the qibochem.selected_ci module has not yet been merged to main. "
            "Use method.kind=hf or vqe."
        )
    raise ConfigError(f"method.kind must be one of {VALID_METHODS}; got {kind!r}")


@dataclass(frozen=True)
class OutputConfig:
    dir: str | None = None
    prefix: str | None = None
    save_circuit: bool = False
    save_parameters: bool = True
    log_level: str = "info"

    def __post_init__(self) -> None:
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigError(f"output.log_level must be one of {VALID_LOG_LEVELS}; got {self.log_level!r}")

    @classmethod
    def parse(cls, d: dict) -> OutputConfig:
        _unknown_keys(d, {"dir", "prefix", "save_circuit", "save_parameters", "log_level"}, "output")
        return cls(
            dir=d.get("dir"),
            prefix=d.get("prefix"),
            save_circuit=bool(d.get("save_circuit", False)),
            save_parameters=bool(d.get("save_parameters", True)),
            log_level=str(d.get("log_level", "info")),
        )


@dataclass(frozen=True)
class TopLevelConfig:
    version: int
    molecule: MoleculeConfig
    hamiltonian: HamiltonianConfig
    method: Any  # HFMethodConfig | VQEMethodConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.version != SUPPORTED_VERSION:
            raise ConfigError(f"version: only schema version {SUPPORTED_VERSION} is supported; got {self.version!r}")

    @classmethod
    def parse(cls, d: dict) -> TopLevelConfig:
        if not isinstance(d, dict):
            raise ConfigError("top-level: expected a mapping")
        _unknown_keys(d, {"version", "molecule", "hamiltonian", "method", "output"}, "top-level")
        return cls(
            version=_coerce_int(_require(d, "version", "top-level"), "version"),
            molecule=MoleculeConfig.parse(_require(d, "molecule", "top-level")),
            hamiltonian=HamiltonianConfig.parse(d.get("hamiltonian", {})),
            method=_parse_method(_require(d, "method", "top-level")),
            output=OutputConfig.parse(d.get("output", {})),
        )
