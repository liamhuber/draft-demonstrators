import itertools
import warnings
from collections.abc import Mapping, Sequence

import ase
import flowrep as fr
import numpy as np
from pyiron_workflow_atomistics import engine as engine_mod
from pyiron_workflow_atomistics.physics import free_energy as free_energy_mod

from . import shared, uris


###
# Generalise pyiron_workflow.physics.bulk.generate_structures
###

StrainSpec = tuple[float, float] | Mapping[str, Sequence[float]]

VOLUME_CONSERVING_MODES = frozenset({"c_over_a", "b_over_a"})


def _mode_scales(mode: str, eps: float) -> tuple[float, float, float]:
    """Per-lattice-vector scale factors for one strain mode at magnitude ``eps``."""
    f = 1.0 + eps
    if mode == "iso":
        return (f, f, f)
    if mode == "a":
        return (f, 1.0, 1.0)
    if mode == "b":
        return (1.0, f, 1.0)
    if mode == "c":
        return (1.0, 1.0, f)
    if mode == "c_over_a":  # c up by f, a and b down by sqrt(f): volume conserving
        g = f ** (-0.5)
        return (g, g, f)
    if mode == "b_over_a":  # b up by f, a down by f: volume conserving
        return (1.0 / f, f, 1.0)
    raise ValueError(
        f"Unknown strain mode {mode!r}; expected one of 'iso', 'a', 'b', 'c', "
        "'c_over_a', 'b_over_a'."
    )


def _is_plain_range(strain_range) -> bool:
    return (
            isinstance(strain_range, (tuple, list))
            and len(strain_range) == 2
            and all(isinstance(v, (int, float)) for v in strain_range)
    )


def _normalise_spec(strain_range, num_points) -> dict[str, tuple[float, float, int]]:
    """Polymorphic spec -> ``{mode: (lo, hi, n)}``."""
    if _is_plain_range(strain_range):
        lo, hi = strain_range
        return {"iso": (float(lo), float(hi), int(num_points))}
    spec = {}
    for mode, values in dict(strain_range).items():
        vals = tuple(values)
        if len(vals) not in (2, 3):
            raise ValueError(
                f"strain spec for {mode!r} must be (lo, hi) or (lo, hi, num); got {vals!r}"
            )
        n = int(vals[2]) if len(vals) == 3 else int(num_points)
        spec[str(mode)] = (float(vals[0]), float(vals[1]), n)
    return spec


def apply_strains(base: ase.Atoms, strains: Mapping[str, float]) -> ase.Atoms:
    """Apply a composition of named strain modes to ``base``'s cell."""
    scales = np.ones(3, dtype=float)
    for mode, eps in strains.items():
        scales *= np.asarray(_mode_scales(mode, float(eps)), dtype=float)
    strained = base.copy()
    strained.set_cell(np.asarray(strained.get_cell()) * scales[:, None], scale_atoms=True)
    return strained


@fr.atomic("structure_list")
def generate_structures(
    base_structure: ase.Atoms,
    strain_range: StrainSpec = (-0.03, 0.03),
    num_points: int = 7,
) -> list[ase.Atoms]:
    """Strained cells over an arbitrary product grid of named strain modes.

    ``strain_range=(-0.03, 0.03)`` reproduces the legacy ``axes=["iso"]`` path
    cell-for-cell. ``{"a": (-0.02, 0.02), "c": (-0.02, 0.02)}`` gives the 2-D grid.
    """
    spec = _normalise_spec(strain_range, num_points)
    modes = list(spec)
    grids = [np.linspace(*spec[mode]) for mode in modes]
    strained_structures = [
        apply_strains(base_structure, dict(zip(modes, combo)))
        for combo in itertools.product(*grids)
    ]

    # Guardrails
    if not _is_plain_range(strain_range):
        overlap = set(dict(strain_range)) & VOLUME_CONSERVING_MODES
        if overlap:
            raise ValueError(
                f"Volume-conserving modes {sorted(overlap)} belong in `shape_modes`, "
                "not `strain_range`: the outer grid must be the volume axis, and the "
                "inner relaxation already optimises shape at fixed volume."
            )
    volumes = np.array([s.get_volume() / len(s) for s in strained_structures])
    if np.any(np.diff(volumes) <= 0):
        raise ValueError(
            f"`strain_range` must produce a strictly increasing volume grid; got "
            f"{volumes.tolist()} Å³/atom. Multi-axis outer specs generally will not."
        )

    return strained_structures

###
# Interject between generating structures and evaluating E-V curves
###

@fr.atomic("shape_candidates", "shape_strains")
def expand_shape_candidates(
    structures: list[ase.Atoms],
    shape_modes: Sequence[str] = (),
    shape_window: float = 0.06,
    num_shape_points: int = 5,
):
    """Volume-conserving shape variants of each input cell.

    Returns ``(candidate_groups, strain_labels)``, both indexed
    ``[volume][candidate]``. The pure, engine-free half of the shape relaxation:
    it defines the round-0 sampling plan that node 3 then refines.

    With ``shape_modes=()`` each group is the single input cell and the whole
    downstream path collapses to the legacy behaviour.
    """
    modes = list(shape_modes)
    for mode in modes:
        if mode not in VOLUME_CONSERVING_MODES:
            raise ValueError(
                f"Shape mode {mode!r} is not volume conserving; legal shape modes "
                f"are {sorted(VOLUME_CONSERVING_MODES)}. Volume must stay the outer "
                "coordinate or the QHA volume grid is no longer well defined."
            )
    if not modes:
        return [[s] for s in structures], [[{}] for _ in structures]

    grid = np.linspace(-shape_window, shape_window, num_shape_points)
    groups, labels = [], []
    for structure in structures:
        combos = [
            dict(zip(modes, combo)) for combo in itertools.product(*([grid] * len(modes)))
        ]
        groups.append([apply_strains(structure, combo) for combo in combos])
        labels.append(combos)
    return groups, labels

###
# Instead of
# pyiron_workflow.physics.free_energy.quasiharmonic._static_energies_per_volume
###

def _quadratic_stationary_point(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Least-squares quadratic through (X, y); its minimum, or None if not convex."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    n_samples, k = X.shape
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    design = np.column_stack(
        [np.ones(n_samples)]
        + [X[:, i] for i in range(k)]
        + [X[:, i] * X[:, j] for i, j in pairs]
    )
    if n_samples < design.shape[1]:
        return None
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    gradient = coefficients[1 : 1 + k]
    hessian = np.zeros((k, k))
    for index, (i, j) in enumerate(pairs):
        c = coefficients[1 + k + index]
        if i == j:
            hessian[i, i] = 2.0 * c
        else:
            hessian[i, j] = hessian[j, i] = c
    try:
        if np.any(np.linalg.eigvalsh(hessian) <= 0.0):
            return None
        return np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError:
        return None


def _evaluate_shape(
    structure: ase.Atoms, engine: engine_mod.Engine, tag: str, counter
) -> float:
    sub_engine = engine.with_working_directory(
        f"{tag}/shape_{next(counter):04d}"
    )
    out = engine_mod.calculate(structure=structure, engine=sub_engine)
    if not out.converged:
        raise RuntimeError(
            f"Static-energy calc failed at {tag} "
            f"(volume {structure.get_volume():.3f} Å³)."
        )
    return float(out.final_energy) / len(structure)


@fr.atomic("energies_per_atom", "volumes_per_atom", "relaxed_structures")
def static_shape_relaxed_energies(
        structures: list[ase.Atoms],
        shape_candidates: list[list[ase.Atoms]],
        shape_strains: list[list[dict]],
        engine: engine_mod.Engine,
        shape_modes: Sequence[str] = (),
        shape_window: float = 0.06,
        num_shape_points: int = 5,
        refine_rounds: int = 2,
):
    """Minimise the static energy over shape at fixed volume, per volume.

    Returns ``(energies_per_atom, volumes_per_atom, relaxed_structures)``. The
    first two feed ``_fit_qha`` on the per-atom basis it needs; the third feeds
    ``_harmonic_grid_over_volumes`` so phonons run on the *relaxed* path.
    """
    modes = list(shape_modes)
    energies, volumes, relaxed = [], [], []

    for i, base in enumerate(structures):
        counter = itertools.count()

        candidates = list(shape_candidates[i])   # round 0, pre-built by node 2
        labels = list(shape_strains[i])
        best_energy, best_strain = np.inf, {}

        if not modes:
            best_energy, best_strain = _evaluate_shape(candidates[0], engine, f"vol_E_{i:03d}", counter), {}
        else:
            centre = np.zeros(len(modes))
            window = float(shape_window)
            for round_index in range(max(1, refine_rounds)):
                if round_index > 0:
                    grids = [
                        np.linspace(c - window, c + window, num_shape_points)
                        for c in centre
                    ]
                    labels = [
                        dict(zip(modes, combo)) for combo in itertools.product(*grids)
                    ]
                    candidates = [apply_strains(base, combo) for combo in labels]

                X = np.array([[label[m] for m in modes] for label in labels])
                y = np.array([_evaluate_shape(c, engine, f"vol_E_{i:03d}", counter) for c in candidates])

                k = int(np.argmin(y))
                if y[k] < best_energy:
                    best_energy, best_strain = float(y[k]), labels[k]

                star = _quadratic_stationary_point(X, y)
                if star is not None and np.all(np.abs(star - centre) <= 2.0 * window):
                    trial = dict(zip(modes, star))
                    trial_energy = _evaluate_shape(
                        apply_strains(base, trial), engine, f"vol_E_{i:03d}", counter
                    )
                    if trial_energy < best_energy:
                        best_energy, best_strain = trial_energy, trial
                    centre = np.asarray(star, dtype=float)
                else:
                    centre = X[k]
                window *= 0.5

        best_structure = apply_strains(base, best_strain) if modes else base
        energies.append(best_energy)
        volumes.append(float(best_structure.get_volume()) / len(best_structure))
        relaxed.append(best_structure)

    return np.asarray(energies), np.asarray(volumes), relaxed


###
# Putting it together: a more flexible quasiharmonic workflow
###

@fr.workflow
def quasiharmonic_free_energy(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    fc2_supercell_matrix,
    temperatures=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
    pressure: float = 0.0,
    strain_range: StrainSpec = (-0.03, 0.03),
    num_points: int = 7,
    shape_modes: Sequence[str] = (),
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_refine_rounds: int = 2,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    eos_type: str = "vinet",
    working_directory: str = ".",
    subdir: str = "quasiharmonic_free_energy",
    keep_handles: bool = False,
):
    """
    Re-written in direct anaology to
    ``pyiron_workflow.physics.free_energy.quasiharmonic.quasiharmonic_free_energy``,
    except with a more generic attack at generating strained structures and applying
    the quasistatic approach to minimize their possible shapes with respect to a fixed
    volume. This is necessary to extend the existing workflow to non-cubic materials.

    Gibbs free energy G(T,P), V*(T,P), B(T,P), α(T,P) via phonopy.qha.QHA.

    Pressure is in **GPa** (phonopy.qha native). At ``pressure=0.0`` the
    ``gibbs_free_energy_array`` field is the Helmholtz free energy F(T).

    The returned ``FreeEnergyOutput`` populates ``free_energy_array`` directly
    from ``gibbs_free_energy_array`` for compatibility with the calphy ``ts``
    mode shape — at finite pressure this is Gibbs, at zero pressure it is
    Helmholtz.
    """
    simfolder, sub_engine = free_energy_mod.harmonic._resolve_simfolder(
        engine=engine,
        working_directory=working_directory,
        subdir=subdir,
    )
    strained_structures = generate_structures(
        base_structure=structure,
        strain_range=strain_range,
        num_points=num_points,
    )
    shape_candidates, shape_strains = expand_shape_candidates(
        structures=strained_structures,
        shape_modes=shape_modes,
        shape_window=shape_window,
        num_shape_points=num_shape_points,
    )
    energies_per_atom, volumes_per_atom, relaxed_structures = static_shape_relaxed_energies(
        structures=strained_structures,
        shape_candidates=shape_candidates,
        shape_strains=shape_strains,
        engine=sub_engine,
        shape_modes=shape_modes,
        shape_window=shape_window,
        num_shape_points=num_shape_points,
        refine_rounds=shape_refine_rounds,
    )
    F_TV, S_TV, Cv_TV = free_energy_mod.quasiharmonic._harmonic_grid_over_volumes(
        strained_structures=relaxed_structures,
        engine=sub_engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperatures=temperatures,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        working_directory=simfolder,
    )
    qha = free_energy_mod.quasiharmonic._fit_qha(
        energies=energies_per_atom,
        volumes=volumes_per_atom,
        free_energy_per_T_V=F_TV,
        entropy_per_T_V=S_TV,
        cv_per_T_V=Cv_TV,
        temperatures=temperatures,
        pressure_GPa=pressure,
        eos_type=eos_type,
    )
    free_energy_output = free_energy_mod.quasiharmonic._pack_qha_output(
        structure=structure,
        qha_results=qha,
        volumes=volumes_per_atom,
        free_energy_per_T_V=F_TV,
        entropy_per_T_V=S_TV,
        cv_per_T_V=Cv_TV,
        temperatures=temperatures,
        pressure_GPa=pressure,
        simfolder=simfolder,
        keep_handles=keep_handles,
    )
    return free_energy_output


###
# Pre-relaxing cells at a given pressure to get a good starting point for scaling
###

GPA_PER_EV_PER_ANG3 = 160.21766208
EV_TO_KJ_MOL = 96.48533212331002


def _relax_shape_at_fixed_volume(
    base: ase.Atoms,
    engine: engine_mod.Engine,
    tag: str,
    shape_modes: Sequence[str],
    shape_window: float,
    num_shape_points: int,
    refine_rounds: int,
    initial_candidates: list[ase.Atoms] | None = None,
    initial_strains: list[dict] | None = None,
) -> tuple[ase.Atoms, float]:
    """Minimise over volume-conserving shape strains of ``base``.

    ``evaluate`` maps an Atoms to an energy per atom. Each round re-centres on
    the running best strain and halves the window. Returns
    ``(relaxed_structure, energy_per_atom)``; with no shape modes this is one
    evaluation of ``base``.
    """
    modes = list(shape_modes)
    counter = itertools.count()
    if not modes:
        return base, _evaluate_shape(base, engine, tag, counter)

    centre = np.zeros(len(modes))
    window = float(shape_window)
    best_energy, best_strain = np.inf, dict.fromkeys(modes, 0.0)
    candidates, labels = initial_candidates, initial_strains

    for round_index in range(max(1, refine_rounds)):
        if round_index > 0 or candidates is None:
            grids = [
                np.linspace(c - window, c + window, num_shape_points) for c in centre
            ]
            labels = [dict(zip(modes, combo)) for combo in itertools.product(*grids)]
            candidates = [apply_strains(base, combo) for combo in labels]

        X = np.array([[label[m] for m in modes] for label in labels])
        y = np.array([_evaluate_shape(c, engine, tag, counter) for c in candidates])

        k = int(np.argmin(y))
        if y[k] < best_energy:
            best_energy, best_strain = float(y[k]), labels[k]

        star = _quadratic_stationary_point(X, y)
        if star is not None and np.all(np.abs(star - centre) <= 2.0 * window):
            trial = dict(zip(modes, star))
            trial_energy = _evaluate_shape(apply_strains(base, trial), engine, tag, counter)
            if trial_energy < best_energy:
                best_energy, best_strain = trial_energy, trial
            centre = np.asarray(star, dtype=float)
        else:
            centre = X[k]
        window *= 0.5
        candidates = None

    return apply_strains(base, best_strain), best_energy


def _interior_minimum(x, y, degree: int | None = None) -> tuple[float, float]:
    """Polynomial-fit ``y(x)``; return ``(x_min, y_min)`` for the interior minimum.

    Raises if the minimum sits on a grid edge — the scan window does not bracket
    it and any answer would be extrapolation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if degree is None:
        degree = int(min(4, max(2, x.size - 2)))

    coefficients = np.polyfit(x, y, degree)
    derivative = np.polyder(coefficients)
    curvature = np.polyder(derivative)
    lo, hi = float(x.min()), float(x.max())

    interior = [
        float(r.real)
        for r in np.roots(derivative)
        if abs(r.imag) < 1e-9
        and lo < r.real < hi
        and np.polyval(curvature, r.real) > 0.0
    ]
    if not interior:
        argmin = int(np.argmin(y))
        raise RuntimeError(
            "No interior minimum in the scan window: sampled minimum sits at "
            f"x={x[argmin]:.4f} (window {lo:.4f}..{hi:.4f}). Widen `scan_range`, or "
            f"shift it toward {'larger' if argmin == x.size - 1 else 'smaller'} values."
        )
    best = min(interior, key=lambda r: float(np.polyval(coefficients, r)))
    best_value = float(np.polyval(coefficients, best))

    if min(best - lo, hi - best) < 0.05 * (hi - lo):
        warnings.warn(
            f"Optimum x={best:.4f} lies within 5% of the scan edge ({lo:.4f}..{hi:.4f}); "
            "the fit is near-extrapolating. Widen `scan_range`.",
            stacklevel=2,
        )
    return best, best_value


@fr.atomic("optimised_structure", "volume_per_atom", "gibbs_per_atom")
def optimise_cell_at_pressure(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    pressure: float = 0.0,
    temperature: float | None = None,
    fc2_supercell_matrix=None,
    scan_range: tuple[float, float] = (-0.06, 0.08),
    num_points: int = 7,
    shape_modes: Sequence[str] = (),
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_refine_rounds: int = 2,
    displacement_distance: float = 0.03,
    is_plusminus="auto",
    working_directory: str = ".",
):
    """Cell minimising G(T,P) = E(V) + F_vib(T,V) + PV, shape relaxed at each V.

    Pressure in **GPa**, sign convention matching ``quasiharmonic_free_energy``
    (negative = tension).

    ``temperature=None`` minimises the static enthalpy E + PV — no phonons, cheap,
    and usually enough to centre a grid, since thermal expansion is a few percent
    while a strain grid spans tens. Pass a temperature (the midpoint of your
    production range) to include F_vib; that costs one harmonic calculation per
    scan point and requires ``fc2_supercell_matrix``.
    """
    if temperature is not None and fc2_supercell_matrix is None:
        raise ValueError(
            "temperature was given but fc2_supercell_matrix is None: the vibrational "
            "term needs a supercell to compute force constants in."
        )
    pressure_ev_per_ang3 = float(pressure) / GPA_PER_EV_PER_ANG3

    scan_structures = generate_structures(
        base_structure=structure, strain_range=scan_range, num_points=num_points
    )
    shape_candidates, shape_strains = expand_shape_candidates(
        structures=scan_structures,
        shape_modes=shape_modes,
        shape_window=shape_window,
        num_shape_points=num_shape_points,
    )

    energies, volumes, relaxed = [], [], []
    for i, scan_structure in enumerate(scan_structures):
        best_structure, best_energy = _relax_shape_at_fixed_volume(
            base=scan_structure,
            engine=engine,
            tag=f"opt_cell/vol_{i:03d}",
            shape_modes=shape_modes,
            shape_window=shape_window,
            num_shape_points=num_shape_points,
            refine_rounds=shape_refine_rounds,
            initial_candidates=shape_candidates[i],
            initial_strains=shape_strains[i],
        )
        energies.append(best_energy)
        volumes.append(best_structure.get_volume() / len(best_structure))
        relaxed.append(best_structure)

    energies, volumes = np.asarray(energies), np.asarray(volumes)
    free_energy = energies.copy()

    if temperature is not None:
        F_TV, _, _ = free_energy_mod.quasiharmonic._harmonic_grid_over_volumes(
            strained_structures=relaxed,
            engine=engine,
            fc2_supercell_matrix=fc2_supercell_matrix,
            temperatures=(float(temperature),),
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            working_directory=working_directory,
        )
        free_energy = free_energy + F_TV[0, :] / EV_TO_KJ_MOL

    gibbs = free_energy + pressure_ev_per_ang3 * volumes
    target_volume, gibbs_per_atom = _interior_minimum(volumes, gibbs)

    # Rebuild at the optimal volume and re-relax the shape there, rather than
    # interpolating a shape strain between grid points.
    scale = (target_volume / (structure.get_volume() / len(structure))) ** (1.0 / 3.0)
    seed = structure.copy()
    seed.set_cell(np.asarray(structure.get_cell()) * scale, scale_atoms=True)
    optimised_structure, _ = _relax_shape_at_fixed_volume(
        base=seed,
        engine=engine,
        tag="opt_cell/final",
        shape_modes=shape_modes,
        shape_window=shape_window,
        num_shape_points=num_shape_points,
        refine_rounds=shape_refine_rounds,
    )
    volume_per_atom = optimised_structure.get_volume() / len(optimised_structure)
    return optimised_structure, float(volume_per_atom), float(gibbs_per_atom)


### Legacy

# @fr.workflow
# def optimise_bulk(
#     species,
#     crystalstructure,
#     engine,
#     a0,
#     cubic=False,
#     orthorhombic=False,
# ):
#     initial_structure = bulk.get_bulk(
#         species, crystalstructure, a=a0, cubic=cubic, orthorhombic=orthorhombic
#     )
#     equil_struct, _, _, _, _, _, _, _ = bulk.optimise_cubic_lattice_parameter(
#         initial_structure,
#         species,
#         crystalstructure,
#         engine,
#         cubic=cubic,
#         orthorhombic=orthorhombic,
#     )
#     return equil_struct
#
#
# @fr.workflow
# def harmonic_free_energy(
#     species,
#     crystalstructure,
#     temperatures,
#     engine,
#     fc2_sc,
#     a0=4,
#     cubic=False,
#     orthorhombic=False,
# ):
#     equil_struct = optimise_bulk(species, crystalstructure, engine, a0, cubic, orthorhombic)
#     free_energy_output = free_energy_mod.harmonic_free_energy(
#         equil_struct,
#         engine,
#         fc2_sc,
#         temperatures=temperatures,
#     )
#     return free_energy_output, equil_struct
#
#
# @fr.dataclass(frozen=True)
# class FreeEnergResult:
#     structure: ase.Atoms
#     harmonic: np.ndarray
#     quasiharmonic: np.ndarray
#
# @fr.workflow
# def fcc_hcp_stability(
#     species,
#     temperatures,
#     engine,
#     fc2_sc,
#
#     fcc_a0=4, hcp_a0=2
# ):
#     # for crystal_structure, a0 in zip()
#     # optimised_fcc = initial_structure = bulk.get_bulk(
#     #     species, "fcc", a=a0, cubic=cubic, orthorhombic=orthorhombic
#     # )
#     # fcc_harmonic_free_energy, fcc_structure = harmonic_free_energy(
#     #     species, "fcc", temperatures, engine, fc2_sc, a0=fcc_a0, cubic=True
#     # )
#     # hcp_harmonic_free_energy, hcp_structure = harmonic_free_energy(
#     #     species, "hcp", temperatures, engine, fc2_sc, a0=hcp_a0, orthorhombic=True
#     # )
#     # fcc_quasiharmonic_free_energy, fcc_structure = quasiharmonic_free_energy()
#     # return fcc_free_energy, hcp_free_energy, fcc_structure, hcp_structure
#     return None
#
#
# @fr.workflow
# def quasiharmonic_free_energy(
#     species,
#     crystalstructure,
#     temperatures,
#     engine,
#     fc2_sc,
#     a0=4,
#     cubic=False,
#     orthorhombic=False,
# ):
#     equil_struct = optimise_bulk(species, crystalstructure, engine, a0, cubic, orthorhombic)
#     free_energy_output = free_energy_mod.quasiharmonic_free_energy(
#         equil_struct,
#         engine,
#         fc2_sc,
#         temperatures=temperatures,
#     )
#     return free_energy_output, equil_struct

