"""Self-consistent quasiharmonic Gibbs free energy.

Differs from ``phase_nodes.quasiharmonic_free_energy`` in three ways:

1. Volume-conserving shape strains are optimised against the *free* energy
   ``F(V', T)`` at each temperature, not against the static energy ``E(V', 0)``.
2. The volume minimisation is done here on a per-temperature polynomial fit
   rather than by ``phonopy.qha.QHA``. That is forced: optimising shape per
   temperature makes the static energy ``E*(V, T)`` two-dimensional, while
   ``QHA`` accepts only a one-dimensional ``electronic_energies``.
3. The volume grid is re-centred *and narrowed*, and the calculation repeated,
   until ``G(T, P)`` stops moving — so the sampling window sits on top of the
   answer at a width it has actually been tested at, and the residual fit error
   is the same for every phase being compared. Converging on position alone is
   not enough: a centred window reproduces itself, so the drift goes to zero
   with the fit bias intact. See :func:`recentre`, and read
   ``GibbsResult.fit_residual`` for what the fit error actually is.

Everything here is in eV/atom and Å³/atom. See
``.claude/superpowers/2026-08-20-self-consistent-gibbs-design.md``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import warnings
from collections.abc import Sequence
from typing import Literal

import ase
import flowrep as fr
import numpy as np
from pyiron_workflow_atomistics import engine as engine_mod
from pyiron_workflow_atomistics.physics import free_energy as free_energy_mod

from . import phase_nodes

_MAX_TAG_DECIMALS = 24

ShapeObjective = Literal["free_energy", "static"]
"""
Used to determine whether shape optimization at a fixed volume is perfomed with 
vibrational free energy (temperature-dependent) or static energy evaluations 
(temperature independent).
"""


class BracketError(RuntimeError):
    """The volume grid did not bracket an interior minimum of G(V)."""


MAX_BRACKET_STRAIN = 0.10
"""Cap, in strain units, on how far `gibbs_at_pressure` and `recentre` will widen a window."""

MIN_STRAIN_HALFWIDTH = 0.008
"""Floor, in strain units, on how far `recentre` will narrow a window.

Narrowing cuts the fit residual, but a window that keeps shrinking eventually
puts every grid point inside the numerical noise of the energies and leaves the
volume fit ill-conditioned. 0.008 keeps a seven-point grid spanning roughly 5%
in volume, which is still comfortably wider than thermal expansion.
"""


@dataclasses.dataclass(frozen=True)
class ShapeScan:
    """Static and vibrational energies over shape strain at one fixed volume.

    ``vib_free_energies`` is always ``(n_shape, n_temperature)``. Under
    ``objective="static"`` the single computed row is broadcast across the shape
    axis, which lets ``minimise_shape`` treat both objectives identically.
    """

    strains: np.ndarray  # (n_shape,)
    static_energies: np.ndarray  # (n_shape,) eV/atom
    vib_free_energies: np.ndarray  # (n_shape, n_temperature) eV/atom
    structures: list[ase.Atoms]
    volume_per_atom: float
    objective: ShapeObjective


@dataclasses.dataclass(frozen=True)
class ShapeOptimum:
    """Shape-minimised free energy at one fixed volume, per temperature."""

    optimal_strains: np.ndarray  # (n_temperature,)
    free_energies: np.ndarray  # (n_temperature,) eV/atom, this is F*(V, T)
    volume_per_atom: float
    fell_back: np.ndarray  # (n_temperature,) bool


@dataclasses.dataclass(frozen=True)
class GibbsIteration:
    """One pass over a volume grid at fixed pressure."""

    volumes: np.ndarray  # (n_volume,) Å³/atom
    free_energies: np.ndarray  # (n_volume, n_temperature) eV/atom
    shape_strains: np.ndarray  # (n_volume, n_temperature)
    gibbs: np.ndarray  # (n_temperature,) eV/atom
    optimal_volumes: np.ndarray  # (n_temperature,) Å³/atom
    optimal_shape_strains: np.ndarray  # (n_temperature,)
    strain_range: tuple[float, float]
    fit_residuals: np.ndarray  # (n_temperature,) eV/atom
    shape_fell_back: np.ndarray  # (n_volume, n_temperature) bool


@dataclasses.dataclass(frozen=True)
class GibbsResult:
    """Converged G(T) at one pressure, plus the iteration history.

    ``gibbs`` is the accurate quantity: ``G`` is stationary at the minimum, so a
    small error in where the minimum sits barely moves its value.
    ``optimal_volumes`` is not — it is a *fitted* quantity, read off the
    polynomial rather than sampled, and it inherits the fit error directly
    instead of quadratically. Its discrepancy from the true minimiser is bounded
    by ``fit_residual``; treat it as diagnostic, and do not read thermal
    expansion off it without checking that ``fit_residual`` is small compared to
    the ``ΔG`` being resolved.

    ``fit_residual`` is the rms misfit of the volume polynomial on the final
    iteration, maximised over temperature. ``shape_fell_back_count`` counts the
    (volume, temperature) pairs on the final iteration where shape minimisation
    could not use its parabola and fell back to the sampled argmin;
    ``fc2_supercell_matrix`` and ``displacement_distance`` record the phonon
    settings actually used, so a result can be checked after the fact against
    the phase it is compared with.
    """

    temperatures: Sequence[float]  # (n_temperature,)
    pressure: float  # GPa
    gibbs: np.ndarray  # (n_temperature,) eV/atom
    optimal_volumes: np.ndarray  # (n_temperature,) Å³/atom
    optimal_shape_strains: np.ndarray  # (n_temperature,)
    iterations: list[GibbsIteration]
    converged: bool
    gibbs_drift: float  # eV/atom
    fit_residual: float  # eV/atom
    shape_fell_back_count: int
    fc2_supercell_matrix: np.ndarray  # (3, 3)
    displacement_distance: float  # Å


@fr.dataclass(frozen=True)
class GibbsSweep:
    """G(T, P) for one phase over a pressure sweep."""

    temperatures: Sequence[float]  # (n_temperature,)
    pressures: Sequence[float]  # (n_pressure,)
    gibbs: np.ndarray  # (n_temperature, n_pressure) eV/atom
    optimal_volumes: np.ndarray  # (n_temperature, n_pressure) Å³/atom
    results: list[GibbsResult]


def supercell_repetitions(
    structure: ase.Atoms, target_length: float = 14.0
) -> tuple[int, int, int]:
    """Force-constant supercell repetitions giving a box of at least ``target_length``.

    ``n_i = ceil(target_length / |a_i|)``, so every supercell axis is at least
    ``target_length``, and within one lattice parameter of it.

    Use the **same** ``target_length`` for every phase in a comparison. Force
    constants truncated at the supercell boundary carry an error that cancels in
    ``ΔG`` only to the extent that the phases share a real-space cutoff.
    ``ceil`` gives *comparable*, not identical, extent: at ``target_length =
    14.0`` fcc comes out (4, 4, 4) → 17.43 Å, hcp (5, 5, 3) → 15.38/15.38/15.07
    Å, and bcc (5, 5, 5) → 16.78 Å. The residual from that mismatch is
    ~0.03–0.05 meV/atom, inside the error budget here, but it is a residual and
    not an exact cancellation.

    Fixed repetitions do not even get that far: at 57 GPa a ``(2, 2, 2)`` fcc
    cell spans 8.6 Å while a ``(4, 2, 2)`` hcp cell spans 12.1 Å, and the
    resulting ``ΔF_vib(fcc - hcp)`` differs by 0.71 meV/atom — about 0.9 GPa of
    transition pressure — from its converged value, with the opposite sign.

    The default of 14 Å is converged for Pb: raising it to 18 Å moves
    ``ΔF_vib(fcc - hcp)`` by only 0.03 meV/atom.
    """
    lengths = np.linalg.norm(np.asarray(structure.get_cell()), axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(
            f"structure must have a non-degenerate cell; got row norms {lengths.tolist()}"
        )
    repetitions = tuple(
        int(max(1, math.ceil(float(target_length) / float(length))))
        for length in lengths
    )
    return repetitions  # type: ignore[return-value]


def _static_energy_per_atom(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    working_directory: str,
    tag: str,
) -> float:
    """Total energy of one cell, divided by its atom count."""
    sub_engine = engine.with_working_directory(os.path.join(working_directory, tag))
    output = engine_mod.calculate(structure=structure, engine=sub_engine)
    if not output.converged:
        raise RuntimeError(
            f"Static-energy calculation failed at {tag} "
            f"(volume {structure.get_volume():.3f} Å³)."
        )
    return float(output.final_energy) / len(structure)


def _vibrational_free_energy(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    *,
    temperatures: Sequence[float],
    fc2_supercell_matrix,
    displacement_distance: float,
    is_plusminus,
    working_directory: str,
    tag: str,
) -> np.ndarray:
    """Harmonic ``F_vib(T)`` in eV/atom for one cell.

    ``harmonic_free_energy`` already returns eV per primitive-cell atom, so no
    unit conversion happens here. The kJ/mol conversion in
    ``quasiharmonic._harmonic_grid_over_volumes`` exists only to feed
    ``phonopy.qha``, which this module does not use.
    """
    cell_directory = os.path.join(working_directory, tag)
    sub_engine = engine.with_working_directory(cell_directory)
    output = free_energy_mod.harmonic.harmonic_free_energy(
        structure=structure,
        engine=sub_engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperatures=temperatures,
        displacement_distance=displacement_distance,
        is_plusminus=is_plusminus,
        working_directory=".",
        subdir="harmonic",
    )
    return np.asarray(output.free_energy_array, dtype=float)


def _parabola_vertex(
    x: np.ndarray, y: np.ndarray, lo: float, hi: float
) -> tuple[float, float, bool]:
    """Least-squares parabola through ``(x, y)``; its vertex if it falls in ``[lo, hi]``.

    Returns ``(x_min, y_min, fell_back)``. ``fell_back`` is True when the fit was
    unusable or unavailable — a two-point scan, a downward-opening parabola, or a
    vertex outside the sampled window — in which case the sampled ``argmin`` is
    returned instead. The exception is a single-point scan (``x.size == 1``):
    that is the exact answer for a zero-dimensional shape space, so
    ``fell_back`` is False.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    argmin = int(np.argmin(y))
    if x.size == 1:
        return float(x[argmin]), float(y[argmin]), False
    if x.size == 2:
        return float(x[argmin]), float(y[argmin]), True
    coefficients = np.polyfit(x, y, 2)
    curvature = float(coefficients[0])
    if curvature <= 0.0:
        return float(x[argmin]), float(y[argmin]), True
    vertex = float(-coefficients[1] / (2.0 * curvature))
    if not lo <= vertex <= hi:
        return float(x[argmin]), float(y[argmin]), True
    return vertex, float(np.polyval(coefficients, vertex)), False


def shape_scan_at_volume(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    *,
    temperatures: Sequence[float],
    fc2_supercell_matrix,
    shape_mode: phase_nodes.StrainMode | None = None,
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_objective: ShapeObjective = "free_energy",
    displacement_distance: float = 0.01,
    is_plusminus="auto",
    working_directory: str = ".",
    tag: str = "shape_scan",
) -> ShapeScan:
    """Static and vibrational energies over volume-conserving shape strain.

    The volume of every candidate equals the volume of ``structure``, so the
    outer volume coordinate stays well defined.

    ``shape_objective="free_energy"`` runs one phonon calculation per shape
    point, giving a temperature-dependent shape optimum.
    ``shape_objective="static"`` runs the phonon calculation only at the
    static-energy optimum and broadcasts it, reproducing the quasi-static
    approximation used by ``phase_nodes`` so the two can be compared.

    ``shape_mode=None`` gives a single point at zero strain, which is exact for
    cubic phases and costs nothing extra.
    """
    if shape_objective not in ("free_energy", "static"):
        raise ValueError(
            f"shape_objective must be 'free_energy' or 'static'; got {shape_objective!r}"
        )
    if shape_mode is not None:
        shape_mode = phase_nodes.StrainMode(shape_mode)
        if not shape_mode.is_volume_conserving:
            raise ValueError(
                f"shape_mode {shape_mode!r} is not volume conserving; legal modes are "
                f"{phase_nodes.StrainMode.VOLUME_CONSERVING_MODES}. Volume must remain "
                "the outer coordinate or the volume grid is no longer well defined."
            )

    if shape_mode is None:
        strains = np.zeros(1)
        structures = [structure.copy()]
    else:
        strains = np.linspace(-shape_window, shape_window, num_shape_points)
        structures = [
            phase_nodes.apply_strains(structure, {shape_mode: float(strain)})
            for strain in strains
        ]

    static_energies = np.array(
        [
            _static_energy_per_atom(
                candidate, engine, working_directory, f"{tag}/shape_{index:03d}"
            )
            for index, candidate in enumerate(structures)
        ]
    )

    n_temperature = len(np.asarray(temperatures, dtype=float))
    if shape_objective == "free_energy":
        vib_free_energies = np.vstack(
            [
                _vibrational_free_energy(
                    candidate,
                    engine,
                    temperatures=temperatures,
                    fc2_supercell_matrix=fc2_supercell_matrix,
                    displacement_distance=displacement_distance,
                    is_plusminus=is_plusminus,
                    working_directory=working_directory,
                    tag=f"{tag}/shape_{index:03d}",
                )
                for index, candidate in enumerate(structures)
            ]
        )
    else:
        best_strain, _, _ = _parabola_vertex(
            strains, static_energies, float(strains.min()), float(strains.max())
        )
        best_structure = (
            structure.copy()
            if shape_mode is None
            else phase_nodes.apply_strains(structure, {shape_mode: best_strain})
        )
        row = _vibrational_free_energy(
            best_structure,
            engine,
            temperatures=temperatures,
            fc2_supercell_matrix=fc2_supercell_matrix,
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            working_directory=working_directory,
            tag=f"{tag}/shape_static",
        )
        vib_free_energies = np.tile(row, (strains.size, 1))

    if vib_free_energies.shape != (strains.size, n_temperature):
        raise RuntimeError(
            f"vibrational grid shape {vib_free_energies.shape} does not match "
            f"({strains.size}, {n_temperature}); phonon output length drifted."
        )

    return ShapeScan(
        strains=strains,
        static_energies=static_energies,
        vib_free_energies=vib_free_energies,
        structures=structures,
        volume_per_atom=float(structure.get_volume()) / len(structure),
        objective=shape_objective,
    )


def minimise_shape(scan: ShapeScan) -> ShapeOptimum:
    """Minimise ``E(eps) + F_vib(eps, T)`` over shape strain, per temperature.

    ``F*`` is read off the parabola rather than recalculated: it is smooth in
    ``eps``, which keeps the downstream volume fit well behaved, and it is free.

    Under ``objective="static"`` the vibrational term is constant along the
    strain axis, so minimising ``E + F_vib`` recovers the static optimum exactly
    and no special case is needed.
    """
    if scan.vib_free_energies.shape[0] != scan.strains.size:
        raise ValueError(
            f"vib_free_energies has {scan.vib_free_energies.shape[0]} rows but there "
            f"are {scan.strains.size} shape strains; the array must be indexed "
            "[shape, temperature]."
        )
    total = scan.static_energies[:, None] + scan.vib_free_energies
    n_temperature = total.shape[1]
    lo, hi = float(scan.strains.min()), float(scan.strains.max())

    optimal_strains = np.empty(n_temperature)
    free_energies = np.empty(n_temperature)
    fell_back = np.zeros(n_temperature, dtype=bool)
    for index in range(n_temperature):
        strain, energy, fallback = _parabola_vertex(
            scan.strains, total[:, index], lo, hi
        )
        optimal_strains[index] = strain
        free_energies[index] = energy
        fell_back[index] = fallback

    if bool(fell_back.any()):
        warnings.warn(
            f"Shape minimisation fell back to the sampled minimum at "
            f"{int(fell_back.sum())} of {n_temperature} temperatures "
            f"(volume {scan.volume_per_atom:.4f} Å³/atom, strain window "
            f"[{lo:.4f}, {hi:.4f}]). Widen `shape_window` or add shape points.",
            stacklevel=2,
        )

    return ShapeOptimum(
        optimal_strains=optimal_strains,
        free_energies=free_energies,
        volume_per_atom=scan.volume_per_atom,
        fell_back=fell_back,
    )


def gibbs_iteration(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    *,
    pressure: float,
    temperatures: Sequence[float],
    fc2_supercell_matrix,
    strain_range: tuple[float, float] = (-0.04, 0.04),
    num_points: int = 7,
    fit_degree: int = 3,
    shape_mode: phase_nodes.StrainMode | None = None,
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_objective: ShapeObjective = "free_energy",
    displacement_distance: float = 0.01,
    is_plusminus="auto",
    working_directory: str = ".",
    tag: str = "iteration",
) -> GibbsIteration:
    """One pass over a volume grid: shape-minimise at each volume, then fit in V.

    ``G(T) = min over V of [F*(V, T) + P*V]``, where ``F*`` is the
    shape-minimised free energy from :func:`minimise_shape`. The minimisation is
    a per-temperature polynomial fit of degree ``fit_degree`` over the volume
    grid, taking the interior minimum.

    ``phonopy.qha.QHA`` is deliberately not used. Optimising shape per
    temperature makes the static energy ``E*(V, T)`` two-dimensional, and
    ``QHA`` accepts only a one-dimensional ``electronic_energies``.

    ``fit_degree`` must be at most ``num_points - 2``. At ``num_points - 1`` the
    polynomial interpolates the grid exactly and beyond that the fit is
    underdetermined; ``np.polyfit`` only whispers a ``RankWarning``, and
    ``_interior_minimum`` then happily reports the "interior minimum" of a
    meaningless curve. That is checked up front rather than discovered later.

    Pressure is in **GPa**; positive compresses.
    """
    if int(fit_degree) >= int(num_points) - 1:
        raise ValueError(
            f"fit_degree must be at most num_points - 2; got fit_degree="
            f"{fit_degree} with num_points={num_points}. A degree-{fit_degree} "
            f"polynomial through {num_points} points is interpolating or "
            "underdetermined, and its 'interior minimum' would be an artefact of "
            "the fit rather than a property of the energy surface."
        )
    scan_structures = phase_nodes.generate_structures(
        base_structure=structure,
        strain_range=tuple(strain_range),
        num_points=num_points,
    )
    n_volume = len(scan_structures)
    n_temperature = len(temperatures)

    volumes = np.empty(n_volume)
    free_energies = np.empty((n_volume, n_temperature))
    shape_strains = np.empty((n_volume, n_temperature))
    shape_fell_back = np.zeros((n_volume, n_temperature), dtype=bool)
    for index, scan_structure in enumerate(scan_structures):
        scan = shape_scan_at_volume(
            scan_structure,
            engine,
            temperatures=temperatures,
            fc2_supercell_matrix=fc2_supercell_matrix,
            shape_mode=shape_mode,
            shape_window=shape_window,
            num_shape_points=num_shape_points,
            shape_objective=shape_objective,
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            working_directory=working_directory,
            tag=f"{tag}/vol_{index:03d}",
        )
        optimum = minimise_shape(scan)
        volumes[index] = optimum.volume_per_atom
        free_energies[index] = optimum.free_energies
        shape_strains[index] = optimum.optimal_strains
        shape_fell_back[index] = optimum.fell_back

    pressure_ev_per_ang3 = float(pressure) / phase_nodes.GPA_PER_EV_PER_ANG3
    shape_degree = int(min(2, max(1, n_volume - 2)))

    gibbs = np.empty(n_temperature)
    optimal_volumes = np.empty(n_temperature)
    optimal_shape_strains = np.empty(n_temperature)
    fit_residuals = np.empty(n_temperature)
    for index in range(n_temperature):
        gibbs_grid = free_energies[:, index] + pressure_ev_per_ang3 * volumes
        try:
            volume, energy = phase_nodes._interior_minimum(
                volumes, gibbs_grid, degree=fit_degree
            )
        except RuntimeError as error:
            raise BracketError(str(error)) from error
        optimal_volumes[index] = volume
        gibbs[index] = energy
        # Same polynomial `_interior_minimum` just used, refit so its misfit is
        # reported rather than assumed: this is the dominant error in `gibbs`
        # and it does not cancel between phases.
        fit_coefficients = np.polyfit(volumes, gibbs_grid, fit_degree)
        fit_residuals[index] = float(
            np.sqrt(np.mean((np.polyval(fit_coefficients, volumes) - gibbs_grid) ** 2))
        )
        optimal_shape_strains[index] = float(
            np.polyval(
                np.polyfit(volumes, shape_strains[:, index], shape_degree), volume
            )
        )

    return GibbsIteration(
        volumes=volumes,
        free_energies=free_energies,
        shape_strains=shape_strains,
        gibbs=gibbs,
        optimal_volumes=optimal_volumes,
        optimal_shape_strains=optimal_shape_strains,
        strain_range=(float(strain_range[0]), float(strain_range[1])),
        fit_residuals=fit_residuals,
        shape_fell_back=shape_fell_back,
    )


def _narrowed_bound(bound: float) -> float:
    """``0.6 * bound``, never smaller in magnitude than ``MIN_STRAIN_HALFWIDTH``."""
    narrowed = 0.6 * float(bound)
    if abs(narrowed) >= MIN_STRAIN_HALFWIDTH or bound == 0.0:
        return narrowed
    return math.copysign(MIN_STRAIN_HALFWIDTH, bound)


def recentre(
    structure: ase.Atoms,
    iteration: GibbsIteration,
    *,
    shape_mode: phase_nodes.StrainMode | None = None,
    max_strain: float = MAX_BRACKET_STRAIN,
) -> tuple[ase.Atoms, tuple[float, float]]:
    """Next seed structure and strain range, centred on the volumes just found.

    The grid's point count is held fixed; its centre and its width both move.

    Width is adjusted by two mutually exclusive rules, applied after the centre
    has been moved onto the optimal volumes:

    * **Widen** by 50% (capped at ``max_strain``) if *any* optimal volume falls
      outside the central 60% of the sampled grid — the window no longer
      comfortably brackets the answer.
    * **Narrow** by 40% (floored at ``MIN_STRAIN_HALFWIDTH``) if *all* optimal
      volumes fall inside the central 20% — the window is far wider than it
      needs to be, and the excess width is paid for in fit error.

    The two conditions cannot both hold: the first needs a point outside the
    central 60%, the second needs every point inside the central 20%.

    Narrowing is not a cosmetic refinement. The degree-3 fit over
    ``strain_range=(-0.04, 0.04)`` carries an rms residual of 0.8–2.1 meV/atom
    against a 0.1 meV/atom target, and that residual does not cancel between
    phases — narrowing to ±0.015 moves the measured fcc/hcp boundary by 0.46 GPa
    at 0 K and 1.14 GPa at 300 K. Without a narrowing rule the loop is blind to
    it: once the grid is centred, each pass reproduces the same grid and hence
    the same biased fit, so the drift falls to zero with the bias fully intact
    and ``converged=True`` is reported anyway. Converging on window *width* as
    well as position is what makes that convergence claim honest — the old
    criterion only certified that ``G`` was insensitive to *where* the window
    sat, never to *how wide* it was. Watch ``GibbsIteration.fit_residuals`` to
    see the width converge.

    Feeding the per-temperature optimal volumes back as the grid itself would go
    too far the other way, collapsing its span to the couple of percent of
    thermal expansion and leaving the polynomial fit ill-conditioned; hence the
    floor.
    """
    if shape_mode is not None:
        shape_mode = phase_nodes.StrainMode(shape_mode)
        if not shape_mode.is_volume_conserving:
            raise ValueError(
                f"shape_mode {shape_mode!r} is not volume conserving; legal modes are "
                f"{phase_nodes.StrainMode.VOLUME_CONSERVING_MODES}. Volume must remain "
                "the outer coordinate or the volume grid is no longer well defined."
            )

    target_volume = float(np.mean(iteration.optimal_volumes))
    current_volume = float(structure.get_volume()) / len(structure)
    scale = (target_volume / current_volume) ** (1.0 / 3.0)

    seed = structure.copy()
    seed.set_cell(np.asarray(seed.get_cell()) * scale, scale_atoms=True)
    if shape_mode is not None:
        middle = iteration.optimal_shape_strains.size // 2
        seed = phase_nodes.apply_strains(
            seed, {shape_mode: float(iteration.optimal_shape_strains[middle])}
        )

    lo, hi = iteration.strain_range
    grid_lo = float(iteration.volumes.min())
    grid_hi = float(iteration.volumes.max())
    span = grid_hi - grid_lo
    optimum_lo = float(iteration.optimal_volumes.min())
    optimum_hi = float(iteration.optimal_volumes.max())

    outside_central_60 = (
        optimum_lo < grid_lo + 0.2 * span or optimum_hi > grid_hi - 0.2 * span
    )
    inside_central_20 = (
        optimum_lo >= grid_lo + 0.4 * span and optimum_hi <= grid_hi - 0.4 * span
    )
    if outside_central_60:
        bound = abs(max_strain)
        lo = float(np.clip(lo * 1.5, -bound, bound))
        hi = float(np.clip(hi * 1.5, -bound, bound))
    elif inside_central_20:
        lo = _narrowed_bound(lo)
        hi = _narrowed_bound(hi)
    return seed, (float(lo), float(hi))


def gibbs_converged(
    previous: GibbsIteration, current: GibbsIteration, tolerance: float
) -> tuple[bool, float]:
    """``(converged, drift)`` where drift is ``max over T of |G_new - G_old|``."""
    drift = float(np.max(np.abs(current.gibbs - previous.gibbs)))
    return drift < float(tolerance), drift


def gibbs_at_pressure(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    *,
    pressure: float,
    temperatures: Sequence[float],
    supercell_target_length: float = 14.0,
    strain_range: tuple[float, float] = (-0.04, 0.04),
    num_points: int = 7,
    fit_degree: int = 3,
    shape_mode: phase_nodes.StrainMode | None = None,
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_objective: ShapeObjective = "free_energy",
    displacement_distance: float = 0.01,
    is_plusminus="auto",
    max_iterations: int = 8,
    gibbs_tolerance: float = 1e-4,
    working_directory: str = ".",
    tag: str = "gibbs",
) -> GibbsResult:
    """``G(T)`` at one pressure, iterating until the volume grid sits on the answer.

    Repeats :func:`gibbs_iteration`, re-centring *and* resizing the grid each
    time (see :func:`recentre`), until ``max over T of |G_new - G_old|`` drops
    below ``gibbs_tolerance``. On the success path at least two iterations always
    run: a single pass cannot demonstrate that the answer is independent of the
    starting grid. A starting grid that cannot bracket an interior minimum even
    at the widest window (see below) may return fewer than two — there is
    nothing more productive to try.

    ``max_iterations`` defaults to 8 rather than the 5 that mere re-centring
    needed. :func:`recentre` now narrows the window as well, and narrowing takes
    several passes to reach the floor: from the default ±0.04 the sequence runs
    0.04 → 0.024 → 0.0144 → 0.0086 → ``MIN_STRAIN_HALFWIDTH``, and the loop
    needs a pass at the settled width to measure a drift there. Cutting the
    budget short converges the position while leaving the width — and so the fit
    bias — wherever it happened to stop.

    ``GibbsResult.fit_residual`` reports the rms misfit of the final volume fit.
    Read it: it is the dominant error in ``gibbs``, it does not cancel between
    phases, and unlike ``gibbs_drift`` it does not go quiet just because the loop
    has stopped moving.

    ``gibbs_tolerance`` defaults to 1e-4 eV/atom — 0.1 meV/atom, roughly 0.1 GPa
    of transition pressure for Pb, about five times below the size of the
    ``ΔG`` signal being resolved.

    The force-constant supercell is chosen **once**, from the initial
    ``structure``, using ``supercell_target_length`` — and reused unchanged for
    every iteration, including after re-centring. ``supercell_repetitions``
    rounds up with ``ceil``, so recomputing it per iteration would let a
    re-centring that crosses an integer boundary flip the repetition count and
    silently change the real-space force-constant cutoff mid-loop: up to 0.71
    meV/atom by this module's own numbers (see :func:`supercell_repetitions`),
    seven times the default ``gibbs_tolerance``. Pinning it to the initial
    structure keeps the cutoff identical across iterations of this loop *and*
    across the phases being compared, which is what makes ``ΔG`` cancellation
    correct in the first place. Use the same ``supercell_target_length`` for
    every phase you intend to compare.

    Failure to converge warns and returns the last iteration rather than
    raising, so one poorly behaved point cannot abort a pressure sweep. The same
    holds if a starting grid is too narrow to bracket even one temperature's
    optimum: :func:`gibbs_iteration` raises :class:`BracketError` in that case
    (no interior minimum to report), so it is retried here with a widened
    window — the same 1.5x step, capped at ``MAX_BRACKET_STRAIN``, that
    :func:`recentre` uses once an iteration has actually run. Only a widened
    window that repeats the previous one (the cap has already been hit) gives
    up, warns, and returns whatever iterations already succeeded. Any other
    ``RuntimeError`` — a non-converged static or phonon calculation, for
    instance — is not a bracketing problem and is left to propagate; widening
    the volume grid would not fix it and would only re-run an expensive grid
    for no reason.
    """
    base_directory = os.path.abspath(os.path.join(working_directory, tag))
    os.makedirs(base_directory, exist_ok=True)

    # Pinned to the initial structure, not recomputed per iteration: see
    # docstring above for why a per-iteration recomputation would be wrong.
    fc2_supercell_matrix = np.diag(
        supercell_repetitions(structure, target_length=supercell_target_length)
    )

    seed = structure.copy()
    current_range = (float(strain_range[0]), float(strain_range[1]))
    iterations: list[GibbsIteration] = []
    converged = False
    drift = float("inf")
    attempt = 0

    while len(iterations) < max(2, int(max_iterations)):
        try:
            iteration = gibbs_iteration(
                seed,
                engine,
                pressure=pressure,
                temperatures=temperatures,
                fc2_supercell_matrix=fc2_supercell_matrix,
                strain_range=current_range,
                num_points=num_points,
                fit_degree=fit_degree,
                shape_mode=shape_mode,
                shape_window=shape_window,
                num_shape_points=num_shape_points,
                shape_objective=shape_objective,
                displacement_distance=displacement_distance,
                is_plusminus=is_plusminus,
                working_directory=base_directory,
                tag=f"iter_{attempt:02d}",
            )
        except BracketError as error:
            attempt += 1
            lo, hi = current_range
            bound = abs(MAX_BRACKET_STRAIN)
            widened = (
                float(np.clip(lo * 1.5, -bound, bound)),
                float(np.clip(hi * 1.5, -bound, bound)),
            )
            if widened == current_range:
                warnings.warn(
                    f"Gibbs energy at {pressure} GPa: the volume grid could not "
                    f"bracket an interior minimum even at the widest strain range "
                    f"{current_range} ({error}). Returning the last successful "
                    "iteration, if any.",
                    stacklevel=2,
                )
                break
            current_range = widened
            continue

        attempt += 1
        iterations.append(iteration)
        if len(iterations) >= 2:
            converged, drift = gibbs_converged(
                iterations[-2], iterations[-1], gibbs_tolerance
            )
            if converged:
                break
        seed, current_range = recentre(seed, iteration, shape_mode=shape_mode)

    if not iterations:
        raise RuntimeError(
            f"Gibbs energy at {pressure} GPa: no volume grid, even at the widest "
            f"strain range ({-MAX_BRACKET_STRAIN}, {MAX_BRACKET_STRAIN}), bracketed "
            "an interior minimum. There is no iteration to return."
        )

    if not converged:
        if math.isinf(drift):
            warnings.warn(
                f"Gibbs energy at {pressure} GPa did not converge: only "
                f"{len(iterations)} iteration(s) succeeded before the volume grid "
                "gave up widening, too few to measure a drift between passes. "
                "Returning the last iteration.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Gibbs energy did not converge at {pressure} GPa after "
                f"{len(iterations)} iterations: drift {drift:.3e} eV/atom vs "
                f"tolerance {gibbs_tolerance:.3e}. Returning the last iteration. "
                "Raise `max_iterations`, or widen `strain_range` if the optimal "
                "volumes are drifting rather than settling.",
                stacklevel=2,
            )

    last = iterations[-1]
    return GibbsResult(
        temperatures=temperatures,
        pressure=float(pressure),
        gibbs=last.gibbs,
        optimal_volumes=last.optimal_volumes,
        optimal_shape_strains=last.optimal_shape_strains,
        iterations=iterations,
        converged=converged,
        gibbs_drift=drift,
        fit_residual=float(np.max(last.fit_residuals)),
        shape_fell_back_count=int(np.count_nonzero(last.shape_fell_back)),
        fc2_supercell_matrix=fc2_supercell_matrix,
        displacement_distance=float(displacement_distance),
    )


def _format_pressure(pressure: float, decimals: int, int_width: int) -> str:
    sign = "n" if pressure < 0 else ""
    width = int_width + 1 + decimals
    body = f"{abs(pressure):0{width}.{decimals}f}".replace(".", "_")
    return f"p_{sign}{body}"


def _pressure_tags(tag: str, pressures: Sequence[float]) -> list[str]:
    """Build one filesystem-safe tag component per pressure.

    Decimal precision is the smallest that keeps every tag distinct. Integer
    parts are zero-padded to a common width so the tags sort lexicographically.
    """
    if not pressures:
        return []

    for pressure in pressures:
        if not math.isfinite(pressure):
            raise ValueError(f"pressures must all be finite, got {pressure}")

    if len(set(pressures)) != len(pressures):
        raise ValueError(f"pressures contains duplicate values: {pressures}")

    int_width = max(len(f"{abs(pressure):.0f}") for pressure in pressures)

    for decimals in range(1, _MAX_TAG_DECIMALS + 1):
        tags = [f"{tag}/{_format_pressure(p, decimals, int_width)}" for p in pressures]
        if len(set(tags)) == len(tags):
            return tags

    raise ValueError(
        f"pressures are not separable within {_MAX_TAG_DECIMALS} decimals: "
        f"{pressures}"
    )


@fr.workflow
def gibbs_over_pressures(
    structure: ase.Atoms,
    engine: engine_mod.Engine,
    pressures: Sequence[float],
    *,
    temperatures: Sequence[float],
    supercell_target_length: float = 14.0,
    strain_range: tuple[float, float] = (-0.04, 0.04),
    num_points: int = 7,
    fit_degree: int = 3,
    shape_mode: phase_nodes.StrainMode | None = None,
    shape_window: float = 0.06,
    num_shape_points: int = 5,
    shape_objective: ShapeObjective = "free_energy",
    displacement_distance: float = 0.01,
    is_plusminus="auto",
    max_iterations: int = 8,
    gibbs_tolerance: float = 1e-4,
    working_directory: str = ".",
    tag: str = "sweep",
) -> GibbsSweep:
    """``G(T, P)`` for one phase, by running :func:`gibbs_at_pressure` per pressure.

    Each pressure starts from the same input ``structure``; the self-consistent
    loop moves it to the right volume and shrinks the volume window onto it.
    Results stack into ``(n_temperature, n_pressure)`` arrays.

    ``max_iterations`` defaults to 8 for the same reason as in
    :func:`gibbs_at_pressure`: :func:`recentre` needs several passes to narrow
    the window from the default ±0.04 down to ``MIN_STRAIN_HALFWIDTH``, and a
    truncated budget leaves the fit bias in place while still reporting a small
    drift. Check ``result.fit_residual`` alongside ``result.converged`` for every
    pressure in the sweep.
    """
    pressure_tags = _pressure_tags(tag, pressures)

    results: list[GibbsResult] = []
    gibbs_results: list[np.ndarray] = []
    volume_results: list[np.ndarray] = []
    for pressure, pressure_tag in zip(pressures, pressure_tags):
        result = gibbs_at_pressure(
            structure,
            engine,
            pressure=pressure,
            temperatures=temperatures,
            supercell_target_length=supercell_target_length,
            strain_range=strain_range,
            num_points=num_points,
            fit_degree=fit_degree,
            shape_mode=shape_mode,
            shape_window=shape_window,
            num_shape_points=num_shape_points,
            shape_objective=shape_objective,
            displacement_distance=displacement_distance,
            is_plusminus=is_plusminus,
            max_iterations=max_iterations,
            gibbs_tolerance=gibbs_tolerance,
            working_directory=working_directory,
            tag=pressure_tag,
        )
        results.append(result)
        gibbs_results.append(result.gibbs)
        volume_results.append(result.optimal_volumes)

    gibbs = np.column_stack(gibbs_results)
    optimal_volumes = np.column_stack(volume_results)
    gibbs_sweep = GibbsSweep(
        temperatures=temperatures,
        pressures=pressures,
        gibbs=gibbs,
        optimal_volumes=optimal_volumes,
        results=results,
    )
    return gibbs_sweep


def phase_boundary(sweep_a: GibbsSweep, sweep_b: GibbsSweep) -> np.ndarray:
    """Transition pressure per temperature, where ``G_a - G_b`` crosses zero.

    Linear interpolation between the two bracketing pressures. Three edge cases
    are resolved rather than left to the interpolation:

    * **No bracketed crossing** → ``nan``. The pressure range does not contain
      the transition at that temperature, and any other answer would be
      extrapolation.
    * **A row that is exactly zero everywhere** (the two phases are exactly
      degenerate across the whole sweep) → the *first* pressure. They are
      degenerate everywhere, so the first pressure is as good an answer as any.
    * **A row whose last point is exactly zero**, with no sign change before it
      → the *last* pressure. That point is the crossing; there is no interval
      beyond it to interpolate into, but reporting ``nan`` for an exact zero
      would be perverse.

    If ``G_a - G_b`` changes sign more than once along the sweep, the first
    crossing is returned and a ``UserWarning`` names all the candidates. In a
    ``ΔG`` this small a spurious flip near zero is entirely plausible, and the
    first crossing is not necessarily the physical one.
    """
    if not np.allclose(sweep_a.pressures, sweep_b.pressures):
        raise ValueError(
            "Both sweeps must share a pressure grid; got "
            f"{sweep_a.pressures} and {sweep_b.pressures}."
        )
    if not np.allclose(sweep_a.temperatures, sweep_b.temperatures):
        raise ValueError(
            "Both sweeps must share a temperature grid; got "
            f"{sweep_a.temperatures} and {sweep_b.temperatures}."
        )
    delta = sweep_a.gibbs - sweep_b.gibbs
    pressures = sweep_a.pressures
    boundary = np.full(delta.shape[0], np.nan)
    for index in range(delta.shape[0]):
        row = delta[index]
        crossings = np.flatnonzero(
            (np.sign(row[:-1]) * np.sign(row[1:]) < 0.0) | (row[:-1] == 0.0)
        )
        if crossings.size == 0:
            if row[-1] == 0.0:
                boundary[index] = float(pressures[-1])
            continue
        if crossings.size > 1:
            candidates = [float(pressures[int(c)]) for c in crossings]
            warnings.warn(
                f"G_a - G_b changes sign {crossings.size} times at temperature "
                f"index {index} (T = {sweep_a.temperatures[index]:.1f} K): "
                f"crossings bracketed at pressures {candidates} GPa. Taking the "
                "first. With a ΔG this small a spurious flip near zero is "
                "plausible; check the ΔG curve before trusting the boundary.",
                stacklevel=2,
            )
        first = int(crossings[0])
        low, high = row[first], row[first + 1]
        if high == low:
            boundary[index] = float(pressures[first])
            continue
        fraction = float(-low / (high - low))
        boundary[index] = float(
            pressures[first] + fraction * (pressures[first + 1] - pressures[first])
        )
    return boundary
