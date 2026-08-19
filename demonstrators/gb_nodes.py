from collections.abc import Iterable

import ase
import flowrep as fr
import freud
import numpy as np
import structuretoolkit as stk
from ase import build
from ase.geometry import cell as ase_cell
from pyiron_workflow_atomistics import engine as engine_mod
from pyiron_workflow_atomistics import structure as structure_mod

from . import shared


@fr.atomic
def find_unique_sites(structure: ase.Atoms) -> list[int]:
    """by Steinhardt parameters"""
    raw_metrics = stk.get_steinhardt_parameter_structure(
        structure,
        n_clusters=None,
    )
    # get_steinhardt_parameter_structure asserts to return (n qs, n atoms)-shaped array
    # but verifiably returns n atoms >> len(structure)
    # Nonetheless, searching across unique values seems to robustly return results
    # inside the original index count
    metrics = raw_metrics.T
    seen = {}
    for i, metric in enumerate(metrics):
        new_ = True
        for j, tagged in seen.items():
            if np.allclose(tagged[0], metric):
                new_ = False
                break
        if new_:
            seen[i] = [metric, 1]
        else:
            seen[j][1] += 1

    return list(seen.keys())


@fr.workflow
def get_relaxed_bulk(
    species: str,
    repetitions: int,
    engine: engine_mod.Engine,
    minimize_input: engine_mod.CalcInputMinimize,
) -> tuple[
    ase.Atoms,
    float,
    list[int],
]:
    cubic_unit = build.bulk(species, cubic=True)
    unique_sites = find_unique_sites(cubic_unit)
    supercell = structure_mod.create_supercell(cubic_unit, repetitions)
    relaxed_structure, relaxed_energy = shared.calculate_with_input(
        structure=supercell,
        engine=engine,
        calc_input=minimize_input,
        label="relax_bulk",
    )
    return relaxed_structure, relaxed_energy, unique_sites


@fr.dataclass
class GBParameters:
    axis: tuple[int, int, int]
    sigma: int
    plane: tuple[int, int, int]
    grain_thickness: int
    # Could be expanded further in line with `structuretoolkit.grainboundary`
    supercell_repeats: tuple[int, int, int]


@fr.atomic
def clean_cell(structure: ase.Atoms, rtol: float = 1e-6) -> None:
    """Zero out near-zero cell entries that can be artefacts of GB creation."""
    cell = np.array(structure.cell)  # copy; atoms.cell is a live Cell view
    cell[np.abs(cell) < rtol * np.abs(cell).max()] = 0.0
    cleaned_structure = structure.copy()
    cleaned_structure.set_cell(cell, scale_atoms=False)
    return cleaned_structure


@fr.workflow
def get_relaxed_gb(
    species: str,
    gb_parameters: GBParameters,
    engine: engine_mod.Engine,
    minimize_input: engine_mod.CalcInputMinimize,
) -> tuple[ase.Atoms, float, list[int]]:
    cubic_unit = build.bulk(species, cubic=True)
    raw_structure = stk.grainboundary(
        axis=gb_parameters.axis,
        sigma=gb_parameters.sigma,
        plane=gb_parameters.plane,
        initial_struct=cubic_unit,
        uc_a=gb_parameters.grain_thickness,
        uc_b=gb_parameters.grain_thickness,
    )
    structure = clean_cell(raw_structure)
    supercell = structure_mod.create_supercell(
        structure,
        gb_parameters.supercell_repeats,
    )
    unique_sites = find_unique_sites(supercell)
    relaxed_structure, relaxed_energy = shared.calculate_with_input(
        structure=supercell,
        engine=engine,
        calc_input=minimize_input,
        label="relax_gb",
    )
    return relaxed_structure, relaxed_energy, unique_sites


@fr.atomic
def analyze_voronoi(
    structure: ase.Atoms,
    sites: list[int],
) -> list[float]:
    """Per-atom Voronoi volumes for a fully periodic cell."""
    # freud requires an upper-triangular box matrix (columns = box vectors);
    # cellpar_to_cell gives a lower-triangular, right-handed cell (rows = vectors).
    tri_cell = ase_cell.cellpar_to_cell(structure.cell.cellpar())
    box = freud.box.Box.from_matrix(tri_cell.T)
    points = box.make_absolute(structure.get_scaled_positions(wrap=True))

    voro = freud.locality.Voronoi()
    voro.compute((box, points))
    return np.asarray(voro.volumes)[sites].tolist()
    # the latest release of pyscal can't handle non-rectangular boxes
    # and I haven't found a way to configure pyiron_workflow_atomistic's calc input
    # to handle isotropic relaxation only
    # thus, our internal structuretoolkit tool doesn't work -- use freud instead


@fr.atomic
def relaxation_substitution_label(context: str, solute: str, site_index: int):
    return f"relax_{context}_with_{solute}_at_{site_index}"


@fr.workflow
def relax_substitution(
    structure: ase.Atoms,
    substitute: str,
    site_index: int,
    engine: engine_mod.Engine,
    minimize_input: engine_mod.CalcInputMinimize,
    context_tag: str,
) -> tuple[ase.Atoms, float]:
    substituted_structure = structure_mod.substitutional_swap(
        structure, site_index, substitute
    )
    label = relaxation_substitution_label(context_tag, substitute, site_index)
    relaxed_structure, relaxed_energy = shared.calculate_with_input(
        structure=substituted_structure,
        engine=engine,
        calc_input=minimize_input,
        label=label,
    )
    return relaxed_structure, relaxed_energy


@fr.atomic
def data_at_energy_minima(
    energies: Iterable[float], volumes: Iterable[float]
) -> tuple[float, float]:
    if len(energies) != len(volumes):
        raise ValueError("energies and volumes must be the same length")

    idx = int(np.argmin(np.array(energies)))
    energy_min = energies[idx]
    volume_min = volumes[idx]
    return energy_min, volume_min


@fr.atomic
def calculate_segregation_energy(
    bulk_energy: float,
    gb_energy: float,
    solvated_energy: float,
    segregated_energy: float,
) -> float:
    """Negative = favourable convention"""
    return (segregated_energy + bulk_energy) - (gb_energy + solvated_energy)


@fr.workflow
def volumetric_segregation(
    # Chemistry
    host: str,
    solutes: Iterable[str],
    # Geometry
    bulk_reps: int,
    gb_parameters: GBParameters,
    # Model
    engine: engine_mod.Engine,
    clean_minimize_input: engine_mod.CalcInputMinimize,
    solute_minimize_input: engine_mod.CalcInputMinimize,
) -> tuple[ase.Atoms, list[int], list[float], list[list[float]]]:
    bulk_structure, bulk_energy, bulk_sites = get_relaxed_bulk(
        host, bulk_reps, engine, clean_minimize_input
    )
    bulk_volumes = analyze_voronoi(bulk_structure, bulk_sites)

    gb_structure, gb_energy, gb_sites = get_relaxed_gb(
        host,
        gb_parameters,
        engine,
        clean_minimize_input,
    )
    site_volumes = analyze_voronoi(gb_structure, gb_sites)

    solute_segregation_energies = []
    solute_excess_volumes = []
    for solute in solutes:

        solvated_energies = []
        for bulk_site in bulk_sites:
            _, solvated_energy = relax_substitution(
                bulk_structure,
                solute,
                bulk_site,
                engine,
                solute_minimize_input,
                "bulk",
            )
            solvated_energies.append(solvated_energy)
        solvated_energy, reference_volume = data_at_energy_minima(
            solvated_energies, bulk_volumes
        )

        segregation_energies = []
        excess_volumes = []
        for gb_site, site_volume in zip(gb_sites, site_volumes):
            _segregated_structure, segregated_energy = relax_substitution(
                gb_structure,
                solute,
                gb_site,
                engine,
                solute_minimize_input,
                "gb",
            )

            segregation_energy = calculate_segregation_energy(
                bulk_energy, gb_energy, solvated_energy, segregated_energy
            )
            segregation_energies.append(segregation_energy)

            excess_volume = fr.std.sub(site_volume, reference_volume)
            excess_volumes.append(excess_volume)

        solute_segregation_energies.append(segregation_energies)
        solute_excess_volumes.append(excess_volumes)

    return (
        gb_structure,
        gb_sites,
        solute_excess_volumes,
        solute_segregation_energies,
    )
