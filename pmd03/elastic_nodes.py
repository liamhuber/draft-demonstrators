"""Workflows for the elastic constants of bulk crystals."""

from typing import Annotated

import ase
import flowrep as fr
from ase import build
from pyiron_workflow_atomistics import engine as engine_mod
from pyiron_workflow_atomistics.physics import bulk, elastic

from . import shared, uris


@fr.workflow
def elastic_constants(
    engine: engine_mod.ASEEngine,
    structure: Annotated[ase.Atoms, {"uri": uris.URI.atomic_structure}],
    # Physically, we're looking for a 3d structure, beyond that it's up to the user
    # if what they give in will give back physically meaningful numbers, IMO
    relaxation_config: engine_mod.CalcInputMinimize | engine_mod.CalcInputStatic,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
    # Semantikon has no concept for a collection of URIs
    # So while ASMO has a "strain" entry, it is not usable here
):
    """Relax ``structure``, then fit its elastic stiffness tensor from stress-strain data.

    The relaxed reference is deformed by each of the normal and shear strains, the
    stress of every deformed cell is evaluated statically, and a linear fit relative
    to the reference's residual stress gives the stiffness tensor.

    Returns the relaxed reference structure, the raw fit, and a summary ``dict`` of
    derived elastic properties (e.g. the Voigt-Reuss-Hill bulk modulus ``K_VRH``).
    """
    relax_engine = elastic.with_calc_input(engine=engine, calc_input=relaxation_config)
    relaxed_output = engine_mod.calculate(structure=structure, engine=relax_engine)
    ref_structure = fr.std.get_attr(relaxed_output, "final_structure")
    eq_stress = elastic._reference_stress_gpa(relaxed_output)

    deformed_structures, strains = elastic.generate_mp_deformations(
        structure=ref_structure, norm_strains=norm_strains, shear_strains=shear_strains
    )
    static_config = engine_mod.CalcInputStatic()
    deform_engine = shared.with_calc_input(engine=engine, calc_input=static_config)

    deformation_results = bulk.evaluate_structures(
        structures=deformed_structures, engine=deform_engine
    )
    stresses = elastic.extract_stresses_gpa(deformation_results)
    fit = elastic.fit_elastic_tensor(
        strains=strains,
        stresses=stresses,
        structure=ref_structure,
        eq_stress=eq_stress,
    )
    summary = elastic.elastic_constants_summary(fit, ref_structure)

    return ref_structure, fit, summary


@fr.atomic("unit_cell")
def bulk_unit(
    symbol: Annotated[str, {"uri": uris.URI.chemical_symbol}],
) -> Annotated[ase.Atoms, {"uri": uris.URI.atomic_structure}]:
    # also "bulk"... and "3D (data)"
    """The primitive cell of an element in its ASE reference crystal structure."""
    return build.bulk(symbol)


@fr.atomic("bulk_modulus")
def get_bulk_modulus(
    elastic_summary: dict,
) -> Annotated[float, {"uri": uris.URI.bulk_modulus}]:
    """The Voigt-Reuss-Hill bulk modulus (GPa) from an elastic summary."""
    return elastic_summary["K_VRH"]


@fr.workflow
def unary_elastic_tensor(
    engine: engine_mod.ASEEngine,
    symbol: Annotated[str, {"uri": uris.URI.chemical_symbol}],
    relaxation_config: engine_mod.CalcInputMinimize | engine_mod.CalcInputStatic,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
) -> tuple[list[list[float]], Annotated[float, {"uri": uris.URI.bulk_modulus}]]:
    """Elastic tensor and bulk modulus of an element in its reference crystal structure.

    Returns the 6x6 stiffness tensor in the IEEE standard orientation and the
    Voigt-Reuss-Hill bulk modulus, both in GPa.
    """
    structure = bulk_unit(symbol)
    _ref_structure, _fit, summary = elastic_constants(
        engine=engine,
        structure=structure,
        relaxation_config=relaxation_config,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
    )
    bulk_modulus = get_bulk_modulus(summary)
    tensor_ieee = fr.std.getitem(summary, "elastic_tensor_ieee")
    return tensor_ieee, bulk_modulus
