import ase
import flowrep as fr
import semantikon
from ase import build

from pyiron_workflow_atomistics import engine as engine_mod
from pyiron_workflow_atomistics.physics import bulk, elastic

from . import uris

### Wrappers


@fr.dataclass
class CalcInputStatic(engine_mod.CalcInputStatic): ...


### Restructured


@fr.workflow
def elastic_constants(
    engine: engine_mod.ASEEngine,
    structure: semantikon.u(ase.Atoms, uris=uris.atomic_structure),
    # Physically, we're looking for a 3d structure, beyond that it's up to the user
    # if what they give in will give back physically meaningful numbers, IMO
    relaxation_config: engine_mod.CalcInputMinimize | engine_mod.CalcInputStatic,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
    # Neither PMDco nor TTO have "strain" entries...
):
    relax_engine = elastic.with_calc_input(engine=engine, calc_input=relaxation_config)
    relaxed_output = engine_mod.calculate(structure=structure, engine=relax_engine)
    ref_structure = fr.std.get_attr(relaxed_output, "final_structure")
    eq_stress = elastic._reference_stress_gpa(relaxed_output)

    deformed_structures, strains = elastic.generate_mp_deformations(
        structure=ref_structure, norm_strains=norm_strains, shear_strains=shear_strains
    )
    static_config = CalcInputStatic()
    deform_engine = elastic.with_calc_input(engine=engine, calc_input=static_config)

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


### New


@fr.atomic("unit_cell")
def bulk_unit(symbol: str) -> semantikon.u(ase.Atoms, uris=uris.atomic_structure):
    # also "bulk"... and "3D (data)"
    return build.bulk(symbol)


@fr.atomic("bulk_modulus")
def get_bulk_modulus(
    elastic_summary: dict,
) -> semantikon.u(float, uri=uris.bulk_modulus):
    return elastic_summary["K_VRH"]


@fr.workflow
def unary_elastic_tensor(
    engine: engine_mod.ASEEngine,
    symbol: semantikon.u(str, uri=uris.chemical_composition),
    relaxation_config: engine_mod.CalcInputMinimize | engine_mod.CalcInputStatic,
    norm_strains: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06),
) -> tuple[list[list[float]], semantikon.u(float, uri=uris.bulk_modulus)]:
    structure = bulk_unit(symbol)
    _, fit, summary = elastic_constants(
        engine=engine,
        structure=structure,
        relaxation_config=relaxation_config,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
    )
    bulk_modulus = get_bulk_modulus(summary)
    tensor_ieee = fr.std.getitem(summary, "elastic_tensor_ieee")
    return tensor_ieee, bulk_modulus
