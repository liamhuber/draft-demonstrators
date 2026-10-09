"""Nodes shared between the demonstration workflows."""

import dataclasses
from typing import Annotated

import ase
import flowrep as fr
from pyiron_workflow_atomistics import engine as engine_mod

from . import uris

type InputType = (
    engine_mod.CalcInputStatic | engine_mod.CalcInputMinimize | engine_mod.CalcInputMD
)


@fr.atomic
def with_calc_input(
    engine: engine_mod.Engine,
    calc_input: InputType,
):
    """Return a copy of a dataclass engine with its EngineInput replaced.

    Lets the elastic macro switch a single user-supplied engine between
    full-relax and fixed-cell-relax modes without the user wiring two engines.

    ``engine`` must be a dataclass (e.g. ``ASEEngine``); a clear ``TypeError``
    is raised otherwise.
    """
    if not isinstance(engine, engine_mod.Engine) or not dataclasses.is_dataclass(
        engine
    ):
        raise TypeError(
            f"with_calc_input requires a dataclass engine ( (got {type(engine).__name__}); "
            "ASEEngine is a dataclass."
        )
    return dataclasses.replace(engine, EngineInput=calc_input)


@fr.workflow
def calculate_with_input(
    structure: Annotated[ase.Atoms, {"uri": uris.URI.atomic_structure}],
    engine: engine_mod.Engine,
    calc_input: InputType,
    label: str,
):
    """Calculate ``structure`` with ``engine`` switched to ``calc_input``.

    The calculation runs in a sub-engine whose working directory is ``label``. Returns
    the final structure and final energy.
    """
    sub_engine = engine_mod.subengine(engine, label)
    used_engine = with_calc_input(sub_engine, calc_input)
    output = engine_mod.calculate(structure=structure, engine=used_engine)
    relaxed_structure = fr.std.get_attr(output, "final_structure")
    relaxed_energy = fr.std.get_attr(output, "final_energy")
    return relaxed_structure, relaxed_energy
