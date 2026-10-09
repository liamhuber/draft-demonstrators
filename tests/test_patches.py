import copy
import dataclasses
import json
import os
import pickle
import unittest

from ase import build, optimize

from demonstrators import patches

_EAM_POTENTIAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources",
    "Pb_II_Wang_2018.eam.alloy",
)


@dataclasses.dataclass
class _Holder:
    calculator: patches.SerializableEMT | patches.SerializableEAM


class _SerializableCalculatorTests:
    """Shared tests; subclasses provide a calculator factory and a structure."""

    calculator_class: type
    json_class: str

    def make(self):
        raise NotImplementedError

    def structure(self):
        raise NotImplementedError

    def used(self):
        calc = self.make()
        atoms = self.structure()
        atoms.calc = calc
        atoms.get_potential_energy()
        return calc

    def energy(self, calc):
        atoms = self.structure()
        atoms.rattle(stdev=0.05, seed=0)
        atoms.calc = calc
        return atoms.get_potential_energy()

    def test_json_dumps(self):
        items = json.loads(json.dumps(self.make()))
        self.assertEqual(items["class"], self.json_class)
        self.assertIn("parameters", items)

    def test_set_updates_items(self):
        calc = self.make()
        calc.set(skin=0.5)
        self.assertEqual(json.loads(json.dumps(calc))["parameters"]["skin"], 0.5)

    def test_identity_semantics(self):
        a, b = self.make(), self.make()
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)
        self.assertTrue(a)
        self.assertIn(f"{self.calculator_class.__name__} object at", repr(a))

    def test_asdict_roundtrip(self):
        holder = _Holder(self.make())
        rebuilt = dataclasses.asdict(holder)["calculator"]
        self.assertIsInstance(rebuilt, self.calculator_class)
        self.assertEqual(json.dumps(rebuilt), json.dumps(holder.calculator))
        json.dumps(dataclasses.asdict(holder))

    def test_copy_and_pickle_after_calculation(self):
        calc = self.used()
        reference = self.energy(self.make())
        for clone in (copy.deepcopy(calc), pickle.loads(pickle.dumps(calc))):
            self.assertIsInstance(clone, self.calculator_class)
            self.assertEqual(json.dumps(clone), json.dumps(calc))
            self.assertAlmostEqual(self.energy(clone), reference)

    def test_still_computes(self):
        self.assertIsInstance(self.used().results["energy"], float)


class TestSerializableEMT(_SerializableCalculatorTests, unittest.TestCase):
    calculator_class = patches.SerializableEMT
    json_class = "ase.calculators.emt.EMT"

    def make(self):
        return patches.SerializableEMT()

    def structure(self):
        return build.bulk("Au", cubic=True)

    def test_json_dumps(self):
        self.assertEqual(
            json.loads(json.dumps(patches.SerializableEMT())),
            {
                "class": "ase.calculators.emt.EMT",
                "parameters": {"asap_cutoff": False},
            },
        )

    def test_set_updates_items(self):
        calc = patches.SerializableEMT()
        calc.set(asap_cutoff=True)
        self.assertTrue(json.loads(json.dumps(calc))["parameters"]["asap_cutoff"])

    def test_asdict_keeps_parameters(self):
        holder = _Holder(patches.SerializableEMT(asap_cutoff=True))
        rebuilt = dataclasses.asdict(holder)["calculator"]
        self.assertTrue(rebuilt.parameters["asap_cutoff"])


class TestSerializableEAM(_SerializableCalculatorTests, unittest.TestCase):
    calculator_class = patches.SerializableEAM
    json_class = "ase.calculators.eam.EAM"

    def make(self):
        return patches.SerializableEAM(potential=_EAM_POTENTIAL, form="alloy")

    def structure(self):
        return build.bulk("Pb", "fcc", a=4.95, cubic=True)

    def test_json_dumps_includes_form(self):
        items = json.loads(json.dumps(self.make()))
        self.assertEqual(items["form"], "alloy")
        self.assertEqual(items["parameters"]["potential"], _EAM_POTENTIAL)


@dataclasses.dataclass
class _OptimizerHolder:
    optimizer_class: patches.JSONableClass


class _Outer:
    class Inner:
        pass


class TestJSONableClass(unittest.TestCase):
    def test_json_dumps(self):
        wrapped = patches.JSONableClass(optimize.BFGS)
        self.assertEqual(
            json.loads(json.dumps(wrapped)), {"class": "ase.optimize.bfgs.BFGS"}
        )

    def test_call_instantiates_wrapped_class(self):
        atoms = build.bulk("Au", cubic=True)
        atoms.calc = patches.SerializableEMT()
        optimizer = patches.JSONableClass(optimize.BFGS)(atoms, logfile=None)
        self.assertIsInstance(optimizer, optimize.BFGS)

    def test_asdict_roundtrip(self):
        holder = _OptimizerHolder(patches.JSONableClass(optimize.BFGS))
        rebuilt = dataclasses.asdict(holder)["optimizer_class"]
        self.assertIsInstance(rebuilt, patches.JSONableClass)
        self.assertIs(rebuilt.cls, optimize.BFGS)

    def test_copy_and_pickle(self):
        wrapped = patches.JSONableClass(optimize.BFGS)
        for clone in (copy.deepcopy(wrapped), pickle.loads(pickle.dumps(wrapped))):
            self.assertIs(clone.cls, optimize.BFGS)

    def test_hash_and_repr(self):
        a = patches.JSONableClass(optimize.BFGS)
        b = patches.JSONableClass(optimize.BFGS)
        self.assertEqual(len({a, b}), 1)
        self.assertEqual(repr(a), "JSONableClass(ase.optimize.bfgs.BFGS)")

    def test_nested_qualname(self):
        wrapped = patches.JSONableClass(_Outer.Inner)
        self.assertIs(patches.JSONableClass(dict(wrapped).items()).cls, _Outer.Inner)

    def test_unimportable_path(self):
        with self.assertRaises(ImportError):
            patches.JSONableClass({"class": "not_a_module_xyz.Thing"})


if __name__ == "__main__":
    unittest.main()
