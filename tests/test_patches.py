import copy
import dataclasses
import json
import pickle
import unittest

from ase import build

from demonstrators import patches


@dataclasses.dataclass
class _Holder:
    calculator: patches.JSONableEMT


class TestJSONableEMT(unittest.TestCase):
    def test_json_dumps(self):
        calc = patches.JSONableEMT()
        self.assertEqual(
            json.loads(json.dumps(calc)),
            {
                "class": "ase.calculators.emt.EMT",
                "parameters": {"asap_cutoff": False},
            },
        )

    def test_set_updates_items(self):
        calc = patches.JSONableEMT()
        calc.set(asap_cutoff=True)
        self.assertTrue(json.loads(json.dumps(calc))["parameters"]["asap_cutoff"])

    def test_identity_semantics(self):
        a, b = patches.JSONableEMT(), patches.JSONableEMT()
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)
        self.assertTrue(a)
        self.assertIn("JSONableEMT object at", repr(a))

    def test_asdict_roundtrip(self):
        holder = _Holder(patches.JSONableEMT(asap_cutoff=True))
        rebuilt = dataclasses.asdict(holder)["calculator"]
        self.assertIsInstance(rebuilt, patches.JSONableEMT)
        self.assertTrue(rebuilt.parameters["asap_cutoff"])
        json.dumps(dataclasses.asdict(holder))

    def test_copy_and_pickle(self):
        calc = patches.JSONableEMT(asap_cutoff=True)
        for clone in (copy.deepcopy(calc), pickle.loads(pickle.dumps(calc))):
            self.assertEqual(json.dumps(clone), json.dumps(calc))

    def test_still_computes(self):
        atoms = build.bulk("Au", cubic=True)
        atoms.calc = patches.JSONableEMT()
        self.assertIsInstance(atoms.get_potential_energy(), float)


if __name__ == "__main__":
    unittest.main()
