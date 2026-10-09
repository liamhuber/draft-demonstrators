import importlib

from ase.calculators import eam, emt


class _SerializableCalculator:
    """
    ASE calculators are neither JSON serializable nor reliably picklable (some hold
    closures) -- patch that.

    `json.dumps` only accepts native types, so the calculator doubles as a `dict` whose
    items describe it (class, parameters, and any extra `__init__` arguments). Pickling
    and copying rebuild the calculator from those same items, so cached state (atoms,
    results, splines, neighbor lists) is dropped and recomputed on first use. Identity
    semantics (equality, hashing, truthiness, repr) are restored from `object`.

    Concrete classes list `dict` *after* the ASE calculator, so calculator methods
    (e.g. `EAM.update`) take precedence over `dict` methods of the same name.
    """

    _calculator_class: type
    _init_args: tuple[str, ...] = ()
    """`__init__` arguments that aren't stored in ASE's `parameters`."""

    def __init__(self, _items=None, **kwargs):
        # dataclasses.asdict rebuilds dict subclasses as `type(obj)(items)`
        if _items is not None:
            items = dict(_items)
            init_args = {arg: items[arg] for arg in self._init_args}
            kwargs = {**items["parameters"], **init_args, **kwargs}
        dict.__init__(self)
        self._calculator_class.__init__(self, **kwargs)

    def set(self, **kwargs):
        changed_parameters = self._calculator_class.set(self, **kwargs)
        self._sync_items()
        return changed_parameters

    def _sync_items(self):
        cls = self._calculator_class
        dict.clear(self)
        dict.update(
            self,
            {
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "parameters": dict(self.parameters),
                **{arg: getattr(self, arg) for arg in self._init_args},
            },
        )

    def __reduce__(self):
        return type(self), (dict(self),)

    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__
    __repr__ = object.__repr__

    def __bool__(self):
        return True


class SerializableEMT(_SerializableCalculator, emt.EMT, dict):
    """ASE's EMT calculator, JSON serializable and picklable."""

    _calculator_class = emt.EMT


class SerializableEAM(_SerializableCalculator, eam.EAM, dict):
    """
    ASE's EAM calculator, JSON serializable and picklable.

    Only file-based potentials are JSON serializable, and the `potential` path is
    stored as given -- a relative path only resolves from the same working directory.
    """

    _calculator_class = eam.EAM
    _init_args = ("form",)


class JSONableClass(dict):
    """
    Classes (e.g. `ASEEngine.optimizer_class`) aren't JSON serializable -- patch that.

    Wraps a class in a `dict` holding its import path; calling the wrapper instantiates
    the wrapped class.
    """

    def __init__(self, cls_or_items):
        # dataclasses.asdict rebuilds dict subclasses as `type(obj)(items)`
        if isinstance(cls_or_items, type):
            path = f"{cls_or_items.__module__}.{cls_or_items.__qualname__}"
        else:
            path = dict(cls_or_items)["class"]
        super().__init__({"class": path})
        self.cls = _import_from_path(path)

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

    def __hash__(self):
        return hash(self["class"])

    def __repr__(self):
        return f"{type(self).__name__}({self['class']})"


def _import_from_path(path):
    module_name, _, qualname = path.rpartition(".")
    while module_name:
        try:
            obj = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            module_name, _, outer = module_name.rpartition(".")
            qualname = f"{outer}.{qualname}"
    else:
        raise ImportError(f"Could not import {path}")
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj
