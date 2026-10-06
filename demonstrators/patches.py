import importlib

from ase.calculators import eam, emt


class PicklableEAM(eam.EAM):
    """
    ASE's EAM calculator uses closures that make it unpickleable -- patch that.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("embedded_energy", "electron_density", "phi", "d", "q"):
            state.pop(key, None)
            state.pop(f"d_{key}", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.form == "fs":
            self.set_fs_splines()
        else:
            self.set_splines()
        if self.form == "adp":
            self.set_adp_splines()


class JSONableEMT(emt.EMT, dict):
    """
    ASE's EMT calculator isn't JSON serializable -- patch that.

    `json.dumps` only accepts native types, so the calculator doubles as a `dict` whose
    items describe it (class and parameters). Identity semantics (equality, hashing,
    truthiness, repr) are restored from `object`.
    """

    def __init__(self, _items=None, **kwargs):
        # dataclasses.asdict rebuilds dict subclasses as `type(obj)(items)`
        if _items is not None:
            kwargs = {**dict(_items)["parameters"], **kwargs}
        dict.__init__(self)
        emt.EMT.__init__(self, **kwargs)

    def set(self, **kwargs):
        changed_parameters = super().set(**kwargs)
        self._sync_items()
        return changed_parameters

    def _sync_items(self):
        dict.clear(self)
        dict.update(
            self,
            {
                "class": f"{emt.EMT.__module__}.{emt.EMT.__qualname__}",
                "parameters": dict(self.parameters),
            },
        )

    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__
    __repr__ = object.__repr__

    def __bool__(self):
        return True


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
