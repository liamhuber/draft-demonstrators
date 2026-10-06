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
