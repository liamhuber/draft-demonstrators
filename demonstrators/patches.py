from ase.calculators import eam

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