import importlib.metadata

from .api import GBParameters as GBParameters
from .api import JSONableClass as JSONableClass
from .api import SerializableEAM as SerializableEAM
from .api import SerializableEMT as SerializableEMT
from .api import gibbs_over_pressures as gibbs_over_pressures
from .api import phase_boundary as phase_boundary
from .api import unary_elastic_tensor as unary_elastic_tensor
from .api import volumetric_segregation as volumetric_segregation

# Re-exports are explicit (PEP 484 `X as X`); an empty `__all__` keeps star-imports inert
__all__: list[str] = []

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"
