"""Infrastructure shared by the MPMD disaggregations (AF-EP, Engram).

One MPI world holds the attention ranks first and one block of ranks per
peer role after them. Each disaggregation exchanges on the pair communicator
of the attention ranks and its peer role (:meth:`MpiWorld.pair_comm`), through
a :class:`PairSession` that carries the transport metadata the native
endpoints parse; a :class:`ClientRegistry` hands the attention-side client to
the model; :func:`run_role_service` is the peer-rank process skeleton.
"""

from .registry import ClientRegistry
from .service import join_mpi_world, run_role_service
from .session import PairSession

__all__ = ["ClientRegistry", "PairSession", "join_mpi_world", "run_role_service"]
