// Test-only PMPI probe. Compile with mpicc -shared -fPIC -std=c11, preload
// into each test rank, and set XCPU_EP_TRACE_ROUNDS=1 XCPU_EP_MPI_PROBE=1.
#include <mpi.h>
#include <stdatomic.h>
#include <stdint.h>

// allreduce / alltoall / alltoallv / fence / ialltoallv / wait
static _Atomic uint64_t calls[6];

void ep_moe_probe_reset(void) {
  for (int i = 0; i < 6; ++i) atomic_store(&calls[i], 0);
}
uint64_t ep_moe_probe_count(int i) { return atomic_load(&calls[i]); }

int MPI_Allreduce(const void* send, void* recv, int n, MPI_Datatype type,
                  MPI_Op op, MPI_Comm comm) {
  atomic_fetch_add(&calls[0], 1);
  return PMPI_Allreduce(send, recv, n, type, op, comm);
}
int MPI_Alltoall(const void* send, int sn, MPI_Datatype st,
                 void* recv, int rn, MPI_Datatype rt, MPI_Comm comm) {
  atomic_fetch_add(&calls[1], 1);
  return PMPI_Alltoall(send, sn, st, recv, rn, rt, comm);
}
int MPI_Alltoallv(const void* send, const int* sn, const int* sd, MPI_Datatype st,
                  void* recv, const int* rn, const int* rd, MPI_Datatype rt,
                  MPI_Comm comm) {
  atomic_fetch_add(&calls[2], 1);
  return PMPI_Alltoallv(send, sn, sd, st, recv, rn, rd, rt, comm);
}
int MPI_Ialltoallv(const void* send, const int* sn, const int* sd, MPI_Datatype st,
                   void* recv, const int* rn, const int* rd, MPI_Datatype rt,
                   MPI_Comm comm, MPI_Request* request) {
  atomic_fetch_add(&calls[4], 1);
  return PMPI_Ialltoallv(send, sn, sd, st, recv, rn, rd, rt, comm, request);
}
int MPI_Wait(MPI_Request* request, MPI_Status* status) {
  atomic_fetch_add(&calls[5], 1);
  return PMPI_Wait(request, status);
}
int MPI_Win_fence(int assert_flags, MPI_Win win) {
  atomic_fetch_add(&calls[3], 1);
  return PMPI_Win_fence(assert_flags, win);
}
