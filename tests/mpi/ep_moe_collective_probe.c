// Test-only PMPI probe. Compile with mpicc -shared -fPIC -std=c11, preload
// into each test rank, and set XCPU_EP_TRACE_ROUNDS=1 XCPU_EP_MPI_PROBE=1.
//
// XCPU_EP_PROBE_LOG=<lines>: additionally print the first <lines> dispatch
// count exchanges (MPI_Alltoall of 1 int per rank, the v5 per-round send
// bytes) and alltoallv send/recv nonzero counts to stderr, prefixed with the
// rank. Header-only keepalives, payload messages and silent destinations are
// then distinguishable in end-to-end logs.
#include <mpi.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// allreduce / alltoall / alltoallv / fence / ialltoallv / wait
static _Atomic uint64_t calls[6];

void ep_moe_probe_reset(void) {
  for (int i = 0; i < 6; ++i) atomic_store(&calls[i], 0);
}
uint64_t ep_moe_probe_count(int i) { return atomic_load(&calls[i]); }

static _Atomic int64_t log_budget = -1;  // -1: read XCPU_EP_PROBE_LOG once
static _Atomic uint64_t logged = 0;

static int64_t take_log_budget(void) {
  int64_t budget = atomic_load(&log_budget);
  if (budget >= 0) return budget;
  const char* value = getenv("XCPU_EP_PROBE_LOG");
  budget = (value != NULL && *value != '\0') ? atoll(value) : 0;
  atomic_store(&log_budget, budget);
  return budget;
}

static void log_line(const char* text) {
  int rank = 0;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  uint64_t index = atomic_fetch_add(&logged, 1) + 1;
  fprintf(stderr, "[ep-probe rank=%d line=%llu] %s\n", rank,
          (unsigned long long)index, text);
}

int MPI_Allreduce(const void* send, void* recv, int n, MPI_Datatype type,
                  MPI_Op op, MPI_Comm comm) {
  atomic_fetch_add(&calls[0], 1);
  return PMPI_Allreduce(send, recv, n, type, op, comm);
}
int MPI_Alltoall(const void* send, int sn, MPI_Datatype st,
                 void* recv, int rn, MPI_Datatype rt, MPI_Comm comm) {
  atomic_fetch_add(&calls[1], 1);
  int64_t budget = take_log_budget();
  if (sn == 1 && st == MPI_INT && (int64_t)atomic_load(&logged) < budget) {
    int size = 0;
    PMPI_Comm_size(comm, &size);
    const int* counts = (const int*)send;
    char text[2048];
    int offset = snprintf(text, sizeof text, "alltoall sendcounts=[");
    for (int r = 0; r < size && offset < (int)sizeof text - 24; ++r)
      offset += snprintf(text + offset, sizeof text - offset, "%s%d",
                         r ? "," : "", counts[r]);
    snprintf(text + offset, sizeof text - offset, "]");
    log_line(text);
  }
  return PMPI_Alltoall(send, sn, st, recv, rn, rt, comm);
}
int MPI_Alltoallv(const void* send, const int* sn, const int* sd, MPI_Datatype st,
                  void* recv, const int* rn, const int* rd, MPI_Datatype rt,
                  MPI_Comm comm) {
  atomic_fetch_add(&calls[2], 1);
  int64_t budget = take_log_budget();
  if ((int64_t)atomic_load(&logged) < budget) {
    int size = 0;
    PMPI_Comm_size(comm, &size);
    int nz_send = 0, nz_recv = 0;
    for (int r = 0; r < size; ++r) {
      nz_send += sn[r] != 0;
      nz_recv += rn[r] != 0;
    }
    char text[128];
    snprintf(text, sizeof text, "alltoallv nz_send=%d/%d nz_recv=%d/%d",
             nz_send, size, nz_recv, size);
    log_line(text);
  }
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
