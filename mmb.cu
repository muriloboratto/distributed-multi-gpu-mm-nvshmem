#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define TILE_DIM 32

extern void ABMultiply(double *a, double *b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w);

static int valid_library(char c)
{
  return c == 'M' || c == 'C' || c == 'N' || c == 'S';
}

int main(int argc, char *argv[])
{
  double start_time = 0, stop_time = 0, elapsed_time;
  int myRank, nRanks;
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;

  if (argc < 4) 
  {
    fprintf(stderr,
      "Usage: %s <deviceId> <matrix_size> <ABC libraries>\n"
      "  M=MPI, C=CUDA-Aware MPI, N=NCCL, S=NVSHMEM\n"
      "  Example: %s 0 8192 SNS\n", argv[0], argv[0]);
    return EXIT_FAILURE;
  }

  /* ---------------------------------------------------------------------- */
  /* 1. MPI initialization                                                  */
  /* ---------------------------------------------------------------------- */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Get_processor_name(name, &resultlen);

  int deviceId   = atoi(argv[1]);
  int matrix_size = atoi(argv[2]);

  char communication_libraries[4] = {0,0,0,0};
  strncpy(communication_libraries, argv[3], 3);
  communication_libraries[3] = '\0';

  if (strlen(communication_libraries) != 3 || !valid_library(communication_libraries[0]) || !valid_library(communication_libraries[1]) || !valid_library(communication_libraries[2])) 
  {
    if (myRank == 0)
      fprintf(stderr,"Invalid communication string. Use exactly 3 characters from M,C,N,S.\n");

    MPI_Finalize();

    return EXIT_FAILURE;
  }

  if (matrix_size % nRanks != 0) 
  {
    if (myRank == 0)
      fprintf(stderr,"matrix_size must be divisible by nRanks in this version.\n");
    
    MPI_Finalize();
    
    return EXIT_FAILURE;
  }

  /* Select CUDA device before initializing NVSHMEM. */
  cudaSetDevice(deviceId);

  /* ---------------------------------------------------------------------- */
  /* 2. NVSHMEM initialization                                              */
  /* ---------------------------------------------------------------------- */
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int myPE = nvshmem_my_pe();
  int nPEs = nvshmem_n_pes();

  if (myPE != myRank || nPEs != nRanks) 
  {
    fprintf(stderr, "Rank/PE mismatch: MPI rank=%d/%d, NVSHMEM PE=%d/%d\n", myRank,nRanks,myPE,nPEs);
    nvshmem_finalize();
  
    MPI_Finalize();
  
    return EXIT_FAILURE;
  }

  /* ---------------------------------------------------------------------- */
  /* 3. NCCL initialization                                                 */
  /* ---------------------------------------------------------------------- */
  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;

  cudaStreamCreate(&s);

  if (myRank == 0)
    ncclGetUniqueId(&id);

  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, nRanks, id, myRank);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  printf("mmb node=%s MPI=%d/%d NVSHMEM=%d/%d device=%d:%s libraries=%s\n", name,myRank,nRanks,myPE,nPEs,deviceId,deviceProp.name,communication_libraries);

  /* ---------------------------------------------------------------------- */
  /* 4. Matrix dimensions                                                   */
  /* ---------------------------------------------------------------------- */
  const int m  = matrix_size;
  const int n  = matrix_size;
  const int k  = matrix_size;
  const int w  = TILE_DIM;
  const int mi = matrix_size / nRanks;

  const size_t bytes_A  = (size_t)m  * k * sizeof(double);
  const size_t bytes_B  = (size_t)k  * n * sizeof(double);
  const size_t bytes_C  = (size_t)m  * n * sizeof(double);
  const size_t bytes_lA = (size_t)mi * k * sizeof(double);
  const size_t bytes_lC = (size_t)mi * n * sizeof(double);

  /* ---------------------------------------------------------------------- */
  /* 5. Traditional host/device buffers for MPI/CUDA-Aware MPI/NCCL        */
  /* ---------------------------------------------------------------------- */
  double *A  = (double *)malloc(bytes_A);
  double *B  = (double *)calloc((size_t)k*n, sizeof(double));
  double *C  = (double *)malloc(bytes_C);
  double *lA = (double *)calloc((size_t)mi*k, sizeof(double));
  double *lC = (double *)calloc((size_t)mi*n, sizeof(double));

  if (!A || !B || !C || !lA || !lC) {
    fprintf(stderr,"Host allocation failed on rank %d\n", myRank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  double *d_A, *d_B, *d_C, *d_lA, *d_lC;
  cudaMalloc((void **)&d_A,  bytes_A);
  cudaMalloc((void **)&d_B,  bytes_B);
  cudaMalloc((void **)&d_C,  bytes_C);
  cudaMalloc((void **)&d_lA, bytes_lA);
  cudaMalloc((void **)&d_lC, bytes_lC);

  /* ---------------------------------------------------------------------- */
  /* 6. NVSHMEM symmetric GPU buffers                                       */
  /* ---------------------------------------------------------------------- */
  double *s_A  = (double *)nvshmem_malloc(bytes_A);
  double *s_B  = (double *)nvshmem_malloc(bytes_B);
  double *s_C  = (double *)nvshmem_malloc(bytes_C);
  double *s_lA = (double *)nvshmem_malloc(bytes_lA);
  double *s_lC = (double *)nvshmem_malloc(bytes_lC);

  if (!s_A || !s_B || !s_C || !s_lA || !s_lC) 
  {
    fprintf(stderr,"NVSHMEM symmetric allocation failed on PE %d\n", myPE);
    nvshmem_global_exit(EXIT_FAILURE);
  }

  /* ---------------------------------------------------------------------- */
  /* 7. Initial values                                                      */
  /* ---------------------------------------------------------------------- */
  if (myRank == 0) 
  {
    for (size_t i = 0; i < (size_t)m*k; ++i) A[i] = 1.0;
    for (size_t i = 0; i < (size_t)k*n; ++i) B[i] = 2.0;

    /* Standard CUDA buffers */
    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

    /* PE 0's symmetric buffers */
    cudaMemcpy(s_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(s_B, B, bytes_B, cudaMemcpyHostToDevice);
  }

  /* Make sure PE 0 has finished initializing s_A/s_B before remote gets. */
  nvshmem_barrier_all();

  const int loop_count = 10;
  MPI_Barrier(MPI_COMM_WORLD);

  start_time = MPI_Wtime();

  for (int iter = 1; iter <= loop_count; ++iter) 
  {

    /* ==================================================================== */
    /* 8. A: distribute rows                                                */
    /* ==================================================================== */
    switch (communication_libraries[0]) 
    {
      case 'M':
        if (myRank == 0)
          cudaMemcpy(A, d_A, bytes_A, cudaMemcpyDeviceToHost);

        MPI_Scatter(A, mi*k, MPI_DOUBLE, lA, mi*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cudaMemcpy(d_lA, lA, bytes_lA, cudaMemcpyHostToDevice);
        break;

      case 'C':
        MPI_Scatter(d_A, mi*k, MPI_DOUBLE, d_lA, mi*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        break;

      case 'N':
        /* NCCL has no Scatter: broadcast all A, then select local row block. */
        ncclBroadcast(d_A, d_A, (size_t)m*k,ncclDouble, 0, comm, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(d_lA, d_A + (size_t)myRank*mi*k, bytes_lA, cudaMemcpyDeviceToDevice);
        break;

      case 'S':
        /* One-sided read: each PE pulls only its own block from PE 0. */
        nvshmem_double_get(s_lA, s_A + (size_t)myPE*mi*k, (size_t)mi*k, 0);
        //nvshmem_barrier_all();
        break;
    }

    /* ==================================================================== */
    /* 9. B: broadcast entire matrix                                        */
    /* ==================================================================== */
    switch (communication_libraries[1]) 
    {
      case 'M':
        if (myRank == 0)
          cudaMemcpy(B, d_B, bytes_B, cudaMemcpyDeviceToHost);

        MPI_Bcast(B, k*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myRank != 0)
          cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
        
        break;

      case 'C':
        MPI_Bcast(d_B, k*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        break;

      case 'N':
        ncclBroadcast(d_B, d_B, (size_t)k*n, ncclDouble, 0, comm, s);
        cudaStreamSynchronize(s);
        break;

      case 'S':
        /* Each non-root PE pulls B directly from PE 0's symmetric buffer. */
        if (myPE != 0)
          nvshmem_double_get(s_B, s_B, (size_t)k*n, 0);
        
        //nvshmem_barrier_all();
        break;
    }

    /* ==================================================================== */
    /* 10. Local GPU multiplication                                         */
    /* ==================================================================== */
    double *kernel_A = (communication_libraries[0] == 'S') ? s_lA : d_lA;
    double *kernel_B = (communication_libraries[1] == 'S') ? s_B  : d_B;
    double *kernel_C = (communication_libraries[2] == 'S') ? s_lC : d_lC;

    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    ABMultiply(kernel_A, kernel_B, kernel_C, mi, n, k, lda, ldb, ldc, w);

    /* Ensure the CUDA kernel has completed before communicating C. */
    cudaDeviceSynchronize();

    /* ==================================================================== */
    /* 11. C: collect partial results                                       */
    /* ==================================================================== */
    switch (communication_libraries[2]) 
    {
      case 'M':
        cudaMemcpy(lC, d_lC, bytes_lC, cudaMemcpyDeviceToHost);

        MPI_Gather(lC, mi*n, MPI_DOUBLE, C, mi*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myRank == 0)
          cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);
        break;

      case 'C':
        MPI_Gather(d_lC, mi*n, MPI_DOUBLE, d_C, mi*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        break;

      case 'N':
        ncclAllGather(d_lC, d_C, (size_t)mi*n, ncclDouble, comm, s);
        cudaStreamSynchronize(s);
        break;

      case 'S':
        /* Make every PE's local C block visible before PE 0 pulls them. */
        nvshmem_barrier_all();

        if (myPE == 0) 
        {
          for (int pe = 0; pe < nPEs; ++pe) 
          {
            nvshmem_double_get(s_C + (size_t)pe*mi*n, s_lC, (size_t)mi*n, pe);
          }
        }

        //nvshmem_barrier_all();
        break;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  stop_time = MPI_Wtime();

  elapsed_time = stop_time - start_time;
  double avg_time_per_transfer = elapsed_time / (double)loop_count;

  if (myRank == 0) 
     printf("\n\n mmb myRank=%d \t Libraries=%s \t RESULT: Matrix size: %d \t Time (seconds): %8.3f \n\n", myRank, communication_libraries, matrix_size, avg_time_per_transfer);

  /* ---------------------------------------------------------------------- */
  /* 12. Cleanup                                                            */
  /* ---------------------------------------------------------------------- */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_lA);
  cudaFree(d_lC);

  free(A);
  free(B);
  free(C);
  free(lA);
  free(lC);

  nvshmem_free(s_A);
  nvshmem_free(s_B);
  nvshmem_free(s_C);
  nvshmem_free(s_lA);
  nvshmem_free(s_lC);

  ncclCommDestroy(comm);
  cudaStreamDestroy(s);

  nvshmem_finalize();
  MPI_Finalize();

  return 0;

}
