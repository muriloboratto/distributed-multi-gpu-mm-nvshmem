#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/*****************************************************************************/

int main(int argc, char *argv[])
{
    double start_time = 0.0, stop_time = 0.0;
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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Get_processor_name(name, &resultlen);

    const int deviceId    = atoi(argv[1]);
    const int matrix_size = atoi(argv[2]);

    char communication_libraries[4] = {0, 0, 0, 0};
    strncpy(communication_libraries, argv[3], 3);
    communication_libraries[3] = '\0';

    if (strlen(communication_libraries) != 3 ||
        !valid_library(communication_libraries[0]) ||
        !valid_library(communication_libraries[1]) ||
        !valid_library(communication_libraries[2])) 
    {
        
        if (myRank == 0)
            fprintf(stderr,"Invalid communication string. Use exactly 3 characters from M,C,N,S.\n");
        
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (matrix_size % nRanks != 0) 
    {
        if (myRank == 0)
            fprintf(stderr, "matrix_size must be divisible by nRanks in this version.\n");
        
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int uses_nvshmem = (strchr(communication_libraries, 'S') != NULL);
    const int uses_nccl    = (strchr(communication_libraries, 'N') != NULL);

    const char libA = communication_libraries[0];
    const char libB = communication_libraries[1];
    const char libC = communication_libraries[2];

    cudaSetDevice(deviceId);

    /* ------------------------------------------------------------------ */
    /* NVSHMEM: initialize only when at least one operation uses 'S'.     */
    /* ------------------------------------------------------------------ */

    int myPE = myRank;
    int nPEs = nRanks;

    if (uses_nvshmem) 
    {
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &mpi_comm;

        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        myPE = nvshmem_my_pe();
        nPEs = nvshmem_n_pes();

        if (myPE != myRank || nPEs != nRanks) 
        {
            fprintf(stderr, "Rank/PE mismatch: MPI rank=%d/%d, NVSHMEM PE=%d/%d\n", myRank, nRanks, myPE, nPEs);
            nvshmem_finalize();
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    /* ------------------------------------------------------------------ */
    /* NCCL: initialize only when at least one operation uses 'N'.        */
    /* ------------------------------------------------------------------ */

    ncclComm_t comm = NULL;
    cudaStream_t nccl_stream = NULL;

    if (uses_nccl) 
    {
        ncclUniqueId id;

        cudaStreamCreate(&nccl_stream);

        if (myRank == 0)
            ncclGetUniqueId(&id);

        MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclCommInitRank(&comm, nRanks, id, myRank);
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    printf("mmb node=%s MPI=%d/%d NVSHMEM=%d/%d device=%d:%s libraries=%s\n", name, myRank, nRanks, myPE, nPEs, deviceId, deviceProp.name, communication_libraries);

    /* ------------------------------------------------------------------ */
    /* Matrix dimensions                                                  */
    /* ------------------------------------------------------------------ */

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

    /* ------------------------------------------------------------------ */
    /* Host buffers: needed only by conventional MPI ('M').               */
    /* ------------------------------------------------------------------ */

    double *A = NULL, *B = NULL, *C = NULL, *lA = NULL, *lC = NULL;

    if (libA == 'M') 
    {
        A  = (double *)malloc(bytes_A);
        lA = (double *)malloc(bytes_lA);

        if (!A || !lA) 
        {
            fprintf(stderr, "Host allocation for A failed on rank %d\n", myRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (libB == 'M') 
    {
        B = (double *)malloc(bytes_B);
        
        if (!B) 
        {
            fprintf(stderr, "Host allocation for B failed on rank %d\n", myRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (libC == 'M') 
    {
        C  = (double *)malloc(bytes_C);
        lC = (double *)malloc(bytes_lC);

        if (!C || !lC) 
        {
            fprintf(stderr, "Host allocation for C failed on rank %d\n", myRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Standard CUDA buffers: allocate only for non-NVSHMEM operations.   */
    /* ------------------------------------------------------------------ */

    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    double *d_lA = NULL, *d_lC = NULL;

    if (libA != 'S') 
    {
        cudaMalloc((void **)&d_A, bytes_A);
        cudaMalloc((void **)&d_lA, bytes_lA);
    }

    if (libB != 'S')
        cudaMalloc((void **)&d_B, bytes_B);

    if (libC != 'S') 
    {
        cudaMalloc((void **)&d_C, bytes_C);
        cudaMalloc((void **)&d_lC, bytes_lC);
    }

    /* ------------------------------------------------------------------ */
    /* NVSHMEM symmetric buffers: allocate only for operations using 'S'. */
    /* ------------------------------------------------------------------ */

    double *s_A = NULL, *s_B = NULL, *s_C = NULL;
    double *s_lA = NULL, *s_lC = NULL;

    if (libA == 'S') 
    {
        s_A  = (double *)nvshmem_malloc(bytes_A);
        s_lA = (double *)nvshmem_malloc(bytes_lA);

        if (!s_A || !s_lA) 
        {
            fprintf(stderr, "NVSHMEM symmetric allocation for A failed on PE %d\n", myPE);
            nvshmem_global_exit(EXIT_FAILURE);
        }
    }

    if (libB == 'S') 
    {
        s_B = (double *)nvshmem_malloc(bytes_B);

        if (!s_B) 
        {
            fprintf(stderr, "NVSHMEM symmetric allocation for B failed on PE %d\n", myPE);
            nvshmem_global_exit(EXIT_FAILURE);
        }
    }

    if (libC == 'S') 
    {
        s_C  = (double *)nvshmem_malloc(bytes_C);
        s_lC = (double *)nvshmem_malloc(bytes_lC);

        if (!s_C || !s_lC) 
        {
            fprintf(stderr, "NVSHMEM symmetric allocation for C failed on PE %d\n", myPE);
            nvshmem_global_exit(EXIT_FAILURE);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Initial values: initialize only the storage selected by A and B.   */
    /* ------------------------------------------------------------------ */

    if (myRank == 0) 
    {
        double *tmp_A = (double *)malloc(bytes_A);
        double *tmp_B = (double *)malloc(bytes_B);

        if (!tmp_A || !tmp_B) 
        {
            fprintf(stderr, "Initialization host allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (size_t i = 0; i < (size_t)m * k; ++i)
            tmp_A[i] = 1.0;

        for (size_t i = 0; i < (size_t)k * n; ++i)
            tmp_B[i] = 2.0;

        if (libA == 'S')
            cudaMemcpy(s_A, tmp_A, bytes_A, cudaMemcpyHostToDevice);
        else
            cudaMemcpy(d_A, tmp_A, bytes_A, cudaMemcpyHostToDevice);

        if (libB == 'S')
            cudaMemcpy(s_B, tmp_B, bytes_B, cudaMemcpyHostToDevice);
        else
            cudaMemcpy(d_B, tmp_B, bytes_B, cudaMemcpyHostToDevice);

        free(tmp_A);
        free(tmp_B);
    }

    if (uses_nvshmem)
        nvshmem_barrier_all();

    const int loop_count = 10;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int iter = 1; iter <= loop_count; ++iter) 
    {

        /* ================================================================ */
        /* A: distribute rows                                               */
        /* ================================================================ */

        switch (libA) 
        {
            case 'M':
                if (myRank == 0)
                    cudaMemcpy(A, d_A, bytes_A, cudaMemcpyDeviceToHost);

                MPI_Scatter(A, mi * k, MPI_DOUBLE, lA, mi * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                cudaMemcpy(d_lA, lA, bytes_lA, cudaMemcpyHostToDevice);
                break;

            case 'C':
                MPI_Scatter(d_A, mi * k, MPI_DOUBLE, d_lA, mi * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                break;

            case 'N':
                ncclBroadcast(d_A, d_A, (size_t)m * k, ncclDouble, 0, comm, nccl_stream);
                cudaStreamSynchronize(nccl_stream);
                cudaMemcpy(d_lA, d_A + (size_t)myRank * mi * k, bytes_lA, cudaMemcpyDeviceToDevice);
                break;

            case 'S':
                nvshmem_double_get(s_lA, s_A + (size_t)myPE * mi * k, (size_t)mi * k, 0);
                break;
        }

        /* ================================================================ */
        /* B: broadcast entire matrix                                       */
        /* ================================================================ */

        switch (libB) 
        {
            case 'M':
                if (myRank == 0)
                    cudaMemcpy(B, d_B, bytes_B, cudaMemcpyDeviceToHost);

                MPI_Bcast(B, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if (myRank != 0)
                    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
                break;

            case 'C':
                MPI_Bcast(d_B, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                break;

            case 'N':
                ncclBroadcast(d_B, d_B, (size_t)k * n, ncclDouble, 0, comm, nccl_stream);
                cudaStreamSynchronize(nccl_stream);
                break;

            case 'S':
                if (myPE != 0)
                    nvshmem_double_get(s_B, s_B, (size_t)k * n, 0);
                break;
        }

        /* ================================================================ */
        /* Local GPU multiplication                                         */
        /* ================================================================ */

        double *kernel_A = (libA == 'S') ? s_lA : d_lA;
        double *kernel_B = (libB == 'S') ? s_B  : d_B;
        double *kernel_C = (libC == 'S') ? s_lC : d_lC;

        ABMultiply(kernel_A, kernel_B, kernel_C, mi, n, k, k, n, n, w);

        cudaGetLastError();
        cudaDeviceSynchronize();

        /* ================================================================ */
        /* C: collect partial results                                        */
        /* ================================================================ */
        switch (libC) 
        {
            case 'M':
                cudaMemcpy(lC, d_lC, bytes_lC, cudaMemcpyDeviceToHost);

                MPI_Gather(lC, mi * n, MPI_DOUBLE, C, mi * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if (myRank == 0)
                    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);
                break;

            case 'C':
                MPI_Gather(d_lC, mi * n, MPI_DOUBLE, d_C, mi * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                break;

            case 'N':
                ncclAllGather(d_lC, d_C, (size_t)mi * n, ncclDouble, comm, nccl_stream);

                cudaStreamSynchronize(nccl_stream);
                break;

            case 'S':
                nvshmem_barrier_all();

                if (myPE == 0) 
                {
                    for (int pe = 0; pe < nPEs; ++pe) 
                        nvshmem_double_get(s_C + (size_t)pe * mi * n, s_lC, (size_t)mi * n, pe);
                    
                }
                break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_time = MPI_Wtime();

    const double elapsed_time = stop_time - start_time;
    const double avg_time_per_transfer = elapsed_time / (double)loop_count;

    if (myRank == 0)
        printf("\n\nmmb myRank=%d\tLibraries=%s\t RESULT: Matrix size: %d\tTime (seconds): %8.3f\n\n", myRank, communication_libraries, matrix_size, avg_time_per_transfer);

    /* ------------------------------------------------------------------ */
    /* Cleanup: free only buffers that were actually allocated.           */
    /* ------------------------------------------------------------------ */
    
    if (d_A)  cudaFree(d_A);
    if (d_B)  cudaFree(d_B);
    if (d_C)  cudaFree(d_C);
    if (d_lA) cudaFree(d_lA);
    if (d_lC) cudaFree(d_lC);

    free(A);
    free(B);
    free(C);
    free(lA);
    free(lC);

    if (uses_nvshmem) 
    {
        if (s_A)  nvshmem_free(s_A);
        if (s_lA) nvshmem_free(s_lA);
        if (s_B)  nvshmem_free(s_B);
        if (s_C)  nvshmem_free(s_C);
        if (s_lC) nvshmem_free(s_lC);
    }

    if (uses_nccl) 
    {
        ncclCommDestroy(comm);
        cudaStreamDestroy(nccl_stream);
    }

    if (uses_nvshmem)
        nvshmem_finalize();

    MPI_Finalize();
    
    return 0;
}
