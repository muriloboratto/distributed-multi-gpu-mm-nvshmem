/******************************************************************************
 *
 * Distributed Multi-GPU Matrix Multiplication Benchmark
 *
 * Description:
 *   Distributed matrix multiplication benchmark for evaluating different
 *   communication libraries in multi-GPU systems.
 *
 *   Supported communication libraries. The three-character argument specifies the communication library used
 *   by the benchmark. For example:
 *
 *     MMM = MPI
 *     CCC = CUDA-Aware MPI
 *     NNN = NCCL
 *     SSS = NVSHMEM
 *
 * Compilation:
 *
 *   [murilo.boratto@sdumont]$ make
 *
 * Execution:
 *
 *   Example using one node with four GPUs and one MPI process per GPU:
 *
 *   [murilo.boratto@sdumont]$ mpirun -np 1 ./mmb 0 2048 MMM \
 *                                  : -np 1 ./mmb 1 2048 MMM \
 *                                  : -np 1 ./mmb 2 2048 MMM \
 *                                  : -np 1 ./mmb 3 2048 MMM
 *
 *   Arguments:
 *
 *    ./mmb <device_id> <matrix_size> <libraries>
 *
 *   where:
 *
 *     device_id    = CUDA GPU device assigned to the MPI process
 *     matrix_size  = matrix dimension (e.g., 2048)
 *     libraries    = communication library combination (MMM, CCC, NNN, SSS)
 *
 *   In the example above:
 *
 *     MPI Rank 0 -> GPU 0
 *     MPI Rank 1 -> GPU 1
 *     MPI Rank 2 -> GPU 2
 *     MPI Rank 3 -> GPU 3
 *
 *   Therefore, four MPI processes are launched, each associated with one
 *   NVIDIA GPU.
 *
 ******************************************************************************/

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
#define DEFAULT_PIPELINE_K_CHUNK 2048

static int pipeline_k_chunk_from_env(int k)
{
    int chunk = DEFAULT_PIPELINE_K_CHUNK;
    const char *env = getenv("MM_PIPELINE_K_CHUNK");

    if (env && *env) 
    {
        const int requested = atoi(env);
        if (requested > 0)
            chunk = requested;
    }

    if (chunk > k)
        chunk = k;

    if (chunk >= TILE_DIM)
        chunk = (chunk / TILE_DIM) * TILE_DIM;

    return (chunk > 0) ? chunk : k;
}

static void prefetch_B_chunk(double *dst,
                             const double *s_B,
                             int k_offset,
                             int k_chunk,
                             int n,
                             int source_pe,
                             int my_pe,
                             cudaStream_t stream)
{
    const double *src = s_B + (size_t)k_offset * n;
    const size_t count = (size_t)k_chunk * n;

    if (my_pe == source_pe) 
    {
        cudaMemcpyAsync(dst, src, count * sizeof(double),cudaMemcpyDeviceToDevice, stream);
    } else 
    {
        nvshmemx_double_get_nbi_on_stream(dst, src, count, source_pe, stream);
        nvshmemx_quiet_on_stream(stream);
    }
}

extern int validate_matrix_C(const double *d_C, int rows, int cols, double expected, double abs_tol, double rel_tol, int rank);

extern void ABMultiply(double *a, double *b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w);

extern void ABMultiplyAccumulateAsync(const double *a, const double *b_chunk, double *c,
                                      int m, int n, int k_total, int k_offset, int k_chunk,
                                      int lda, int ldb, int ldc, int w, cudaStream_t stream);

static int valid_library(char c)
{
    return c == 'M' || c == 'C' || c == 'N' || c == 'S';
}

/************************************************************************************************/

int main(int argc, char *argv[])
{
    double start_time = 0.0, stop_time = 0.0;
    int myRank, nRanks;
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlen;

    if (argc < 4) 
    {
        fprintf(stderr,"Usage: %s <deviceId> <matrix_size> <ABC libraries>\n"
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
            fprintf(stderr,"Invalid communication string. Use exactly 3 characters from M, C, N, S.\n");
        
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
    const int pipeline_k_chunk = pipeline_k_chunk_from_env(k);

    if (myRank == 0 && libB == 'S')
        printf("mmb NVSHMEM overlap: double buffering, K-chunk=%d, asynchronous GET/quiet on stream\n", pipeline_k_chunk);

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
    /* NVSHMEM B pipeline resources.                                      */
    /* ------------------------------------------------------------------ */

    double *b_stage[2] = {NULL, NULL};
    cudaStream_t comm_stream = NULL, compute_stream = NULL;
    cudaEvent_t b_ready[2] = {NULL, NULL};
    cudaEvent_t compute_done[2] = {NULL, NULL};

    if (libB == 'S')
    {
        const size_t stage_bytes = (size_t)pipeline_k_chunk * n * sizeof(double);

        cudaMalloc((void **)&b_stage[0], stage_bytes);
        cudaMalloc((void **)&b_stage[1], stage_bytes);
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

        for (int i = 0; i < 2; ++i) 
        {
            cudaEventCreateWithFlags(&b_ready[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming);
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
                /* Chunked transfer is performed below by the double-buffer pipeline. */
                break;
        }

        /* ================================================================ */
        /* Local GPU multiplication                                         */
        /* ================================================================ */

        double *kernel_A = (libA == 'S') ? s_lA : d_lA;
        double *kernel_B = (libB == 'S') ? s_B  : d_B;
        double *kernel_C = (libC == 'S') ? s_lC : d_lC;

        if (libB == 'S')
        {
            cudaMemsetAsync(kernel_C, 0, bytes_lC, compute_stream);

            const int num_chunks = (k + pipeline_k_chunk - 1) / pipeline_k_chunk;
            int buffer_used[2] = {0, 0};

            const int first_k = (k < pipeline_k_chunk) ? k : pipeline_k_chunk;
            
            prefetch_B_chunk(b_stage[0], s_B, 0, first_k, n, 0, myPE, comm_stream);
            
            cudaEventRecord(b_ready[0], comm_stream);
            
            buffer_used[0] = 1;

            for (int chunk = 0; chunk < num_chunks; ++chunk)
            {
                const int current  = chunk & 1;
                const int k_offset = chunk * pipeline_k_chunk;
                const int k_chunk  = ((k - k_offset) < pipeline_k_chunk) ? (k - k_offset) : pipeline_k_chunk;

                cudaStreamWaitEvent(compute_stream, b_ready[current], 0);

                ABMultiplyAccumulateAsync(kernel_A, b_stage[current], kernel_C,
                                          mi, n, k, k_offset, k_chunk,
                                          k, n, n, w, compute_stream);
                
                cudaEventRecord(compute_done[current], compute_stream);

                const int next_chunk = chunk + 1;

                if (next_chunk < num_chunks)
                {
                    const int next = 1 - current;
                    const int next_offset = next_chunk * pipeline_k_chunk;
                    const int next_k = ((k - next_offset) < pipeline_k_chunk)
                                     ? (k - next_offset) : pipeline_k_chunk;

                    if (buffer_used[next])
                        cudaStreamWaitEvent(comm_stream, compute_done[next], 0);

                    prefetch_B_chunk(b_stage[next], s_B, next_offset, next_k, n, 0, myPE, comm_stream);

                    cudaEventRecord(b_ready[next], comm_stream);

                    buffer_used[next] = 1;
                }
            }

            cudaStreamSynchronize(compute_stream);
        }
        else
        {
            ABMultiply(kernel_A, kernel_B, kernel_C, mi, n, k, k, n, n, w);
            cudaGetLastError();
            cudaDeviceSynchronize();
        }

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
    /* Numerical validation of the final matrix C on rank 0.              */
    /* A[i,j] = 1 and B[i,j] = 2, therefore C[i,j] = 2 * matrix_size.     */
    /* ------------------------------------------------------------------ */

    if (myRank == 0)
    {
        const double *validation_C = (libC == 'S') ? s_C : d_C;
        const double expected_C = 2.0 * (double)k;
        const double abs_tol = 1.0e-9;
        const double rel_tol = 1.0e-12;

        validate_matrix_C(validation_C, m, n, expected_C, abs_tol, rel_tol, myRank);
    }

    /* ------------------------------------------------------------------ */
    /* Cleanup: free only buffers that were actually allocated.           */
    /* ------------------------------------------------------------------ */
    
    if (libB == 'S')
    {
        cudaStreamSynchronize(comm_stream);
        cudaStreamSynchronize(compute_stream);
        
        for (int i = 0; i < 2; ++i) 
        {
            cudaEventDestroy(b_ready[i]);
            cudaEventDestroy(compute_done[i]);
            cudaFree(b_stage[i]);
        }
        
        cudaStreamDestroy(comm_stream);
        cudaStreamDestroy(compute_stream);
    }

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
