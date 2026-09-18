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

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication CUDA kernels
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

__global__ void simpleMultiply(double *a, double *b, double *c, int n)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) 
    {
        double sum = 0.0;

        for (int i = 0; i < n; ++i)
            sum += a[row * n + i] * b[i * n + col];

        c[row * n + col] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tiled matrix multiplication kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void sharedABMultiply(double *a, double *b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w)
{
    extern __shared__ char shared_memory_space[];

    double *aTile = (double *)shared_memory_space;
    double *bTile = (double *)&shared_memory_space[w * w * sizeof(double)];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    const int num_tiles = (k + w - 1) / w;

    for (int tile = 0; tile < num_tiles; ++tile) 
    {
        const int a_col = tile * w + threadIdx.x;
        const int b_row = tile * w + threadIdx.y;

        if (row < m && a_col < k)
            aTile[threadIdx.y * w + threadIdx.x] =
                a[row * lda + a_col];
        else
            aTile[threadIdx.y * w + threadIdx.x] = 0.0;

        if (b_row < k && col < n)
            bTile[threadIdx.y * w + threadIdx.x] =
                b[b_row * ldb + col];
        else
            bTile[threadIdx.y * w + threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < w; ++i)
            sum += aTile[threadIdx.y * w + i] * bTile[i * w + threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        c[row * ldc + col] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// ABMultiply Sync.
////////////////////////////////////////////////////////////////////////////////

void ABMultiply(double *a, double *b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w)
{
    dim3 block(w, w);
    dim3 grid((n + w - 1) / w, (m + w - 1) / w);

    sharedABMultiply<<< grid, block, 2 * w * w * sizeof(double)>>>(a, b, c, m, n, k, lda, ldb, ldc, w);
}


////////////////////////////////////////////////////////////////////////////////
// Kernel Overlaping Comunication/COmputation.
// Computes C += A[:, k_offset:k_offset+k_chunk] * B_chunk.
////////////////////////////////////////////////////////////////////////////////

__global__ void sharedABMultiplyAccumulate(const double *a,
                                           const double *b_chunk,
                                           double *c,
                                           int m, int n, int k_total,
                                           int k_offset, int k_chunk,
                                           int lda, int ldb, int ldc, int w)
{
    extern __shared__ char shared_memory_space[];
    double *aTile = (double *)shared_memory_space;
    double *bTile = (double *)&shared_memory_space[w * w * sizeof(double)];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    double partial = 0.0;

    const int num_tiles = (k_chunk + w - 1) / w;
    
    for (int tile = 0; tile < num_tiles; ++tile) 
    {
        const int kk_a = tile * w + threadIdx.x;
        const int kk_b = tile * w + threadIdx.y;

        if (row < m && kk_a < k_chunk && (k_offset + kk_a) < k_total)
            aTile[threadIdx.y * w + threadIdx.x] =
                a[(size_t)row * lda + k_offset + kk_a];
        else
            aTile[threadIdx.y * w + threadIdx.x] = 0.0;

        if (kk_b < k_chunk && col < n)
            bTile[threadIdx.y * w + threadIdx.x] =
                b_chunk[(size_t)kk_b * ldb + col];
        else
            bTile[threadIdx.y * w + threadIdx.x] = 0.0;

        __syncthreads();
        for (int i = 0; i < w; ++i)
            partial += aTile[threadIdx.y * w + i] * bTile[i * w + threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n)
        c[(size_t)row * ldc + col] += partial;
}

////////////////////////////////////////////////////////////////////////////////
// ABMultiply Assync.
////////////////////////////////////////////////////////////////////////////////

void ABMultiplyAccumulateAsync(const double *a, const double *b_chunk, double *c,
                               int m, int n, int k_total,
                               int k_offset, int k_chunk,
                               int lda, int ldb, int ldc, int w,
                               cudaStream_t stream)
{
    dim3 block(w, w);
    dim3 grid((n + w - 1) / w, (m + w - 1) / w);
    sharedABMultiplyAccumulate<<<grid, block, 2 * w * w * sizeof(double), stream>>>(a, b_chunk, c, m, n, k_total, k_offset, k_chunk,
                                                                                    lda, ldb, ldc, w);
}

////////////////////////////////////////////////////////////////////////////////
// validate_C_kernel
////////////////////////////////////////////////////////////////////////////////


__global__ void validate_C_kernel(const double *C,
                       size_t num_elements,
                       double expected,
                       double abs_tol,
                       double rel_tol,
                       unsigned long long *error_count,
                       double *max_abs_error,
                       double *max_rel_error)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements)
        return;

    double value = C[idx];
    double abs_error = fabs(value - expected);
    double rel_error = 0.0;

    if (fabs(expected) > 0.0)
        rel_error = abs_error / fabs(expected);

    if ((abs_error > abs_tol) && (rel_error > rel_tol))
        atomicAdd(error_count, 1ULL);

    unsigned long long *addr_abs = (unsigned long long *)max_abs_error;
    unsigned long long old_abs = *addr_abs;
    unsigned long long assumed_abs;

    do
    {
        assumed_abs = old_abs;

        if (__longlong_as_double(assumed_abs) >= abs_error)
            break;

        old_abs = atomicCAS(addr_abs,
                            assumed_abs,
                            __double_as_longlong(abs_error));

    } while (assumed_abs != old_abs);

    unsigned long long *addr_rel = (unsigned long long *)max_rel_error;
    unsigned long long old_rel = *addr_rel;
    unsigned long long assumed_rel;

    do
    {
        assumed_rel = old_rel;

        if (__longlong_as_double(assumed_rel) >= rel_error)
            break;

        old_rel = atomicCAS(addr_rel,
                            assumed_rel,
                            __double_as_longlong(rel_error));

    } while (assumed_rel != old_rel);
}

////////////////////////////////////////////////////////////////////////////////
// validate_matrix_C function
////////////////////////////////////////////////////////////////////////////////

int validate_matrix_C(const double *d_C,
                      int rows,
                      int cols,
                      double expected,
                      double abs_tol,
                      double rel_tol,
                      int rank)
{
    size_t num_elements = (size_t)rows * (size_t)cols;

    unsigned long long *d_error_count;
    double *d_max_abs_error;
    double *d_max_rel_error;

    cudaMalloc((void **)&d_error_count, sizeof(unsigned long long));
    cudaMalloc((void **)&d_max_abs_error, sizeof(double));
    cudaMalloc((void **)&d_max_rel_error, sizeof(double));

    cudaMemset(d_error_count, 0, sizeof(unsigned long long));
    cudaMemset(d_max_abs_error, 0, sizeof(double));
    cudaMemset(d_max_rel_error, 0, sizeof(double));

    const int threads = 256;
    size_t blocks = (num_elements + threads - 1) / threads;

    validate_C_kernel<<<blocks, threads>>>(d_C,
                                           num_elements,
                                           expected,
                                           abs_tol,
                                           rel_tol,
                                           d_error_count,
                                           d_max_abs_error,
                                           d_max_rel_error);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Rank %d: validation kernel error: %s\n",
                rank,
                cudaGetErrorString(err));

        cudaFree(d_error_count);
        cudaFree(d_max_abs_error);
        cudaFree(d_max_rel_error);
        return 0;
    }

    cudaDeviceSynchronize();

    unsigned long long error_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    cudaMemcpy(&error_count,
               d_error_count,
               sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(&max_abs_error,
               d_max_abs_error,
               sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(&max_rel_error,
               d_max_rel_error,
               sizeof(double),
               cudaMemcpyDeviceToHost);

    if (rank == 0)
    {
        printf("\n");
        printf("============================================================\n");
        printf("NUMERICAL VALIDATION\n");
        printf("============================================================\n");
        printf("Matrix dimensions  : %d x %d\n", rows, cols);
        printf("Expected C[i,j]    : %.12f\n", expected);
        printf("Elements checked   : %llu\n", (unsigned long long)num_elements);
        printf("Invalid elements   : %llu\n", error_count);
        printf("Max absolute error : %.6e\n", max_abs_error);
        printf("Max relative error : %.6e\n", max_rel_error);
        printf("Absolute tolerance : %.6e\n", abs_tol);
        printf("Relative tolerance : %.6e\n", rel_tol);

        if (error_count == 0)
            printf("Result             : PASS\n");
        else
            printf("Result             : FAIL\n");

        printf("============================================================\n");
    }

    cudaFree(d_error_count);
    cudaFree(d_max_abs_error);
    cudaFree(d_max_rel_error);

    return (error_count == 0);
}
