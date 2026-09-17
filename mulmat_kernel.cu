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
