////////////////////////////////////////////////////////////////////////////////
// simpleMultiply kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void simpleMultiply (double *a, double* b, double *c,int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n)
    {
    	double sum = 0.0;

    	for (int i = 0; i < n; i++)
        	sum += a[row * n + i] * b[i * n + col];

    	c[row * n + col] = sum;
  	}

}

////////////////////////////////////////////////////
//  sharedABMultiply kernel
///////////////////////////////////////////////////

__global__ void sharedABMultiply (double *a, double* b, double *c, int m, int n, int k, int lda, int ldb, int ldc, int w)
{
	extern __shared__ char shared_memory_space[];
	double *aTile = (double *) shared_memory_space;
	double *bTile = (double *) &(shared_memory_space[w * w * sizeof(double)]);

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	double sum = 0.0f;
	int num_tiles = k/w;
	int tile,i;
	
	c[row * ldc + col] = 0.;

	for (tile = 0; tile < num_tiles; tile++)
	{
		aTile[threadIdx.y * w + threadIdx.x] = a[row * lda  +	tile * w    	+	threadIdx.x];
		bTile[threadIdx.y * w + threadIdx.x] = b[col		+	tile * w * ldb	+	threadIdx.y * ldb];

		__syncthreads();

		sum=0.;
		for (i = 0; i < w; i++)  
			sum += aTile[threadIdx.y * w + i] * bTile[i * w + threadIdx.x];

        __syncthreads();

		c[row * ldc + col] += sum;

	}

}

////////////////////////////////////////////////////
//  ABMultiply kernel
///////////////////////////////////////////////////

void ABMultiply (double *a, double *b, double *c, int m, int n, int k,int lda, int ldb,int ldc, int w)
{
       dim3 grid(n/w, m/w);
       dim3 block(w, w);

       sharedABMultiply <<< grid, block,2 * w * w * sizeof(double) >>> (a, b, c, m, n, k, lda, ldb, ldc, w);

       cudaDeviceSynchronize();

}
