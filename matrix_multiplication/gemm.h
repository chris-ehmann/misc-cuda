// Generic GEMM kernel. No tiling, caching, etc.

__global__
void gemm( int * A, int * B, int * C, int N, int M, int P )
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < M && j < P) {
        int sum = 0;
        for(int cur = 0; cur < N; cur++)
        {
            sum += A[i * N + cur] * B[j + cur * M];
        }
        __syncthreads();
        C[i * P + j] = sum;
    }

    return;
}