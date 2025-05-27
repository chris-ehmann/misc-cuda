// Tensor cores can really speed things up. 
// The GEMM kernel in the other file takes about 
// ~10 seconds to compute multiplication
// between two 16384x16384 matrices. 

// This kernel takes 2/5ths of a second.

const int tensor_int8_K = 16;
const int tensor_int8_M = 16;
const int tensor_int8_N = 16;

__global__
void tensor_int8_gemm( unsigned char * A, unsigned char * B, int * C, int M, int N, int K )
{
    int w_M = ( blockIdx.x * blockDim.x + threadIdx.x ) / 32;
    int w_N = blockIdx.y * blockDim.y + threadIdx.y;
    int c_row = w_M * 16;
    int c_column = w_N * 16;

    nvcuda::wmma::fragment< nvcuda::wmma::matrix_a, tensor_int8_M, tensor_int8_N, tensor_int8_K, unsigned char, nvcuda::wmma::col_major > a_fragment;
    nvcuda::wmma::fragment< nvcuda::wmma::matrix_b, tensor_int8_M, tensor_int8_N, tensor_int8_K, unsigned char, nvcuda::wmma::col_major > b_fragment;
    nvcuda::wmma::fragment< nvcuda::wmma::accumulator, tensor_int8_M, tensor_int8_N, tensor_int8_K, int > c_fragment;

    nvcuda::wmma::fill_fragment( c_fragment, 0 );

    //Loop over shared dimension of A and B (i.e., K)

    for( int curr = 0; curr < K; curr += tensor_int8_K )
    {
      int a_row = w_M * tensor_int8_M;
      int b_column = w_N * tensor_int8_N;

      //Check bounds

      if ( a_row < M && curr < K && curr < K && b_column < N ) {
          nvcuda::wmma::load_matrix_sync( a_fragment, A + a_row + curr * M, M );
          nvcuda::wmma::load_matrix_sync( b_fragment, B + curr + b_column * K, K );
          nvcuda::wmma::mma_sync( c_fragment, a_fragment, b_fragment, c_fragment );
      }
    }

    //Store resulting fragment in correct spot in C

    if( c_row < M && c_column < N ) {
      nvcuda::wmma::store_matrix_sync( C + c_row + c_column * M, c_fragment, M, nvcuda::wmma::mem_col_major );
    }

    return;
}