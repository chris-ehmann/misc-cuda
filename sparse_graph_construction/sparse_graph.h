// Kernels to generate a sparse graph from an input list of m edges
// Sparse graph is represented in a Compressed Sparse Row (CSR) format,
// using implementation details given in this paper:
// https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf

struct Graph
{
  std::size_t n;
  std::size_t m;
  node_t * neighbours_start_at;
  node_t * neighbours;
};

__global__
void reduce( Graph g, edge_t const * edge_list, std::size_t m )
{
  int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(th_id < m) {
    edge_t edge = edge_list[th_id];
    int temp = atomicSub(&g.neighbours_start_at[edge.x], 1);
    g.neighbours[temp - 1] = edge.y;
    __syncthreads();
  }

  return;
}

__global__
void create_neighbours( Graph g, edge_t const * edge_list, std::size_t m )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(th_id < g.n)
    {
      g.neighbours_start_at[th_id] = 0;
    }
    __syncthreads();

    if(th_id < m)
    {
      g.neighbours[th_id] = 0;
      edge_t edge = edge_list[th_id];
      atomicAdd(&g.neighbours_start_at[edge.x], 1);
    }

    __syncthreads();

    if(th_id < g.n) {
      for(int j = 1; j < g.n; j <<= 1)
      {
        if(th_id >= j)
        {
          atomicAdd(&g.neighbours_start_at[th_id], g.neighbours_start_at[th_id - j]);
        }
      }
    }
    __syncthreads();

    return;
}

 void build_sparse_graph( Graph g, edge_t const * edge_list, std::size_t m )
 {
    std::size_t const block_size = 1024;
    std::size_t const num_blocks =  ( m + block_size - 1 ) / block_size;

    create_neighbours<<<num_blocks, block_size>>>(g, edge_list, m);
    cudaDeviceSynchronize();
    reduce<<<num_blocks, block_size>>>(g, edge_list, m);
    cudaDeviceSynchronize();

    return;
 }