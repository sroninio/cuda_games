#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that adds 1 to each element
__global__ void unite_step_kernel(float *d_array, int n, int solved_block_size) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_indx = (t_idx / solved_block_size) * solved_block_size;
    int pusher_indx = base_indx + solved_block_size - 1
    int pushed_indx = base_indx + solved_block_size + t_indx % solved_block_size;
    if (pushed_indx < n) {
        d_array[pushed_indx] += d_array[pusher_indx]
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    
    float *h_array = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)i;
    }
    
    float *d_array;
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);
    
    printf("Launching kernels with %d blocks and %d threads per block\n\n", 
           blocksPerGrid, threadsPerBlock);
    for (int solved_block_size = 1, solved_block_size < N; solved_block_size *= 2) {
        int threadsPerBlock = 256;
        int blocksPerGrid = ((N/2 + 1) + threadsPerBlock - 1) / threadsPerBlock;
        unite_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N, solved_block_size);
        cudaDeviceSynchronize();
    }

    printf("Finished kernels with %d blocks and %d threads per block\n\n", 
           blocksPerGrid, threadsPerBlock);

    int sum = 0;
    bool is_good = true;
    for (int i = 0; i < N; i++) {
        sum += i;
        if (h_array[i] != sum) {is_good = False; break;}
    }
    
    if (is_good) {
        printf("✓ Test PASSED!\n");
    } else {
        printf("✗ Test FAILED!\n");
    }

    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    free(h_array);
    return 0;
}

