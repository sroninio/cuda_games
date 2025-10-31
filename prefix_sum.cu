#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA kernel that adds 1 to each element
__global__ void unite_step_kernel(int *d_array, int n, int solved_block_size) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_indx = (t_idx / solved_block_size) * (2 * solved_block_size);
    int pusher_indx = base_indx + solved_block_size - 1;
    int pushed_indx = base_indx + solved_block_size + t_idx % solved_block_size;
    if (pushed_indx < n) {
        d_array[pushed_indx] += d_array[pusher_indx];
    }
}

void prepareArray(int * arr,size_t n)
{
    for (int i = 0; i < n; i++) {
        arr[i] = (int)i;
    }
}

void verifyArray(int * arr, size_t n)
{
    int sum = 0;
    bool is_good = true;
    for (int i = 0; i < n; i++) {
        sum += i;
        if (arr[i] != sum) {is_good = false; break;}
    }
    if (is_good) {
        printf("✓ Test PASSED!\n");
    } else {
        printf("✗ Test FAILED!\n");
    }
}

int main() {
    const int N = 16;
    const size_t bytes = N * sizeof(int);

    int *h_array = (int*)malloc(bytes);
    prepareArray(h_array, N);

    int *d_array;
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Launching kernels\n");
    
    // Start timing the kernel loop
    cudaEventRecord(start);
    for (int solved_block_size = 1; solved_block_size < N; solved_block_size *= 2) {
        int threadsPerBlock = 256;
        int blocksPerGrid = ((N/2 + 1) + threadsPerBlock - 1) / threadsPerBlock;
        unite_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N, solved_block_size);
        cudaDeviceSynchronize();
        cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    }
    
    // Stop timing the kernel loop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("Finished kernels\n");
    printf("Kernel loop execution time: %.3f ms\n\n", kernelTime);

    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    
    // Time the verification
    clock_t verify_start = clock();
    verifyArray(h_array, N);
    clock_t verify_end = clock();
    double verifyTime = ((double)(verify_end - verify_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Verification time: %.3f ms\n", verifyTime);
    
    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_array);
    free(h_array);
    return 0;
}