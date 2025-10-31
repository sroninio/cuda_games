#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char *argv[]) {
    // Check command-line arguments
    if (argc != 4) {
        printf("Usage: %s <N> <NUM_REQUESTS> <NUM_STREAMS>\n", argv[0]);
        printf("  N             - Array size\n");
        printf("  NUM_REQUESTS  - Number of arrays to process\n");
        printf("  NUM_STREAMS   - Number of CUDA streams\n");
        return 1;
    }
    
    // Parse command-line arguments
    const int N = atoi(argv[1]);
    const int NUM_REQUESTS = atoi(argv[2]);
    const int NUM_STREAMS = atoi(argv[3]);
    
    printf("Running with N=%d, NUM_REQUESTS=%d, NUM_STREAMS=%d\n\n", N, NUM_REQUESTS, NUM_STREAMS);
    
    const size_t bytes = N * sizeof(int);
    cudaStream_t streams[NUM_STREAMS];

    for (int iter = 0; iter < NUM_STREAMS; iter++){
        cudaStreamCreate(&(streams[iter]));
    }

    int **h_arrays = (int**)malloc(NUM_REQUESTS * sizeof(int *));
    int **d_arrays = (int**)malloc(NUM_REQUESTS * sizeof(int *)); 
    clock_t prepare_start = clock();
    for (int iter = 0; iter < NUM_REQUESTS; iter++){
        h_arrays[iter] = (int*)malloc(bytes);
        prepareArray(h_arrays[iter], N);
        cudaMalloc(d_arrays + iter, bytes);
        cudaMemcpy(d_arrays[iter], h_arrays[iter], bytes, cudaMemcpyHostToDevice);
    }
    clock_t prepare_end = clock();
    double prepareTime = ((double)(prepare_end - prepare_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Prepare time: %.3f ms\n", prepareTime);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Launching kernels\n");
    cudaEventRecord(start);
    for (int req = 0; req < NUM_REQUESTS; req++) {
        for (int solved_block_size = 1; solved_block_size < N; solved_block_size *= 2) {
            int threadsPerBlock = 256;
            int blocksPerGrid = ((N/2 + 1) + threadsPerBlock - 1) / threadsPerBlock;
            unite_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[req % NUM_STREAMS]>>>(d_arrays[req], N, solved_block_size);
        }
    }    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("Finished kernels\n");
    printf("Kernel loop execution time: %.3f ms\n\n", kernelTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    clock_t verify_start = clock();
    for (int iter = 0; iter < NUM_REQUESTS; iter++){
        cudaMemcpy(h_arrays[iter], d_arrays[iter], bytes, cudaMemcpyDeviceToHost);
        verifyArray(h_arrays[iter], N); 
        cudaFree(d_arrays[iter]);
        free(h_arrays[iter]);
    } 
    clock_t verify_end = clock();
    double verifyTime = ((double)(verify_end - verify_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Verification time: %.3f ms\n", verifyTime);
    
    free(h_arrays);  
    free(d_arrays);

    for (int iter = 0; iter < NUM_STREAMS; iter++){
        cudaStreamDestroy(streams[iter]);
    }

    return 0;
}
