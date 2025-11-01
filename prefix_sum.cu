#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
        //printf("✓ Test PASSED!\n");
    } else {
        //printf("✗ Test FAILED!\n");
    }
}

void solve(int N, cudaStream_t * pstream, int * d_array)
{
    for (int solved_block_size = 1; solved_block_size < N; solved_block_size *= 2) {
        int threadsPerBlock = 256;
        int blocksPerGrid = ((N/2 + 1) + threadsPerBlock - 1) / threadsPerBlock;
        if (pstream) {
            unite_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, *pstream>>>(d_array, N, solved_block_size);
        } else {
            unite_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N, solved_block_size); 
        }
    }    
}

int main(int argc, char *argv[]) {
    // Check command-line arguments
    if (argc != 6) {
        printf("Usage: %s <N> <NUM_REQUESTS> <NUM_STREAMS> <IF_CUDA_GRAPH> <N_ITERATIONS>\n", argv[0]);
        printf("  N             - Array size\n");
        printf("  NUM_REQUESTS  - Number of arrays to process\n");
        printf("  NUM_STREAMS   - Number of CUDA streams\n");
        printf("  IF_CUDA_GRAPH - Use CUDA graphs (0=no, 1=yes)\n");
        printf("  N_ITERATIONS  - Number of times to repeat the kernel execution\n");
        return 1;
    }
    
    // Parse command-line arguments
    const int N = atoi(argv[1]);
    const int NUM_REQUESTS = atoi(argv[2]);
    const int NUM_STREAMS = atoi(argv[3]);
    const int IF_CUDA_GRAPH = atoi(argv[4]);
    const int N_ITERATIONS = atoi(argv[5]);

    printf("Running with N=%d, NUM_REQUESTS=%d, NUM_STREAMS=%d, IF_CUDA_GRAPH=%d, N_ITERATIONS=%d\n\n", 
        N, NUM_REQUESTS, NUM_STREAMS, IF_CUDA_GRAPH, N_ITERATIONS);
    
    cudaGraphExec_t graphExecs[NUM_REQUESTS];
    // Initialize to NULL
    for (int i = 0; i < NUM_REQUESTS; i++) {
        graphExecs[i] = NULL;
    }
    cudaGraph_t graph;

    
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
    

    if (IF_CUDA_GRAPH) {
        cudaStream_t captureStream; 
        CUDA_CHECK(cudaStreamCreate(&captureStream));
        printf("Creating CUDA graphs...\n");
        for (int req = 0; req < NUM_REQUESTS; req++) {
            CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));
            solve(N, &captureStream, d_arrays[req]);  // Must use captureStream here!
            CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&(graphExecs[req]), graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        CUDA_CHECK(cudaStreamDestroy(captureStream));
        printf("All graphs created successfully\n");
    }

    
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Launching kernels\n");
    cudaEventRecord(start);
    for (int iteration = 0; iteration < N_ITERATIONS; iteration++) {
        for (int req = 0; req < NUM_REQUESTS; req++) {
            if (IF_CUDA_GRAPH){
                CUDA_CHECK(cudaGraphLaunch(graphExecs[req], streams[req % NUM_STREAMS]));
                
            } else {
                solve(N, &(streams[req % NUM_STREAMS]), d_arrays[req]);
            }
        }
        cudaDeviceSynchronize();
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
    if (IF_CUDA_GRAPH) {
        for (int req = 0; req < NUM_REQUESTS; req++) {
            cudaGraphExecDestroy(graphExecs[req]);
        }
    }
    return 0;
}
