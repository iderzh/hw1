#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#else
/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>
#endif

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000

typedef unsigned char uchar;
#define OUT

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        return 1;                                                                           \
    }                                                                                       \
} while (0)
#else
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)
#endif

#define SQR(a) ((a) * (a))

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
double inline get_time_msec (void) {
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1000.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }

    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount();
    }
}
#else
double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}
#endif

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

__host__ __device__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__host__ __device__ uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

__host__ float histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    float distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((float)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

/* Your __device__ functions and __global__ kernels here */
__global__ void image_to_histogram_batch(uchar *imagearray1, uchar *imagearray2, OUT float *distance) {
    __shared__ uchar local_image1[IMG_DIMENSION * IMG_DIMENSION], local_image2[IMG_DIMENSION * IMG_DIMENSION];
    __shared__ int local_hist1[256], local_hist2[256];
    const int i = threadIdx.x % IMG_DIMENSION;
    const int j = threadIdx.x / IMG_DIMENSION;
    const unsigned long index_image = blockIdx.x * blockDim.x + threadIdx.x;

    local_image1[threadIdx.x] = imagearray1[index_image];
    local_image2[threadIdx.x] = imagearray2[index_image];
    if (threadIdx.x < 256) {
    	local_hist1[threadIdx.x] = 0;
    	local_hist2[threadIdx.x] = 0;
    }
    __threadfence();

    uchar pattern1 = local_binary_pattern(local_image1, i, j);
    uchar pattern2 = local_binary_pattern(local_image2, i, j);
	atomicAdd(&local_hist1[pattern1], 1);
	atomicAdd(&local_hist2[pattern2], 1);
	__syncthreads();

    float ldistance;
    if (threadIdx.x < 256) {
    	if ((local_hist1[threadIdx.x] + local_hist2[threadIdx.x]) !=0 ) {
    		ldistance = ((float)SQR(local_hist1[threadIdx.x] - local_hist2[threadIdx.x])) / (local_hist1[threadIdx.x] + local_hist2[threadIdx.x]);
    	}
    	atomicAdd(distance, ldistance);
    }
}

__global__ void image_to_histogram_sharedm(uchar *image, OUT int *hist) {
    __shared__ uchar local_image[IMG_DIMENSION * IMG_DIMENSION];
    __shared__ int local_hist[256];
    const int i = threadIdx.x % IMG_DIMENSION;
    const int j = threadIdx.x / IMG_DIMENSION;
    const int h = threadIdx.x / 4;
    local_image[threadIdx.x] = image[threadIdx.x];
    local_hist[h] = 0;
    __threadfence();
    uchar pattern = local_binary_pattern(local_image, i, j);
    atomicAdd(&local_hist[pattern], 1);
    __syncthreads();
    if (threadIdx.x < 256) {
        hist[threadIdx.x] = local_hist[threadIdx.x];
    }
}

__global__ void image_to_histogram_simple(uchar *image1, OUT int *hist1) {
    const int i = threadIdx.x % IMG_DIMENSION;
    const int j = threadIdx.x / IMG_DIMENSION;
    uchar pattern = local_binary_pattern(image1, i, j);
    atomicAdd(&hist1[pattern], 1);
}

__global__ void histogram_distance(int *hist1, int *hist2, OUT float *distance) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float ldistance;
    if ((hist1[i] + hist2[i]) !=0 ) {
        ldistance = ((float)SQR(hist1[i] - hist2[i])) / (hist1[i] + hist2[i]);
    }
    atomicAdd(distance, ldistance);
}

int main() {
    uchar *images1;
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance;
#if 1
    /* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        image_to_histogram(&images1[i * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[i * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);
#endif
#if 1
    /* using GPU task-serial */
    printf("\n=== GPU Task Serial ===\n");
    total_distance = 0.0;

    do {
        uchar *gpu_image1, *gpu_image2;
        int *gpu_hist1, *gpu_hist2;
        float *gpu_hist_distance;
        float cpu_hist_distance;

        CUDA_CHECK( cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_hist1, sizeof (int) * 256) );
        CUDA_CHECK( cudaMalloc(&gpu_hist2, sizeof (int) * 256) );
        CUDA_CHECK( cudaMalloc(&gpu_hist_distance, sizeof (float)) );

        t_start = get_time_msec();
        for (int i = 0; i < N_IMG_PAIRS; i++) {
            CUDA_CHECK( cudaMemcpy(gpu_image1, &images1[i * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(gpu_image2, &images2[i * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemset((void*)gpu_hist1, 0, sizeof (int) * 256) );
            CUDA_CHECK( cudaMemset((void*)gpu_hist2, 0, sizeof (int) * 256) );
            CUDA_CHECK( cudaMemset((void*)gpu_hist_distance, 0, sizeof (float)) );

            image_to_histogram_simple<<<1, 1024>>>(gpu_image1, gpu_hist1);
            image_to_histogram_simple<<<1, 1024>>>(gpu_image2, gpu_hist2);

            CUDA_CHECK(cudaDeviceSynchronize());

            histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);

            CUDA_CHECK( cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof (float), cudaMemcpyDeviceToHost) );
            total_distance += cpu_hist_distance;
        }
        t_finish = get_time_msec();
        printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        printf("total time %f [msec]\n", t_finish - t_start);
        CUDA_CHECK( cudaFree(gpu_image1) );
        CUDA_CHECK( cudaFree(gpu_image2) );
        CUDA_CHECK( cudaFree(gpu_hist1) );
        CUDA_CHECK( cudaFree(gpu_hist2) );
        CUDA_CHECK( cudaFree(gpu_hist_distance) );
    } while (0);
#endif
#if 1
    /* using GPU task-serial + images and histograms in shared memory */
    printf("\n=== GPU Task Serial with shared memory ===\n");
    total_distance = 0.0;
    do {
        uchar *gpu_image1, *gpu_image2;
        int *gpu_hist1, *gpu_hist2;
        float *gpu_hist_distance;
        float cpu_hist_distance;

        CUDA_CHECK( cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_hist1, sizeof (int) * 256) );
        CUDA_CHECK( cudaMalloc(&gpu_hist2, sizeof (int) * 256) );
        CUDA_CHECK( cudaMalloc(&gpu_hist_distance, sizeof (float)) );

        t_start = get_time_msec();
        for (int i = 0; i < N_IMG_PAIRS; i++) {
            CUDA_CHECK( cudaMemcpy(gpu_image1, &images1[i * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(gpu_image2, &images2[i * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemset((void*)gpu_hist_distance, 0, sizeof (float)) );

            image_to_histogram_sharedm<<<1, 1024>>>(gpu_image1, gpu_hist1);
            image_to_histogram_sharedm<<<1, 1024>>>(gpu_image2, gpu_hist2);

            CUDA_CHECK(cudaDeviceSynchronize());

            histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);

            CUDA_CHECK( cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof (float), cudaMemcpyDeviceToHost) );
            total_distance += cpu_hist_distance;
        }
        t_finish = get_time_msec();
        CUDA_CHECK( cudaFree(gpu_image1) );
        CUDA_CHECK( cudaFree(gpu_image2) );
        CUDA_CHECK( cudaFree(gpu_hist1) );
        CUDA_CHECK( cudaFree(gpu_hist2) );
        CUDA_CHECK( cudaFree(gpu_hist_distance) );
    } while (0);
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);
#endif
#if 1
    /* using GPU + batching */
    printf("\n=== GPU Batching ===\n");
    total_distance = 0.0;
    do {
        uchar *gpu_image_array1, *gpu_image_array2;
        int *gpu_hist_array1, *gpu_hist_array2;
        float *gpu_total_distance;
        float cpu_hist_distance;

        CUDA_CHECK( cudaMalloc(&gpu_image_array1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_image_array2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION) );
        CUDA_CHECK( cudaMalloc(&gpu_hist_array1, sizeof (int) * 256 * N_IMG_PAIRS) );
        CUDA_CHECK( cudaMalloc(&gpu_hist_array2, sizeof (int) * 256 * N_IMG_PAIRS) );
        CUDA_CHECK( cudaMalloc(&gpu_total_distance, sizeof (float)) );

        CUDA_CHECK( cudaMemcpy(gpu_image_array1, images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(gpu_image_array2, images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemset(gpu_total_distance,0 , sizeof(float)));

        t_start = get_time_msec();
        image_to_histogram_batch<<<10000, IMG_DIMENSION * IMG_DIMENSION >>>(gpu_image_array1, gpu_image_array2, gpu_total_distance);
        CUDA_CHECK( cudaMemcpy(&cpu_hist_distance, gpu_total_distance, sizeof (float), cudaMemcpyDeviceToHost) );
        t_finish = get_time_msec();

        total_distance += cpu_hist_distance;

        CUDA_CHECK( cudaFree(gpu_image_array1) );
        CUDA_CHECK( cudaFree(gpu_image_array2) );
        CUDA_CHECK( cudaFree(gpu_hist_array1) );
        CUDA_CHECK( cudaFree(gpu_hist_array2) );
        CUDA_CHECK( cudaFree(gpu_total_distance) );
    } while (0);
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);
#endif

    CUDA_CHECK( cudaFreeHost(images1) );
    CUDA_CHECK( cudaFreeHost(images2) );
    CUDA_CHECK( cudaDeviceReset() );

    return 0;
}
