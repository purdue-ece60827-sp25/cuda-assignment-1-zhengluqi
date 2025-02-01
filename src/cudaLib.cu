
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_kernel(float* x, float* y, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = factor * x[idx] + y[idx];
    }
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
    size_t bytes = vectorSize * sizeof(float);
    float *d_x, *d_y, factor = 2.0f;

    std::vector<float> h_x(vectorSize), h_y(vectorSize), h_result(vectorSize);
    vectorInit(h_x.data(), vectorSize);
    vectorInit(h_y.data(), vectorSize);
    std::copy(h_y.begin(), h_y.end(), h_result.begin());

    gpuAssert(cudaMalloc(&d_x, bytes), __FILE__, __LINE__);
    gpuAssert(cudaMalloc(&d_y, bytes), __FILE__, __LINE__);

    gpuAssert(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    gpuAssert(cudaMemcpy(d_y, h_result.data(), bytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    int threads = 256;
    int blocks = (vectorSize + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads>>>(d_x, d_y, factor, vectorSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    gpuAssert(cudaMemcpy(h_result.data(), d_y, bytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    int errors = verifyVector(h_x.data(), h_y.data(), h_result.data(), factor, vectorSize);
    std::cout << "Errors found: " << errors << " out of " << vectorSize << "\n";

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pSumSize) return;

    uint64_t hit_count = 0;
    curandState_t rng;
    curand_init(clock64(), idx, 0, &rng);

    for (int i = 0; i < sampleSize; ++i) {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        if ((x * x + y * y) <= 1.0f) {
            ++hit_count;
        }
    }
    pSums[idx] = hit_count;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t numThreads = pSumSize / reduceSize;

    if (idx >= numThreads) return;

    uint64_t localSum = 0;
    for (uint64_t j = 0; j < reduceSize; ++j) {
        localSum += pSums[idx * reduceSize + j];
    }
    totals[idx] = localSum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	double approxPi = 0;
	
	//      Insert code here
 // Allocate device memory for pSums and totals
    size_t genMemSize = generateThreadCount * sizeof(uint64_t);
    size_t redMemSize = reduceThreadCount * sizeof(uint64_t);
    std::vector<uint64_t> hostTotals(reduceThreadCount);
    uint64_t* d_pSums = nullptr;
    uint64_t* d_totals = nullptr;

    gpuAssert(cudaMalloc(&d_pSums, genMemSize), __FILE__, __LINE__);
    gpuAssert(cudaMalloc(&d_totals, redMemSize), __FILE__, __LINE__);

    dim3 genGridDim((generateThreadCount + 127) / 128, 1, 1);
    dim3 genBlockDim(128, 1, 1);
    generatePoints<<<genGridDim, genBlockDim>>>(d_pSums, generateThreadCount, sampleSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    dim3 redGridDim((reduceThreadCount + 127) / 128, 1, 1);
    dim3 redBlockDim(128, 1, 1);
    reduceCounts<<<redGridDim, redBlockDim>>>(d_pSums, d_totals, generateThreadCount, reduceSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    gpuAssert(cudaMemcpy(hostTotals.data(), d_totals, redMemSize, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    uint64_t totalHits = 0;
    for (uint64_t i = 0; i < reduceThreadCount; ++i) {
        totalHits += hostTotals[i];
    }

    approxPi = (static_cast<double>(totalHits) / 
                 (static_cast<double>(generateThreadCount) * sampleSize)) * 4.0;

    cudaFree(d_pSums);
    cudaFree(d_totals);
    return approxPi;
}
