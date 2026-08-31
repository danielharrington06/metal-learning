#import <Metal/Metal.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

double dotProductCPU(const std::vector<float>& a, const std::vector<float>& b) {
    double result = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        result += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return result;
}

double dotProductGPU(
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> multiplicationPipeline,
    id<MTLComputePipelineState> reductionPipeline,
    id<MTLBuffer> bufferA,
    id<MTLBuffer> bufferB,
    id<MTLBuffer> bufferProducts,
    id<MTLBuffer> reductionBufferA,
    id<MTLBuffer> reductionBufferB,
    NSUInteger elementCount,
    NSUInteger threadgroupSize
) {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Multiplication
    id<MTLComputeCommandEncoder> multiplicationEncoder = [commandBuffer computeCommandEncoder];

    [multiplicationEncoder setComputePipelineState:multiplicationPipeline];

    [multiplicationEncoder setBuffer:bufferA offset:0 atIndex:0];
    [multiplicationEncoder setBuffer:bufferB offset:0 atIndex:1];
    [multiplicationEncoder setBuffer:bufferProducts offset:0 atIndex:2];
    [multiplicationEncoder setBytes:&elementCount length:sizeof(elementCount) atIndex:3];

    NSUInteger numberOfThreadgroups = (elementCount + threadgroupSize - 1) / threadgroupSize;

    [multiplicationEncoder dispatchThreadgroups: MTLSizeMake(numberOfThreadgroups, 1, 1) threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1)];
    
    [multiplicationEncoder endEncoding];

    // Reduction
    NSUInteger currentCount = elementCount;

    id<MTLBuffer> currentInput = bufferProducts;
    id<MTLBuffer> currentOutput = reductionBufferA;

    while (currentCount > 1) {

        NSUInteger currentThreadgroups = (currentCount + threadgroupSize - 1) / threadgroupSize;

        id<MTLComputeCommandEncoder> reductionEncoder = [commandBuffer computeCommandEncoder];

        [reductionEncoder setComputePipelineState:reductionPipeline];

        [reductionEncoder setBuffer:currentInput offset:0 atIndex:0];
        [reductionEncoder setBuffer:currentOutput offset:0 atIndex:1];

        [reductionEncoder setBytes:&currentCount length:sizeof(currentCount) atIndex:2];

        [reductionEncoder setBytes:&threadgroupSize length:sizeof(threadgroupSize) atIndex:3];

        [reductionEncoder dispatchThreadgroups: MTLSizeMake(currentThreadgroups, 1, 1) threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1)];

        [reductionEncoder endEncoding];

        currentCount = currentThreadgroups;
        std::swap(currentInput, currentOutput);
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    float* result = static_cast<float*>(currentInput.contents);

    return result[0];
}

id<MTLDevice> createDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Failed to find a Metal device.\n";
    }
    return device;
}

id<MTLLibrary> loadLibrary(id<MTLDevice> device) {
    NSError* error = nil;

    NSURL* libraryURL =  [NSURL fileURLWithPath:@"./bin/dot_product.metallib"];

    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];

    if (!library) {
        std::cerr << "Failed to load Metal library.\n";

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }
    }
    return library;
}

id<MTLComputePipelineState> createPipeline(id<MTLDevice> device, id<MTLLibrary> library, const char* functionName) {
    id<MTLFunction> function = [library newFunctionWithName: [NSString stringWithUTF8String:functionName]];

    if (!function) {
        std::cerr << "Failed to find Metal function: " << functionName << '\n';

        return nil;
    }

    NSError* error = nil;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction: function error:&error];

    if (!pipeline) {
        std::cerr << "Failed to create pipeline for: " << functionName << '\n';

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }
    }

    return pipeline;
}

int main() {

    // variables
    const size_t count = 10'000'000;
    const size_t warmupCount = 10;
    const size_t testCount = 100;

    using Clock = std::chrono::high_resolution_clock;
    auto cpuStart = Clock::now();

    std::vector<double> cpuTimes(testCount);
    std::vector<double> gpuTimes(testCount);

    std::vector<float> a(count);
    std::vector<float> b(count);

    for (size_t i = 0; i < count; i++) {
        a[i] = static_cast<float>(i % 100);
        b[i] = static_cast<float>((i*3) % 100);
    }

    // --- get the GPU
    id<MTLDevice> device = createDevice();

    if (!device) {
        return 1;
    }
    
    // create a command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    if (!commandQueue) {
        std::cerr << "Failed to create command queue\n";
        return 1;
    }

    // load Metal library
    id<MTLLibrary> library = loadLibrary(device);

    if (!library) {
        return 1;
    }

    // create compute pipelines

    id<MTLComputePipelineState> multiplicationPipeline = createPipeline(device, library, "multiply_vectors");

    if (!multiplicationPipeline) {
        return 1;
    }

    id<MTLComputePipelineState> reductionPipeline = createPipeline(device, library, "reduce_sum");

    if (!reductionPipeline) {
        return 1;
    }

    // --- setup buffers

    NSUInteger elementCount = a.size();

    // threadgroups and size
    NSUInteger threadgroupSize = std::min(multiplicationPipeline.maxTotalThreadsPerThreadgroup, elementCount);
    NSUInteger numberOfThreadgroups = (elementCount + threadgroupSize - 1) / threadgroupSize;

    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data() length: elementCount * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data() length: elementCount * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferProducts = [device newBufferWithLength: elementCount * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reductionBufferA = [device newBufferWithLength: elementCount * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reductionBufferB = [device newBufferWithLength: elementCount * sizeof(float) options:MTLResourceStorageModeShared];

    for (int i = 0; i < warmupCount; i++) {
        dotProductCPU(a, b);
        dotProductGPU(commandQueue, multiplicationPipeline, reductionPipeline, bufferA, bufferB, bufferProducts, reductionBufferA, reductionBufferB, elementCount, threadgroupSize);
    }

    double cpuResult;
    double gpuResult;

    for (int i = 0; i < testCount; i++) {
        // CPU
        auto cpuStart = Clock::now();

        cpuResult = dotProductCPU(a, b);

        auto cpuEnd = Clock::now();

        cpuTimes[i] = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

        // GPU
        auto gpuStart = Clock::now();

        gpuResult = dotProductGPU(
            commandQueue,
            multiplicationPipeline,
            reductionPipeline,
            bufferA,
            bufferB,
            bufferProducts,
            reductionBufferA,
            reductionBufferB,
            elementCount,
            threadgroupSize
        );

        auto gpuEnd = Clock::now();

        gpuTimes[i] =
            std::chrono::duration<double, std::milli>(
                gpuEnd - gpuStart
            ).count();
    }

    double cpuAverage = std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0) / testCount;

    double gpuAverage = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0) / testCount;

    std::cout << "Dot Product Benchmark\n";
    std::cout << "Vectors of size " << elementCount << '\n';
    std::cout << warmupCount << " Warmup Runs\n";
    std::cout << testCount << " Test Runs\n\n";

    std::cout << "CPU Time Average: " << cpuAverage << "ms\n";
    std::cout << "GPU Time Average: " << gpuAverage << "ms\n";

    double speedup = cpuAverage / gpuAverage;
    std::cout << "GPU speedup: " << speedup << "x\n";

    return 0;
}