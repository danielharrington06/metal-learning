#import <Metal/Metal.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

float dotProductCPU(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }

    return result;
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

    // test data
    const size_t count = 10'000'000;

    std::vector<float> a(count);
    std::vector<float> b(count);

    for (size_t i = 0; i < count; i++) {
        a[i] = static_cast<float>(i % 100);
        b[i] = static_cast<float>((i*3) % 100);
    }

    // --- CPU computation

    // first execute on CPU
    float cpuResult = dotProductCPU(a, b);

    std::cout << "CPU result: "
              << cpuResult
              << '\n';

    // --- get the GPU
    id<MTLDevice> device = createDevice();

    if (!device) {
        return 1;
    }

    std::cout << "GPU: " << [[device name] UTF8String] << '\n';
    
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

    NSUInteger count = a.size();

    // threadgroups and size
    NSUInteger threadgroupSize = std::min(multiplicationPipeline.maxTotalThreadsPerThreadgroup, count);
    NSUInteger numberOfThreadgroups = (count + threadgroupSize - 1) / threadgroupSize;

    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data() length: count * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data() length: count * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferProducts = [device newBufferWithLength: count * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reductionBufferA = [device newBufferWithLength: count * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reductionBufferB = [device newBufferWithLength: count * sizeof(float) options:MTLResourceStorageModeShared];

    // create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    if (!commandBuffer) {
        std::cerr << "Failed to create command buffer.\n";
        return 1;
    }

    // --- multiplication

    // create a compute command encoder
    id<MTLComputeCommandEncoder> multiplicationEncoder = [commandBuffer computeCommandEncoder];

    if (!multiplicationEncoder) {
        std::cerr << "Failed to create multiplication compute encoder.\n";
        return 1;
    }

    // tell encoder what kernel to run
    [multiplicationEncoder setComputePipelineState:multiplicationPipeline];

    // give kernel its buffers
    [multiplicationEncoder setBuffer:bufferA offset:0 atIndex:0];
    [multiplicationEncoder setBuffer:bufferB offset:0 atIndex:1];
    [multiplicationEncoder setBuffer:bufferProducts offset:0 atIndex:2];

    [multiplicationEncoder setBytes:&count length:sizeof(count) atIndex:3];

    // dispatch GPU threads
    [multiplicationEncoder dispatchThreadgroups: MTLSizeMake(numberOfThreadgroups, 1, 1) threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1)];

    // fimish and submit the work
    [multiplicationEncoder endEncoding]; // finished recording commands

    // --- reduction

    NSUInteger currentCount = count;

    id<MTLBuffer> currentInput = bufferProducts;
    id<MTLBuffer> currentOutput = reductionBufferA;

    while (currentCount > 1) {
        NSUInteger currentThreadgroups = (currentCount + threadgroupSize - 1)/threadgroupSize;

        id<MTLComputeCommandEncoder> reductionEncoder = [commandBuffer computeCommandEncoder];

        if (!reductionEncoder) {
            std::cerr << "Failed to create reduction encoder.\n";
            return 1;
        }

        [reductionEncoder setComputePipelineState:reductionPipeline];
        [reductionEncoder setBuffer:currentInput offset:0 atIndex: 0];
        [reductionEncoder setBuffer:currentOutput offset:0 atIndex: 1];
        [reductionEncoder setBytes:&currentCount length:sizeof(currentCount) atIndex: 2];
        [reductionEncoder setBytes:&threadgroupSize length:sizeof(threadgroupSize) atIndex: 3];

        [reductionEncoder dispatchThreadgroups:MTLSizeMake(currentThreadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];

        [reductionEncoder endEncoding];

        currentCount = currentThreadgroups;

        std::swap(currentInput, currentOutput);
    }

    // --- execute GPU work

    [commandBuffer commit]; // submit the command buffer to GPU
    [commandBuffer waitUntilCompleted]; // wait for the GPU to finish before we try to read the result

    // --- read results
    float* gpuResult = static_cast<float*>(currentInput.contents);

    std::cout << "GPU result:" << gpuResult[0] << '\n';

    return 0;
}