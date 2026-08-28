#import <Metal/Metal.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

float dotProductCPU(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i]
    }

    return result;
}

int main() {
    // test data
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {5, 9, -5, 2};

    // first execute on CPU
    float cpuResult = dotProductCPU(a, b);

    std::cout << "CPU result: "
              << cpuResult
              << '\n';

    // get the GPU
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        std::cerr << "Failed to find a Metal device.\n";
        return 1;
    }

    std::cout << "GPU: " << [[device name] UTF8String] << '\n';
    
    // create a command queue

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    if (!commandQueue) {
        std::cerr << "Failed to create command queue\n";
        return 1;
    }

    // load the compiled Metal library
    NSError* error = nil;

    NSURL* libraryURL =  [NSURL fileURLWithPath:@"./bin/dot_product.metallib"];

    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];

    if (!library) {
        std::cerr << "Failed to load Metal library.\n";

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }

        return 1;
    }

    // find multiply_vectors function (kernel)
    id<MTLFunction> function = [library newFunctionWithName:@"multiply_vectors"];

    if (!function) {
        std::cerr << "failed to find multiply_vectors function.\n";
        return 1;
    }

    // create the compute pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    if (!pipeline) {
        std::cerr << "Failed to create compute pipeline.\n";

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }
        return 1;
    }

    // buffers
    NSUInteger count = a.size();
    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data() length: count * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data() length: count * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferProducts = [device newBufferWithLength: count * sizeof(float) options: MTLResourceStorageModeShared];
    // use MTLResourceStorageModeShared because Apple Silicon uses unified memory

    // create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    if (!commandBuffer) {
        std::cerr << "Failed to create command buffer.\n";
        return 1;
    }

    // create a compute command encoder
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (!encoder) {
        std::cerr << "Failed to create compute encoder.\n";
        return 1;
    }

    // tell encoder what kernel to run
    [encoder setComputePipelineState:pipeline];

    // give kernel its buffers
    [encoder setBuffer: bufferA offset:0 atIndex:0];
    [encoder setBuffer: bufferB offset:0 atIndex:1];
    [encoder setBuffer: bufferProducts offset:0 atIndex:2];

    [encoder setBytes:&count length:sizeof(count) atIndex:3]; // TODO work out what this does

    // calculate threadgroup size
    NSUInteger threadgroupSize = pipeline.maxTotalThreadsPerThreadgroup
    threadgroupSize = std::min(threadgroupSize, count);

    // dispatch GPU threads
    NSUInteger threadgroups = (count + threadgroupSize - 1) / threadgroupSize;
    [encoder dispatchThreadgroups: MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1)];

    // fimish and submit the work
    [encoder endEncoding]; // finished recording commands
    [commandBuffer commit]; // submit the command buffer to GPU
    [commandBuffer waitUntilCompleted]; // wait for the GPU to finish before we try to read the result

    // read the GPU result
    float* gpuResult = static_cast<float*>(bufferProducts.contents);

    std::cout << "GPU partial products:\n";

    for (NSUInteger i = 0; i < count; ++i) {
        std::cout << products[i] << ' ';
    }

    std::cout << '\n';

    return 0;
}