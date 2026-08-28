#import <Metal/Metal.h>

#include <cmath>
#include <iostream>
#include <vector>

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

    NSURL* libraryURL =  [NSURL fileURLWithPath:@"./bin/vector_transform.metallib"];

    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];

    if (!library) {
        std::cerr << "Failed to load Metal library.\n";

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }

        return 1;
    }

    // find vector_transform function (kernel)
    id<MTLFunction> function = [library newFunctionWithName:@"vector_transform"];

    if (!function) {
        std::cerr << "failed to find vector_transform function.\n";
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
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data() length: input.size() * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength: input.size() * sizeof(float) options: MTLResourceStorageModeShared];
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
    [encoder setBuffer: inputBuffer offset:0 atIndex:0];
    [encoder setBuffer: outputBuffer offset:0 atIndex:1];

    // dispatch GPU threads
    NSUInteger count = input.size(); // for a vector of n elements, count is n
    [encoder dispatchThreads: MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    // fimish and submit the work
    [encoder endEncoding]; // finished recording commands
    [commandBuffer commit]; // submit the command buffer to GPU
    [commandBuffer waitUntilCompleted]; // wait for the GPU to finish before we try to read the result

    // read the GPU result
    float* gpuResult = static_cast<float*>(outputBuffer.contents);

    for (NSUInteger i = 0; i < count; i++) {
        std::cout << gpuResult[i] << '\n';
    }

    return 0;
}