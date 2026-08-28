#import <Metal/Metal.h>

#include <iostream>
#include <vector>

int main() {
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

    NSURL* libraryURL =  [NSURL fileURLWithPath:@"./vector_add.metallib"];

    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];

    // i will replace the loading mechanism soon but for now report the situation
    if (!library) {
        std::cerr << "Failed to load Metal library.\n";

        if (error) {
            std::cerr << [[error localizedDescription] UTF8String] << '\n';
        }

        return 1;
    }

    // find vector_add function
    if<MTLFunction> function = [library newFunctionWithName:@"vector_add"]

    if (!function) {
        std::cerr << "failed to find vector_add function.\n";
        return 1
    }

    // create the compute pipeline
    id<MTLComputePipelineState> pipeline = [device newComputPipelineStateWithFunction:function error:&error];

    if (!pipeline) {
        std:cerr << "Failed to create compute pipeline.\n";

        if (error) {
            std::cerr << [[error locaizedDescription] UTF8String] << '\n';
        }
        return 1;
    }

    // input data
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    std::vector<float> result(4);

    // buffers
    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data() length: a.size() * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data() length: b.size() * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength: result.size() * sizeof(float) options: MTLResourceStorageModeShared];
    // use MTLResourceStorageModeShared because Apple Silicon uses unified memory

    // create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer]

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
    [encoder setComputePipelineState:pipeline]

    // give kernel its buffers
    [encoder setBuffer: bufferA offset:0 atIndex:0]
    [encoder setBuffer: bufferB offset:0 atIndex:1]
    [encoder setBuffer: bufferResult offset:0 atIndex:2]

    // dispatch GPU threads
    NSUInteger count = a.size(); // for a vector of n elements, count is n
    [encoder dispatchThreads: MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]

    // fimish and submit the work
    [encoder endEncoding]; // finished recording commands
    [commandBuffer commit]; // submit the command buffer to GPU
    [commandBuffer waitUntilCompleted]; // wait for the GPU to finish before we try to read the result

    // read the result
    float* output = static_cast<float*>(bufferResult.contents);
    for (NSUInteger i = 0; i < count; i++) {
        std::cout << output[i] << ' ';
    }
    
    std::cout << '\n';

    return 0;
}