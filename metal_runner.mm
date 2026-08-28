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

    std::cout << "GPU: "
              << [[device name] UTF8String]
              << '\n';
    
    // create a command queue

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    if (!commandQueue) {
        std::cerr < "Failed to create command queue\n";
        return 1;
    }

    // load the compiled Metal library
    // TODO continue from here
}