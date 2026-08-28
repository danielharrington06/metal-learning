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

    return 0;
}